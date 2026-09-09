"""CodeExecutorAgent - 代码执行 Agent，在沙箱中运行论文代码。

对齐方案「Phase 4: 代码执行」：
- 两阶段执行：smoke test（短时冒烟，快速暴露环境问题）-> full run（完整运行）；
- 捕获标准输出、错误日志、退出码；
- 支持本地子进程（隔离临时目录 + 超时）与 Docker 容器两种沙箱。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List
from src.base_agent import BaseAgent
from src.llm.ollama_client import LLMClient
from src.agents.env_builder import PIP_INDEX_URL, PIP_FIND_LINKS

LOCAL_TIMEOUT_SMOKE = 10
LOCAL_TIMEOUT_FULL = 60
DOCKER_TIMEOUT_SMOKE = 30
DOCKER_TIMEOUT_FULL = 300

# markdown 代码块围栏（可能带 python 语言标注）
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# 行首残留的围栏/引号残片
_FENCE_LEFT = re.compile(r"^\s*(```+|>>>|\.\.\.)\s*", re.MULTILINE)
# 判定"看起来像 Python 代码行"的行首（\w 会匹配中文,故全部用 ASCII 白名单）
_CODE_LINE_START = re.compile(
    r"^\s*(?:"
    r"import\s|from\s|def\s|class\s|if\s|elif\s|else\s*:|for\s|while\s|"
    r"try\s*:|except\s|finally\s*:|with\s|return\s|print\s*\(|raise\s|"
    r"pass\s*$|break\s*$|continue\s*$|del\s|assert\s|global\s|nonlocal\s|"
    r"yield\s|match\s|case\s|lambda\s|@|#|"
    r"[A-Za-z_][A-Za-z0-9_.]*\s*=|"
    r"[A-Za-z_\[\(\"']|[\d+\-.]"
    r")")


class CodeExecutorAgent(BaseAgent):
    """在 Docker 或本地沙箱中执行论文代码。"""

    system_prompt = "在沙箱中安全执行论文代码,输出运行日志、数值结果与退出码"

    def __init__(self, llm_client: LLMClient, logger=None, use_docker: bool = False):
        super().__init__("CodeExecutor", logger)
        self.llm = llm_client
        self.use_docker = use_docker

    def run(self, input_data: dict) -> dict:
        """执行论文代码（smoke test + full run）。

        input_data: {"paper_info", "env_config", "resources", "code"(可选)}
        """
        self.log("execute_code", "START", "开始执行代码", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        env_config = input_data.get("env_config", {}) or {}
        code = input_data.get("code", "") or ""
        if not code:
            code = self._generate_code(paper_info)
        code = self._sanitize_code(code)
        self.env_config = env_config  # 供执行阶段选择镜像/依赖

        smoke = self._execute_code(code, stage="smoke")
        if not smoke["success"]:
            # smoke 失败：不浪费预算跑 full，返回诊断信息
            result = {"stages": [{"stage": "smoke", **smoke}],
                      "success": False, "final": smoke,
                      "code": code}
            self.log_experiment(
                "EXECUTE_CODE", "smoke test 失败,终止 full run",
                inputs={"code": code}, outputs=smoke,
                result={"success": False})
            self.log("execute_code", "ERROR",
                     f"smoke test 失败: {smoke.get('stderr', '')[:120]}",
                     {"stage": "smoke", "exit_code": smoke.get("exit_code")})
            return {**result, "llm_calls": self._delta_llm_calls()}

        full = self._execute_code(code, stage="full")
        stages = [{"stage": "smoke", **smoke}, {"stage": "full", **full}]
        result = {"stages": stages, "success": full["success"],
                  "final": full, "code": code}

        self.log_experiment(
            "EXECUTE_CODE", "完成 smoke + full 两阶段执行",
            inputs={"code": code},
            outputs={"smoke": smoke, "full": full},
            result={"success": full["success"]},
        )
        self.log("execute_code",
                 "SUCCESS" if full["success"] else "ERROR",
                 f"代码执行{'成功' if full['success'] else '失败'} "
                 f"(smoke 通过, full {'通过' if full['success'] else '失败'})",
                 {"smoke_exit": smoke.get("exit_code"),
                  "full_exit": full.get("exit_code"),
                  "stdout_tail": full.get("stdout", "")[-300:]})

        return {**result, "llm_calls": self._delta_llm_calls()}

    # ---------------- 代码生成 ----------------

    def _generate_code(self, paper_info: Dict) -> str:
        prompt = f"""根据论文信息生成一段简短的训练代码用于复现实验。
论文方法: {paper_info.get('method', '未知')}
指标: {paper_info.get('metrics', {})}
数据集: {paper_info.get('dataset', '未知')}

【关键输出约束 - 必须严格遵守】
1. 只输出一份可直接运行的 Python 脚本（完整训练+评估流程,最后打印关键指标）；
2. 输出的每一行都必须是合法 Python 代码,严禁出现任何解释性文字、中文叙述、
   说明语句或自然语言段落；
3. 不要使用 markdown 代码块围栏(``` 或 ```python)包裹输出,不要输出围栏标记；
4. 如需注释仅使用以 # 开头的 Python 注释；
5. 第一行直接开始写代码,不要有开场白。
"""
        return self.llm.chat(prompt, task="code_executor")

    def _sanitize_code(self, raw: str) -> str:
        """将 LLM 原始输出清洗为可执行的纯净 Python 代码。

        兜底处理两类常见污染：
        1. markdown 代码块围栏包裹(```python ... ```)；
        2. 代码块外/代码中的中文叙述行("为了...""假设我们使用..."等自然语言)。
        清洗后产物为纯代码文本;仍含语法错误时如实保留,由执行阶段判定。
        """
        if not raw or not raw.strip():
            return raw or ""
        text = raw.strip()

        # 1) 提取最长的 markdown 代码块（若被围栏包裹）
        fenced = _CODE_FENCE.findall(text)
        if fenced:
            text = max(fenced, key=len).strip()

        # 2) 逐行剥离叙述行,只保留代码行与代码内空行
        cleaned = []
        for ln in text.splitlines():
            stripped = ln.strip()
            if not stripped:
                if cleaned and cleaned[-1].strip():
                    cleaned.append(ln)
                continue
            # 中文叙述行一律丢弃（含中文且非 # 注释）
            if re.search(r"[\u4e00-\u9fff]", stripped) and not stripped.startswith("#"):
                continue
            if _CODE_LINE_START.match(stripped):
                cleaned.append(ln)
        code = "\n".join(cleaned).strip("\n")

        # 3) 语法兜底: 若整体不可编译,再去掉围栏残片/行首行号后重新过滤
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError:
            code = _FENCE_LEFT.sub("", code)
            lines = []
            for ln in code.splitlines():
                ln = re.sub(r"^\s*\d+\s+", "", ln).strip()
                if not ln:
                    continue
                if (_CODE_LINE_START.match(ln)
                        and not (re.search(r"[\u4e00-\u9fff]", ln)
                                 and not ln.startswith("#"))):
                    lines.append(ln)
            code = "\n".join(lines)
        return code

    # ---------------- 执行 ----------------

    def _execute_code(self, code: str, stage: str) -> Dict:
        """执行代码：本地子进程或 Docker 容器，按阶段使用不同超时。"""
        if self.use_docker:
            return self._execute_code_docker(code, stage)
        return self._execute_code_local(code, stage)

    def _execute_code_local(self, code: str, stage: str) -> Dict:
        """在本地临时文件中执行代码（隔离目录，限制超时）。"""
        workdir = tempfile.mkdtemp(prefix="autorepro_exec_")
        script = os.path.join(workdir, "run.py")
        timeout = LOCAL_TIMEOUT_SMOKE if stage == "smoke" else LOCAL_TIMEOUT_FULL
        try:
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=timeout,
                cwd=workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s, {stage})", "exit_code": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "exit_code": -2}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _execute_code_docker(self, code: str, stage: str) -> Dict:
        """在 Docker 容器中执行代码（挂载临时目录，隔离运行）。

        镜像选择：优先使用 env_config.image_tag（如流水线 EnvBuilder 已构建的
        autorepro-env 镜像，内含 requirements 依赖）；否则退回 python:3.11-slim，
        并把 env_config 中的 requirements 注入容器临时安装后执行。
        """
        docker_cmd = self._resolve_docker_cmd()
        if docker_cmd is None:
            return {"success": False, "stdout": "",
                    "stderr": "本机未安装 Docker 或不在 PATH 中", "exit_code": -3}

        env_config = getattr(self, "env_config", None) or {}
        image = env_config.get("image_tag") or "python:3.11-slim"
        reqs = (env_config.get("requirements_txt") or "").strip()
        if not reqs:
            pkgs = env_config.get("required_packages") or []
            if isinstance(pkgs, list):
                reqs = "\n".join(str(p) for p in pkgs if p).strip()

        workdir = tempfile.mkdtemp(prefix="autorepro_docker_")
        script = os.path.join(workdir, "run.py")
        timeout = DOCKER_TIMEOUT_SMOKE if stage == "smoke" else DOCKER_TIMEOUT_FULL
        try:
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            mount = workdir.replace("\\", "/")
            cmd = [docker_cmd, "run", "--rm",
                   "-v", f"{mount}:/app", "-w", "/app"]
            if image == "python:3.11-slim" and reqs:
                with open(os.path.join(workdir, "requirements.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(reqs)
                runner = ["sh", "-c",
                          f"pip install -i {PIP_INDEX_URL} "
                          f"--find-links {PIP_FIND_LINKS} "
                          "-r /app/requirements.txt -q && python run.py"]
            else:
                runner = ["python", "run.py"]
            cmd += [image] + runner
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s, {stage})", "exit_code": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "exit_code": -2}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ---------------- 内部工具 ----------------

    def _delta_llm_calls(self) -> int:
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    def extract_result_files(self, result: Dict) -> List[str]:
        """从执行产物中收集数值结果/文件（对齐方案的输出采集）。"""
        files = []
        for artifact in ("stdout", "stderr"):
            text = result.get(artifact, "") or ""
            for line in text.splitlines():
                if "=" in line and any(ch.isdigit() for ch in line):
                    files.append(line.strip())
        return files