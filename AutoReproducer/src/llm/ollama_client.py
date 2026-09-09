"""LLM 客户端 - 封装 OpenAI 兼容 Chat Completions API，支持 Mock 与真实（API）模式。

真实模式不再依赖本地 Ollama 部署，改用远程 LLM API（OpenAI 兼容接口，
不绑定厂商），通过环境变量或构造参数配置：

    LLM_BASE_URL   API 地址（含 /v1 前缀或网关根地址均可）：
                   例如 https://api.deepseek.com/v1
                       https://qianfan.baidubce.com/v2
                       https://api.openai.com/v1
                       http://localhost:11434   （Ollama 的 /v1 兼容端点）
    LLM_API_KEY    访问密钥；无鉴权的网关心跳 Authorization 头
    LLM_MODEL      模型名：deepseek-chat / ernie-4.0-8k / gpt-4o-mini / qwen2.5:7b 等
    LLM_TIMEOUT    请求超时秒数（默认 120）

Mock 模式按 `task` 任务标识精确分发确定性响应，杜绝关键词误命中；
`task` 缺省时回退到关键词匹配，兼容遗留调用点。
"""
import json
import os
import urllib.request
import urllib.error
from typing import Optional

# 任务标识 -> Mock 响应（确定性、可复现，用于演示完整流水线）
_MOCK_TASKS = {
    "paper_reader": {
        "title": "AutoReproducer: 基于多智能体协作的论文自动复现与优化系统",
        "authors": ["AutoReproducer Team"],
        "method": "多智能体协作框架，包含 1 个 Orchestrator 与 7 个专职 Agent，"
                  "以状态机驱动论文复现与优化闭环",
        "dependencies": ["Python 3.11+", "PyTorch", "Streamlit", "PyPDF2"],
        "metrics": {"accuracy": 0.85, "f1_score": 0.82},
        "dataset": "CIFAR-10",
        "code_url": "https://github.com/example/autoreproducer",
    },
    "resource_finder": {
        "code_repo_url": "https://github.com/example/repo",
        "alternative_repos": ["https://github.com/example/repo-alt"],
        "dataset_url": "https://example.com/dataset",
        "confidence": 0.7,
    },
    "env_builder": {
        "required_packages": ["torch>=2.0.0", "torchvision>=0.15.0",
                              "numpy>=1.24.0", "tqdm>=4.65.0"],
        "python_version": "3.11",
        "dockerfile": (
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install -r requirements.txt"
        ),
        "setup_commands": ["pip install -r requirements.txt"],
        "estimated_disk_gb": 3.0,
    },
    "code_executor": (
        "import numpy as np\n"
        "import time\n"
        "print('Training complete. Test accuracy: 85.2%')\n"
        "print('Final loss: 0.3120')\n"
    ),
    "result_validator": {
        "match": True,
        "differences": [],
        "confidence": 0.85,
        "analysis": "运行结果与论文声明基本一致(复现成功):准确率差异小于 0.5% 阈值",
    },
    "verifier": {"pass": True, "issues": [],
                 "fix_suggestions": [], "confidence": 0.9},
    "optimizer_arms": {
        "suggestions": [
            "将学习率从 0.01 降至 0.005 并配合余弦退火",
            "将 SGD 替换为 AdamW 并加入权重衰减 1e-4",
            "增加 Batch Normalization 层以稳定训练",
        ]
    },
}


def _env_or(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env_or(key, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


class LLMClient:
    """OpenAI 兼容 LLM API 客户端，支持 Mock 模式（默认演示）与真实(API)模式。

    真实模式向 ``{base_url}/chat/completions`` 发送 OpenAI 格式请求
    （body: model/messages/temperature/stream；Auth: Bearer key），按
    ``choices[0].message.content`` 解析返回值。base_url 已含 /v1、/v2 等
    版本前缀时自动兼容，不做重复拼接；未配置 base_url/model 时返回
    明确的配置错误提示，便于快速定位。
    """

    def __init__(self, base_url: str = "", model: str = "",
                 api_key: str = "", timeout: Optional[int] = None,
                 mock_mode: bool = False):
        self.base_url = (base_url or _env_or("LLM_BASE_URL")).rstrip("/")
        self.model = model or _env_or("LLM_MODEL")
        self.api_key = api_key or _env_or("LLM_API_KEY")
        self.timeout = timeout or _env_int("LLM_TIMEOUT", 120)
        self.mock_mode = mock_mode
        self.call_count = 0

    def chat(self, prompt: str, system_prompt: str = "",
             temperature: float = 0.3, task: str = "") -> str:
        """发送聊天请求到 LLM API。

        Args:
            prompt: 用户提示词。
            system_prompt: 系统提示词（真实模式生效）。
            temperature: 采样温度。
            task: 任务标识（如 "paper_reader"）。Mock 模式下按此精确分发；
                  真实模式仅用于统计，不参与调用。
        """
        self.call_count += 1
        if self.mock_mode:
            return self._mock_response(prompt, task)
        return self._real_chat(prompt, system_prompt, temperature)

    # ---------------- 真实模式（远程 API） ----------------

    def _endpoint(self) -> str:
        """OpenAI 兼容 Chat Completions 端点（兼容 /v1、/v2 前缀，不重复拼接）。"""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1") or base.endswith("/v2"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _real_chat(self, prompt: str, system_prompt: str,
                   temperature: float) -> str:
        """真实 API 调用（OpenAI Chat Completions 格式，非流式）。"""
        if not self.base_url:
            return ("[LLM API Error: 未配置 base_url——请传入 base_url 参数或"
                    "设置环境变量 LLM_BASE_URL]")
        if not self.model:
            return ("[LLM API Error: 未配置 model——请传入 model 参数或"
                    "设置环境变量 LLM_MODEL]")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(self._endpoint(), data=data,
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                try:
                    return result["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    return f"[LLM API Error: 响应缺少 choices[0].message.content: {json.dumps(result, ensure_ascii=False)[:200]}]"
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            return f"[LLM API Error: HTTP {e.code} {body}]"
        except Exception as e:
            return f"[LLM API Error: {e}]"

    # ---------------- Mock 模式 ----------------

    def _mock_response(self, prompt: str, task: str = "") -> str:
        """Mock 响应：优先按 task 精确分发；缺失时回退到关键词匹配。"""
        if task and task in _MOCK_TASKS:
            value = _MOCK_TASKS[task]
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

        # ------- 回退：关键词匹配（兼容未传 task 的调用点）-------
        if "质量验证器" in prompt or "待验证输出" in prompt:
            return json.dumps(_MOCK_TASKS["verifier"], ensure_ascii=False)
        if "比对论文声明" in prompt or "提取到的实际指标" in prompt:
            return json.dumps(_MOCK_TASKS["result_validator"], ensure_ascii=False)
        if "代码仓库" in prompt and "推测" in prompt:
            return json.dumps(_MOCK_TASKS["resource_finder"], ensure_ascii=False)
        if "运行环境配置" in prompt or "Dockerfile" in prompt:
            return json.dumps(_MOCK_TASKS["env_builder"], ensure_ascii=False)
        if "训练代码" in prompt or "生成Python代码" in prompt:
            return _MOCK_TASKS["code_executor"]
        if "优化建议" in prompt or "optimize" in prompt.lower():
            return json.dumps(_MOCK_TASKS["optimizer_arms"], ensure_ascii=False)
        if "提取结构化信息" in prompt or "论文内容" in prompt:
            return json.dumps(_MOCK_TASKS["paper_reader"], ensure_ascii=False)
        return f"[Mock Response] 收到请求: {prompt[:50]}..."

    def get_call_count(self) -> int:
        return self.call_count

    def reset_call_count(self) -> None:
        """重置调用计数（用于预算统计口径归零）。"""
        self.call_count = 0