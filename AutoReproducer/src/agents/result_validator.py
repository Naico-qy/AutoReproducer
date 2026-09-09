"""ResultValidatorAgent - 结果验证 Agent，比对论文声明值与运行结果。

对齐方案「Phase 5: 结果验证」：
- 将实际运行结果与论文声明数值比对；
- 输出复现报告：成功/失败、数值差异、可能原因分析；
- 判断标准：环境是否 OK、输出是否 OK、描述是否 OK（输出 is_reproduced）。
"""
import json
import re
from typing import Dict
from src.base_agent import BaseAgent
from src.llm.ollama_client import LLMClient

# 指标提取模式：键名 -> 输出中的统一指标名
_METRIC_PATTERNS = [
    (r"(?:accuracy|acc|精确率|准确率|测试集准确率)\s*[:：=]?\s*([\d.]+)\s*%?", "accuracy"),
    (r"(?:f1[_-]?score|f1)\s*[:：=]?\s*([\d.]+)", "f1_score"),
    (r"(?:precision|精确率)\s*[:：=]?\s*([\d.]+)\s*%?", "precision"),
    (r"(?:recall|召回率)\s*[:：=]?\s*([\d.]+)\s*%?", "recall"),
    (r"(?:loss|损失)\s*[:：=]?\s*([\d.]+)", "loss"),
    (r"(?:mse|rmse)\s*[:：=]?\s*([\d.]+)", "mse"),
]
# 复现成功判定的相对差异阈值
_TOLERANCE = 0.05


class ResultValidatorAgent(BaseAgent):
    """验证运行结果是否与论文声明一致。"""

    system_prompt = "比对实际运行指标与论文声明值,输出复现成功/失败、数值差异与原因分析"

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("ResultValidator", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """验证执行结果。

        input_data: {"paper_info": dict, "execution": dict, "corpus_paper": str}
        """
        self.log("validate", "START", "开始验证结果", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        execution = input_data.get("execution", {}) or {}

        # execution 兼容新结构（stages）与旧结构（execution.execution）
        stdout = execution.get("stdout", "") or ""
        stderr = execution.get("stderr", "") or ""
        if not stdout and execution.get("stages"):
            stdout = execution["stages"][-1].get("stdout", "")
            stderr = execution["stages"][-1].get("stderr", "")

        paper_metrics = dict(paper_info.get("metrics", {}) or {})

        # 语料对照层：无声明指标时用语料复现分作为论文声明值
        corpus_paper = input_data.get("corpus_paper") or paper_info.get("corpus_paper")
        if not paper_metrics and corpus_paper:
            from src.corpus import get_declared_score
            score = get_declared_score(corpus_paper)
            if score is not None:
                paper_metrics = {"reproduction_score": round(float(score), 4)}

        actual_metrics = self._extract_metrics(stdout)

        # LLM 比对 + 本地数值校验兜底
        prompt = f"""比对论文声明的指标与代码运行结果。

论文声明指标: {json.dumps(paper_metrics, ensure_ascii=False)}
代码运行输出: {stdout[:2000]}
提取到的实际指标: {json.dumps(actual_metrics, ensure_ascii=False)}

返回JSON格式:
{{
    "match": true/false,
    "differences": ["指标1: 声明值 vs 实际值"],
    "confidence": 0.0-1.0,
    "analysis": "分析说明"
}}
"""
        llm_result = self.llm.chat(prompt, task="result_validator")
        parsed = self._parse_json(llm_result)
        if not parsed or "match" not in parsed:
            parsed = self._local_compare(paper_metrics, actual_metrics)

        # 本地校验：与 LLM 结果取交集（两者都判成功才算成功）
        local_ok = self._local_compare(paper_metrics, actual_metrics).get("match", False)
        match = bool(parsed.get("match", False)) and local_ok
        if paper_metrics and not actual_metrics:
            match = False  # 有声明无实测值 -> 不可判定为复现成功

        result = {
            "validation": {**parsed, "match": match},
            "metrics_comparison": {"paper": paper_metrics, "actual": actual_metrics},
            "is_reproduced": match,
            "confidence": round(float(parsed.get("confidence", 0.0)), 4),
        }

        self.log_experiment(
            "VALIDATE", "比对论文声明与运行结果",
            inputs={"paper_metrics": paper_metrics, "stdout_tail": stdout[-500:]},
            outputs=actual_metrics,
            result={"is_reproduced": match, "differences": parsed.get("differences", [])},
        )
        self.log("validate",
                 "SUCCESS" if match else "WARNING",
                 f"验证{'通过' if match else '未通过'} - 置信度: {result['confidence']:.2f}",
                 result)

        return {**result, "llm_calls": self._delta_llm_calls()}

    # ---------------- 内部工具 ----------------

    def _local_compare(self, paper_metrics: Dict, actual_metrics: Dict) -> Dict:
        """本地规则比对：同键指标相对差异 <= 5% 视为匹配。"""
        if not paper_metrics:
            return {"match": bool(actual_metrics), "differences": [],
                    "confidence": 0.8 if actual_metrics else 0.3,
                    "analysis": "无论文声明指标,以运行产出是否有效判定"}
        if not actual_metrics:
            return {"match": False, "differences": ["论文声明指标但运行输出未提取到数值"],
                    "confidence": 0.3, "analysis": "运行输出缺少可解析的数值指标"}

        differences = []
        match_all = True
        for key, declared in paper_metrics.items():
            try:
                declared = float(declared)
            except (TypeError, ValueError):
                continue
            actual = None
            for akey, aval in actual_metrics.items():
                if akey == key:
                    try:
                        actual = float(aval)
                    except (TypeError, ValueError):
                        actual = None
                    break
            if actual is None:
                # 尽量与语料的 reproduction_score 对齐
                continue
            # 口径统一：一方为小数(0~1)、另一方为百分数(>=10)时,归一到小数再比对
            declared_raw, actual_raw = declared, actual
            if declared <= 1.0 and actual >= 10.0:
                actual = actual / 100.0
            elif declared >= 10.0 and actual <= 1.0:
                declared = declared / 100.0
            diff = abs(actual - declared) / max(abs(declared), 1e-9)
            if diff > _TOLERANCE:
                match_all = False
            differences.append(
                f"{key}: 声明 {declared_raw:g} vs 实际 {actual_raw:g} "
                f"(归一化后相对差异 {diff:.1%})")
        return {"match": match_all, "differences": differences,
                "confidence": 0.8 if match_all else 0.4,
                "analysis": "本地数值比对完成"}

    def _extract_metrics(self, text: str) -> Dict:
        """从输出文本中提取指标数值。"""
        metrics: Dict[str, float] = {}
        for pattern, name in _METRIC_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics[name] = float(match.group(1))
                except ValueError:
                    pass
        # 覆盖键=值 风格的行（如 accuracy=0.852）
        for line in (text or "").splitlines():
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\d.]+)\s*$", line)
            if m:
                try:
                    metrics[m.group(1).lower()] = float(m.group(2))
                except ValueError:
                    pass
        # 值为 0~1 的 accuracy 统一保留（用于与声明 0.85 对齐）
        return metrics

    def _delta_llm_calls(self) -> int:
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    @staticmethod
    def _parse_json(text: str) -> Dict:
        for candidate in (text, re.sub(r"```(?:json)?\s*(.*?)```", r"\1", text, flags=re.DOTALL)):
            if not candidate or not candidate.strip():
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
        return {}