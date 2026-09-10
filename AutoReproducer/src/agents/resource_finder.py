"""ResourceFinderAgent - 资源查找 Agent，定位代码仓库与数据集。

流程：
1. 从论文原文中提取 URL（正则）；
2. LLM 综合论文信息推测代码仓库与数据集；LLM 失败时使用提取结果降级。
"""
import json
import re
from typing import Dict
from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient


class ResourceFinderAgent(BaseAgent):
    """从论文信息中查找代码仓库、数据集和相关资源。"""

    system_prompt = "根据论文信息定位代码仓库与数据集,输出仓库 URL、数据源列表及置信度"

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("ResourceFinder", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """查找论文相关资源。

        input_data: {"paper_info": dict, "raw_text": str}
        """
        self.log("find_resources", "START", "开始查找资源", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        raw_text = input_data.get("raw_text", "") or ""

        # 1. 从文本中提取 URL
        urls = self._extract_urls(raw_text)
        github_urls = [u for u in urls if "github.com" in u]

        # 2. LLM 推测代码仓库
        prompt = f"""根据论文信息，推测最可能的代码仓库URL和使用的数据集。
论文标题: {paper_info.get('title', '未知')}
方法: {paper_info.get('method', '未知')}

返回JSON格式:
{{
    "code_repo_url": "最可能的GitHub URL或'未找到'",
    "alternative_repos": ["备用仓库1", "备用仓库2"],
    "dataset_url": "数据集URL或'未找到'",
    "confidence": 0.0-1.0
}}
"""
        llm_result = self.llm.chat(prompt, task="resource_finder")
        parsed = self._parse_json(llm_result)
        if not parsed or not parsed.get("code_repo_url"):
            parsed = {
                "code_repo_url": github_urls[0] if github_urls else "未找到",
                "alternative_repos": [],
                "dataset_url": "未找到",
                "confidence": 0.7 if github_urls else 0.3,
            }

        # 合并提取的 URL（真实线索优先）
        parsed["extracted_urls"] = urls[:10]
        parsed["github_urls"] = github_urls
        if github_urls:
            parsed["confidence"] = max(parsed.get("confidence", 0.0), 0.8)

        self.log_experiment(
            "FIND_RESOURCES", "定位代码仓库与数据集",
            inputs={"paper_info": paper_info},
            outputs=parsed,
        )
        self.log("find_resources", "SUCCESS",
                 f"找到 {len(github_urls)} 个 GitHub 仓库,"
                 f"置信度 {parsed.get('confidence', 0):.2f}",
                 {"github_urls": github_urls, "confidence": parsed.get("confidence")})

        return {
            "resources": parsed,
            "llm_calls": self._delta_llm_calls(),
        }

    # ---------------- 内部工具 ----------------

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

    @staticmethod
    def _extract_urls(text: str) -> list:
        """从文本中提取 URL。"""
        url_pattern = r'https?://[^\s\)\]}"]+'
        return list(set(re.findall(url_pattern, text)))