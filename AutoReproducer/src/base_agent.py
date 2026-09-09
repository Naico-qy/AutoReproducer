"""Agent基类 - 所有 Agent 的抽象基类"""
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Optional
from src.audit.audit_logger import AuditLogger

# Docker Desktop / 常见安装路径（Windows 上 docker.exe 常不在系统 PATH 中）
_DOCKER_CANDIDATES = [
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    r"C:\Program Files\Docker\Docker\resources\bin\docker-cli.exe",
    r"/usr/local/bin/docker",
    r"/usr/bin/docker",
]


class BaseAgent(ABC):
    """所有 Agent 的抽象基类。

    每个 Agent 通过 `system_prompt` 声明自身的质量标准；
    Verifier 在执行 Prompt-Free 验证时直接复用该提示词，无需人工设计额外验证词。
    """

    # 本 Agent 的质量标准（系统提示词）
    system_prompt = ""

    def __init__(self, name: str, logger: Optional[AuditLogger] = None):
        self.name = name
        self.logger = logger or AuditLogger()

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """执行 Agent 的核心逻辑，返回结构化结果 dict。"""
        raise NotImplementedError

    def log(self, action: str, status: str, detail: str,
            data: Optional[dict] = None):
        """记录审计日志。"""
        if self.logger:
            self.logger.log(self.name, action, status, detail, data)

    def log_experiment(self, phase: str, decision: str,
                       inputs: Optional[dict] = None,
                       outputs: Optional[dict] = None,
                       result: Optional[dict] = None):
        """记录实验账本（Ledger），供时间轴回放与审计。"""
        if self.logger:
            self.logger.log_experiment(phase, decision, inputs, outputs, result)

    def __repr__(self) -> str:
        return f"Agent({self.name})"

    @staticmethod
    def _resolve_docker_cmd() -> Optional[str]:
        """解析可用的 docker CLI 路径。

        优先取系统 PATH，其次探测 DOCKER_PATH 环境变量与常见安装目录
        （Windows 上 Docker Desktop 的 docker.exe 常不在 PATH 中）。
        返回绝对路径或命令名，未找到返回 None。
        """
        env_path = os.environ.get("DOCKER_PATH", "").strip().strip('"')
        if env_path and os.path.isfile(env_path):
            return env_path
        found = shutil.which("docker")
        if found:
            return found
        for cand in _DOCKER_CANDIDATES:
            if os.path.isfile(cand):
                return cand
        return None