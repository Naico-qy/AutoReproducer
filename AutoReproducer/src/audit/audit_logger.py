"""审计日志模块 - 记录所有 Agent 的决策与执行过程（可审计追踪）。

对齐方案「创新点四：可审计的完整实验追踪」：
- audit 日志：所有 Agent 输入输出 / 决策依据以 JSON Lines 写入 data/logs/；
- experiment_ledger：面向研究过程的完整实验账本（含每次尝试与结果）写入
  data/experiment_ledger/，支持按 session_id 时间轴回放。
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目根目录（src/audit/audit_logger.py -> parents[2] 为仓库内 AutoReproducer 包根）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AuditLogger:
    """审计日志记录器：追踪每一步的决策依据，并落盘为可回放的账本。"""

    def __init__(self, log_dir: Optional[str] = None,
                 ledger_dir: Optional[str] = None):
        # 默认落在项目根 data/ 下，避免随 CWD 漂移
        base = _PROJECT_ROOT / "data"
        self.log_dir = Path(log_dir) if log_dir else (base / "logs")
        self.ledger_dir = Path(ledger_dir) if ledger_dir else (base / "experiment_ledger")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: List[Dict[str, Any]] = []
        self.llm_calls = 0
        self._start_time = time.time()

    def log(self, agent: str, action: str, status: str, detail: str,
            data: Optional[dict] = None) -> dict:
        """记录一条审计日志。"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - self._start_time, 2),
            "agent": agent,
            "action": action,
            "status": status,
            "detail": detail,
            "data": data or {},
        }
        self.entries.append(entry)
        self._flush(entry)
        return entry

    def log_experiment(self, phase: str, decision: str,
                       inputs: Optional[dict] = None,
                       outputs: Optional[dict] = None,
                       result: Optional[dict] = None) -> dict:
        """写入一条实验账本记录（Ledger）：可覆盖 Agent 输入输出与尝试结果。"""
        record = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "decision": decision,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "result": result or {},
        }
        ledger_file = self.ledger_dir / f"ledger_{self.session_id}.jsonl"
        with open(ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def replay(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """按时间轴回放某次实验的完整账本（默认当前会话）。

        返回按写入顺序排列的 ledger 记录，供「任意时间点状态回放与审计」。
        """
        sid = session_id or self.session_id
        ledger_file = self.ledger_dir / f"ledger_{sid}.jsonl"
        records: List[Dict[str, Any]] = []
        if not ledger_file.exists():
            return records
        for line in ledger_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def add_llm_calls(self, count: int) -> None:
        """累计 LLM 调用次数（预算统计口径）。"""
        self.llm_calls += max(0, int(count))

    def _flush(self, entry: dict) -> None:
        """将单条日志追加写入 JSON Lines 文件。"""
        log_file = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_summary(self) -> List[Dict[str, Any]]:
        """获取当前会话的所有日志。"""
        return self.entries

    def get_stats(self) -> dict:
        """获取统计信息（含预算口径的 LLM 调用数）。"""
        total = len(self.entries)
        errors = sum(1 for e in self.entries if e["status"] == "ERROR")
        success = sum(1 for e in self.entries if e["status"] == "SUCCESS")
        return {
            "total_steps": total,
            "errors": errors,
            "success": success,
            "duration_sec": round(time.time() - self._start_time, 2),
            "llm_calls": self.llm_calls,
        }