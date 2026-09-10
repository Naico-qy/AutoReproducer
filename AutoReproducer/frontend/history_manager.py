"""复现历史管理：扫描 ledger/runtime/reports 目录，生成历史记录列表，
提供存储状态、一键清理、报告下载链接。

无 Streamlit 依赖，纯 Python 工具函数，便于单元测试。
"""
import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_project_data_dir() -> Path:
    return Path("data")


def _parse_session_id(filename: str) -> Optional[str]:
    """从文件名提取 session_id 格式 (YYYYMMDD_HHMMSS)。"""
    name = Path(filename).stem
    for prefix in ("ledger_", "session_", "progress_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def _session_time(session_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(session_id, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def list_sessions() -> List[Dict[str, Any]]:
    """扫描 experiment_ledger/、logs/、runtime/，反推所有历史复现会话。"""
    base = get_project_data_dir()
    sessions: Dict[str, Dict[str, Any]] = {}

    # 扫描 experiment_ledger（最权威来源：含论文标题、结果等）
    ledger_dir = base / "experiment_ledger"
    if ledger_dir.exists():
        for f in sorted(ledger_dir.glob("ledger_*.jsonl"), reverse=True):
            sid = _parse_session_id(f.name)
            if not sid:
                continue
            sessions.setdefault(sid, {
                "session_id": sid,
                "paper_title": "",
                "state": "未知",
                "duration_sec": 0.0,
                "llm_calls": 0,
                "log_entries": 0,
                "ledger_size_bytes": 0,
                "progress_file": "",
                "report_path": "",
            })
            # 读取 ledger 全量记录：标题取首个含 outputs.title 的记录，
            # 终态（state/duration/llm_calls）取末条 FINISH 记录的 result。
            sessions[sid]["ledger_size_bytes"] = f.stat().st_size
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    records = [json.loads(line) for line in fh if line.strip()]
                for rec in records:
                    outputs = rec.get("outputs", {}) or {}
                    title = outputs.get("title") or \
                        (outputs.get("paper_info") or {}).get("title", "")
                    if title:
                        sessions[sid]["paper_title"] = title
                        break
                if records:
                    result = records[-1].get("result", {}) or {}
                    sessions[sid]["state"] = result.get("state", "RUNNING")
                    sessions[sid]["duration_sec"] = result.get("duration_sec", 0)
                    sessions[sid]["llm_calls"] = result.get("llm_calls", 0)
            except Exception:
                pass
            sessions[sid]["log_entries"] += _count_lines(f)

    # 扫描 logs/ 补充日志条目数
    log_dir = base / "logs"
    if log_dir.exists():
        for f in sorted(log_dir.glob("session_*.jsonl"), reverse=True):
            sid = _parse_session_id(f.name)
            if not sid or sid not in sessions:
                continue
            sessions[sid]["log_entries"] += _count_lines(f)

    # 扫描 runtime/ 关联 progress 文件
    runtime_dir = base / "runtime"
    if runtime_dir.exists():
        for f in sorted(runtime_dir.glob("progress_*.jsonl"), reverse=True):
            # runtime 文件名用毫秒时间戳，无法精确匹配 session_id
            # 但可以通过文件内容中的 done 事件提取 session_id
            try:
                sid_from_file = _extract_session_from_progress(f)
                if sid_from_file and sid_from_file in sessions:
                    sessions[sid_from_file]["progress_file"] = str(f)
            except Exception:
                pass

    # 扫描 reports/ 关联报告文件
    reports_dir = base / "reports"
    if reports_dir.exists():
        for f in sorted(reports_dir.glob("*.md"), reverse=True):
            # 报告文件名: {title}_{YYYYMMDD_HHMMSS}.md
            # 提取末尾的时间戳部分
            stem = f.stem
            # 从末尾往前找最后一个符合 YYYYMMDD_HHMMSS 的位置
            # 注意：YYYYMMDD_HHMMSS 共 15 字符（8 + 1 + 6）
            sid = ""
            for i in range(len(stem) - 14, -1, -1):
                candidate = stem[i:i+15]
                if len(candidate) == 15 and candidate[8] == '_' and candidate[:8].isdigit() and candidate[9:].isdigit():
                    sid = candidate
                    break
            if sid and sid in sessions:
                sessions[sid]["report_path"] = str(f)
                # 如果 ledger 没抓到标题，尝试从报告文件名推断
                if not sessions[sid]["paper_title"]:
                    title_guess = stem[:len(stem)-len(sid)-1]  # 去掉 _{sid}
                    sessions[sid]["paper_title"] = title_guess.replace("_", " ")

    # 按时间倒序排列
    return sorted(sessions.values(), key=lambda s: s["session_id"], reverse=True)


def _count_lines(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def _extract_session_from_progress(path: Path) -> Optional[str]:
    """从进度文件内容提取 session_id（如果日志条目中有）。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("type") == "log":
                    log = record.get("log", {})
                    ts = log.get("timestamp", "")
                    # 尝试从 timestamp (YYYY-MM-DDTHH:MM:SS) 转 session_id
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            return dt.strftime("%Y%m%d_%H%M%S")
                        except ValueError:
                            pass
        # 没找到就回退到文件修改时间
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return mtime.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return None


def get_storage_stats() -> Dict[str, Any]:
    """各数据子目录的存储占用统计。"""
    base = get_project_data_dir()
    stats = {}
    total = 0
    for sub in ("experiment_ledger", "logs", "runtime", "reports",
                "optimization_demo", "pinn-output"):
        d = base / sub
        if not d.exists():
            stats[sub] = {"files": 0, "bytes": 0}
            continue
        files = list(d.glob("**/*"))
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        stats[sub] = {"files": len([f for f in files if f.is_file()]),
                      "bytes": total_bytes}
        total += total_bytes
    stats["total"] = {"bytes": total}
    return stats


def cleanup_runtime(keep_days: int = 7) -> Tuple[int, int]:
    """清理超过 keep_days 天的 runtime 进度文件，返回 (删除文件数, 释放字节数)。"""
    cutoff = time.time() - keep_days * 86400
    runtime_dir = get_project_data_dir() / "runtime"
    removed = 0
    freed = 0
    if not runtime_dir.exists():
        return 0, 0
    for f in runtime_dir.glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                freed += f.stat().st_size
                f.unlink()
                removed += 1
        except Exception:
            pass
    return removed, freed


def get_session_detail(session_id: str) -> Optional[Dict[str, Any]]:
    """获取单次会话的详细记录（ledger + logs）。"""
    base = get_project_data_dir()
    detail: Dict[str, Any] = {"session_id": session_id, "ledger": [], "logs": []}

    ledger_file = base / "experiment_ledger" / f"ledger_{session_id}.jsonl"
    if ledger_file.exists():
        try:
            with open(ledger_file, "r", encoding="utf-8") as fh:
                detail["ledger"] = [json.loads(line) for line in fh]
        except Exception:
            pass

    log_file = base / "logs" / f"session_{session_id}.jsonl"
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as fh:
                detail["logs"] = [json.loads(line) for line in fh]
        except Exception:
            pass

    return detail if detail["ledger"] or detail["logs"] else None


def format_size(bytes_: int) -> str:
    if bytes_ >= 1024 ** 3:
        return f"{bytes_ / (1024 ** 3):.2f} GB"
    if bytes_ >= 1024 ** 2:
        return f"{bytes_ / (1024 ** 2):.2f} MB"
    if bytes_ >= 1024:
        return f"{bytes_ / 1024:.2f} KB"
    return f"{bytes_} B"
