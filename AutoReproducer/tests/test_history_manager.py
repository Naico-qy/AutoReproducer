"""历史记录管理模块单元测试：list_sessions / get_storage_stats / cleanup_runtime / get_session_detail / format_size。

通过 monkeypatch 将数据目录指向临时目录，不触碰真实 data/。
"""
import json
import time
from pathlib import Path

import pytest

from frontend.history_manager import (
    cleanup_runtime,
    format_size,
    get_project_data_dir,
    get_session_detail,
    get_storage_stats,
    list_sessions,
)


@pytest.fixture
def fake_data(tmp_path: Path, monkeypatch):
    """构造临时数据目录并注入 history_manager."""
    monkeypatch.setattr("frontend.history_manager.get_project_data_dir",
                        lambda: tmp_path)
    return tmp_path


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mk_ledger(data_dir: Path, sid: str, title: str = "测试论文", state: str = "COMPLETED"):
    record = {
        "type": "start",
        "outputs": {"title": title, "paper_info": {"title": title}},
        "result": {"state": state, "duration_sec": 12.5, "llm_calls": 35},
        "timestamp": "2026-09-10T10:00:00",
    }
    _write_jsonl(data_dir / "experiment_ledger" / f"ledger_{sid}.jsonl", [record])


# ---------- list_sessions ----------

def test_list_sessions_empty(fake_data: Path):
    assert list_sessions() == []


def test_list_sessions_detects_ledger(fake_data: Path):
    _mk_ledger(fake_data, "20260910_100000", "梯度下降实验")
    sessions = list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["session_id"] == "20260910_100000"
    assert s["paper_title"] == "梯度下降实验"
    assert s["state"] == "COMPLETED"
    assert s["duration_sec"] == 12.5
    assert s["llm_calls"] == 35


def test_list_sessions_counts_log_lines(fake_data: Path):
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid)
    _write_jsonl(fake_data / "logs" / f"session_{sid}.jsonl",
                 [{"type": "log"} for _ in range(5)])
    sessions = list_sessions()
    assert sessions[0]["log_entries"] == 1 + 5  # ledger 1 行 + log 5 行


def test_list_sessions_matches_report(fake_data: Path):
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid)
    report = fake_data / "reports" / f"线性回归复现_{sid}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# 报告", encoding="utf-8")
    sessions = list_sessions()
    assert sessions[0]["report_path"] == str(report)


def test_list_sessions_ignores_unrelated_files(fake_data: Path):
    _mk_ledger(fake_data, "20260910_100000")
    (fake_data / "experiment_ledger" / "random.txt").write_text("x", encoding="utf-8")
    assert len(list_sessions()) == 1


def test_list_sessions_sorted_desc(fake_data: Path):
    _mk_ledger(fake_data, "20260910_090000", "旧")
    _mk_ledger(fake_data, "20260910_110000", "新")
    sids = [s["session_id"] for s in list_sessions()]
    assert sids == ["20260910_110000", "20260910_090000"]


# ---------- get_storage_stats ----------

def test_storage_stats_counts_files_and_bytes(fake_data: Path):
    p = fake_data / "experiment_ledger" / "ledger_20260910_100000.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x" * 1024, encoding="utf-8")
    stats = get_storage_stats()
    assert stats["experiment_ledger"]["files"] == 1
    assert stats["experiment_ledger"]["bytes"] == 1024
    assert stats["total"]["bytes"] == 1024


def test_storage_stats_missing_dirs_zero(fake_data: Path):
    stats = get_storage_stats()
    for key in ("logs", "runtime", "reports"):
        assert stats[key] == {"files": 0, "bytes": 0}


# ---------- cleanup_runtime ----------

def test_cleanup_runtime_removes_old_only(fake_data: Path):
    runtime = fake_data / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    old = runtime / "progress_old.jsonl"
    new = runtime / "progress_new.jsonl"
    old.write_text("o" * 100, encoding="utf-8")
    new.write_text("n" * 50, encoding="utf-8")
    # 把 old 文件修改时间改为 10 天前
    past = time.time() - 10 * 86400
    import os as _os
    _os.utime(old, (past, past))
    removed, freed = cleanup_runtime(keep_days=7)
    assert removed == 1
    assert freed == 100
    assert not old.exists()
    assert new.exists()


def test_cleanup_runtime_no_dir(fake_data: Path):
    assert cleanup_runtime() == (0, 0)


# ---------- get_session_detail ----------

def test_get_session_detail_returns_none_for_unknown(fake_data: Path):
    assert get_session_detail("20260101_000000") is None


def test_get_session_detail_reads_ledger_and_logs(fake_data: Path):
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid)
    _write_jsonl(fake_data / "logs" / f"session_{sid}.jsonl", [{"type": "log", "msg": "hi"}])
    detail = get_session_detail(sid)
    assert detail is not None
    assert len(detail["ledger"]) == 1
    assert len(detail["logs"]) == 1
    assert detail["logs"][0]["msg"] == "hi"


# ---------- format_size ----------

@pytest.mark.parametrize("bytes_,expected", [
    (0, "0 B"),
    (512, "512 B"),
    (1024, "1.00 KB"),
    (2 * 1024 * 1024, "2.00 MB"),
    (3 * 1024 * 1024 * 1024, "3.00 GB"),
])
def test_format_size(bytes_: int, expected: str):
    assert format_size(bytes_) == expected


# ---------- 解析辅助函数 ----------

def test_parse_session_id_variants():
    from frontend.history_manager import _parse_session_id
    assert _parse_session_id("ledger_20260910_100000.jsonl") == "20260910_100000"
    assert _parse_session_id("session_20260910_100000.jsonl") == "20260910_100000"
    assert _parse_session_id("progress_20260910_100000.jsonl") == "20260910_100000"
    assert _parse_session_id("random.txt") is None