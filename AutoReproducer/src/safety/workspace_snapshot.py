"""WorkspaceSnapshot - 工作区快照与回滚（安全网 A3-2，借鉴 ScholarAgent 的 workspace_snapshot 设计）。

真实优化执行（或任何破坏性尝试）前，对目标工作区做一次快照：
每条文件记录 {相对路径: {"sha256": 指纹, "content": 内容副本}}。

执行失败、验收不通过或补丁被拒绝时，调用 :func:`restore_snapshot` 全量还原:

- 快照存在但当前缺失 / 内容不符的文件 -> 重建为快照版本；
- 快照时刻之后新增的文件 -> 删除。

保证论文原始仓库在任意失败路径下都能恢复原状；快照跳过 .git 等
内部目录，不复制版本控制数据。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# 快照条目的类型别名：{rel_path: {"sha256": str, "content": bytes}}
Snapshot = Dict[str, Dict[str, object]]

_DEFAULT_SKIP_DIRS: Tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
)
_DEFAULT_SKIP_FILES: Tuple[str, ...] = (".DS_Store",)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_workspace_files(root: Path, skip_dirs: Optional[Iterable[str]] = None,
                         skip_files: Optional[Iterable[str]] = None) -> Iterable[Path]:
    """遍历工作区内的普通文件（跳过 .git、缓存目录与临时文件）。"""
    skip_dirs = set(skip_dirs or _DEFAULT_SKIP_DIRS)
    skip_files = set(skip_files or _DEFAULT_SKIP_FILES)
    root = Path(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(d in path.parts for d in skip_dirs):
            continue
        if path.name in skip_files:
            continue
        yield path


def snapshot_workspace(root: Path, skip_dirs: Optional[Iterable[str]] = None,
                       skip_files: Optional[Iterable[str]] = None) -> Snapshot:
    """对工作区生成 {相对路径: {"sha256": 指纹, "content": 内容}} 快照。"""
    root = Path(root)
    snap: Snapshot = {}
    for f in iter_workspace_files(root, skip_dirs=skip_dirs, skip_files=skip_files):
        rel = f.relative_to(root).as_posix()
        data = f.read_bytes()
        snap[rel] = {"sha256": _sha256_bytes(data), "content": data}
    return snap


def changed_files(before: Snapshot, after: Snapshot) -> List[str]:
    """对比两个快照，返回新增 / 删除 / 内容变化的相对路径（字典序）。"""
    changed = []
    for rel in sorted(set(before) | set(after)):
        if before.get(rel, {}).get("sha256") != after.get(rel, {}).get("sha256"):
            changed.append(rel)
    return changed


def restore_snapshot(root: Path, snapshot: Snapshot,
                     skip_dirs: Optional[Iterable[str]] = None,
                     skip_files: Optional[Iterable[str]] = None) -> Tuple[List[str], List[str]]:
    """按快照还原工作区。返回 (已恢复文件列表, 已删除的新增文件列表)。

    - 快照中存在、但当前缺失或内容不符 -> 重建为快照版本（记入 restored）；
    - 快照后新增的文件 -> 删除（记入 removed）。
    """
    root = Path(root)
    restored: List[str] = []
    removed: List[str] = []

    # 1) 还原快照内文件
    for rel, entry in snapshot.items():
        target = root / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(entry["content"])  # type: ignore[arg-type]
            restored.append(rel)
        elif isinstance(entry["content"], bytes) and \
                _sha256_bytes(target.read_bytes()) != entry["sha256"]:
            target.write_bytes(entry["content"])
            restored.append(rel)

    # 2) 删除快照之后新增的文件（先收集再删除，避免遍历中目录变化）
    current = list(iter_workspace_files(root, skip_dirs=skip_dirs,
                                        skip_files=skip_files))
    for f in current:
        rel = f.relative_to(root).as_posix()
        if rel not in snapshot:
            f.unlink()
            removed.append(rel)

    for d in sorted({p.parent for p in current}, key=lambda v: -len(v.parts)):
        try:
            d.rmdir()          # 清理可能变空的目录（非递归，失败忽略）
        except OSError:
            pass

    return restored, removed