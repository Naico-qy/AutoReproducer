"""safety 安全网子包：为 Optimizer 真实执行提供修改边界与回滚保障。

对齐借鉴点 A3（补丁白名单 + 工作区快照回滚）：
- patch_policy：LLM 优化补丁只允许修改 editable 白名单内文件，
  protected 目录/文件一律拒绝；
- workspace_snapshot：真实执行前记录工作区 SHA-256 + 内容快照，
  失败或验收不通过时按快照自动还原。
"""