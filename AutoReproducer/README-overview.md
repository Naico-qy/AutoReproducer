# AutoReproducer 项目总览

> **基于多智能体协作的论文自动复现与优化系统**
> 给一篇论文（PDF / 标题 / 外部代码），自动完成「复现 → 验证 → 优化 → 报告」全流程闭环。

| 项目状态 | 最新更新 | 分支 | 测试基线 |
|---|---|---|---|
| 🟢 核心功能完成 + 工程可信度增强完成 | 2026-09-18 | `wqy` | **478 passed / 1 skipped** |

---

## 目录

1. [项目定位与背景](#1-项目定位与背景)
2. [总体框架](#2-总体框架)
3. [实现内容](#3-实现内容)
4. [方法与关键机制](#4-方法与关键机制)
5. [创新点](#5-创新点)
6. [测试流程](#6-测试流程)
7. [当前进展与规划](#7-当前进展与规划)
8. [快速开始](#8-快速开始)
9. [技术栈与项目结构](#9-技术栈与项目结构)

---

## 1. 项目定位与背景

学术论文的「可复现性」是科研生态的基础，但现状是：

- **手动复现耗时**：读懂论文 + 找到代码 + 搭环境 + 调参 + 比对指标，一篇论文往往数天到数周；
- **环境依赖地狱**：不同论文依赖不同版本框架（PyTorch 1.x / 2.x、CUDA、老库），全局环境互相污染；
- **磁盘与算力瓶颈**：一篇论文 = 代码 + 数据集 + 预训练权重；ImageNet 约 150GB 而 CIFAR-10 仅 170MB，多篇累加后普通笔记本（512GB-1TB）很快耗尽；
- **结论可信度**：谁来保证「复现成功」不是模型自我感觉良好？需要一套不依赖 LLM 自评的确定性验证链。

**AutoReproducer** 用一个编排器 + 8 个专职 Agent 的协作流水线解决上述问题：输入论文后自动查找资源、构建环境、执行代码、比对指标、验证质量、智能优化，全程留审计痕迹，并对「复现结论」做确定性证据裁决。项目已内置 **PaperBench 23 篇真实论文语料库** 作为对照锚点，并完成了第一篇论文（PINN，ICML 2024）的 Docker 真实复现（`reports/pinn_reproduction.md`）。

---

## 2. 总体框架

### 2.1 核心架构：1 个编排器 + 8 个专职 Agent

| 组件 | 职责 |
|---|---|
| 🧭 **Orchestrator（编排器）** | 状态机流转、各阶段输出合并、Prompt-Free 验证闭环、预算统计、存储钩子（按需取资源 / 落 manifest）、用量计量接线 |
| 📖 **PaperReader** | 解析论文 PDF（PyPDF2 主用、pdfplumber 兜底）/ 标题，提取标题、方法、依赖、数值声明、数据集 |
| 🔍 **ResourceFinder** | 定位代码仓库与数据集（确定性发现链：用户URL → PaperWithCode → GitHub 搜索 → curated 回退） |
| 🔧 **EnvBuilder** | 生成运行环境配置（Dockerfile / requirements），5 轮依赖诊断，构建共享底座镜像 |
| ⚡ **CodeExecutor** | 在本地隔离目录或 Docker 沙箱中运行代码（smoke + full 双阶段），缺模块 pip 自愈 |
| ✅ **ResultValidator** | 提取运行输出指标，口径归一化后与论文声明值比对 |
| 🛡️ **Verifier** | Prompt-Free 质量验证：复用各 Agent 的 `system_prompt` 作为质量标准，不引入额外验证提示词 |
| 🧪 **Optimizer** | UCB 预算感知调度下的智能优化（Keep/Reject 决策，真实执行闭环） |
| 📝 **ReportGenerator** | 生成 Markdown 复现 + 优化报告（含审计与预算统计） |

### 2.2 状态机流转

```
INIT → READ_PAPER → FIND_RESOURCES → BUILD_ENV → EXECUTE_CODE → VALIDATE
  →（复现成功）OPTIMIZING → OPTIMIZED → GENERATE_REPORT → COMPLETED
  →（复现失败）GENERATE_REPORT → COMPLETED
```

每个阶段输出都经过 **Prompt-Free 验证**：未通过时按修正建议触发一次修正重试，形成「生成 → 验证 → 修正 → 再验证」闭环（预算内 `MAX_FIX_RETRIES=1`）。

### 2.3 输入与输出

- **输入**：论文标题 / 上传 PDF / 外部代码仓库 URL（任意组合），可选 `corpus_paper` 指定语料对照层论文 id；
- **输出**：Markdown 报告（论文信息 / 资源 / 环境 / 执行 / 验证 / 优化 / 验证闭环 / 审计统计），前端可下载；历史会话在页面可回溯、清理。

---

## 3. 实现内容

### 3.1 复现核心闭环（Phase 1-7）

| 阶段 | 实现要点 |
|---|---|
| 输入透传 | 标题 / PDF / 外部代码三种输入正确进入数据上下文 |
| 资源定位 | 确定性仓库发现链（P1-⑨）+ 语料对照层（corpus.py 读 PaperBench 23 篇） |
| 环境构建 | 5 轮依赖诊断循环「探测 → 分类 → 定点修复 → 重验证」；Dockerfile 生成 |
| 代码执行 | smoke 冒烟（快速暴露环境问题）→ full 完整运行；真实模式在 Docker 沙箱或本地隔离依赖目录执行 |
| 结果验证 | 指标提取 + 口径归一化（论文声明 `0.85` vs 运行 `85.2%` 自动对齐） |
| 智能优化 | 仅复现成功后触发：UCB 预算调度 + BeamUCT 双层搜索（P1-⑥），真实补丁 → 白名单 → 快照 → 重跑 → 真实指标 → Keep/Reject |
| 报告生成 | 全量审计 + 预算统计注入 |

### 3.2 存储管理（三层缓存，存储友好设计）

解决「多篇论文累加后磁盘耗尽」：**按需懒加载 + 三层存储 + 体积瘦身**。

| 层 | 载体 | 内容 | 生命周期 |
|---|---|---|---|
| **L0 热缓存** | 本地磁盘 `data/` | 当前任务的代码 / 数据子集 / 权重 | 任务完成并归档后清理 |
| **L1 温存储** | 移动硬盘 / NAS / 网盘 | 已复现论文完整快照（zip 归档） | `resource_cli` archive / restore |
| **L2 冷存储** | HuggingFace / ModelScope | 只存清单 + 按 ID 可重现下载 | 不落地 |

关键实现（均已完成并提交）：

- **P0-1 ResourceManager**（`src/resource_manager.py`）：`fetch_code` / `fetch_dataset` / `fetch_weights` 懒加载（depth-1 克隆、冒烟子集、权重本地复制或子路径下载），重复 fetch 幂等；manifest 落盘；L0 配额守护 `AUTOREPRO_L0_QUOTA_GB`（默认 20GB，超限按 LRU 给建议不自动删除）；网络不可用时诚实降级为合成冒烟集并标注实际状态，绝不静默伪造大文件；
- **P0-2 编排器存储钩子**（`src/orchestrator.py`）：`paper_id` 注入、FIND_RESOURCES 后自动 fetch、COMPLETED 前生成 manifest，fetch 失败仅告警不阻断；
- **P0-3 隔离依赖安装**（`code_executor.py`）：真实模式 `pip install --target data/deps/<sha1(reqs)>` + `.ready` 标记 + PYTHONPATH 注入；同名依赖清单跨论文只落一份、天然去重、不污染全局 Python；
- **P1-1 共享底座镜像**（`env_builder.py`）：`autorepro-base`（python:3.11-slim + CPU torch/torchvision/numpy/tqdm + 国内源），一次构建多论文复用；论文 Dockerfile `FROM python:*` 自动替换为底座，构建失败自动降级原 Dockerfile（`degraded` 标注不阻断）；
- **P1-2 数据集注册表**（`dataset_registry.py`）：15 类常见数据集别名归一化 / 体积预估 / 子集策略（torchvision 内建懒加载不预下载、full 全量、percent:N 降采样验证趋势、synthetic 零数据），未知数据集诚实降级合成冒烟集；
- **P2 缓存管理 CLI**（`scripts/resource_cli.py`）：`status` / `list` / `manifest` / `archive` / `restore` / `prune` / `quota-check` 七子命令；`quota-check --size-gb N` 拉取前配额预检（不足拒绝退出码 2）。

### 3.3 工程可信度增强（ScholarAgent 融合，P0-①~⑤ + P1-⑥~⑫）

面向「复现结论可信」的工程防线，全部落地并附 pytest：

| 机制 | 位置 | 说明 |
|---|---|---|
| 补丁安全策略（P0-①） | `src/safety/patch_policy.py` | 优化补丁禁改数据/权重/评测文件、禁越目录、单文件体积预算，结构化拒绝决策 |
| 快照指纹（P0-④） | `src/safety/workspace_snapshot.py` | 全工作区 SHA-256 指纹 + 运行中篡改检测（不一致即中止恢复） |
| 经验库（P0-③） | `src/experience/experience_store.py` | JSONL 持久化（`validated` 标记）、上限截断、summarize/best，供 BeamUCT 播种先验 |
| TrialLedger（P0-⑤） | `src/experience/trial_ledger.py` | Keep/Reject 结构化账本（candidate/评估/理由/restored），供报告与经验库消费 |
| 证据链（P0-②） | `src/evidence/` | 证据注册表（SHA-256 + authentic）、裁决门控（无真实执行证据不判 verified）、Claim/Criterion/Evidence 哈希锚定图 |
| BeamUCT 双层搜索（P1-⑥） | `src/optimizer/beam_uct.py` | 方向级 UCB + 参数树 UCT + Beam top-k，经验库摘要播种先验、方向退役 |
| 冻结 Spec + Holdout（P1-⑦） | `src/optimizer/research_spec.py` | ResearchSpec sha256 冻结验收契约，只对最终 best 状态隐藏留出多轮评估（防过拟合验收） |
| 静态依赖解析 + pip 自愈（P1-⑧） | `src/agents/dependency_resolver.py` | AST+requirements 双源解析、stdlib 过滤、py39 归一、运行时缺模块 pip 自愈（≤3 轮） |
| 确定性仓库发现链（P1-⑨） | `src/agents/repo_discovery.py` | 用户URL → PwC → GitHub 搜索 → curated 四级降级链 + revision pin + 溯源标记（`.autorepro-repo-source.json`） |
| 防泄漏 Benchmark（P1-⑩） | `src/benchmark/leakage_safe.py` | 确定性 hash 切分（60/20/20）、隐藏标签私有目录、指标契约冻结后端复算（纯 Python 零依赖） |
| 沙箱加固（P1-⑪） | `src/agents/code_executor.py` | 镜像白名单（拒非法镜像 `exit_code=-5`）、cap-drop ALL、no-new-privileges、只读 rootfs+tmpfs、非 root、CPU/mem/pids 限额、三级降级（level 0→1→2） |
| 用量计量（P1-⑫） | `src/audit/audit_logger.py` + `src/llm/llm_client.py` | plan 级 LLM token 统计（`begin_plan`/`end_plan` 界定阶段，`extract_token_usage` 兼容三种 usage 形态）+ 容器耗时维度；`get_stats()` 汇总结算 |

### 3.4 前端与接口

- **Streamlit 前端**（`app.py` / `frontend/`）：LLM 配置侧边栏、复现进度实时日志、历史会话列表（回溯 / 清理 / 报告下载）、存储仪表板；
- **编程接口**（`src/orchestrator.py`）：`Orchestrator.run()` 一行触发完整流水线，Mock / 真实模式均可用；
- **对外能力**（`scripts/`）：资源缓存 CLI、优化 Demo 脚本、示例论文生成器。

---

## 4. 方法与关键机制

| 机制 | 方法说明 | 解决的问题 |
|---|---|---|
| **Prompt-Free 双层验证** | 复用各 Agent 的 `system_prompt` 充当质量标准，由 Verifier 做生成 → 验证 → 修正 → 再验证，`MAX_FIX_RETRIES=1` | 验证提示词与生成提示词同源，不引入额外偏置，预算可控 |
| **预算感知 UCB 调度** | 多臂老虎机：`UCB = Q + c·√(ln N / nᵢ)`，未拉臂强制探索，预算上限可配（默认 10 次，方案级 LLM 调用上限 100 次） | 优化方向间智能分配预算，探索/利用平衡 |
| **BeamUCT 双层搜索** | 方向级 UCB（快）+ 参数树 UCT（深）+ Beam top-k；`seed()` 以经验库摘要与历史先验播种、`retire()` 退役已穷尽方向 | 搜索空间大时避免从零开始，提升优化质量与收敛速度 |
| **确定性验证（证据链）** | 裁决绝不委托 LLM：criterion 只有在 artifacts 含 *authentic* 执行证据（内容 SHA-256 锚定）时才可判 verified；无真实执行证据 → unverifiable；smoke 天花板 `smoke_verified` | 「复现成功」必须有真实执行证据支撑，杜绝自评幻觉 |
| **冻结 Spec + 隐藏 Holdout** | ResearchSpec 冻结（sha256 契约），运行期间任何验收字段不得修改；只对最终 best 候选做多轮独立 Holdout 评估（均值/标准差判定） | 防「优化过程中悄悄改验收标准」、防拟合单次评估噪音 |
| **防泄漏 Benchmark** | 确定性 hash 切分训练/验证/测试（60/20/20）、隐藏标签存私有目录、指标契约冻结后由后端复算 | 评测标签/指标不被学习和缓存侧写泄漏 |
| **沙箱加固** | 镜像白名单 + 最小权限容器（cap-drop ALL、no-new-privileges、只读 rootfs、tmpfs、非 root nobody、CPU/mem/pids 限额），随 Docker 能力自动降级 | 外来论文代码是不可信代码，隔离运行 + 可审计降级 |
| **用量计量** | plan 级分账：LLM token（三形态 usage 提取 + hook 直连归账）与容器墙钟耗时，按流水线阶段核算 | 资源消耗可审计、可核算（L1/L2 分级成本） |
| **三层缓存 + 懒加载** | 只拉当前任务最小集，完成归档后清理 L0；同名依赖/底座镜像跨论文共享去重 | 512GB-1TB 笔记本可长期运行多论文复现 |
| **隔离依赖安装** | `pip install --target <sha1(reqs)>` + PYTHONPATH 注入 + 磁盘就绪标记 | 论文间依赖版本冲突隔离，不污染全局 Python |

---

## 5. 创新点

1. **复现-优化一体化闭环** — 复现成功自动触发优化（UCB 调度 + 真实补丁执行 + Keep/Reject），一次输入产出复现与优化双报告；
2. **Prompt-Free 双层验证** — 复用各 Agent 系统提示词作为质量标准，全过程零额外验证提示词，验证与生成同源、可审计；
3. **确定性证据裁决** — 复现结论由哈希锚定的真实执行证据链裁决（无证据不判 verified），不依赖 LLM 自评，杜绝「感觉成功」；
4. **冻结契约 + 隐藏 Holdout** — 优化验收标准全场冻结、最终状态多轮独立评估，防验收过拟合与标准漂移；
5. **工程化沙箱 + 防泄漏评测** — 外来代码以最小权限容器隔离执行、评测切分与指标复算防泄漏，兼顾安全与可信；
6. **三层缓存 + 镜像级共享存储体系** — 按需懒加载 + L0/L1/L2 分层 + 共享底座镜像 + 数据集注册表策略化降采样，让消费级笔记本可持续复现多篇论文（量化存储不用于精确数值复现比对，仅作 L3 专家模式可选）；
7. **BeamUCT 双层搜索 + 经验先验播种** — 在 UCB 预算框架上叠加参数树 UCT 与 Beam，并用历史经验播种，优化搜索从「零样本冷启动」变为「经验引导」；
8. **可审计完整实验追踪** — 审计 JSONL + 实验账本（Ledger，`replay()` 时间轴回放）+ plan 级用量计量，全链路留痕可回放可核算。

---

## 6. 测试流程

### 6.1 运行测试

```bash
# 完整测试套件（Mock 模式，无需 LLM API / Docker，CI 全绿）
python -m pytest tests/ -v

# 只看端到端集成用例
python -m pytest tests/ -v -k TestEndToEnd

# 只跑某个模块
python -m pytest tests/test_usage_metering.py -v
```

### 6.2 覆盖范围（27 个测试文件 / 478 用例）

| 分组 | 测试文件（用例数） |
|---|---|
| 复现核心链路 | `test_architecture.py`、`test_reproduction_core.py`、`test_code_executor_deps.py`、`test_code_executor_self_heal.py`、`test_optimizer_real.py` |
| 存储管理 | `test_resource_manager.py`、`test_orchestrator_storage.py`、`test_env_builder_base.py`（13）、`test_dataset_registry.py`（18）、`test_resource_cli.py`（12） |
| 工程可信度 P0 | `test_patch_policy_fusion.py`（23）、`test_workspace_snapshot_fusion.py`（14）、`test_experience_store.py`（16）、`test_trial_ledger.py`（14）、`test_evidence_graph.py`（17） |
| 工程可信度 P1 | `test_beam_uct_fusion.py`（24）、`test_research_spec_fusion.py`（24）、`test_dependency_resolver_fusion.py`（23）、`test_repo_discovery.py`（37）、`test_leakage_safe_benchmark.py`（38）、`test_sandbox_hardening.py`（13）、`test_usage_metering.py`（23） |
| 前端与配置 | `test_app_smoke.py`、`test_backend_pipeline.py`、`test_history_manager.py`、`test_llm_config.py`、`test_safety.py` |

### 6.3 当前基线

- **全量 478 passed / 1 skipped**（2026-09-18 实测，耗时约 76s）；
- 真实模式（Docker 沙箱子集）**21s 全绿**；
- 测试全部运行于 Mock 模式：无需 LLM API、无需 Docker，可离线 CI；
- 新增功能遵循约定：**每条配套 pytest 用例 → 跑绿后 git commit**，测试基线持续守护。

### 6.4 真实模式验证（可选）

设置 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 环境变量（OpenAI 兼容端点，默认指向 DeepSeek），`use_docker=True` 即可在 Docker 沙箱中真实执行论文代码。已完成的真实复现样例：**PINN（Challenges in Training PINNs: A Loss Landscape Perspective，ICML 2024）**——Docker 中 `python:3.12-slim + torch 2.12.0+cpu` 镜像，smoke 模式（每 PDE 100 次迭代）产出三个 PDE 的真实 loss/l2re，端到端验证流水线可用（`reports/pinn_reproduction.md`）。

---

## 7. 当前进展与规划

### 7.1 项目验收目标

- **L1 复现成功率 ≥ 60%**，且**至少 2 篇论文升级到 L2**（分级复现：L1 冒烟级流程验证 → L2 数值级复现 → L3 高精度复现）；
- 单次优化预算默认 `max_trials=10`，方案级 LLM 调用上限 100 次。

### 7.2 里程碑完成度（截至 2026-09-18）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| 核心多智能体闭环 | 1 编排器 + 8 Agent + 验证闭环 + Docker 真实执行 | ✅ 完成 |
| 存储管理改造 | P0-1 ResourceManager / P0-2 编排器钩子 / P0-3 隔离依赖 / P1-1 底座镜像 / P1-2 数据集注册表 / P2 CLI | ✅ 完成 |
| ScholarAgent 融合 P0 | ① 补丁安全 / ② 证据链 / ③ 经验库 / ④ 快照指纹 / ⑤ TrialLedger | ✅ 完成 |
| ScholarAgent 融合 P1 | ⑥ BeamUCT / ⑦ 冻结 Spec+Holdout / ⑧ 依赖自愈 / ⑨ 仓库发现链 / ⑩ 防泄漏评测 / ⑪ 沙箱加固 / ⑫ 用量计量 | ✅ 完成 |

### 7.3 git 历史脉络（33 次提交，分支 `wqy`）

```
cff5d2b docs: README/CHANGELOG 增补 ScholarAgent 融合章节与 P1-⑫ 批次   ← HEAD
7c6c7cc feat(usage): P1-⑫ plan 级 LLM token 计量与容器耗时维度
aad082b feat(sandbox): P1-⑪ Docker 沙箱加固参数
e395e86 feat(benchmark): P1-⑩ 防泄漏 Benchmark 评测
a0f10ef feat(resource): P1-⑨ 确定性仓库发现链
ef1eaf7 feat(deps): P1-⑧ 静态依赖解析+pip 自愈
2342dd0 feat(optimizer): P1-⑦ 冻结 ResearchSpec 契约 + 隐藏 Holdout 验收
a8c88d4 P1-⑥ BeamUCT 双层搜索: 方向级 UCB + 参数树 UCT + Beam top-k + 经验库先验播种
2019a73 … P0-② 证据链 / 8d34031 … P0-⑤ TrialLedger / d4d927c … P0-③ 经验库 /
782cda9 … P0-④ 快照指纹 / b532894 … P0-① 补丁安全
c6fd264 … P2 CLI / 3c0d103 … P1-2 数据集注册表 / 324a3d5 … P1-1 底座镜像 /
0b7b65c … P0-2 编排器钩子 / 8af3f90 … P0-1 ResourceManager / 499507d … P0-3 隔离依赖
0262d9a…9881c97 … 前端可观测性 / 历史管理 / 依赖与 PDF 解析修复 / 语料库
b310fc5 docs: 第一篇真实复现报告（pinn, Docker 真实运行）
ea7fb73 AutoReproducer: 补齐验证-优化闭环 + Docker 真实执行
03d6ca1 AutoReproducer: 多智能体论文自动复现系统（初始）
```

### 7.4 下一步方向（候选）

- 将 P1 完成后的融合闭环在 PaperBench 多篇论文上批量跑真实复现，逼近 L1 ≥60% 验收；
- L3 高精度复现路径（含 L2 升级机制与 GPU 按需租用）；
- 前端用量计量面板展示（plan 级 token / 容器耗时可视化）；
- 经验库跨论文迁移收益的量化评估。

---

## 8. 快速开始

```bash
cd AutoReproducer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 方式一：前端
streamlit run app.py                                # 浏览器访问 http://localhost:8501

# 方式二：编程接口（Mock 演示，无需任何外部依赖）
python -c "
from src.orchestrator import Orchestrator
orch = Orchestrator(mock_mode=True, max_trials=6)
result = orch.run({'paper_title': 'Attention Is All You Need'})
print(result['state']); print(result['data']['report'][:500])
"
```

真实模式需配置 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`（默认 DeepSeek），`use_docker=True` 启用 Docker 沙箱执行。

---

## 9. 技术栈与项目结构

### 技术栈

- **前端**：Streamlit
- **核心**：Python 3.11+
- **LLM**：OpenAI 兼容 API（DeepSeek / 千帆 / OpenAI，环境变量注入，客户端不绑定厂商）
- **沙箱**：Docker（可选，真实执行；镜像白名单 + 加固参数）
- **PDF**：PyPDF2 / pdfplumber
- **测试**：pytest（478 用例，Mock 模式离线全绿）

### 项目结构

```
AutoReproducer/
├── app.py                     # Streamlit 前端
├── src/
│   ├── orchestrator.py        # 编排器（状态机 + 验证闭环 + 预算 + 存储钩子 + 用量接线）
│   ├── resource_manager.py    # 资源懒加载 + L0 缓存/配额/归档（P0-1）
│   ├── dataset_registry.py    # 数据集注册表（P1-2）
│   ├── corpus.py              # 语料对照层（PaperBench 23 篇）
│   ├── agents/                # 8 个 Agent + dependency_resolver + repo_discovery
│   ├── optimizer/             # UCB 调度器 / BeamUCT / ResearchSpec / RealSimulator
│   ├── safety/                # 补丁安全（P0-①）/ 快照指纹（P0-④）
│   ├── experience/            # 经验库（P0-③）/ TrialLedger（P0-⑤）
│   ├── evidence/              # 证据链（P0-②）
│   ├── benchmark/             # 防泄漏评测（P1-⑩）
│   ├── llm/llm_client.py      # LLM 客户端（含 usage 提取与累计）
│   └── audit/audit_logger.py  # 审计 + 实验账本 + plan 级用量计量（P1-⑫）
├── scripts/                   # resource_cli / 优化 demo / 示例论文生成器
├── tests/                     # 27 个测试文件（+ conftest）/ 478 用例
├── references/paperbench/     # PaperBench 23 篇论文复现评测基准与完整提交物
├── reports/                   # 真实复现报告（pinn 等）
└── data/                      # L0 热缓存（repos/datasets/manifests/archive/deps/experience/ledger）
```

---

*本文档为 AutoReproducer 项目总览（README-overview.md），随项目进展更新。详细版本历史见 `CHANGELOG.md`，快速上手与接口细节见 `README.md`。*