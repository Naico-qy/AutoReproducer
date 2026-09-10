# references/ — 外部项目参考资源

本目录收录从其他 fork 项目中提取的有价值资源，供 AutoReproducer 开发与验收时参考。

## 目录结构

```
references/
├── paperbench/               PaperBench 复现评测数据（全部 23 篇论文，随仓库分发）
│   ├── aggregate-final.json         23 篇论文逐篇得分 + 基线对比（66.05% 全场总分）
│   ├── PER_PAPER_COMPARISON.md     逐篇 Δ 对比（vs AiScientist 等基线）
│   ├── REPORT.md                   人工可读头条报告
│   ├── README.md                   PaperBench 布局与再评测协议说明
│   └── submissions/                23 篇论文的完整复现提交物
│       └── <paper-id>/submission/  train.py + eval.py + reproduce.sh + requirements.txt
│                                   + configs/ data/ model/ utils/ …
│       （figures/*.png 架构图约 58 MB，未入库；需要时从原始仓库取）
├── researchstudio-idea/      ResearchStudio-Idea：研究想法生成流水线（论文→可辩护 idea card）
│   ├── README.md             项目说明（5 阶段工作流、证据门控设计）
│   ├── .env.template         环境变量模板（OpenReview/OpenAlex 等连接器）
│   ├── paper_search/         多源论文检索 skill（arXiv/DBLP/OpenAlex/OpenReview/S2/Crossref）
│   └── evaluation/           idea 质量评测数据与协议
└── researchstudio-reel/      ResearchStudio-Reel：论文→海报/视频/博客交付流水线
    ├── README.md             Reel 项目说明
    └── paper2reel/           paper2reel skill 定义（SKILL.md + README.md）
```

## 各资源的可借鉴点

### 1. paperbench/ — 复现评测基准（价值最高）

PaperGuru 在 23 篇 ICML 2024 spotlight 论文上按官方 PaperBench rubric-tree grader 评测，
总分 **66.05%**，超过 Human Expert 41% 与 AiScientist 最佳 33.73%。

**23 篇论文与复现分**（来自 `aggregate-final.json`，可直接作为 AutoReproducer 的外部对照基准）：

| 论文 id | 复现分 | 论文 id | 复现分 |
|---|---:|---|---:|
| semantic-self-consistency | 95.45 | ftrl | 62.66 |
| sequential-neural-score-estimation | 89.32 | fre | 61.51 |
| stay-on-topic-with-classifier-free-guidance | 88.16 | what-will-my-model-forget | 60.98 |
| sample-specific-masks | 86.52 | rice | 57.65 |
| lbcs | 85.74 | bridging-data-gaps | 57.14 |
| bam | 84.72 | **pinn** | **54.29** |
| stochastic-interpolants | 82.99 | all-in-one | 53.96 |
| mechanistic-understanding | 70.19 | robust-clip | 52.55 |
| test-time-model-adaptation | 70.06 | adaptive-pruning | 50.59 |
| self-composing-policies | 65.03 | sapg | 46.49 |
| lca-on-the-line | 63.16 | bbox | 40.34 |
| | | self-expansion | 39.77 |

> `pinn` 已完成真实复现（见 [reports/pinn_reproduction.md](../reports/pinn_reproduction.md)），
> 是本项目与 PaperGuru 基准的首个可比对样本。
>
> 代码侧通过 [`src/corpus.py`](../src/corpus.py) 读取，优先使用本目录（随仓库分发），
> 缺省回退到同级 `PaperGuru-Benchmark/PaperBench/`。
>
> `stay-on-topic-with-classifier-free-guidance` 上游即无 `requirements.txt`
> （依赖写在 `reproduce.sh` / README 中），`corpus.get_requirements()` 返回空串并降级。

对 AutoReproducer 的用途：

- **打分/验收数据**：`aggregate-final.json` 可直接作为复现系统输出对齐的外部基准；
  若 AutoReproducer 复现了其中某篇，可与 PaperGuru 得分对比。
- **提交物结构参考**：`submissions/<paper-id>/submission/` 的标准布局
  （`train.py` + `eval.py` + `reproduce.sh` + `data/` + `model/` + `evaluation/`）
  可借鉴为 AutoReproducer 复现产物的目标目录规范。
- **评测协议参考**：README 里"如何重新打分"一节说明用官方 grader 独立复评，
  与 AutoReproducer 的 ResultValidator 对照，可补强评估可信度（外部 judge 而非自报）。
- **样例论文**：`pinn/` 是我们 `data/pinn-output/` 可对照的真实论文提交物。

> 注意：提交物仅为还原评测输入（带 masking），不附带 agent 系统提示与 runner 代码；
> 若需完整训练数据/权重请回原仓库获取。

### 2. researchstudio-idea/ — 论文检索与研究想法生成

- `paper_search/scripts/search_papers.py`：六源并发论文检索脚本（arXiv / DBLP /
  OpenAlex / OpenReview / Semantic Scholar / Crossref），自带 `source_worker.py`
  并发实现，可直接复用为 AutoReproducer 的"论文来源检索"模块（输入侧）。
- `IdeaSpark` 的 5 阶段工作流（检索→瓶颈诊断→模式选择→生成→质量门控）中的
  **证据门控** 与 **回归自测脚本**（`scripts/regression_check.py` / `selftest_*.py`）
  值得借鉴到 Optimizer / Orchestrator 的验收环节。
- `.env.template`：OpenReview/OpenAlex 连接器凭据管理模板，与项目的 `.env` 方案一致。

### 3. researchstudio-reel/ — 论文交付物流水线

- `paper2reel`：一篇论文 → 可编辑海报 / 视频旁白 / 中英博客 / 交互 reel，
  与 AutoReproducer 报告落盘 + 下载方向的"结果交付"环节思路相通，
  可作为报告形态（Poster / 技术报告 / 视频讲解）的扩展参考。

## 来源

- PaperGuru-Benchmark（clone 自 https://github.com/AutoTrustAI/PaperGuru-Benchmark）
- ResearchStudio（clone 自 https://github.com/microsoft/ResearchStudio）

仅收录公开可复用资料；已去除 `.git` 历史、`__pycache__` 与大型架构图，并扫描确认
无 API key / token / 私钥 / 本机用户路径。

> 完整原始仓库（含 `assets/`、`SurveyBench/`、论文 PDF 等，合计约 370 MB）仍保留在
> 仓库根目录的 `PaperGuru-Benchmark/` 与 `ResearchStudio/`，二者已在根 `.gitignore`
> 中排除，不随 git 分发。