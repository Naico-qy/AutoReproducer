"""AutoReproducer - Streamlit 前端界面

修复与增强：
- 论文标题 / 上传 PDF 正确传递到 PaperReader；
- 侧边栏 LLM API 配置（OpenAI 兼容端点 / Key / 模型）真实生效；
- 展示优化结果（最优方向/改进幅度）与 LLM 预算统计。
"""
import os
import sys
import time
import tempfile

import streamlit as st

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator
from src.llm.ollama_client import LLMClient
from src.audit.audit_logger import AuditLogger
from src.corpus import list_papers

# 页面配置
st.set_page_config(
    page_title="AutoReproducer - 论文自动复现系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 样式
st.markdown("""
<style>
    .status-ok { color: #00ff00; font-weight: bold; }
    .status-error { color: #ff0000; font-weight: bold; }
    .status-running { color: #ffaa00; font-weight: bold; }
    .status-waiting { color: #888888; }
    .agent-card {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    .stApp header {display: none;}
    .main-title {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        margin-top: 0;
    }
    div[data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 400px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session 状态
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False
if "logs" not in st.session_state:
    st.session_state.logs = []
if "current_state" not in st.session_state:
    st.session_state.current_state = "INIT"
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {}
if "mock_mode" not in st.session_state:
    st.session_state.mock_mode = True
if "paper_title" not in st.session_state:
    st.session_state.paper_title = ""


# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/idea.png", width=60)
    st.markdown("## ⚙️ 控制面板")

    # 模式选择
    st.session_state.mock_mode = st.toggle(
        "🧪 Mock模式（无需API）",
        value=st.session_state.mock_mode,
        help="启用Mock模式可直接演示，无需连接任何LLM服务")

    # LLM API 配置（真实模式；OpenAI 兼容接口，不依赖本地部署）
    with st.expander("🔗 LLM API 配置", expanded=not st.session_state.mock_mode):
        base_url = st.text_input(
            "API 地址（OpenAI 兼容）",
            value=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            placeholder="如 https://api.deepseek.com/v1",
            help="支持 DeepSeek / 千帆 / OpenAI 等任意 OpenAI 兼容端点",
            disabled=st.session_state.mock_mode)
        api_key = st.text_input(
            "API Key",
            value=os.environ.get("LLM_API_KEY", ""),
            type="password",
            help="远程 API 的访问密钥（无鉴权服务可留空）",
            disabled=st.session_state.mock_mode)
        model_name = st.text_input(
            "模型名称",
            value=os.environ.get("LLM_MODEL", "deepseek-chat"),
            placeholder="如 deepseek-chat / ernie-4.0-8k / gpt-4o-mini",
            disabled=st.session_state.mock_mode)

    # 预算上限
    max_trials = st.slider(
        "🎯 优化预算（UCB 尝试次数）", min_value=3, max_value=20, value=10,
        help="Optimizer 在复现成功后最多尝试的优化方向次数")

    # 论文输入
    st.markdown("### 📄 论文输入")
    input_mode = st.radio("输入方式", ["论文标题", "上传PDF"], key="input_mode")

    paper_title = st.session_state.paper_title
    uploaded_file = None

    if input_mode == "论文标题":
        paper_title = st.text_input(
            "论文标题",
            value=st.session_state.paper_title,
            placeholder="输入论文标题...",
            key="paper_title_input")
        st.session_state.paper_title = paper_title
    else:
        uploaded_file = st.file_uploader("上传PDF文件", type=["pdf"],
                                         key="pdf_uploader")

    # 语料对照层（可选）：选择真实论文作为轻量锚点
    st.markdown("### 🗂️ 语料对照(可选)")
    _corpus = [p["id"] for p in list_papers()]
    _corpus_choice = st.selectbox(
        "选择 PaperGuru-Benchmark 论文", ["无"] + _corpus, index=0,
        key="corpus_paper_select")
    corpus_paper = None if _corpus_choice == "无" else _corpus_choice

    # 启动 / 重置按钮
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 开始复现", type="primary",
                              use_container_width=True,
                              disabled=st.session_state.running)
    with col2:
        reset_btn = st.button("🔄 重置", use_container_width=True)

    # 系统状态
    st.markdown("---")
    st.markdown("### 📊 系统状态")
    state_colors = {
        "INIT": "⚪", "READ_PAPER": "📖", "FIND_RESOURCES": "🔍",
        "BUILD_ENV": "🔧", "EXECUTE_CODE": "⚡", "VALIDATE": "✅",
        "OPTIMIZING": "🧪", "OPTIMIZED": "🏆",
        "GENERATE_REPORT": "📝", "COMPLETED": "🎉", "ERROR": "❌",
    }
    st.markdown(
        f"**当前状态**: {state_colors.get(st.session_state.current_state, '⚪')} "
        f"`{st.session_state.current_state}`")


# ========== 主界面 ==========
st.markdown('<p class="main-title">🔬 AutoReproducer</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于多智能体协作的论文自动复现与优化系统</p>',
            unsafe_allow_html=True)

# 标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 流水线状态", "📄 复现报告", "📜 审计日志", "🔍 状态机",
])

# ===== Tab 1: 流水线状态 =====
with tab1:
    st.markdown("### 🏗️ 复现流水线（复现 -> 验证 -> 优化 -> 报告）")

    AGENTS = [
        ("📖 PaperReader", "论文解析", "从PDF/标题中提取结构化信息"),
        ("🔍 ResourceFinder", "资源查找", "定位代码仓库和数据集"),
        ("🔧 EnvBuilder", "环境构建", "自动搭建环境 + 依赖诊断"),
        ("⚡ CodeExecutor", "代码执行", "smoke + full 双阶段执行"),
        ("✅ ResultValidator", "结果验证", "比对论文声明值与运行结果"),
        ("🛡️ Verifier", "质量验证", "Prompt-Free 检查质量 + 修正闭环"),
        ("🧪 Optimizer", "智能优化", "UCB 预算调度, Keep/Reject"),
        ("📝 ReportGenerator", "报告生成", "生成复现+优化 Markdown 报告"),
    ]

    cols = st.columns(3)
    for i, (name, title, desc) in enumerate(AGENTS):
        with cols[i % 3]:
            status = st.session_state.agent_status.get(name, "waiting")
            status_icons = {"success": "✅", "error": "❌",
                            "running": "🔄", "waiting": "⏳"}
            status_colors = {
                "success": "border-left: 4px solid #4CAF50;",
                "error": "border-left: 4px solid #f44336;",
                "running": "border-left: 4px solid #FF9800;",
                "waiting": "border-left: 4px solid #9E9E9E;",
            }
            icon = status_icons.get(status, "⏳")
            border = status_colors.get(status, "")
            st.markdown(f"""
            <div class="agent-card" style="{border}">
                <h4>{icon} {name}</h4>
                <small>{title}</small><br>
                <span style="color: #888;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    agent_order = [a[0] for a in AGENTS]
    completed = sum(1 for a in agent_order
                    if st.session_state.agent_status.get(a) == "success")
    progress = completed / len(agent_order) if agent_order else 0
    st.progress(progress, text=f"整体进度: {completed}/{len(agent_order)}")

    # 运行结果展示
    if st.session_state.result:
        result = st.session_state.result
        st.markdown("---")
        st.markdown("### 📊 运行摘要")
        stats = result.get("audit_stats", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("总步骤数", stats.get("total_steps", 0))
        c2.metric("成功", stats.get("success", 0))
        c3.metric("错误", stats.get("errors", 0))
        c4.metric("耗时(秒)", f"{stats.get('duration_sec', 0):.1f}")
        c5.metric("LLM调用", stats.get("llm_calls", 0))

        data = result.get("data", {}) or {}
        optimization = data.get("optimization", {}) or {}
        if optimization.get("optimized"):
            o1, o2, o3 = st.columns(3)
            o1.metric("最优优化方向", str(optimization.get("best_arm", "无"))[:18])
            o2.metric("改进幅度", f"{optimization.get('improvement', 0):.2%}")
            o3.metric("预算使用", f"{optimization.get('budget_used', 0)}/"
                      f"{optimization.get('budget', 0)}")

        if result.get("state") == "COMPLETED":
            st.success("🎉 复现流程成功完成！")
        elif result.get("state") == "ERROR":
            st.error(f"❌ 流程出错: {result.get('error', '未知错误')}")

# ===== Tab 2: 复现报告 =====
with tab2:
    st.markdown("### 📄 复现与优化报告")
    if st.session_state.result and st.session_state.result.get("data", {}).get("report"):
        report = st.session_state.result["data"]["report"]
        st.markdown(report)
    else:
        st.info("运行复现流程后，这里将显示完整的复现与优化报告。")

# ===== Tab 3: 审计日志 =====
with tab3:
    st.markdown("### 📜 审计日志")
    if st.session_state.logs:
        col1, col2 = st.columns(2)
        with col1:
            filter_agent = st.selectbox(
                "按Agent筛选",
                ["全部"] + sorted(
                    set(l.get("agent", "") for l in st.session_state.logs)),
                key="filter_agent_tab3")
        with col2:
            filter_status = st.selectbox(
                "按状态筛选",
                ["全部", "SUCCESS", "ERROR", "START", "RUNNING", "WARNING"],
                key="filter_status_tab3")

        filtered_logs = st.session_state.logs
        if filter_agent != "全部":
            filtered_logs = [l for l in filtered_logs
                             if l.get("agent") == filter_agent]
        if filter_status != "全部":
            filtered_logs = [l for l in filtered_logs
                             if l.get("status") == filter_status]

        for log in filtered_logs:
            status_color = {"SUCCESS": "🟢", "ERROR": "🔴", "START": "🟡",
                            "RUNNING": "🔄", "WARNING": "🟠"}.get(
                                log.get("status", ""), "⚪")
            with st.expander(
                f"{status_color} [{log.get('elapsed_sec', 0):.1f}s] "
                f"{log.get('agent', '?')} - {log.get('action', '?')}"
            ):
                st.json(log)
    else:
        st.info("运行复现流程后，这里将显示详细的审计日志。")

# ===== Tab 4: 状态机 =====
with tab4:
    st.markdown("### 🔍 状态机定义")
    st.markdown("系统使用有限状态机（FSM）管理 Agent 的流转。")
    state_info = """
```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> READ_PAPER
    READ_PAPER --> FIND_RESOURCES
    FIND_RESOURCES --> BUILD_ENV
    BUILD_ENV --> EXECUTE_CODE
    EXECUTE_CODE --> VALIDATE
    VALIDATE --> OPTIMIZING: 复现成功
    VALIDATE --> GENERATE_REPORT: 复现失败
    OPTIMIZING --> OPTIMIZED
    OPTIMIZED --> GENERATE_REPORT
    GENERATE_REPORT --> COMPLETED
    READ_PAPER --> ERROR
    FIND_RESOURCES --> ERROR
    BUILD_ENV --> ERROR
    EXECUTE_CODE --> ERROR
    VALIDATE --> ERROR
    OPTIMIZING --> ERROR
    GENERATE_REPORT --> ERROR
    ERROR --> INIT
    COMPLETED --> [*]
```
"""
    st.markdown(state_info)

    st.markdown("### 状态说明")
    state_data = [
        {"状态": "INIT", "说明": "初始化，等待输入", "Agent": "—"},
        {"状态": "READ_PAPER", "说明": "解析论文PDF/标题,提取结构化信息", "Agent": "PaperReader"},
        {"状态": "FIND_RESOURCES", "说明": "查找代码仓库和数据集", "Agent": "ResourceFinder"},
        {"状态": "BUILD_ENV", "说明": "构建环境 + 5轮依赖诊断", "Agent": "EnvBuilder"},
        {"状态": "EXECUTE_CODE", "说明": "smoke+full 双阶段执行", "Agent": "CodeExecutor"},
        {"状态": "VALIDATE", "说明": "验证结果与论文一致性", "Agent": "ResultValidator"},
        {"状态": "OPTIMIZING", "说明": "复现成功后,UCB 预算调度优化", "Agent": "Optimizer"},
        {"状态": "OPTIMIZED", "说明": "优化完成,产出最优方案", "Agent": "Optimizer"},
        {"状态": "GENERATE_REPORT", "说明": "生成Markdown复现+优化报告", "Agent": "ReportGenerator"},
        {"状态": "COMPLETED", "说明": "流水线完成", "Agent": "—"},
        {"状态": "ERROR", "说明": "出错状态，可重试", "Agent": "—"},
    ]
    st.table(state_data)


# ========== 事件处理 ==========
def _save_uploaded_pdf(uploaded_file) -> str:
    """将上传的 PDF 保存为临时文件，返回路径。"""
    suffix = os.path.splitext(uploaded_file.name or "paper.pdf")[1] or ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="autorepro_paper_")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tmp_path


def run_pipeline(paper_title="", pdf_path="", corpus_paper=None,
                 model_name="", base_url="", api_key="",
                 mock_mode=True, max_trials=10):
    """运行完整复现流水线（复现 -> 验证 -> 优化 -> 报告）。

    model_name/base_url/api_key 缺省时从环境变量 LLM_MODEL / LLM_BASE_URL /
    LLM_API_KEY 读取（客户端内部处理），因此真实模式无需本地 LLM 部署。
    """
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.agent_status = {}

    logger = AuditLogger()
    llm = LLMClient(
        mock_mode=mock_mode,
        model="" if mock_mode else model_name,
        base_url=base_url,
        api_key=api_key,
    )
    orchestrator = Orchestrator(llm_client=llm, mock_mode=mock_mode,
                                logger=logger, max_trials=max_trials)
    st.session_state.orchestrator = orchestrator

    # 修复：paper_title / pdf_path 现在会真实传递给流水线
    data = {
        "paper_title": paper_title,
        "pdf_path": pdf_path,
        "corpus_paper": corpus_paper,
    }

    agents_info = [
        ("READ_PAPER", "📖 PaperReader", "reader"),
        ("FIND_RESOURCES", "🔍 ResourceFinder", "finder"),
        ("BUILD_ENV", "🔧 EnvBuilder", "builder"),
        ("EXECUTE_CODE", "⚡ CodeExecutor", "executor"),
        ("VALIDATE", "✅ ResultValidator", "validator"),
    ]

    for state_name, display_name, agent_key in agents_info:
        st.session_state.current_state = state_name
        st.session_state.agent_status[display_name] = "running"
        yield

        agent = orchestrator.agents[agent_key]
        try:
            result = agent.run(data)

            if state_name == "READ_PAPER":
                data["paper_info"] = result.get("paper_info", {})
                data["raw_text"] = result.get("raw_text", "")
            elif state_name == "FIND_RESOURCES":
                data["resources"] = result.get("resources", {})
            elif state_name == "BUILD_ENV":
                data["env_config"] = result.get("env_config", {})
            elif state_name == "EXECUTE_CODE":
                data["execution"] = result
            elif state_name == "VALIDATE":
                data["validation"] = result

            # Prompt-Free 验证
            verif = orchestrator.agents["verifier"].run({
                "agent_name": agent.name,
                "system_prompt": getattr(agent, "system_prompt", "") or agent.name,
                "output": result,
            })
            data.setdefault("verifications", []).append(
                {"state": state_name, "agent": agent.name, **verif})

            data["total_llm_calls"] = data.get("total_llm_calls", 0) + \
                int(result.get("llm_calls", 0) or 0)

            st.session_state.agent_status[display_name] = "success"
            st.session_state.logs = logger.get_summary()
            yield

        except Exception as e:
            st.session_state.agent_status[display_name] = "error"
            st.session_state.current_state = "ERROR"
            st.session_state.logs = logger.get_summary()
            yield
            break

    st.session_state.agent_status["🛡️ Verifier"] = "success"

    # 优化阶段：仅在复现成功后触发
    if st.session_state.current_state != "ERROR":
        if data.get("validation", {}).get("is_reproduced"):
            st.session_state.current_state = "OPTIMIZING"
            st.session_state.agent_status["🧪 Optimizer"] = "running"
            yield
            try:
                data["optimization"] = orchestrator.agents["optimizer"].run(data)
                st.session_state.current_state = "OPTIMIZED"
                st.session_state.agent_status["🧪 Optimizer"] = "success"
            except Exception as e:
                st.session_state.current_state = "ERROR"
                st.session_state.agent_status["🧪 Optimizer"] = "error"
        else:
            data["optimization"] = {"optimized": False,
                                    "reason": "复现未成功,跳过优化"}
            st.session_state.agent_status["🧪 Optimizer"] = "waiting"
        st.session_state.logs = logger.get_summary()
        yield

    # 报告生成（合并复现 + 优化）
    if st.session_state.current_state != "ERROR":
        st.session_state.current_state = "GENERATE_REPORT"
        st.session_state.agent_status["📝 ReportGenerator"] = "running"
        yield
        try:
            data["report"] = orchestrator.agents["reporter"].run(data) \
                .get("report", "")
            st.session_state.agent_status["📝 ReportGenerator"] = "success"
        except Exception as e:
            st.session_state.current_state = "ERROR"
            st.session_state.agent_status["📝 ReportGenerator"] = "error"
        st.session_state.logs = logger.get_summary()
        yield

    if st.session_state.current_state != "ERROR":
        st.session_state.current_state = "COMPLETED"
        data["audit_summary"] = logger.get_stats()

    st.session_state.result = {
        "state": st.session_state.current_state,
        "error": None,
        "data": data,
        "audit_logs": logger.get_summary(),
        "audit_stats": logger.get_stats(),
    }
    st.session_state.logs = logger.get_summary()
    st.session_state.running = False
    yield


# 启动按钮处理
if start_btn:
    pt = st.session_state.paper_title or ""
    pdf_path = ""
    if not pt and not uploaded_file:
        st.error("请先输入论文标题或上传PDF文件")
    else:
        tmp_pdf = _save_uploaded_pdf(uploaded_file) if uploaded_file else ""
        with st.spinner("正在执行复现流程..."):
            try:
                for _ in run_pipeline(
                        paper_title=pt, pdf_path=tmp_pdf,
                        corpus_paper=corpus_paper,
                        model_name=model_name, base_url=base_url,
                        api_key=api_key,
                        mock_mode=st.session_state.mock_mode,
                        max_trials=max_trials):
                    time.sleep(0.3)
            finally:
                if tmp_pdf and os.path.exists(tmp_pdf):
                    try:
                        os.unlink(tmp_pdf)
                    except OSError:
                        pass
        st.rerun()

# 重置按钮处理
if reset_btn:
    st.session_state.orchestrator = None
    st.session_state.result = None
    st.session_state.running = False
    st.session_state.logs = []
    st.session_state.current_state = "INIT"
    st.session_state.agent_status = {}
    st.rerun()

# 底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em;">
    AutoReproducer v0.2.0 | 基于多智能体协作的论文自动复现与优化系统
</div>
""", unsafe_allow_html=True)