import streamlit as st
import requests
import pandas as pd
import altair as alt

# 1. 页面配置：使用 Wide 模式并设置静谧的主题色
st.set_page_config(
    page_title="RAG Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 自定义高级 CSS
st.markdown("""
<style>
    /* 核心修复：隐藏 Header 背景，但保留按钮可见性 */
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important; /* 背景透明 */
    }

    /* 确保展开按钮（那个 > 符号）是可见的，并且颜色与背景协调 */
    button[kind="header"] {
        color: #4F46E5 !important; 
    }

    /* 侧边栏本身的样式美化 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #E2E8F0;
    }
    
    /* 让主容器更有呼吸感 */
    .main .block-container {
        padding-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：参数配置 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)
    st.title("系统配置")
    st.markdown("---")
    
    system = st.selectbox(
        "🧠 核心引擎",
        ["LangChain", "LlamaIndex", "LightRAG"],
        help="选择底层检索生成的框架"
    )
    
    use_compress = st.toggle("⚙️ 启用上下文压缩", value=True)
    
    st.info("提示：开启压缩算法可以有效节省 Token 并提升响应速度。")

# --- 主界面：交互区 ---
st.title("💠 RAG Intelligence")
st.caption("基于检索增强生成的智能问答系统")

# 使用列布局美化输入行
query = st.text_input("💬 提问", placeholder="请输入您想查询的问题...", label_visibility="collapsed")
col_run, col_empty = st.columns([1, 4])
with col_run:
    run_btn = st.button("🚀 发送指令")

# --- 逻辑处理与结果展示 ---
if run_btn and query:
    with st.spinner("正在检索知识库并生成回答..."):
        try:
            # 模拟请求逻辑（替换为你的实际接口）
            res = requests.post("http://localhost:8000/query", json={
                "query": query,
                "system": system,
                "compress": use_compress
            }).json()
            
            
            # 展示回答区
            st.markdown("### 📌 AI 生成结果")
            st.markdown(f"""<div class="answer-card">{res['answer']}</div>""", unsafe_allow_html=True)
            
            st.write("") # 留白
            
            # --- Token 可视化区 ---
            st.markdown("### 📊 消耗分析")
            
            tokens = res["tokens"]
            if isinstance(tokens, dict):
                # 1. 顶层指标卡片
                max_limit = 8000
                total_tokens = tokens.get("Total", 0)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("总消耗 (Total)", tokens.get("Total", 0))
                m2.metric("提示词 (Prompt)", tokens.get("Prompt", 0))
                m3.metric("生成量 (Completion)", tokens.get("Completion", 0))
                
                # 2. 高级可视化图表 (Altair)
                df = pd.DataFrame(list(tokens.items()), columns=["Category", "Count"])
                
                chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                    x=alt.X('Category:N', title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Count:Q', title="Token Count"),
                    color=alt.Color('Category:N', scale=alt.Scale(range=['#4F46E5', '#7C3AED', '#EC4899']), legend=None)
                ).properties(height=300)
                
                st.altair_chart(chart, width='stretch')
                
                # 3. 进度条展示限额
                usage_pct = min(total_tokens / max_limit, 1.0)
                st.write(f"窗口占用率: {usage_pct*100:.1f}% ({total_tokens} / {max_limit})")
                st.progress(usage_pct)
                
        except Exception as e:
            st.error(f"连接失败：{str(e)}")

elif run_btn and not query:
    st.warning("请输入问题后再运行。")

# --- 页脚 ---
st.markdown("---")
st.caption("© 2026 RAG Demo System")