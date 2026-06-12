import streamlit as st
import json
import re
import time
import uuid
from backend.generation import RAGPipelineManager
from backend.bm25_retriever import retriever

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="RAG Agentic System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS Styling (Optional) ----------------
st.markdown("""
<style>
    .stChatInput {
        position: fixed !important;
        bottom: 30px !important;
        left: 21rem !important;
        right: 0 !important;
        width: calc(100% - 21rem) !important;
        z-index: 999 !important;
        background-color: var(--background-color, #ffffff) !important;
        padding: 0 1rem !important;
    }
    .block-container {padding-top: 2rem;}
    /* 优化 JSON 显示的字体大小 */
    .stJson {font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- Initialize Pipeline (Cached) ----------------
@st.cache_resource
def get_manager_default():
    manager = RAGPipelineManager()
    return manager

try:
    with st.spinner("Initializing Agent System & Loading Index..."):
        manager_default = get_manager_default()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    # 🆕 New Chat Button
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    st.subheader("👥 Group Members")
    st.markdown("""
    1. **Feng Yujie**
    2. **Wu Xuanxuan**
    3. **Fei Yang**
    4. **Gao Jing**
    """)
    
    st.divider()
    
    # 🔧 将配置项放在一个可折叠的设置区域
    with st.expander("⚙️ Settings", expanded=False):
        # 1. Retrieval Method (Placeholder for UI, logic is inside BM25 currently)
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["Dense Retrieval (BGE)", "Sparse (BM25)", "Hybrid (Sparse + Dense)"],
            index=0,
            help="Choose between Semantic Search (BGE), Keyword Search (BM25), or a Hybrid approach."
        )
        
        # 2. Generator Model
        generator_model = st.selectbox(
            "Generator Model",
            ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
            index=0,
            help="Model used for decomposition and reasoning."
        )

# ---------------- Main Interface ----------------

st.title("Agentic RAG Question Answering")
# st.caption("COMP5423 Group Project | Multi-hop Reasoning with Decomposition & Iterative Retrieval")

# Message history management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Session ID management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 如果有详细的执行日志，展示出来
        if "details" in msg:
            details = msg["details"]
            
            # 1. 展示 Agent 工作流日志 (Decomposition & Steps)
            with st.status("✅ Agent Workflow Trace", expanded=False, state="complete"):
                for log in details.get("logs", []):
                    if log["type"] == "plan":
                        st.info(f"**Decomposition:** {log.get('sub_questions', [])}")
                    elif log["type"] == "observe":
                        st.write(f"🔄 **Step {log.get('step')}:** {log.get('sub_question')}")
            
            # 2. 展示最终思考过程 (Thought Process)
            if details.get("thought"):
                st.write("###### 🧠 Final Reasoning")
                st.json(details["thought"])
            
            # 3. 展示所有用到的文档 (Docs)
            if details.get("docs"):
                with st.expander(f"📂 Aggregated Evidence ({len(details['docs'])} docs)"):
                    for doc in details["docs"]:
                        # 兼容 content 或 text 字段
                        txt = doc.get("content", doc.get("text", ""))
                        st.markdown(f"**[{doc['score']:.2f}] {doc['id']}**: {txt}")

# ---------------- Process User Input ----------------
if prompt := st.chat_input("Ask a complex question (e.g., Which film directed by the director of Inception won an Oscar?)..."):
    
    # 1. Display user question
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. System processing
    with st.chat_message("assistant"):
        # 创建一个状态容器，动态展示 Agent 的执行过程
        status_box = st.status("🚀 Agent is thinking...", expanded=True)
        
        # 先定义默认占位，避免异常分支引用未定义变量
        final_answer = ""
        formatted_retrieved_docs = []
        reasoning_steps = []
        rewritten_query = prompt
        plan_history = []
        known_facts = []

        try:
            # --- Call the Agentic Workflow (Iterative Planner) ---
            # 使用缓存的管理器以保留多轮上下文
            rag_manager = manager_default
            try:
                rag_manager.llm_handler.model_name = generator_model
                method_map = {
                    "Dense Retrieval (BGE)": "dense",
                    "Sparse (BM25)": "sparse",
                    "Hybrid (Sparse + Dense)": "hybrid"
                }
                selected_method = method_map.get(retrieval_method, "dense")
            except Exception:
                pass
            run_result = rag_manager.run(st.session_state.session_id, prompt, retrieval_method=selected_method)
            if isinstance(run_result, dict):
                final_answer = run_result.get("final_answer", "")
                formatted_retrieved_docs = run_result.get("formatted_retrieved_docs", [])
                reasoning_steps = run_result.get("reasoning_steps", [])
                rewritten_query = run_result.get("rewritten_query", prompt)
                plan_history = run_result.get("plan_history", [])
                known_facts = run_result.get("known_facts", [])
            elif isinstance(run_result, tuple):
                # 兼容旧版返回：尝试从 manager 的上下文补全展示所需信息
                if len(run_result) == 3:
                    final_answer, formatted_retrieved_docs, reasoning_steps = run_result
                elif len(run_result) == 2:
                    final_answer, formatted_retrieved_docs = run_result
                    reasoning_steps = []
                # 读取最近上下文以获取 rewritten_query 与 plan_history
                try:
                    ctx_list = rag_manager.multi_turn_manager.multi_turn_contexts.get(st.session_state.session_id, [])
                    if ctx_list:
                        last_ctx = ctx_list[-1]
                        rewritten_query = last_ctx.get("rewritten_query", rewritten_query)
                        plan_history = last_ctx.get("plan_history", plan_history)
                except Exception:
                    pass

            # 构建汇总文档（使用 BM25 文档映射）
            doc_map = {d.get("doc_id"): d.get("content", "") for d in getattr(retriever, "documents", [])}
            all_docs = []
            for item in formatted_retrieved_docs:
                try:
                    doc_id, score = item
                except Exception:
                    continue
                content = doc_map.get(doc_id, "")
                all_docs.append({
                    "id": doc_id,
                    "doc_id": doc_id,
                    "score": float(score),
                    "text": content,
                    "content": content
                })

            # 从返回结果/上下文中提取用于展示的信息（已在上方兼容设置）
            # 构造日志以贴近终端输出
            logs = []
            sub_questions = [h.get("sub_question") for h in plan_history if h.get("sub_question")]
            logs.append({
                "type": "plan",
                "step": 1,
                "sub_questions": sub_questions or [rewritten_query]
            })
            for idx, h in enumerate(plan_history):
                logs.append({
                    "type": "observe",
                    "step": idx + 1,
                    "sub_question": h.get("sub_question", ""),
                    "doc_count": h.get("doc_count", 0),
                    "answer": h.get("answer", "")
                })

            thought_data = {"thought_steps": reasoning_steps} if reasoning_steps else {}
            logs.append({
                "type": "final",
                "full_response": final_answer,
                "parsed_answer": final_answer,
                "thought_process": thought_data
            })
            
            # --- 渲染执行日志 (Replay Logs) ---
            thought_data = {}
            for log in logs:
                # Type 1: Planning / Decomposition
                if log["type"] == "plan":
                    status_box.write("📋 **Phase 1: Decomposition**")
                    if "sub_questions" in log:
                        status_box.json(log["sub_questions"])
                    else:
                        status_box.warning("Decomposition failed, falling back to direct query.")
                    status_box.write("---")

                # Type 2: Iterative Execution
                elif log["type"] == "observe":
                    step_num = log.get("step")
                    sub_q = log.get("sub_question")
                    docs_count = log.get("doc_count", 0)
                    answer_text = log.get("answer")
                    
                    status_box.write(f"🔄 **Phase 2 - Step {step_num}: Retrieval**")
                    status_box.markdown(f"> *Sub-question: {sub_q}*")
                    status_box.caption(f"Found {docs_count} relevant documents for this step.")
                    if answer_text:
                        status_box.markdown(f"**Answer:** {answer_text}")
                    
                    # 可选：展示每一步的具体文档（折叠）
                    # with status_box.expander(f"View Docs for Step {step_num}"):
                    #     for d in log.get("docs", []):
                    #         st.text(d.get("text", "")[:100] + "...")

                # Type 3: Final Synthesis
                elif log["type"] == "final":
                    status_box.write("---")
                    status_box.write("🧠 **Phase 3: Final Synthesis**")
                    status_box.write("Generating final answer based on all collected evidence...")
                    thought_data = log.get("thought_process", {})

            # Update status to complete
            status_box.update(label="✅ Agent Workflow Complete", state="complete", expanded=False)

            # ---------------- Display Results ----------------
            
            # 0. 展示改写后的问题（若可用）
            if rewritten_query and rewritten_query != prompt:
                st.caption(f"Rewritten Query: {rewritten_query}")

            # 1. 展示最终的推理步骤 (JSON)
            if thought_data:
                st.write("###### 🧠 Final Reasoning Steps")
                st.json(thought_data)
            
            # 2. 展示最终答案
            st.write("### Answer")
            st.success(final_answer)

            # 3. 展示汇总的证据文档
            with st.expander(f"📂 Aggregated Supporting Documents ({len(all_docs)})"):
                for i, doc in enumerate(all_docs):
                    # 兼容 backend 返回的字段名 (content 或 text)
                    content = doc.get("content", doc.get("text", ""))
                    st.info(f"**Doc {doc.get('id')}** (Score: {doc['score']:.2f})\n\n{content}")

            # Save to Session State
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "details": {
                    "logs": logs,             # 保存完整日志以便回看
                    "thought": thought_data,  # 最终思考
                    "docs": all_docs          # 所有文档
                }
            })

        except Exception as e:
            status_box.update(label="Error Occurred", state="error")
            st.error(f"An error occurred: {str(e)}")
            # 4. 展示已知事实（Known Facts）
            if known_facts:
                with st.expander(f"📌 Known Facts ({len(known_facts)})"):
                    for i, fact in enumerate(known_facts, 1):
                        st.write(f"{i}. {fact}")
