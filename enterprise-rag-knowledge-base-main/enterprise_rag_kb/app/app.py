"""
Enterprise RAG Knowledge Base - Streamlit Web UI
==================================================
Run with: streamlit run app/app.py
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    get_script_run_ctx = None


def _ensure_streamlit_runtime():
    """Exit cleanly if the file is launched with plain Python instead of streamlit run."""
    if get_script_run_ctx is not None and get_script_run_ctx() is None:
        print("This app must be launched with: streamlit run app/app.py")
        sys.exit(0)


_ensure_streamlit_runtime()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG Knowledge Base",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — ENHANCED PREMIUM UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Global - Fix text visibility ── */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 40%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 20px 60px rgba(48, 43, 99, 0.4);
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(124,58,237,0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }
    .hero-header p {
        margin: 0.5rem 0 0 0;
        color: #a5b4fc;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(124,58,237,0.25);
        border: 1px solid rgba(167,139,250,0.4);
        color: #c4b5fd;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 0.6rem;
        position: relative;
        z-index: 1;
    }

    /* ── Answer Box ── */
    .answer-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.85;
        color: #14532d;
        box-shadow: 0 8px 24px rgba(34,197,94,0.12);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    /* ── Source Card ── */
    .source-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 12px 12px 0;
        font-size: 0.92rem;
        color: #1e3a8a;
        line-height: 1.6;
        word-wrap: break-word;
    }

    /* ── Pipeline Step Cards ── */
    .pipeline-step {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #e9d5ff;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .pipeline-step:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(147,51,234,0.15);
    }
    .pipeline-step .step-icon {
        font-size: 1.1rem;
        margin-right: 0.3rem;
    }
    .pipeline-step .step-title {
        font-weight: 700;
        color: #581c87;
        font-size: 0.95rem;
    }
    .pipeline-step .step-desc {
        color: #7c3aed;
        font-size: 0.82rem;
        margin-top: 2px;
        line-height: 1.4;
    }

    /* ── Settings Card ── */
    .settings-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1rem;
        color: #0c4a6e;
        font-size: 0.88rem;
        line-height: 1.8;
    }
    .settings-card b { color: #0369a1; }

    /* ── Confidence Badge ── */
    .conf-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    .conf-high { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .conf-mid  { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .conf-low  { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 0.55rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(124,58,237,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%);
        box-shadow: 0 6px 20px rgba(124,58,237,0.45);
        transform: translateY(-2px);
        color: white !important;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 60%, #24243e 100%);
        border-right: 1px solid rgba(167,139,250,0.15);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #a5b4fc !important;
        font-size: 0.85rem;
        font-weight: 500;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] .stSuccess {
        background: rgba(34,197,94,0.15) !important;
        border: 1px solid rgba(34,197,94,0.3) !important;
    }
    [data-testid="stSidebar"] .stError {
        background: rgba(239,68,68,0.15) !important;
        border: 1px solid rgba(239,68,68,0.3) !important;
    }
    /* Make sidebar metric text visible */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #a5b4fc !important;
    }
    [data-testid="stSidebar"] [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(167,139,250,0.2) !important;
        border-radius: 10px;
        padding: 0.6rem !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 100%) !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(167,139,250,0.15) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        border-radius: 12px;
        padding: 5px;
        border: 1px solid #e9d5ff;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        font-size: 0.88rem;
        color: #6b7280 !important;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #7c3aed !important;
        box-shadow: 0 4px 12px rgba(124,58,237,0.15);
        font-weight: 700;
    }

    /* ── Metric containers ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
        border-radius: 12px;
        padding: 0.9rem;
        border: 1px solid #e9d5ff;
        box-shadow: 0 2px 8px rgba(124,58,237,0.06);
    }
    [data-testid="stMetricValue"] {
        color: #1e1b4b !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #6b21a8 !important;
        font-weight: 500;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
        font-size: 0.92rem;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #6b7280;
        background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
        border-radius: 16px;
        border: 2px dashed #d8b4fe;
    }
    .empty-state h3 { color: #581c87; margin-bottom: 0.5rem; }
    .empty-state p { color: #7c3aed; }

    /* ── Status badges ── */
    .badge-ready {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #15803d;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #86efac;
    }
    .badge-empty {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #fca5a5;
    }

    /* ── History ── */
    .history-q {
        font-weight: 700;
        color: #581c87;
        margin-bottom: 0.4rem;
        font-size: 1rem;
    }
    .history-a {
        color: #374151;
        font-size: 0.95rem;
        line-height: 1.7;
        word-wrap: break-word;
    }

    /* ── Evaluation info box ── */
    .eval-info {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #d8b4fe;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: #581c87;
    }
    .eval-info b { color: #7c3aed; }
    .eval-info li { margin: 0.3rem 0; }

    /* ── Architecture diagram ── */
    .arch-flow {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%);
        border-radius: 14px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        text-align: center;
        border: 1px solid rgba(167,139,250,0.3);
    }
    .arch-flow .flow-text {
        color: #c4b5fd;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.8;
        letter-spacing: 0.5px;
    }
    .arch-flow .flow-arrow { color: #8b5cf6; font-weight: 700; }

    /* ── Data table styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Fix text overflow everywhere ── */
    p, span, div, li, label, .stMarkdown {
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #8b5cf6; }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=" Loading RAG Pipeline... (first load takes ~30s)")
def load_pipeline():
    from src.pipeline.rag_pipeline import RAGPipeline
    return RAGPipeline(auto_load=True)


def init_session():
    defaults = {
        "query_history": [],
        "last_result": None,
        "prefill_query": "",
        "clear_counter": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1> Enterprise RAG Knowledge Base</h1>
    <p>Retrieval-Augmented Generation &nbsp;•&nbsp; Hybrid Search &nbsp;•&nbsp; Cross-Encoder Reranking &nbsp;•&nbsp; LLM Generation</p>
    <div class="hero-badge"> Powered by Llama 3.3 70B &nbsp;|&nbsp; FAISS &nbsp;|&nbsp; Sentence Transformers</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  Control Panel")

    # ── Load Pipeline ──────────────────────────────────────────────────────
    try:
        pipeline = load_pipeline()
        stats = pipeline.get_stats()
        status = stats.get("pipeline_status", "empty")
        st.markdown(
            f"**Status:** <span class='badge-{'ready' if status=='ready' else 'empty'}'>"
            f"{' Ready' if status=='ready' else ' Empty'}</span>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(" Pipeline initialization failed")
        st.exception(e)
        st.info("Run: `pip install -r requirements.txt`")
        st.stop()

    st.markdown("---")

    # ── KB Stats ───────────────────────────────────────────────────────────
    st.markdown("###  Knowledge Base")
    vs = stats.get("vector_store", {})
    c1, c2 = st.columns(2)
    c1.metric(" Chunks",     vs.get("total_chunks", 0))
    c1.metric(" Documents",  vs.get("unique_documents", 0))
    c2.metric(" Sources",   vs.get("unique_sources", 0))
    c2.metric(" Queries",    stats.get("queries_served", 0))

    st.markdown("---")

    # ── Retrieval Settings ─────────────────────────────────────────────────
    st.markdown("###  Retrieval Settings")

    retrieval_mode = st.selectbox(
        "Mode",
        ["hybrid", "dense", "sparse"],
        index=0,
        help="hybrid = BM25 + FAISS | dense = FAISS only | sparse = BM25 only"
    )

    top_k = st.slider(
        "Top-K Chunks", min_value=1, max_value=10, value=5,
        help="Number of document chunks to retrieve"
    )

    use_reranker = st.toggle(
        " Cross-Encoder Reranker", value=False,
        help="Enables precision reranking — recommended ON"
    )

    st.markdown("---")

    # ── Document Upload ────────────────────────────────────────────────────
    st.markdown("###  Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, Markdown, CSV, JSON",
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button(" Ingest Uploaded Files", use_container_width=True):
            with st.spinner("Processing documents..."):
                upload_dir = Path(__file__).parent.parent / "data" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                results = []
                for f in uploaded_files:
                    file_path = upload_dir / f.name
                    with open(file_path, "wb") as out:
                        out.write(f.getbuffer())
                    result = pipeline.ingest_file(str(file_path))
                    results.append(result)
                total_chunks = sum(r.get("chunks_added", 0) for r in results)
                st.success(f" {len(uploaded_files)} file(s) → {total_chunks} chunks added!")
                st.rerun()

    st.markdown("---")

    # ── Quick Load Sample Docs ─────────────────────────────────────────────
    st.markdown("###  Quick Demo")
    if st.button(" Load Sample Documents", use_container_width=True,
                 help="Loads 4 built-in demo documents (policy, tech guide, FAQ, finance)"):
        with st.spinner("Creating & ingesting sample documents..."):
            try:
                from scripts.ingest_documents import create_sample_documents
                create_sample_documents()
                sample_dir = Path(__file__).parent.parent / "data" / "sample_documents"

                # ── Skip already-ingested files ──────────────────────────────
                existing_sources = {doc["source"] for doc in pipeline.list_documents()}
                new_files = [
                    str(f) for f in sample_dir.glob("*.*")
                    if str(f) not in existing_sources
                ]

                if not new_files:
                    st.info(" Sample documents are already loaded!")
                else:
                    total_chunks = 0
                    total_docs = 0
                    for file_path in new_files:
                        result = pipeline.ingest_file(file_path)
                        total_chunks += result.get("chunks_added", 0)
                        total_docs += result.get("documents_loaded", 0)
                    st.success(f" {total_docs} doc(s) → {total_chunks} chunks!")
                    st.rerun()
            except Exception as e:
                st.error(f" Failed: {e}")

    if st.button(" Reset Knowledge Base", use_container_width=True,
                 help="Clears all ingested documents from the index"):
        with st.spinner("Resetting..."):
            pipeline.reset_index()
            st.cache_resource.clear()
            st.success(" Knowledge base cleared!")
            st.rerun()

    st.markdown("---")

    # ── API Config ─────────────────────────────────────────────────────────
    with st.expander(" Groq API Key"):
        current_key = pipeline.llm.groq_api_key
        key_status = " Configured" if current_key else " Not set"
        st.caption(key_status)
        groq_key = st.text_input(
            "New Key",
            type="password",
            placeholder="gsk_...",
            help="Get free key at console.groq.com/keys",
            label_visibility="collapsed"
        )
        if groq_key and st.button("Save Key", use_container_width=True):
            pipeline.llm.groq_api_key = groq_key
            st.success(" Key updated!")

    # ── Model Info ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(" System Info"):
        st.caption(f"**Embedding:** `{stats.get('embedding_model','').split('/')[-1]}`")
        st.caption(f"**LLM:** `{stats.get('llm_model','').split('/')[-1]}`")
        st.caption(f"**Reranker:** `{stats.get('reranker','')}`")
        st.caption(f"**Retrieval:** `{stats.get('retrieval_mode','')}`")


# ── Apply settings to pipeline ────────────────────────────────────────────────
pipeline.cfg.retrieval.retrieval_mode = retrieval_mode
pipeline.cfg.retrieval.use_reranker = use_reranker


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_query, tab_docs, tab_history, tab_arch, tab_eval = st.tabs([
    " Ask a Question",
    " Knowledge Base",
    " Query History",
    " Architecture",
    " Evaluation",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Query
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:
    col_main, col_info = st.columns([3, 1.2])

    with col_main:
        st.markdown("###  Ask Anything About Your Documents")

        # ── Input ──────────────────────────────────────────────────────────
        query_input = st.text_area(
            "Your Question",
            placeholder="e.g., What is the company's data privacy policy?\n      What are the financial highlights for FY2024?",
            height=120,
            label_visibility="collapsed",
            key=f"query_textarea_{st.session_state.clear_counter}",
        )

        col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1, 3])
        with col_btn1:
            search_btn = st.button(" Search", use_container_width=True, type="primary")
        with col_btn2:
            clear_btn = st.button(" Clear", use_container_width=True)

        if clear_btn:
            st.session_state.last_result = None
            st.session_state.clear_counter += 1
            st.rerun()

        # ── Example Questions ───────────────────────────────────────────────
        with st.expander(" Try Example Questions", expanded=False):
            examples = [
                "What is the company's return and refund policy?",
                "What are the data privacy tiers and their requirements?",
                "What were the key financial highlights for FY2024?",
                "How does the hybrid retrieval system work?",
                "What is the password policy for employees?",
                "Which languages does the system support?",
            ]
            
            def set_query(q):
                st.session_state[f"query_textarea_{st.session_state.clear_counter}"] = q
                
            cols = st.columns(2)
            for i, ex in enumerate(examples):
                with cols[i % 2]:
                    st.button(f"→ {ex}", key=f"ex_{i}", on_click=set_query, args=(ex,), use_container_width=True)

        # ── Search ──────────────────────────────────────────────────────────
        if search_btn and query_input.strip():
            if not pipeline.vector_store.is_loaded():
                st.warning(" Knowledge base is empty! Load sample documents from the sidebar first.")
            else:
                with st.spinner(" Searching knowledge base and generating answer..."):
                    result = pipeline.query(query_input.strip(), top_k=top_k)
                    st.session_state.last_result = result
                    st.session_state.query_history.insert(0, {
                        "question": query_input.strip(),
                        "answer": result.get("answer", ""),
                        "confidence": result.get("confidence", 0),
                        "time_ms": result.get("retrieval_time_ms", 0),
                        "model": result.get("model", "N/A"),
                        "source_count": len(result.get("sources", [])),
                    })

        elif search_btn and not query_input.strip():
            st.warning("Please enter a question first.")

        # ── Result Display ───────────────────────────────────────────────────
        if st.session_state.last_result:
            result = st.session_state.last_result
            st.markdown("---")

            # Confidence + metrics row
            conf = result.get("confidence", 0)
            if conf > 0.6:
                conf_label = " High Confidence"
                conf_class = "conf-high"
            elif conf > 0.3:
                conf_label = " Medium Confidence"
                conf_class = "conf-mid"
            else:
                conf_label = " Low Confidence"
                conf_class = "conf-low"

            model_name = result.get("model", "N/A")
            model_short = model_name.split("/")[-1] if "/" in model_name else model_name

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(" Confidence",    f"{conf:.1%}")
            m2.metric(" Response Time", f"{result.get('retrieval_time_ms', 0)} ms")
            m3.metric(" Model",         model_short)
            m4.metric(" Sources",       len(result.get("sources", [])))

            # Confidence badge
            st.markdown(
                f"<div style='margin: 0.5rem 0;'><span class='conf-badge {conf_class}'>{conf_label} — {conf:.1%}</span></div>",
                unsafe_allow_html=True
            )

            # Answer box
            st.markdown("####  Generated Answer")
            answer_text = result.get("answer", "No answer generated.").replace("\n", "<br>")
            st.markdown(f"""
            <div class="answer-box">
                {answer_text}
            </div>
            """, unsafe_allow_html=True)

            # Sources
            sources = result.get("sources", [])
            if sources:
                st.markdown(f"####  Retrieved Sources ({len(sources)} chunks)")
                for i, src in enumerate(sources):
                    relevance = src.get("relevance_score", 0)
                    rel_pct = f"{relevance:.1%}"
                    filename = src.get("filename") or Path(src.get("source", "Unknown")).name
                    page_str = f" • Page {src['page']}" if src.get("page") else ""

                    with st.expander(
                        f" [{i+1}] {filename}{page_str}  —  Relevance: {rel_pct}"
                    ):
                        st.markdown(f"""
                        <div class="source-card">
                            <b> File:</b> {src.get('source', 'N/A')}<br>
                            <b> Chunk ID:</b> <code>{src.get('chunk_id', 'N/A')}</code><br>
                            <b> Relevance:</b> {relevance:.4f}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("** Content Preview:**")
                        st.code(src.get("preview", ""), language=None)

    # ── Info panel ─────────────────────────────────────────────────────────
    with col_info:
        st.markdown("###  Pipeline Steps")
        steps = [
            ("", "Embed Query",  "Question → 384-dim dense vector via Sentence-BERT"),
            ("", "Retrieve",     "FAISS + BM25 hybrid search over knowledge base"),
            ("", "Rerank",       "Cross-encoder reranking for precision boost"),
            ("", "Generate",     "Llama 3.3 70B creates grounded answer via Groq"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="step-icon">{icon}</span>
                <span class="step-title">{title}</span><br>
                <span class="step-desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("###  Current Settings")
        st.markdown(f"""
        <div class="settings-card">
            <b>Mode:</b> {retrieval_mode}<br>
            <b>Top-K:</b> {top_k} chunks<br>
            <b>Reranker:</b> {' ON' if use_reranker else ' OFF'}<br>
            <b>KB Size:</b> {vs.get('total_chunks', 0)} chunks<br>
            <b>LLM:</b> Llama 3.3 70B
        </div>
        """, unsafe_allow_html=True)

        if not pipeline.vector_store.is_loaded():
            st.markdown("---")
            st.warning(" **Knowledge base is empty!**\n\nUse the sidebar → **Load Sample Documents** to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("###  Ingested Documents")

    docs = pipeline.list_documents()
    if docs:
        df = pd.DataFrame(docs)

        # Summary row
        d1, d2, d3 = st.columns(3)
        d1.metric(" Total Documents", len(docs))
        d2.metric(" Total Chunks",    sum(d["chunk_count"] for d in docs))
        d3.metric(" File Types",     len(set(d.get("file_type","") for d in docs)))

        st.markdown("---")

        # Documents table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source":      st.column_config.TextColumn(" Source Path",  width="large"),
                "filename":    st.column_config.TextColumn(" Filename",      width="medium"),
                "file_type":   st.column_config.TextColumn(" Type",          width="small"),
                "chunk_count": st.column_config.NumberColumn(" Chunks",       width="small", format="%d"),
            }
        )

        # Bar chart — chunks per document (only if > 1 doc)
        if len(docs) > 1:
            st.markdown("####  Chunks per Document")
            chart_data = pd.DataFrame({
                "Document": [d.get("filename") or str(d["source"]).split("/")[-1] for d in docs],
                "Chunks":   [d["chunk_count"] for d in docs],
            }).set_index("Document")
            st.bar_chart(chart_data, color="#7c3aed")
        else:
            st.info("Upload more documents to see the chunks distribution chart.")

    else:
        st.markdown("""
        <div class="empty-state">
            <h3> No Documents Ingested Yet</h3>
            <p>Use the sidebar to upload files or load sample documents.</p>
        </div>
        """, unsafe_allow_html=True)

    # Vector store raw stats
    st.markdown("---")
    with st.expander(" Vector Store — Raw Stats"):
        st.json(pipeline.get_stats())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Query History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("###  Query History")

    history = st.session_state.query_history
    if history:
        avg_conf = sum(h["confidence"] for h in history) / len(history)
        avg_time = sum(h.get("time_ms", 0) for h in history) / len(history)

        h1, h2, h3 = st.columns(3)
        h1.metric(" Total Queries",    len(history))
        h2.metric(" Avg Confidence",  f"{avg_conf:.1%}")
        h3.metric(" Avg Response",     f"{avg_time:.0f} ms")

        st.markdown("---")

        for i, h in enumerate(history):
            conf = h["confidence"]
            conf_emoji = "" if conf > 0.6 else ("" if conf > 0.3 else "")
            q_preview = h["question"][:60] + "..." if len(h["question"]) > 60 else h["question"]
            with st.expander(
                f"Q{len(history)-i}: {q_preview}  |  {conf_emoji} {conf:.0%}  |  {h.get('time_ms',0)}ms"
            ):
                st.markdown(f'<div class="history-q"> {h["question"]}</div>', unsafe_allow_html=True)
                ans_preview = h["answer"][:600] + ("..." if len(h["answer"]) > 600 else "")
                st.markdown(f'<div class="history-a"> {ans_preview}</div>', unsafe_allow_html=True)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Confidence",  f"{conf:.4f}")
                col_b.metric("Time",        f'{h.get("time_ms",0)} ms')
                col_c.metric("Sources",     h.get("source_count", "N/A"))

        st.markdown("---")
        if st.button(" Clear All History"):
            st.session_state.query_history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
            <h3> No Query History Yet</h3>
            <p>Ask questions in the <b> Ask a Question</b> tab to see history here.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("###  System Architecture")
    st.markdown("A visual overview of how the Enterprise RAG pipeline processes your queries.")

    st.markdown("---")

    # Architecture flow
    st.markdown("""
    <div class="arch-flow">
        <div class="flow-text">
             <b>Documents</b> &nbsp;<span class="flow-arrow">→</span>&nbsp;
             <b>Chunking</b> &nbsp;<span class="flow-arrow">→</span>&nbsp;
             <b>Embeddings</b> &nbsp;<span class="flow-arrow">→</span>&nbsp;
             <b>FAISS Index</b>
            <br>
             <b>Query</b> &nbsp;<span class="flow-arrow">→</span>&nbsp;
             <b>Hybrid Search</b> &nbsp;<span class="flow-arrow">→</span>&nbsp;
             <b>Reranker</b> &nbsp;<span class="flow-arrow">→</span>&nbsp;
             <b>LLM Answer</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Component details
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("####  Ingestion Pipeline")
        components_ingestion = [
            (" Document Loader", "Multi-format support: PDF, TXT, MD, CSV, JSON, DOCX"),
            (" Text Preprocessor", "Unicode normalization, whitespace cleanup, noise removal"),
            (" Recursive Splitter", "Intelligent chunking with configurable size and overlap"),
            (" Embedding Engine", "Sentence-Transformers (all-MiniLM-L6-v2) → 384-dim vectors"),
            (" FAISS Vector Store", "Facebook AI Similarity Search with persistent index"),
        ]
        for title, desc in components_ingestion:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="step-title">{title}</span><br>
                <span class="step-desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("####  Query Pipeline")
        components_query = [
            (" Hybrid Retriever", "Dense (FAISS) + Sparse (BM25) with configurable fusion"),
            (" Cross-Encoder Reranker", "ms-marco-MiniLM-L-6-v2 for precision re-scoring"),
            (" LLM Generation", "Groq API → Llama 3.3 70B (70 billion parameters)"),
            (" Confidence Estimator", "Retrieval score + answer quality → confidence %"),
            (" Fallback Chain", "Groq → Gemini → Extractive (always provides an answer)"),
        ]
        for title, desc in components_query:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="step-title">{title}</span><br>
                <span class="step-desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tech Stack
    st.markdown("####  Technology Stack")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div class="settings-card" style="text-align:center;">
            <b> Python 3.13</b><br>
            <span style="font-size:0.8rem;">Core Language</span>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="settings-card" style="text-align:center;">
            <b> Transformers</b><br>
            <span style="font-size:0.8rem;">NLP Models</span>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="settings-card" style="text-align:center;">
            <b> FAISS</b><br>
            <span style="font-size:0.8rem;">Vector Search</span>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="settings-card" style="text-align:center;">
            <b> Groq API</b><br>
            <span style="font-size:0.8rem;">LLM Inference</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("###  RAG Pipeline Evaluation")

    st.markdown("""
    <div class="eval-info">
    <b> Evaluation Metrics Used:</b>
    <ul>
        <li><b>Context Precision</b> — What fraction of retrieved chunks are relevant?</li>
        <li><b>Context Recall</b> — How much of the ground truth is in the context?</li>
        <li><b>Answer Faithfulness</b> — Is the answer grounded in retrieved context?</li>
        <li><b>Answer Relevancy</b> — Does the answer address the question?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Guard: KB must not be empty ────────────────────────────────────────
    if not pipeline.vector_store.is_loaded():
        st.warning(" Knowledge base is empty. Load documents first before evaluating.")
    else:
        # ── Manual Evaluation ──────────────────────────────────────────────
        st.markdown("####  Manual Evaluation")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            eval_question = st.text_input(
                "Test Question",
                placeholder="e.g., What is the return policy?",
            )
        with col_e2:
            eval_gt = st.text_area(
                "Ground Truth Answer",
                placeholder="Enter the expected correct answer...",
                height=100,
            )

        if st.button(" Evaluate This Sample", type="primary"):
            if not eval_question or not eval_gt:
                st.warning("Please fill in both the question and the ground truth answer.")
            else:
                with st.spinner("Running evaluation..."):
                    try:
                        from src.evaluation.evaluator import RAGEvaluator, EvalSample

                        # fast_mode=True: skips reranker + LLM API call (~1s vs 30s)
                        result = pipeline.query(eval_question, fast_mode=True)
                        contexts = [s["preview"] for s in result.get("sources", [])]

                        evaluator = RAGEvaluator(use_semantic_similarity=False)
                        sample = EvalSample(
                            question=eval_question,
                            ground_truth_answer=eval_gt,
                            generated_answer=result.get("answer", ""),
                            retrieved_contexts=contexts,
                        )
                        eval_result = evaluator.evaluate_sample(sample)

                        # Display metrics with color-coded grades
                        st.markdown("** Evaluation Results:**")
                        e1, e2, e3, e4, e5 = st.columns(5)
                        metrics = [
                            (e1, "Ctx Precision", eval_result.context_precision),
                            (e2, "Ctx Recall",    eval_result.context_recall),
                            (e3, "Faithfulness",  eval_result.answer_faithfulness),
                            (e4, "Relevancy",     eval_result.answer_relevancy),
                            (e5, " Overall",    eval_result.overall_score),
                        ]
                        for col, name, val in metrics:
                            grade = "A" if val > 0.8 else ("B" if val > 0.6 else ("C" if val > 0.4 else "D"))
                            col.metric(name, f"{val:.2f}", f"Grade: {grade}")

                        # Show the generated answer too
                        st.markdown("**Generated Answer:**")
                        st.info(result.get("answer", ""))

                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

        st.markdown("---")

        # ── Batch Evaluation ────────────────────────────────────────────────
        st.markdown("####  Batch Evaluation from JSON File")
        st.caption('Upload a JSON file with format: `[{"question": "...", "answer": "..."}]`')

        eval_file = st.file_uploader(
            "Upload evaluation dataset (JSON)",
            type=["json"],
            key="eval_upload"
        )

        if eval_file and st.button(" Run Batch Evaluation", type="primary"):
            with st.spinner("Running batch evaluation..."):
                try:
                    import tempfile
                    from src.evaluation.evaluator import RAGEvaluator

                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        tmp.write(eval_file.getbuffer())
                        tmp_path = tmp.name

                    evaluator = RAGEvaluator(use_semantic_similarity=False)
                    samples = evaluator.load_eval_dataset(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)

                    progress = st.progress(0, text="Evaluating samples...")
                    for idx, sample in enumerate(samples):
                        result = pipeline.query(sample.question, fast_mode=True)
                        sample.generated_answer = result.get("answer", "")
                        sample.retrieved_contexts = [s["preview"] for s in result.get("sources", [])]
                        progress.progress((idx + 1) / len(samples), text=f"Sample {idx+1}/{len(samples)}")

                    eval_results = evaluator.evaluate_dataset(samples)
                    progress.empty()

                    st.success(f" Evaluated {eval_results['num_samples']} samples")

                    # Summary metrics
                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("Ctx Precision",  f"{eval_results['context_precision']:.3f}")
                    r2.metric("Ctx Recall",     f"{eval_results['context_recall']:.3f}")
                    r3.metric("Faithfulness",   f"{eval_results['answer_faithfulness']:.3f}")
                    r4.metric("Relevancy",      f"{eval_results['answer_relevancy']:.3f}")
                    r5.metric(" Overall",      f"{eval_results['overall_score']:.3f}")

                    # Per-sample table
                    if eval_results.get("per_sample"):
                        st.markdown("**Per-Sample Results:**")
                        per_df = pd.DataFrame(eval_results["per_sample"])
                        st.dataframe(per_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Batch evaluation failed: {e}")
