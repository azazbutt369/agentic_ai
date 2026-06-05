"""
==============================================================================
ML Paper Tutor Agent — Streamlit Frontend
==============================================================================

This is the main user interface for the Paper Tutor Agent.
It provides:
    - PDF upload and ingestion
    - Paper library management
    - Chat interface for Q&A, summarization, and comparison
    - Source citation display
    - Conversation history management
    - Ollama connection status

LAUNCH:
    cd paper_agent
    streamlit run ui/app.py
==============================================================================
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from config.settings import ensure_directories, LLM_MODEL, EMBEDDING_MODEL
from ingestion.pipeline import ingest_pdf, save_uploaded_pdf
from embeddings.embedder import get_embedding_model
from vectorstore.faiss_store import (
    create_index,
    save_index,
    load_index,
    add_documents,
    get_index_stats,
    index_exists,
)
from generation.chain import PaperTutorChain, ChainResponse
from generation.llm_client import check_ollama_connection
from generation.remote_inference import check_remote_connection
from memory.chat_memory import ChatMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# Page Configuration
# ===========================================================================

st.set_page_config(
    page_title="ML Paper Tutor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# Custom CSS — Premium dark theme with glassmorphism
# ===========================================================================

st.markdown("""
<style>
    /* ---------- Import Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- Global ---------- */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Sidebar styling ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0ff !important;
    }

    /* ---------- Status badges ---------- */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    .status-connected {
        background: rgba(46, 213, 115, 0.15);
        color: #2ed573;
        border: 1px solid rgba(46, 213, 115, 0.25);
    }
    .status-disconnected {
        background: rgba(255, 71, 87, 0.15);
        color: #ff4757;
        border: 1px solid rgba(255, 71, 87, 0.25);
    }

    /* ---------- Paper chips ---------- */
    .paper-chip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s ease;
        font-size: 0.85rem;
    }
    .paper-chip:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(108, 92, 231, 0.4);
    }
    .paper-chip .paper-name {
        color: #e0e0ff;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 180px;
    }
    .paper-chip .paper-pages {
        color: #7f8c8d;
        font-size: 0.75rem;
        flex-shrink: 0;
    }

    /* ---------- Mode selector cards ---------- */
    .mode-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.02);
        margin: 4px 0;
        transition: all 0.25s ease;
        cursor: pointer;
    }
    .mode-card:hover {
        background: rgba(108, 92, 231, 0.1);
        border-color: rgba(108, 92, 231, 0.3);
    }
    .mode-card .mode-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #e0e0ff;
        margin-bottom: 3px;
    }
    .mode-card .mode-desc {
        font-size: 0.75rem;
        color: #7f8c8d;
        line-height: 1.4;
    }

    /* ---------- Header area ---------- */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe, #74b9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #7f8c8d;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* ---------- Stats row ---------- */
    .stats-row {
        display: flex;
        gap: 12px;
        justify-content: center;
        margin: 1rem 0 1.5rem 0;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 20px;
        text-align: center;
        min-width: 120px;
    }
    .stat-card .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #a29bfe;
    }
    .stat-card .stat-label {
        font-size: 0.72rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* ---------- Source citation expanders ---------- */
    .source-card {
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 10px;
        background: rgba(108, 92, 231, 0.06);
        border-left: 3px solid #6c5ce7;
        font-size: 0.84rem;
        line-height: 1.6;
    }
    .source-card .source-header {
        font-weight: 600;
        color: #a29bfe;
        font-size: 0.8rem;
        margin-bottom: 6px;
    }
    .source-card .source-score {
        color: #2ed573;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .source-card .source-text {
        color: #bdc3c7;
        font-size: 0.82rem;
    }

    /* ---------- Welcome card ---------- */
    .welcome-card {
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        margin: 2rem auto;
        max-width: 600px;
    }
    .welcome-card .welcome-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .welcome-card h3 {
        color: #e0e0ff;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .welcome-card p {
        color: #7f8c8d;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ---------- Chat message refinements ---------- */
    .stChatMessage {
        border-radius: 14px !important;
        margin-bottom: 8px;
    }

    /* ---------- Dividers ---------- */
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session State Initialization
# ===========================================================================


def init_session_state():
    """Initialize all session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_history = []          # List of (role, content) tuples
        st.session_state.memory = ChatMemory()
        st.session_state.index = None               # FAISS index
        st.session_state.chain = None                # PaperTutorChain
        st.session_state.papers = {}                 # {filename: metadata}
        st.session_state.mode = "qa"                 # Active mode
        st.session_state.total_chunks = 0
        st.session_state.embedding_model = None
        st.session_state.last_sources = []           # Sources from last query
        st.session_state.use_remote = False            # Remote GPU inference
        st.session_state.remote_url = ""               # ngrok URL

        ensure_directories()

        # Try to load existing index
        if index_exists():
            try:
                model = get_embedding_model()
                st.session_state.embedding_model = model
                st.session_state.index = load_index(embedding_model=model)
                stats = get_index_stats(st.session_state.index)
                st.session_state.total_chunks = stats["total_vectors"]
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")


init_session_state()


# ===========================================================================
# Sidebar
# ===========================================================================


def render_sidebar():
    """Render the sidebar with upload, paper library, and settings."""

    with st.sidebar:
        # --- Logo / Title ---
        st.markdown("## 📄 Paper Tutor")
        st.markdown(
            '<p style="color:#7f8c8d; font-size:0.82rem; margin-top:-10px;">'
            'AI-powered ML research assistant</p>',
            unsafe_allow_html=True,
        )

        st.divider()

        # --- Ollama Status ---
        status = check_ollama_connection()
        if status["connected"] and status["error"] is None:
            st.markdown(
                '<span class="status-badge status-connected">'
                f'● Connected — {LLM_MODEL}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-disconnected">'
                '● Disconnected</span>',
                unsafe_allow_html=True,
            )
            if status["error"]:
                st.caption(f"⚠️ {status['error']}")

        st.divider()

        # --- GPU Inference Toggle ---
        st.markdown("### ⚡ GPU Inference")

        use_remote = st.toggle(
            "Use Colab GPU",
            value=st.session_state.use_remote,
            help="Offload LLM inference to a Google Colab GPU for faster responses",
        )

        if use_remote != st.session_state.use_remote:
            st.session_state.use_remote = use_remote
            st.session_state.chain = None  # Reset chain to switch mode

        if use_remote:
            remote_url = st.text_input(
                "ngrok URL",
                value=st.session_state.remote_url,
                placeholder="https://abc123.ngrok-free.app",
                label_visibility="collapsed",
            )
            if remote_url != st.session_state.remote_url:
                st.session_state.remote_url = remote_url
                st.session_state.chain = None

            # Check remote connection
            if remote_url:
                remote_status = check_remote_connection(remote_url)
                if remote_status["connected"]:
                    st.markdown(
                        f'<span class="status-badge status-connected">'
                        f'● GPU Connected · {remote_status.get("latency_ms", 0):.0f}ms</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="status-badge status-disconnected">'
                        '● GPU Disconnected</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"⚠️ {remote_status['error'][:100]}")
            else:
                st.caption("Paste the ngrok URL from your Colab notebook above.")

        # --- PDF Upload ---
        st.markdown("### 📤 Upload Paper")
        uploaded_file = st.file_uploader(
            "Drop a PDF here",
            type=["pdf"],
            label_visibility="collapsed",
            key="pdf_uploader",
        )

        col_strip, col_ingest = st.columns([1, 1])
        with col_strip:
            strip_refs = st.checkbox(
                "Strip refs",
                value=False,
                help="Remove the References section before indexing",
            )

        if uploaded_file is not None:
            with col_ingest:
                if st.button("⚡ Ingest", use_container_width=True, type="primary"):
                    _handle_upload(uploaded_file, strip_refs)

        st.divider()

        # --- Paper Library ---
        st.markdown("### 📚 Papers")

        if st.session_state.papers:
            for filename, meta in st.session_state.papers.items():
                st.markdown(
                    f'<div class="paper-chip">'
                    f'  <span class="paper-name" title="{filename}">📄 {filename}</span>'
                    f'  <span class="paper-pages">{meta.get("pages", "?")}p</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No papers ingested yet.")

        st.divider()

        # --- Mode Selector ---
        st.markdown("### 🎯 Mode")

        mode = st.radio(
            "Select mode",
            options=["qa", "summarize", "compare"],
            format_func=lambda x: {
                "qa": "💬 Ask Questions",
                "summarize": "📝 Summarize Paper",
                "compare": "⚖️ Compare Methods",
            }[x],
            index=["qa", "summarize", "compare"].index(st.session_state.mode),
            label_visibility="collapsed",
        )
        st.session_state.mode = mode

        # Mode descriptions
        mode_descs = {
            "qa": "Ask specific questions about your papers. Follow-up questions are supported.",
            "summarize": "Get a structured summary with key contributions, methodology, and results.",
            "compare": "Compare approaches across different papers side by side.",
        }
        st.caption(mode_descs[mode])

        st.divider()

        # --- Actions ---
        st.markdown("### ⚙️ Actions")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.session_state.last_sources = []
            st.rerun()

        st.divider()

        # --- Index Stats ---
        if st.session_state.index:
            stats = get_index_stats(st.session_state.index)
            st.markdown(
                f'<div style="font-size:0.78rem; color:#7f8c8d;">'
                f'📊 <strong>{stats["total_vectors"]}</strong> chunks indexed · '
                f'{stats["dimension"]}d vectors · '
                f'{len(st.session_state.papers)} papers'
                f'</div>',
                unsafe_allow_html=True,
            )


def _handle_upload(uploaded_file, strip_refs: bool):
    """Process an uploaded PDF through the ingestion pipeline."""
    with st.spinner(f"Ingesting {uploaded_file.name}..."):
        try:
            # Save the uploaded file
            saved_path = save_uploaded_pdf(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )

            # Run ingestion pipeline
            result = ingest_pdf(
                saved_path,
                strip_references=strip_refs,
            )

            chunks = result["chunks"]
            metadata = result["metadata"]

            # Load embedding model (cached)
            if st.session_state.embedding_model is None:
                st.session_state.embedding_model = get_embedding_model()

            # Create or update index
            if st.session_state.index is None:
                st.session_state.index = create_index(
                    chunks,
                    embedding_model=st.session_state.embedding_model,
                )
            else:
                add_documents(st.session_state.index, chunks)

            # Save index to disk
            save_index(st.session_state.index)

            # Update session state
            st.session_state.papers[uploaded_file.name] = metadata
            st.session_state.total_chunks += len(chunks)

            # Reset chain to use updated index
            st.session_state.chain = None

            st.success(
                f"✅ **{uploaded_file.name}** ingested! "
                f"{len(chunks)} chunks from {metadata['pages']} pages."
            )

        except Exception as e:
            st.error(f"❌ Ingestion failed: {e}")
            logger.error(f"Ingestion error: {e}", exc_info=True)


# ===========================================================================
# Main Chat Area
# ===========================================================================


def render_header():
    """Render the main header with stats."""
    st.markdown(
        '<div class="main-header">'
        '<h1>📄 ML Paper Tutor</h1>'
        '<p>Upload ML papers and ask questions — powered by local AI</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Stats row
    n_papers = len(st.session_state.papers)
    n_chunks = st.session_state.total_chunks
    n_chats = len(st.session_state.chat_history) // 2

    st.markdown(
        f'<div class="stats-row">'
        f'  <div class="stat-card">'
        f'    <div class="stat-value">{n_papers}</div>'
        f'    <div class="stat-label">Papers</div>'
        f'  </div>'
        f'  <div class="stat-card">'
        f'    <div class="stat-value">{n_chunks}</div>'
        f'    <div class="stat-label">Chunks</div>'
        f'  </div>'
        f'  <div class="stat-card">'
        f'    <div class="stat-value">{n_chats}</div>'
        f'    <div class="stat-label">Exchanges</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_chat():
    """Render the chat interface."""

    # Show welcome card if no papers uploaded
    if not st.session_state.papers:
        st.markdown(
            '<div class="welcome-card">'
            '  <div class="welcome-icon">🚀</div>'
            '  <h3>Welcome to Paper Tutor</h3>'
            '  <p>Upload an ML research paper (PDF) from the sidebar to get started.<br>'
            '  Then ask questions, request summaries, or compare methods.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Display chat history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            st.markdown(content)

    # Display sources from last query
    if st.session_state.last_sources:
        with st.expander("📎 Sources from last response", expanded=False):
            for src in st.session_state.last_sources:
                score_pct = int(src["score"] * 100)
                st.markdown(
                    f'<div class="source-card">'
                    f'  <div class="source-header">'
                    f'    📄 {src["source"]} · Page {src["page"]}'
                    f'    <span class="source-score"> · {score_pct}% match</span>'
                    f'  </div>'
                    f'  <div class="source-text">{src["preview"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Chat input
    mode_placeholders = {
        "qa": "Ask a question about your papers...",
        "summarize": "e.g., Summarize the key contributions of this paper",
        "compare": "e.g., Compare the attention mechanisms described in these papers",
    }

    if prompt := st.chat_input(
        mode_placeholders.get(st.session_state.mode, "Ask a question..."),
    ):
        _handle_query(prompt)


def _handle_query(question: str):
    """Process a user query through the RAG chain."""

    # Display user message
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    # Ensure chain is initialized
    if st.session_state.chain is None:
        if st.session_state.index is None:
            with st.chat_message("assistant", avatar="🤖"):
                st.warning("Please upload a paper first!")
            return

        st.session_state.chain = PaperTutorChain(
            index=st.session_state.index,
            memory=st.session_state.memory,
            mode=st.session_state.mode,
            use_remote=st.session_state.use_remote,
            remote_url=st.session_state.remote_url,
        )

    chain = st.session_state.chain

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        if st.session_state.use_remote and st.session_state.remote_url:
            # STREAMING REMOTE PATH — tokens appear as they're generated
            try:
                token_gen, buffer, sources, results = chain.ask_stream(
                    question=question,
                    mode=st.session_state.mode,
                )
                answer = st.write_stream(token_gen)

                # Store in memory after streaming completes
                if st.session_state.mode == "qa":
                    chain.memory.add_exchange(question, answer)

                st.session_state.last_sources = sources

            except Exception as e:
                st.error(f"❌ Remote inference error: {e}")
                answer = f"*Error: {e}*"
        else:
            # LOCAL OLLAMA PATH — existing behavior, unchanged
            with st.spinner("Thinking..."):
                response = chain.ask(
                    question=question,
                    mode=st.session_state.mode,
                )

            if response.error:
                st.error(f"❌ Error: {response.error}")
                answer = f"*Error: {response.error}*"
            else:
                st.markdown(response.answer)
                answer = response.answer
                st.session_state.last_sources = response.sources

    # Store in chat history
    st.session_state.chat_history.append(("assistant", answer))

    st.rerun()


# ===========================================================================
# Main App
# ===========================================================================


def main():
    render_sidebar()
    render_header()
    render_chat()


if __name__ == "__main__":
    main()
