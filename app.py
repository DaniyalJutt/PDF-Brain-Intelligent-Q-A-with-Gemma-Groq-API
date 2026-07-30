import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root theme */
:root {
    --bg-primary:    #0d1117;
    --bg-surface:    #161b22;
    --bg-card:       #1c2230;
    --border:        #30363d;
    --accent:        #58a6ff;
    --accent-soft:   rgba(88,166,255,0.12);
    --accent-glow:   rgba(88,166,255,0.25);
    --success:       #3fb950;
    --warning:       #d29922;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
    --text-code:     #79c0ff;
    --font-body:     'Inter', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
}

/* Global reset */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1100px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Hero header ── */
.hero-block {
    padding: 2.5rem 2rem 2rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: linear-gradient(135deg, var(--bg-surface) 0%, #0f1923 100%);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 60% at 80% 20%, var(--accent-glow), transparent);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    margin: 0 0 0.4rem;
}
.hero-title span { color: var(--accent); }
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--accent-soft);
    border: 1px solid rgba(88,166,255,0.3);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 99px;
    margin-bottom: 0.8rem;
}

/* ── Status chip ── */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.5rem;
}
.status-ready {
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.35);
    color: var(--success);
}
.status-pending {
    background: rgba(210,153,34,0.12);
    border: 1px solid rgba(210,153,34,0.35);
    color: var(--warning);
}
.dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
.dot-green { background: var(--success); box-shadow: 0 0 6px var(--success); animation: pulse 2s infinite; }
.dot-yellow { background: var(--warning); }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.45; } }

/* ── Input ── */
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
    outline: none !important;
}
.stTextInput > label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.15s, transform 0.1s;
    letter-spacing: 0.01em;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* Secondary button variant via container class */
.btn-secondary .stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* ── Answer card ── */
.answer-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--text-primary);
}
.answer-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    font-weight: 500;
}

/* ── Metrics row ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
}
.metric-box {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--accent);
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Source chunks ── */
.chunk-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text-muted);
    position: relative;
}
.chunk-index {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--accent);
    background: var(--accent-soft);
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    display: inline-block;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.8rem 1rem !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── API Keys ──────────────────────────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")


# ─── LLM & Prompt ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_llm(api_key):
    return ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")

llm = load_llm(groq_api_key)

prompt_template = ChatPromptTemplate.from_template("""
You are a precise document analyst. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I couldn't find this in the provided documents."
Be concise, accurate, and cite relevant details.

<context>
{context}
</context>

Question: {input}
""")


# ─── Vector Embedding ──────────────────────────────────────────────────────────
def vector_embedding(data_path: str, chunk_size: int, chunk_overlap: int, max_docs: int):
    """Build FAISS vector store from PDFs in data_path."""
    with st.spinner("Loading documents…"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader(data_path)
        docs = loader.load()

    if not docs:
        st.error(f"No PDF files found in `{data_path}`. Please add PDFs and try again.")
        return

    with st.spinner(f"Splitting {len(docs)} pages into chunks…"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs[:max_docs])

    with st.spinner(f"Embedding {len(chunks)} chunks — this may take a moment…"):
        vectors = FAISS.from_documents(chunks, embeddings)

    # Store in session
    st.session_state.vectors = vectors
    st.session_state.doc_count = len(docs)
    st.session_state.chunk_count = len(chunks)
    st.session_state.embed_model = "models/embedding-001"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    data_path = st.text_input("📁 PDF Directory", value="./data",
                              help="Folder containing your PDF files")

    st.markdown("**Chunking Settings**")
    chunk_size = st.slider("Chunk Size (tokens)", 300, 2000, 1000, 100,
                           help="Larger chunks = more context, slower retrieval")
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50,
                              help="Overlap between adjacent chunks")
    max_docs = st.slider("Max Pages to Index", 5, 100, 20, 5,
                         help="Cap to avoid rate limits")

    st.markdown("---")
    embed_ready = "vectors" in st.session_state

    if embed_ready:
        st.markdown(f"""
        <div class="status-chip status-ready">
            <span class="dot dot-green"></span> Vector DB Ready
        </div>
        <div style="margin-top:0.6rem; font-size:0.78rem; color:#8b949e;">
            📄 {st.session_state.get('doc_count','?')} pages &nbsp;·&nbsp;
            🧩 {st.session_state.get('chunk_count','?')} chunks
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-chip status-pending">
            <span class="dot dot-yellow"></span> Not Indexed
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        vector_embedding(data_path, chunk_size, chunk_overlap, max_docs)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#8b949e; line-height:1.6;">
    <strong style="color:#58a6ff">Model</strong><br>Llama 3 · 8B · Groq<br><br>
    <strong style="color:#58a6ff">Embeddings</strong><br>Google Gemini text-embedding-001<br><br>
    <strong style="color:#58a6ff">Vector Store</strong><br>FAISS (in-memory)
    </div>
    """, unsafe_allow_html=True)


# ─── Main Area ────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-block">
    <div class="hero-badge">🔍 RAG · Retrieval-Augmented Generation</div>
    <h1 class="hero-title">Doc<span>Mind</span></h1>
    <p class="hero-subtitle">Ask questions across your documents — powered by Llama 3 & Google Embeddings</p>
</div>
""", unsafe_allow_html=True)

# Quick-start hint if not indexed
if "vectors" not in st.session_state:
    st.info("👈 **Get started:** Place your PDFs in the `./data` folder, then click **Build Index** in the sidebar.")

# Question input
st.markdown("#### Ask a Question")
question = st.text_input(
    "question",
    placeholder="e.g.  What are the main findings of the report?",
    label_visibility="collapsed",
)

col_ask, col_clear, _ = st.columns([1.4, 1, 5])
with col_ask:
    ask_clicked = st.button("🔍 Ask", use_container_width=True)
with col_clear:
    if st.button("✕ Clear", use_container_width=True):
        for key in ["last_answer", "last_context", "last_response_time", "last_question"]:
            st.session_state.pop(key, None)
        st.rerun()

# ─── Inference ────────────────────────────────────────────────────────────────
if ask_clicked and question:
    if "vectors" not in st.session_state:
        st.error("Please build the vector index first using the sidebar.")
    else:
        with st.spinner("Searching documents and generating answer…"):
            doc_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = st.session_state.vectors.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)
            t0 = time.perf_counter()
            response = retrieval_chain.invoke({"input": question})
            elapsed = time.perf_counter() - t0

        st.session_state.last_answer = response["answer"]
        st.session_state.last_context = response["context"]
        st.session_state.last_response_time = elapsed
        st.session_state.last_question = question

# ─── Answer Display ───────────────────────────────────────────────────────────
if "last_answer" in st.session_state:
    elapsed = st.session_state.last_response_time
    ctx_docs = st.session_state.last_context

    # Metrics row
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-value">{elapsed:.2f}s</div>
            <div class="metric-label">Response Time</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(ctx_docs)}</div>
            <div class="metric-label">Sources Used</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(st.session_state.last_answer.split())}</div>
            <div class="metric-label">Answer Words</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Answer card
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-label">▸ Answer</div>
        {st.session_state.last_answer}
    </div>
    """, unsafe_allow_html=True)

    # Source chunks expander
    st.markdown("<div style='margin-top:1.2rem'>", unsafe_allow_html=True)
    with st.expander(f"📄 View {len(ctx_docs)} Source Chunks Used"):
        for i, doc in enumerate(ctx_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-index">CHUNK {i:02d} · {os.path.basename(source)} · page {page}</div>
                {doc.page_content}
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)