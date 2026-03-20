import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ─────────────────────────────────────────────
# ENV & PAGE CONFIG
# ─────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="DocLens · RAG Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — Research Intelligence Terminal
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-base:        #07090f;
    --bg-panel:       #0d1117;
    --bg-glass:       rgba(13, 17, 23, 0.82);
    --bg-card:        rgba(16, 22, 35, 0.9);
    --border-subtle:  rgba(0, 212, 255, 0.12);
    --border-glow:    rgba(0, 212, 255, 0.45);
    --cyan:           #00d4ff;
    --cyan-dim:       rgba(0, 212, 255, 0.15);
    --amber:          #ffb347;
    --amber-dim:      rgba(255, 179, 71, 0.12);
    --red-accent:     #ff4f6a;
    --text-primary:   #e8edf5;
    --text-secondary: #7a8a9e;
    --text-muted:     #404e62;
    --font-display:   'Syne', sans-serif;
    --font-body:      'DM Sans', sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── App Background — animated gradient mesh ── */
.stApp {
    background-color: var(--bg-base);
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 20%, rgba(0,212,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 80%, rgba(255,179,71,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(0,90,140,0.04) 0%, transparent 70%);
    font-family: var(--font-body);
    color: var(--text-primary);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c14 0%, #0a0f1c 100%);
    border-right: 1px solid var(--border-subtle);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.2rem;
}

/* ── Sidebar brand header ── */
.sidebar-brand {
    font-family: var(--font-display);
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem;
}
.sidebar-brand span {
    color: var(--cyan);
}
.sidebar-tagline {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--cyan);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* ── API Key input ── */
[data-testid="stSidebar"] .stTextInput label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

[data-testid="stSidebar"] .stTextInput input {
    background: rgba(0, 212, 255, 0.04) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    color: var(--cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.08) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border-subtle);
    border-radius: 10px;
    background: rgba(0,212,255,0.02);
    padding: 0.5rem;
    transition: border-color 0.3s ease, background 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.35);
    background: rgba(0,212,255,0.04);
}
[data-testid="stFileUploader"] label {
    font-family: var(--font-mono);
    font-size: 0.65rem !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploadDropzone"] p {
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
}

/* ── Stat badges in sidebar ── */
.stat-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    margin-bottom: 0.45rem;
    font-family: var(--font-mono);
}
.stat-badge .stat-num {
    font-size: 1rem;
    font-weight: 500;
    color: var(--cyan);
    min-width: 2.5rem;
}
.stat-badge .stat-label {
    font-size: 0.6rem;
    color: var(--text-secondary);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    line-height: 1.3;
}
.stat-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 6px var(--cyan);
    flex-shrink: 0;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.75); }
}

/* ── Main page header ── */
.page-header {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border-subtle);
}
.page-title {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1;
}
.page-title em {
    font-style: normal;
    color: var(--cyan);
}
.page-subtitle {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--cyan);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.status-pill-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--cyan);
    animation: pulse 1.8s ease-in-out infinite;
}

/* ── Session ID input (main area) ── */
div[data-testid="stTextInput"] label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="stTextInput"] input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--amber) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    max-width: 320px;
    transition: border-color 0.25s ease;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(255,179,71,0.5) !important;
    box-shadow: 0 0 0 3px rgba(255,179,71,0.06) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.07) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
div.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}

/* Message content cards */
[data-testid="stChatMessage"] .stMarkdown {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    font-family: var(--font-body);
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text-primary);
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, rgba(255,179,71,0.2), rgba(255,179,71,0.05)) !important;
    border: 1px solid rgba(255,179,71,0.3) !important;
    color: var(--amber) !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,212,255,0.05)) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: var(--cyan) !important;
}

/* ── Info / Success / Warning boxes ── */
.stAlert {
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    border-left-width: 3px !important;
}
div[data-testid="stInfo"] {
    background: rgba(0,212,255,0.04) !important;
    border-color: var(--cyan) !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stSuccess"] {
    background: rgba(0,255,140,0.04) !important;
    border-color: #00ff8c !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stWarning"] {
    background: rgba(255,179,71,0.04) !important;
    border-color: var(--amber) !important;
    color: var(--text-secondary) !important;
}

/* ── Expander (debug panels) ── */
.streamlit-expanderHeader {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-secondary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    transition: border-color 0.2s ease, color 0.2s ease;
}
.streamlit-expanderHeader:hover {
    border-color: var(--border-glow) !important;
    color: var(--cyan) !important;
}
.streamlit-expanderContent {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 0.8rem 1rem !important;
}

/* ── Code blocks ── */
code, pre {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    color: var(--cyan) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.2rem 0 !important;
}

/* ── File tag chips ── */
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.6rem;
    background: var(--amber-dim);
    border: 1px solid rgba(255,179,71,0.25);
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--amber);
    letter-spacing: 0.05em;
    margin: 0.18rem;
    white-space: nowrap;
}
.files-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.5rem;
}

/* ── Chat container wrapper ── */
.chat-container {
    max-width: 860px;
    margin: 0 auto;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    opacity: 0.5;
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}
.empty-title {
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}
.empty-desc {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
}

/* ── Chunk detail cards in expander ── */
.chunk-card {
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border-subtle);
    border-left: 2px solid var(--cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
}
.chunk-meta {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--cyan);
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}
.chunk-text {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.55;
}

/* ── Sidebar uploaded file list ── */
.uploaded-file-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.6rem;
    background: var(--amber-dim);
    border: 1px solid rgba(255,179,71,0.15);
    border-radius: 6px;
    margin-bottom: 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--amber);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.uploaded-file-icon { flex-shrink: 0; opacity: 0.7; }

/* ── Welcome intro banner ── */
.intro-banner {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.intro-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(0,212,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.intro-banner-title {
    font-family: var(--font-display);
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}
.intro-banner-desc {
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.intro-steps {
    display: flex;
    gap: 1.2rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.intro-step {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
}
.intro-step-num {
    width: 18px; height: 18px;
    background: var(--cyan-dim);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.55rem;
    color: var(--cyan);
    font-weight: 600;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class='sidebar-brand'>Doc<span>Lens</span></div>
        <div class='sidebar-tagline'>◈ RAG Intelligence Terminal</div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Authentication</div>", unsafe_allow_html=True)
    api_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_…")

    st.markdown("<div class='sidebar-section'>Document Upload</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF Files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show uploaded file chips
    if uploaded_files:
        for f in uploaded_files:
            name = f.name if len(f.name) <= 28 else f.name[:25] + "…"
            st.markdown(
                f"<div class='uploaded-file-item'>"
                f"<span class='uploaded-file-icon'>◪</span> {name}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='sidebar-section'>Index Stats</div>", unsafe_allow_html=True)

    # Placeholders updated later after indexing
    pages_badge   = st.empty()
    chunks_badge  = st.empty()

    # Default/initial badge values
    pages_badge.markdown(
        "<div class='stat-badge'>"
        "<div class='stat-dot'></div>"
        "<div class='stat-num'>—</div>"
        "<div class='stat-label'>Pages<br/>Loaded</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    chunks_badge.markdown(
        "<div class='stat-badge'>"
        "<div class='stat-dot'></div>"
        "<div class='stat-num'>—</div>"
        "<div class='stat-label'>Chunks<br/>Indexed</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:var(--font-mono);font-size:0.55rem;"
        "color:var(--text-muted);letter-spacing:0.08em;line-height:1.8'>"
        "MODEL · llama-3.1-8b-instant<br>"
        "EMBED · all-MiniLM-L6-v2<br>"
        "STORE · ChromaDB · MMR k=6"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# MAIN AREA HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <div class='page-header'>
        <div>
            <div class='page-subtitle'>◈ Retrieval-Augmented Generation</div>
            <div class='page-title'>Doc<em>Lens</em></div>
        </div>
        <div style='margin-bottom:0.25rem'>
            <div class='status-pill'>
                <div class='status-pill-dot'></div>
                System Online
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GATE: API KEY
# ─────────────────────────────────────────────
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.markdown("""
        <div class='intro-banner'>
            <div class='intro-banner-title'>◈ Welcome to DocLens</div>
            <div class='intro-banner-desc'>
                An intelligent RAG system that lets you interrogate your PDF documents with precision.
                Ask complex questions and receive grounded, context-aware answers drawn exclusively from your files.
            </div>
            <div class='intro-steps'>
                <div class='intro-step'><div class='intro-step-num'>1</div> Enter your Groq API Key in the sidebar</div>
                <div class='intro-step'><div class='intro-step-num'>2</div> Upload one or more PDF documents</div>
                <div class='intro-step'><div class='intro-step-num'>3</div> Ask questions in natural language</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.warning("⚠  Enter your Groq API Key in the sidebar to continue.")
    st.stop()

# ─────────────────────────────────────────────
# GATE: FILES
# ─────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
        <div class='intro-banner'>
            <div class='intro-banner-title'>◈ No Documents Loaded</div>
            <div class='intro-banner-desc'>
                Upload one or more PDF files using the sidebar panel.
                DocLens will automatically chunk, embed, and index them for retrieval.
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.info("📄  Upload PDF files in the sidebar to begin.")
    st.stop()

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

embeddings = get_embeddings()

llm = ChatGroq(
    api_key=SecretStr(api_key),
    model="llama-3.1-8b-instant",
)

# ─────────────────────────────────────────────
# DOCUMENT LOADING + INDEXING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index(file_names_and_bytes: tuple, _embeddings):
    """
    Accepts a tuple of (filename, bytes) pairs so st.cache_resource
    can hash the arguments — file objects are not hashable.
    """
    all_docs, tmp_paths = [], []
    for file_name, file_bytes in file_names_and_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(file_bytes)
        tmp.close()
        tmp_paths.append(tmp.name)
        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = file_name
        all_docs.extend(docs)

    for p in tmp_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    splits = splitter.split_documents(all_docs)

    INDEX_DIR = "chroma_index"
    vs = Chroma.from_documents(splits, _embeddings, persist_directory=INDEX_DIR)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
    return len(all_docs), splits, retriever


with st.spinner("⚙  Indexing documents…"):
    # Read bytes outside the cached function — file objects can't be hashed or cached
    file_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    n_pages, splits, retriever = build_index(file_data, embeddings)

# Update sidebar badges
pages_badge.markdown(
    f"<div class='stat-badge'>"
    f"<div class='stat-dot'></div>"
    f"<div class='stat-num'>{n_pages}</div>"
    f"<div class='stat-label'>Pages<br/>Loaded</div>"
    f"</div>",
    unsafe_allow_html=True,
)
chunks_badge.markdown(
    f"<div class='stat-badge'>"
    f"<div class='stat-dot'></div>"
    f"<div class='stat-num'>{len(splits)}</div>"
    f"<div class='stat-label'>Chunks<br/>Indexed</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question into a standalone search query using the chat history for context. "
     "Return only the rewritten query, no extra text."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT RAG assistant. Answer using ONLY the provided context.\n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope — not found in the provided documents.'\n"
     "Do NOT use outside knowledge.\n\n"
     "RESPONSE RULES:\n"
     "- Give a THOROUGH, DETAILED answer — never a single line.\n"
     "- Always explain the why and how, not just the what.\n"
     "- Use bullet points or numbered steps when listing multiple ideas.\n"
     "- If the context contains examples, include them in your answer.\n"
     "- Minimum 3-5 sentences for any factual question.\n"
     "- End with a brief summary sentence.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}
if "display_messages" not in st.session_state:
    st.session_state.display_messages = {}

def get_history(sid):
    if sid not in st.session_state.chathistory:
        st.session_state.chathistory[sid] = ChatMessageHistory()
    return st.session_state.chathistory[sid]

def get_display(sid):
    if sid not in st.session_state.display_messages:
        st.session_state.display_messages[sid] = []
    return st.session_state.display_messages[sid]

# ─────────────────────────────────────────────
# CHAT UI
# ─────────────────────────────────────────────
col_session, _ = st.columns([2, 5])
with col_session:
    session_id = st.text_input("🆔  Session ID", value="default_session", label_visibility="visible")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# Render previous messages
display_msgs = get_display(session_id)

if not display_msgs:
    st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>◈</div>
            <div class='empty-title'>No messages yet</div>
            <div class='empty-desc'>Ask a question about your uploaded documents</div>
        </div>
    """, unsafe_allow_html=True)
else:
    for msg in display_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("debug"):
                with st.expander("🧪  Rewritten Query & Retrieval"):
                    st.markdown(
                        f"<div class='chunk-meta'>Standalone Query</div>"
                        f"<div class='chunk-text'>{msg['debug']['standalone_q']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='chunk-meta' style='margin-top:0.7rem'>"
                        f"Retrieved {msg['debug']['n_chunks']} chunk(s)</div>",
                        unsafe_allow_html=True,
                    )
                with st.expander("📑  Retrieved Chunks"):
                    for i, chunk in enumerate(msg["debug"]["chunks"], 1):
                        src = chunk["source"]
                        page = chunk["page"]
                        text = chunk["text"]
                        st.markdown(
                            f"<div class='chunk-card'>"
                            f"<div class='chunk-meta'>#{i} · {src} · page {page}</div>"
                            f"<div class='chunk-text'>{text}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

def _extract_content(result):
    if isinstance(result, list):
        if result and isinstance(result[0], dict) and "content" in result[0]:
            return result[0]["content"].strip()
        elif result and isinstance(result[0], str):
            return result[0].strip()
        return str(result).strip()
    if hasattr(result, "content"):
        c = result.content
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            if c and isinstance(c[0], dict) and "content" in c[0]:
                return c[0]["content"].strip()
            if c and isinstance(c[0], str):
                return c[0].strip()
        return str(c).strip()
    return str(result).strip()

# ─────────────────────────────────────────────
# CHAT INPUT HANDLER
# ─────────────────────────────────────────────
user_q = st.chat_input("Ask a question about your documents…")

if user_q:
    history = get_history(session_id)
    msgs    = get_display(session_id)

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_q)

    msgs.append({"role": "user", "content": user_q})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            # 1) Rewrite question
            rewrite_msgs = contextualize_q_prompt.format_messages(
                chat_history=history.messages, input=user_q
            )
            standalone_q = _extract_content(llm.invoke(rewrite_msgs))

            # 2) Retrieve
            docs = retriever.invoke(standalone_q)

            if not docs:
                answer = "Out of scope — not found in the provided documents."
                st.markdown(answer)
                history.add_user_message(user_q)
                history.add_ai_message(answer)
                msgs.append({"role": "assistant", "content": answer})
                st.stop()

            # 3) Build context + answer
            context_str = _join_docs(docs)
            qa_msgs = qa_prompt.format_messages(
                chat_history=history.messages, input=user_q, context=context_str
            )
            answer = _extract_content(llm.invoke(qa_msgs))
            st.markdown(answer)

        # Debug panels
        chunk_data = [
            {
                "source": d.metadata.get("source_file", "Unknown"),
                "page":   d.metadata.get("page", "?"),
                "text":   d.page_content[:500] + ("…" if len(d.page_content) > 500 else ""),
            }
            for d in docs
        ]

        with st.expander("🧪  Rewritten Query & Retrieval"):
            st.markdown(
                f"<div class='chunk-meta'>Standalone Query</div>"
                f"<div class='chunk-text'>{standalone_q}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='chunk-meta' style='margin-top:0.7rem'>"
                f"Retrieved {len(docs)} chunk(s)</div>",
                unsafe_allow_html=True,
            )

        with st.expander("📑  Retrieved Chunks"):
            for i, chunk in enumerate(chunk_data, 1):
                st.markdown(
                    f"<div class='chunk-card'>"
                    f"<div class='chunk-meta'>#{i} · {chunk['source']} · page {chunk['page']}</div>"
                    f"<div class='chunk-text'>{chunk['text']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Persist to session state
    msgs.append({
        "role": "assistant",
        "content": answer,
        "debug": {
            "standalone_q": standalone_q,
            "n_chunks": len(docs),
            "chunks": chunk_data,
        },
    })
    history.add_user_message(user_q)
    history.add_ai_message(answer)
