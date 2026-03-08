# import (python built-ins)
import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
## imports langchain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# setup : env + streamlit page

load_dotenv() 
st.set_page_config(page_title=" 📝 RAG Q&A ",layout="wide")
st.title("📝 RAG Q&A with Multiple PDFs + Chat History")

# Sidbar config: Groq API Key input

with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs -> Ask questions -> Get Answers")

# Accept keey from input or .env
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning(" Please enter your Groq API Key (or set GROQ_API_KEY in .env) ")
    st.stop()

# embeddings ad llm initialization

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(
    groq_api_key=api_key, # type: ignore
    model_name="llama-3.3-70b-versatile" # type: ignore
)

# upload PDFs (multiple)

uploaded_files = st.file_uploader(
    " 📚 Upload PDF files",
    type = "pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDFs to begin")
    st.stop()

all_docs = []
tmp_paths = []

for pdf in uploaded_files:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()
    tmp_paths.append(tmp.name)    

    loader = PyPDFLoader(tmp.name)
    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf.name

    all_docs.extend(docs)

st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")

# Clean up temp files
for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass

#chunking (split text)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120
)

splits = text_splitter.split_documents(all_docs)


# ── Vectorstore 
INDEX_DIR = "chroma_index"

vectorstore = Chroma.from_documents(
    splits,
    embeddings,
    persist_directory=INDEX_DIR
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

st.sidebar.write(f"🔍 Indexed {len(splits)} chunks for retrieval")

# ── Helper: format docs for stuffing ───────────────────────────────────────────
def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# prompts

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question into a standalone search query using the chat history for the context."
     "Return only the rewritten query, no extra text."),
     MessagesPlaceholder("chat_history"), 
     ("human","{input}") 
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT RAG assistant. You must answer using ONLY the provided context.\n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope - not found in provided documents.'\n"
     "Do NOT use outside knowledge. \n\n"
     "Context:\n{context}"),
     MessagesPlaceholder("chat_history"),
     ("human","{input}") 
])

# session state for chat history (multi sessions)
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

CHAT_MEMORY_DIR = "chat_memory"
os.makedirs(CHAT_MEMORY_DIR, exist_ok=True)

def _session_file(session_id: str):
    safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id.strip())
    safe_id = safe_id or "default_session"
    return os.path.join(CHAT_MEMORY_DIR, f"{safe_id}.json")

def _load_persistent_history(session_id: str):
    history = ChatMessageHistory()
    path = _session_file(session_id)
    if not os.path.exists(path):
        return history
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            history.messages = messages_from_dict(raw)
    except Exception:
        pass
    return history

def _save_persistent_history(session_id: str, history: ChatMessageHistory):
    path = _session_file(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages_to_dict(history.messages), f, ensure_ascii=False, indent=2)

def _list_saved_sessions():
    session_names = []
    for name in os.listdir(CHAT_MEMORY_DIR):
        if name.endswith(".json"):
            session_names.append(name[:-5])
    return sorted(session_names)

def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = _load_persistent_history(session_id)
    return st.session_state.chathistory[session_id]

# chat ui
session_id = st.text_input(" 🆔 Session ID ", value="default_session")
user_q = st.chat_input("💬 Ask a question...")


# ── Session state for chat history here

session_id = st.sidebar.text_input("Session ID", value=session_id)
history = get_history(session_id)

st.sidebar.subheader("Saved Sessions")
saved_sessions = _list_saved_sessions()
if saved_sessions:
    for s in saved_sessions:
        st.sidebar.write(f"- {s}")
else:
    st.sidebar.caption("No saved sessions yet.")

if st.sidebar.button("Clear This Session Memory"):
    history.clear()
    _save_persistent_history(session_id, history)
    st.rerun()

for msg in history.messages:
    role = "assistant" if msg.type in ("ai", "assistant") else "user"
    st.chat_message(role).write(msg.content)

if user_q:

    # 1) Rewrite question with history
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )
    standalone_q = llm.invoke(rewrite_msgs).content.strip() # type: ignore

    #Retrieve chunks
    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "Out of scope — not found in provided documents."
        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)
        _save_persistent_history(session_id, history)
        st.stop()

    # 3) Build context string
    context_str = _join_docs(docs)

    # Asking final question with stuffed context
    qa_msgs = qa_prompt.format_messages(
        chat_history= history.messages,
        input=user_q,
        context=context_str
    )
    answer = llm.invoke(qa_msgs).content
    
    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)
    _save_persistent_history(session_id, history)

    # Debug panels
    with st.expander("🧪 Debug: Rewritten Query & Retrieval"):
        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📑 Retrieved Chunks"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))


