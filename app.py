import html
import os
import re
import tempfile
import time
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from document_loader import load_pdfs
from vector_store import (
    chunk_documents,
    build_vector_store,
    save_vector_store,
    load_vector_store,
)
from rag_pipeline import NO_DOCUMENT_ANSWER, RAGPipeline
from gemini_client import GeminiClient, GeminiError


# ============================================================
# Configuration
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Knowledge Chat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(markup: str) -> None:
    """Render a controlled HTML fragment."""
    st.markdown(markup, unsafe_allow_html=True)


def stream_generator(text: str, delay: float = 0.008):
    """Provide a lightweight streaming effect for the latest answer."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def get_secret(name: str, default: str = "") -> str:
    """Read an environment variable, then fall back to Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def safe_message_html(text: str) -> str:
    """Convert a small safe subset of Markdown into readable chat HTML."""
    safe = html.escape(text or "")
    safe = re.sub(r"(?m)^###\s+(.+?)(?=\n|$)", r"<h4>\1</h4>", safe)
    safe = re.sub(r"(?m)^##\s+(.+?)(?=\n|$)", r"<h3>\1</h3>", safe)
    safe = re.sub(r"(?m)^#\s+(.+?)(?=\n|$)", r"<h2>\1</h2>", safe)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
    safe = safe.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{safe}</p>"


def render_bubble(role: str, content: str) -> None:
    """Render one message with users aligned right and the assistant left."""
    is_user = role == "user"
    role_class = "user" if is_user else "assistant"
    avatar = "You" if is_user else "AI"
    label = "You" if is_user else "Knowledge Chat"

    render_html(
        f"""
        <div class="message-row {role_class}">
            <div class="message-avatar {role_class}">{avatar}</div>
            <div class="message-column">
                <div class="message-label">{label}</div>
                <div class="message-bubble {role_class}">
                    {safe_message_html(content)}
                </div>
            </div>
        </div>
        """
    )


def render_sources(sources: list[dict[str, Any]], message_index: int) -> None:
    """Render retrieved document references below an assistant response."""
    if not sources:
        return

    with st.expander(f"Sources used · {len(sources)}", expanded=False):
        for source in sources:
            source_name = html.escape(str(source.get("source", "Document")))
            page = html.escape(str(source.get("page", "?")))
            render_html(
                f'<span class="source-badge">{source_name} · page {page}</span>'
            )


def create_chat() -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "name": "New Conversation",
        "messages": [],
    }
    return chat_id


# ============================================================
# Styling
# ============================================================

render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --kc-bg: #f7f8fa;
        --kc-panel: #ffffff;
        --kc-border: #e5e7eb;
        --kc-text: #111827;
        --kc-muted: #6b7280;
        --kc-accent: #2563eb;
        --kc-accent-soft: #eff6ff;
        --kc-assistant: #ffffff;
        --kc-user: #2563eb;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--kc-bg);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #fbfbfc;
        border-right: 1px solid var(--kc-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }

    .block-container {
        max-width: 1020px;
        padding: 2.25rem 2rem 7rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.25rem 0 2.5rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        color: white;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        border-radius: 13px;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.22);
    }

    .brand-title {
        color: var(--kc-text);
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .brand-subtitle {
        color: var(--kc-muted);
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    .empty-state {
        margin: 12vh auto 2.5rem;
        max-width: 720px;
        text-align: center;
    }

    .empty-state h1 {
        color: var(--kc-text);
        font-size: clamp(2rem, 4vw, 3.2rem);
        letter-spacing: -0.05em;
        margin-bottom: 0.75rem;
    }

    .empty-state p {
        color: var(--kc-muted);
        font-size: 1rem;
        line-height: 1.7;
    }

    .message-row {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        width: 100%;
        margin: 1.35rem 0;
    }

    .message-row.user {
        justify-content: flex-end;
    }

    .message-column {
        display: flex;
        flex-direction: column;
        max-width: min(780px, 82%);
    }

    .message-row.user .message-column {
        align-items: flex-end;
    }

    .message-avatar {
        flex: 0 0 34px;
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 11px;
        color: white;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-top: 1.2rem;
    }

    .message-avatar.user {
        background: #111827;
    }

    .message-avatar.assistant {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
    }

    .message-label {
        color: var(--kc-muted);
        font-size: 0.72rem;
        font-weight: 600;
        margin: 0 0 0.35rem 0.25rem;
    }

    .message-bubble {
        border: 1px solid var(--kc-border);
        border-radius: 18px;
        color: var(--kc-text);
        font-size: 0.96rem;
        line-height: 1.7;
        padding: 0.85rem 1.05rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .message-bubble p {
        margin: 0;
    }

    .message-bubble p + p {
        margin-top: 0.65rem;
    }

    .message-bubble h2,
    .message-bubble h3,
    .message-bubble h4 {
        margin: 0 0 0.35rem;
        color: inherit;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .message-bubble h2:not(:first-child),
    .message-bubble h3:not(:first-child),
    .message-bubble h4:not(:first-child) {
        margin-top: 0.85rem;
    }

    .message-bubble.assistant {
        background: var(--kc-assistant);
        border-top-left-radius: 6px;
    }

    .message-bubble.user {
        color: white;
        background: var(--kc-user);
        border-color: var(--kc-user);
        border-top-right-radius: 6px;
    }

    .message-bubble code {
        padding: 0.12rem 0.35rem;
        border-radius: 5px;
        background: rgba(15, 23, 42, 0.08);
        font-size: 0.88em;
    }

    .message-bubble.user code {
        background: rgba(255, 255, 255, 0.18);
    }

    .source-badge {
        display: inline-block;
        margin: 0.15rem 0.35rem 0.15rem 0;
        padding: 0.3rem 0.65rem;
        border: 1px solid #dbeafe;
        border-radius: 999px;
        color: #1d4ed8;
        background: #eff6ff;
        font-size: 0.76rem;
        font-weight: 600;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-pill.warn {
        background: #fffbeb;
        color: #b45309;
    }

    .hint-card {
        padding: 0.75rem 0.9rem;
        margin-top: 0.5rem;
        border: 1px solid var(--kc-border);
        border-radius: 12px;
        color: var(--kc-muted);
        background: white;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    div[data-testid="stChatInput"] {
        border-top: 0;
    }

    div[data-testid="stChatInput"] textarea {
        border: 1px solid #d1d5db;
        border-radius: 16px;
        background: white;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.09);
    }
    </style>
    """
)


# ============================================================
# Session state
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None
if "chats" not in st.session_state:
    first_chat_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_chat_id: {"name": "New Conversation", "messages": []}
    }
    st.session_state.current_chat_id = first_chat_id

if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = next(iter(st.session_state.chats))

MAX_UPLOAD_MB = 25
MAX_UPLOAD_FILES = 10
GROQ_KEY_SET = bool(get_secret("GROQ_API_KEY"))
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_KEY_SET = bool(GEMINI_API_KEY)

# Reuse a saved FAISS index after a rerun or app restart.
if st.session_state.vector_store is None:
    saved_store = load_vector_store()
    if saved_store is not None:
        st.session_state.vector_store = saved_store
        st.session_state.rag_pipeline = RAGPipeline(saved_store)
        st.session_state.chunk_count = getattr(
            getattr(saved_store, "index", None), "ntotal", None
        )

current_chat = st.session_state.chats[st.session_state.current_chat_id]
current_messages = current_chat["messages"]


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    render_html(
        """
        <div class="brand">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-title">Knowledge Chat</div>
                <div class="brand-subtitle">Your documents, explained simply</div>
            </div>
        </div>
        """
    )

    if st.button("＋  New chat", use_container_width=True):
        new_id = create_chat()
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()
    st.caption("CONVERSATIONS")

    for chat_id, chat_data in reversed(list(st.session_state.chats.items())):
        is_active = chat_id == st.session_state.current_chat_id
        prefix = "● " if is_active else "○ "
        if st.button(
            prefix + chat_data["name"],
            key=f"chat_{chat_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.divider()
    st.caption("KNOWLEDGE BASE")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Upload up to {MAX_UPLOAD_FILES} PDFs. Each file must be {MAX_UPLOAD_MB} MB or smaller.",
    )

    upload_is_valid = bool(uploaded_files) and len(uploaded_files) <= MAX_UPLOAD_FILES and all(
        uploaded_file.size <= MAX_UPLOAD_MB * 1024 * 1024
        for uploaded_file in uploaded_files
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(f"Please upload no more than {MAX_UPLOAD_FILES} PDFs at a time.")
        for uploaded_file in uploaded_files:
            if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.warning(
                    f"{uploaded_file.name} is too large. The limit is {MAX_UPLOAD_MB} MB per file."
                )

    process_clicked = st.button(
        "Index documents",
        disabled=not upload_is_valid,
        use_container_width=True,
    )

    if process_clicked:
        with st.spinner("Reading and indexing your documents..."):
            temp_paths: list[str] = []
            try:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_paths.append(tmp.name)

                docs = load_pdfs(temp_paths)
                if not docs:
                    st.error("No readable text was found in the uploaded PDFs.")
                else:
                    chunks = chunk_documents(docs)
                    store = build_vector_store(chunks)
                    save_vector_store(store)

                    st.session_state.vector_store = store
                    st.session_state.rag_pipeline = RAGPipeline(store)
                    st.session_state.chunk_count = len(chunks)
                    st.success(f"Indexed {len(chunks)} document chunks.")
            except Exception as exc:
                st.error(f"Document processing failed: {exc}")
            finally:
                for path in temp_paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    if st.session_state.vector_store is not None:
        chunk_label = st.session_state.chunk_count or "saved"
        render_html(
            f'<span class="status-pill">✓ {chunk_label} chunks indexed</span>'
        )
    else:
        render_html('<span class="status-pill warn">No documents indexed</span>')

    st.divider()
    st.caption("AI CONNECTIONS")
    if GEMINI_KEY_SET:
        st.success(f"Gemini configured · {GEMINI_MODEL}")
    else:
        st.warning("Add GEMINI_API_KEY to your .env file.")

    if GROQ_KEY_SET:
        st.caption("Groq is available for grounded document retrieval answers.")
    else:
        st.caption(
            "Groq is optional. Gemini can write the final answer using only retrieved PDF passages."
        )

    if st.button("Clear current chat", use_container_width=True):
        current_chat["messages"].clear()
        current_chat["name"] = "New Conversation"
        st.rerun()


# ============================================================
# Main conversation area
# ============================================================

if not current_messages:
    render_html(
        """
        <div class="empty-state">
            <h1>What would you like to understand?</h1>
            <p>
                Ask a question about your uploaded documents.
                I will answer only from retrieved PDF passages and explain the supported answer in clear, simple language.
            </p>
        </div>
        """
    )

for index, message in enumerate(current_messages):
    render_bubble(message["role"], message["content"])
    if message["role"] == "assistant":
        render_sources(message.get("sources", []), index)


# ============================================================
# Answer orchestration
# ============================================================

question = st.chat_input("Message Knowledge Chat...")

if question:
    question = question.strip()
    if not question:
        st.stop()

    if not GEMINI_KEY_SET and not GROQ_KEY_SET:
        st.error("Add GEMINI_API_KEY or GROQ_API_KEY to your .env file before asking a question.")
        st.stop()

    if not current_messages:
        current_chat["name"] = question[:32] + ("..." if len(question) > 32 else "")

    current_messages.append({"role": "user", "content": question})
    render_bubble("user", question)

    kb_answer = ""
    retrieved_context = ""
    sources: list[dict[str, Any]] = []
    retrieval_status = ""

    if st.session_state.vector_store is not None:
        try:
            pipeline = st.session_state.rag_pipeline
            if pipeline is None:
                pipeline = RAGPipeline(st.session_state.vector_store)
                st.session_state.rag_pipeline = pipeline

            with st.spinner("Checking your knowledge base..."):
                kb_answer, sources, retrieved_context = pipeline.answer_with_context(question)

            kb_answer = (kb_answer or "").strip()
            retrieved_context = (retrieved_context or "").strip()
            sources = sources or []
        except Exception as exc:
            # Gemini remains usable for general questions if retrieval fails.
            retrieval_status = f"The knowledge base could not be queried for this turn: {exc}"
    else:
        retrieval_status = "No documents are indexed for this turn."

    # The project requirement is strict: no retrieved passages means no model
    # answer. General questions are intentionally refused in this mode.
    if not retrieved_context or not sources:
        answer = NO_DOCUMENT_ANSWER
        current_messages.append(
            {"role": "assistant", "content": answer, "sources": []}
        )
        with st.chat_message("assistant"):
            st.write(answer)
        st.rerun()

    history_for_model = current_messages[:-1][-6:]
    client = GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL) if GEMINI_KEY_SET else None

    try:
        with st.spinner("Writing a document-grounded answer..."):
            try:
                if client is None:
                    raise GeminiError("Gemini is not configured.")
                answer = client.generate_answer(
                    question=question,
                    knowledge_base_answer=kb_answer,
                    sources=sources,
                    retrieved_context=retrieved_context,
                    history=[],
                    retrieval_status=retrieval_status,
                )
            except GeminiError as gemini_exc:
                # If Gemini is unavailable, use Groq only with the same strict
                # document-only prompt when Groq is configured.
                fallback_pipeline = st.session_state.rag_pipeline
                if fallback_pipeline is None:
                    fallback_pipeline = RAGPipeline(st.session_state.vector_store)
                if not fallback_pipeline.has_groq:
                    raise gemini_exc
                answer = fallback_pipeline.answer_general(
                    question=question,
                    knowledge_base_answer=kb_answer,
                    sources=sources,
                    retrieved_context=retrieved_context,
                    history=[],
                    retrieval_status=retrieval_status,
                )

        with st.chat_message("assistant"):
            st.write_stream(stream_generator(answer))

        current_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
        st.rerun()
    except GeminiError as exc:
        error_message = f"Gemini could not generate an answer, and no Groq fallback is available: {exc}"
        st.error(error_message)
        current_messages.append(
            {"role": "assistant", "content": error_message, "sources": sources}
        )
    except Exception as exc:
        error_message = f"Unexpected error while generating the answer: {exc}"
        st.error(error_message)
        current_messages.append(
            {"role": "assistant", "content": error_message, "sources": sources}
        )
