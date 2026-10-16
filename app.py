"""
app.py
======
Streamlit interface for the AUTOSAR HLD Document Analysis Assistant (RAG).
Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
    REFUSAL_REASONS,
    Answer,
    ConfigurationError,
    GeminiClient,
    LLMError,
    PDFExtractionError,
    RAGConfig,
    RAGPipeline,
    SentenceTransformerEmbedder,
)

load_dotenv()

st.set_page_config(
    page_title="AUTOSAR HLD Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #0f2b5b 0%, #1d4ed8 55%, #0ea5e9 100%);
    padding: 1.6rem 1.8rem; border-radius: 16px; color: #fff; margin-bottom: 1.1rem;
    box-shadow: 0 8px 24px rgba(15, 43, 91, 0.25);
}
.hero h1 { margin: 0; font-size: 1.9rem; color: #fff; }
.hero p  { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 1rem; }
.badge { display: inline-block; padding: 0.18rem 0.65rem; border-radius: 999px;
         font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; }
.badge-ok   { background: #dcfce7; color: #166534; }
.badge-warn { background: #fef3c7; color: #92400e; }
.badge-info { background: #e0f2fe; color: #075985; }
.pagepill { display: inline-block; background: #eef2ff; color: #3730a3; border-radius: 8px;
            padding: 0.12rem 0.55rem; margin: 0.15rem 0.3rem 0.15rem 0; font-size: 0.82rem; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLE_QUESTIONS = [
    "What is the purpose of the Runtime Environment (RTE)?",
    "Describe the layered architecture described in the document.",
    "How do software components communicate with each other?",
    "Which diagnostic services or modules are described?",
]


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def load_embedder(model_name: str) -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder(model_name)
    embedder.load()
    return embedder


def init_state() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        embedder = load_embedder(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
        st.session_state.pipeline = RAGPipeline(embedder, llm=None, config=RAGConfig.from_env())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)
    return st.session_state.pipeline


def get_llm(api_key: str, model: str) -> GeminiClient:
    cache = st.session_state.get("llm_cache")
    if cache and cache[0] == (api_key, model):
        return cache[1]
    client = GeminiClient(api_key=api_key, model=model)
    st.session_state.llm_cache = ((api_key, model), client)
    return client


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def index_files(pipeline: RAGPipeline, files) -> None:
    progress = st.progress(0.0, text="Starting...")
    for i, f in enumerate(files):
        progress.progress(i / len(files), text=f"Indexing {f.name} ...")
        try:
            report = pipeline.ingest_pdf(f.name, f.getvalue())
        except PDFExtractionError as exc:
            st.error(f"**{f.name}**: {exc}")
            continue
        except Exception as exc:  # defensive: never crash the UI on one bad file
            st.error(f"**{f.name}**: unexpected error - {exc}")
            continue
        if report.skipped_duplicate:
            st.info(f"{f.name} is already indexed.")
        else:
            st.toast(f"Indexed {f.name}: {report.pages_with_text}/{report.pages_total} pages, {report.chunks} chunks", icon="✅")
    progress.empty()


def render_sidebar(pipeline: RAGPipeline) -> tuple[str, str]:
    with st.sidebar:
        st.header("⚙️ Setup")

        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key and not env_key.lower().startswith("your_"):
            api_key = env_key
            st.success("Gemini API key loaded from .env")
        else:
            api_key = st.text_input("Gemini API key", type="password", help="Get a key from Google AI Studio.")
        model = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))

        st.divider()
        st.subheader("📄 Documents")
        files = st.file_uploader(
            "Upload AUTOSAR HLD PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Index documents", type="primary", disabled=not files):
            index_files(pipeline, files)

        if pipeline.documents:
            st.caption("Indexed documents")
            for doc_id, rep in list(pipeline.documents.items()):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{rep.source}**  \n{rep.pages_with_text}/{rep.pages_total} pages · {rep.chunks} chunks")
                if c2.button("🗑", key=f"del_{doc_id}", help="Remove this document"):
                    pipeline.remove_document(doc_id)
                    st.rerun()

        st.divider()
        with st.expander("Advanced retrieval settings"):
            pipeline.config.top_k = st.slider("Passages retrieved (top-k)", 1, 10, pipeline.config.top_k)
            pipeline.config.min_score = st.slider(
                "Minimum relevance to answer",
                0.0, 0.8, float(pipeline.config.min_score), 0.05,
                help="Higher = stricter refusals. Lower = answers more often, with more risk of weak evidence.",
            )

        c1, c2 = st.columns(2)
        if c1.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
        if c2.button("Reset all"):
            pipeline.clear()
            st.session_state.messages = []
            st.session_state.uploader_key += 1
            st.rerun()
    return api_key, model


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_answer(answer: Answer) -> None:
    if answer.answerable:
        st.markdown('<span class="badge badge-ok">✔ Grounded in documents</span>'
                    f'<span class="badge badge-info">{answer.latency_s:.1f}s</span>', unsafe_allow_html=True)
        st.markdown(answer.answer)

        pages = {}
        for c in answer.citations:
            pages.setdefault(c.source, set()).add(c.page)
        pills = "".join(
            f'<span class="pagepill">{_esc(src)} · p. {", ".join(str(p) for p in sorted(pgs))}</span>'
            for src, pgs in pages.items()
        )
        st.markdown(f"**Cited pages:** {pills}", unsafe_allow_html=True)

        for c in answer.citations:
            with st.expander(f"[{c.index}] {c.source} - Page {c.page}  ·  relevance {c.score:.2f}"):
                st.write(c.excerpt)
    else:
        st.markdown('<span class="badge badge-warn">⚠ Insufficient evidence</span>', unsafe_allow_html=True)
        st.warning(answer.answer)
        reason = REFUSAL_REASONS.get(answer.refusal_reason or "", "")
        if reason:
            st.caption(f"Reason: {reason}")
        if answer.retrieved:
            with st.expander("Closest passages found (not used as evidence)"):
                for r in answer.retrieved:
                    st.markdown(f"**{r.chunk.source} - Page {r.chunk.page}** · similarity {r.score:.2f}")
                    st.write(r.chunk.text[:300] + ("..." if len(r.chunk.text) > 300 else ""))


def _esc(text: str) -> str:
    import html
    return html.escape(text, quote=True)


def transcript_markdown(messages: List[dict]) -> str:
    lines = ["# AUTOSAR HLD Assistant - Chat Transcript", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"## Q: {m['content']}", ""]
        else:
            lines += [m["content"], ""]
            ans: Optional[Answer] = m.get("answer")
            if ans and ans.citations:
                lines.append("**Sources:**")
                lines += [f"- {c.source}, page {c.page}" for c in ans.citations]
                lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Question handling
# --------------------------------------------------------------------------- #
def handle_question(pipeline: RAGPipeline, question: str, api_key: str, model: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            pipeline.llm = get_llm(api_key, model)
            with st.spinner("Searching documents and generating a grounded answer..."):
                answer = pipeline.ask(question)
        except ConfigurationError as exc:
            st.error(str(exc))
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc}", "answer": None})
            return
        except LLMError as exc:
            st.error(f"The language model request failed. {exc}")
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc}", "answer": None})
            return
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc}", "answer": None})
            return
        render_answer(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer.answer, "answer": answer})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    pipeline = init_state()
    api_key, model = render_sidebar(pipeline)

    st.markdown(
        """
        <div class="hero">
          <h1>🚗 AUTOSAR HLD Document Analysis Assistant</h1>
          <p>Upload AUTOSAR High-Level Design PDFs and ask questions. Answers are grounded in the
          documents with page-level citations, and the assistant refuses when evidence is missing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    docs = list(pipeline.documents.values())
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(docs))
    m2.metric("Pages indexed", sum(d.pages_with_text for d in docs))
    m3.metric("Chunks in FAISS index", len(pipeline.store))

    if not pipeline.has_documents:
        st.info("**Get started:** 1) add your Gemini API key in the sidebar (or `.env`), "
                "2) upload one or more AUTOSAR HLD PDFs, 3) click **Index documents**, 4) ask a question.")
    else:
        st.caption("Try an example question:")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"ex_{i}"):
                st.session_state.pending_question = q

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and m.get("answer") is not None:
                render_answer(m["answer"])
            else:
                st.markdown(m["content"])

    question = st.chat_input(
        "Ask a question about your AUTOSAR HLD documents..." if pipeline.has_documents
        else "Index at least one PDF to start asking questions",
        disabled=not pipeline.has_documents,
    )
    if not question:
        question = st.session_state.pop("pending_question", None)
    if question:
        handle_question(pipeline, question, api_key, model)

    if st.session_state.messages:
        st.download_button(
            "⬇ Download transcript (Markdown)",
            data=transcript_markdown(st.session_state.messages),
            file_name="autosar_rag_transcript.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()