"""
Automated tests for the RAG workflow. They run fully offline:
  * a deterministic hashing embedder replaces sentence-transformers
  * a fake LLM replaces Gemini
  * PDFs are generated in-memory (no external files needed)
Run with:  pytest -v
"""

import hashlib
import json
import re

import numpy as np
import pytest

from rag_pipeline import (
    REFUSAL_MESSAGE,
    ConfigurationError,
    GeminiClient,
    LLMError,
    NoTextFoundError,
    PageText,
    PDFExtractionError,
    RAGConfig,
    RAGPipeline,
    VectorStore,
    chunk_page,
    chunk_pages,
    extract_pages,
    parse_model_output,
)

STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "is", "are", "for", "what", "does", "which", "how", "between"}


class HashingEmbedder:
    """Deterministic bag-of-words embedder so tests need no model download."""

    dim = 4096

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in STOPWORDS:
                    continue
                bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                out[i, bucket] += 1.0
        return out


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, system, prompt):
        self.calls.append((system, prompt))
        if isinstance(self.response, Exception):
            raise self.response
        if isinstance(self.response, dict):
            return json.dumps(self.response)
        return self.response


def build_pdf(pages):
    """Create a minimal valid multi-page PDF (Helvetica text) in memory."""

    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    content_ids = [5 + 2 * i for i in range(n)]
    objs = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{p} 0 R" for p in page_ids), n),
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(pages):
        lines = text.split("\n") if text else []
        body = "".join(f"({esc(line)}) Tj T*\n" for line in lines)
        stream = f"BT /F1 11 Tf 50 750 Td 14 TL\n{body}ET"
        objs[page_ids[i]] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_ids[i]} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objs[content_ids[i]] = f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid in sorted(objs):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n{objs[oid]}\nendobj\n".encode("latin-1")
    xref = len(out)
    size = max(objs) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for oid in range(1, size):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


PAGE_TEXTS = [
    "AUTOSAR layered architecture separates application software\nfrom basic software and hardware.",
    "The Runtime Environment RTE provides communication services\nbetween software components and acts as middleware.",
    "The Diagnostic Event Manager DEM stores diagnostic trouble codes\nand event status information.",
]


@pytest.fixture
def config():
    return RAGConfig(chunk_size=500, chunk_overlap=50, top_k=3, min_score=0.2)


@pytest.fixture
def pdf_bytes():
    return build_pdf(PAGE_TEXTS)


@pytest.fixture
def pipeline(config, pdf_bytes):
    p = RAGPipeline(HashingEmbedder(), llm=None, config=config)
    p.ingest_pdf("hld.pdf", pdf_bytes)
    return p


# ------------------------------ extraction -------------------------------- #
def test_extract_pages_preserves_page_numbers(pdf_bytes):
    pages = extract_pages(pdf_bytes)
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Runtime Environment" in pages[1].text
    assert "Diagnostic Event Manager" in pages[2].text


def test_extract_rejects_empty_bytes():
    with pytest.raises(PDFExtractionError):
        extract_pages(b"")


def test_extract_rejects_corrupt_pdf():
    with pytest.raises(PDFExtractionError):
        extract_pages(b"this is definitely not a pdf")


def test_pdf_without_text_is_rejected(config):
    blank = build_pdf(["", ""])
    pipeline = RAGPipeline(HashingEmbedder(), config=config)
    with pytest.raises(NoTextFoundError):
        pipeline.ingest_pdf("scanned.pdf", blank)
    assert not pipeline.has_documents


# -------------------------------- chunking -------------------------------- #
def test_chunks_keep_correct_page_numbers():
    pages = [
        PageText(1, " ".join(f"ALPHA sentence number {i}." for i in range(60))),
        PageText(2, " ".join(f"BETA sentence number {i}." for i in range(60))),
    ]
    chunks = chunk_pages(pages, "doc", "x.pdf", size=300, overlap=60)
    assert len(chunks) > 2
    for c in chunks:
        assert c.page in (1, 2)
        assert ("ALPHA" in c.text) == (c.page == 1)
        assert ("BETA" in c.text) == (c.page == 2)


def test_chunk_size_is_respected_and_overlap_exists():
    text = " ".join(f"Sentence {i} talks about AUTOSAR modules." for i in range(80))
    chunks = chunk_page(text, size=300, overlap=80)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].split(". ")[-1] in chunks[1] or chunks[0][-30:] in chunks[1]


def test_very_long_unbroken_text_is_split():
    chunks = chunk_page("X" * 2500, size=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c) for c in chunks) >= 2500


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        chunk_page("text", size=100, overlap=100)
    with pytest.raises(ValueError):
        RAGConfig(chunk_size=500, chunk_overlap=600)
    with pytest.raises(ValueError):
        RAGConfig(top_k=0)


# --------------------------- vector store / FAISS -------------------------- #
def test_vector_store_returns_most_relevant_chunk(pipeline):
    hits = pipeline.store.search("runtime environment RTE communication services", 3)
    assert hits[0].chunk.page == 2
    assert hits[0].score > hits[-1].score or len(hits) == 1


def test_empty_store_search_returns_nothing():
    assert VectorStore(HashingEmbedder()).search("anything", 3) == []


def test_remove_document_rebuilds_index(pipeline, pdf_bytes):
    doc_id = next(iter(pipeline.documents))
    assert pipeline.remove_document(doc_id) is True
    assert not pipeline.has_documents
    assert pipeline.store.search("runtime", 3) == []


def test_duplicate_upload_is_not_indexed_twice(pipeline, pdf_bytes):
    before = len(pipeline.store)
    report = pipeline.ingest_pdf("copy.pdf", pdf_bytes)
    assert report.skipped_duplicate is True
    assert len(pipeline.store) == before


# --------------------------- grounded answering ---------------------------- #
def test_answer_has_correct_page_citation(pipeline):
    llm = FakeLLM({"answerable": True, "answer": "The RTE provides communication services [1].", "citations": [1]})
    pipeline.llm = llm
    result = pipeline.ask("runtime environment RTE communication services")
    assert result.answerable is True
    assert [c.page for c in result.citations] == [2]
    assert result.citations[0].source == "hld.pdf"
    assert "Page 2" in llm.calls[0][1]          # the evidence was sent to the model
    assert "ONLY" in llm.calls[0][0]            # grounding rules were sent


def test_refuses_without_calling_llm_when_irrelevant(pipeline):
    llm = FakeLLM({"answerable": True, "answer": "x [1]", "citations": [1]})
    pipeline.llm = llm
    result = pipeline.ask("banana smoothie recipe blender")
    assert result.answerable is False
    assert result.refusal_reason == "low_relevance"
    assert result.answer == REFUSAL_MESSAGE
    assert llm.calls == []


def test_refuses_when_model_says_not_answerable(pipeline):
    pipeline.llm = FakeLLM({"answerable": False, "answer": "Not covered.", "citations": []})
    result = pipeline.ask("runtime environment RTE communication services")
    assert result.answerable is False
    assert result.refusal_reason == "model_declined"
    assert result.citations == []


def test_refuses_when_citations_are_fabricated(pipeline):
    pipeline.llm = FakeLLM({"answerable": True, "answer": "Something [99].", "citations": [99]})
    result = pipeline.ask("runtime environment RTE communication services")
    assert result.answerable is False
    assert result.refusal_reason == "no_valid_citations"


def test_refuses_when_answer_has_no_citations(pipeline):
    pipeline.llm = FakeLLM({"answerable": True, "answer": "Plain claim without evidence.", "citations": []})
    result = pipeline.ask("runtime environment RTE communication services")
    assert result.answerable is False


def test_inline_markers_used_when_citation_list_missing(pipeline):
    pipeline.llm = FakeLLM({"answerable": True, "answer": "RTE offers communication [1]."})
    result = pipeline.ask("runtime environment RTE communication services")
    assert result.answerable is True
    assert result.citations[0].page == 2


def test_invalid_json_from_model_is_refused(pipeline):
    pipeline.llm = FakeLLM("Sure! The RTE does things.")
    result = pipeline.ask("runtime environment RTE communication services")
    assert result.answerable is False
    assert result.refusal_reason == "invalid_model_output"


def test_json_in_code_fence_is_parsed():
    raw = '```json\n{"answerable": true, "answer": "ok [1]", "citations": [1]}\n```'
    assert parse_model_output(raw)["answerable"] is True
    with pytest.raises(ValueError):
        parse_model_output("")


def test_no_documents_refusal(config):
    pipeline = RAGPipeline(HashingEmbedder(), llm=FakeLLM("{}"), config=config)
    result = pipeline.ask("What is the RTE?")
    assert result.answerable is False
    assert result.refusal_reason == "no_documents"


def test_empty_question_raises(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_llm_failure_propagates(pipeline):
    pipeline.llm = FakeLLM(LLMError("quota exceeded"))
    with pytest.raises(LLMError):
        pipeline.ask("runtime environment RTE communication services")


def test_missing_llm_raises_configuration_error(pipeline):
    pipeline.llm = None
    with pytest.raises(ConfigurationError):
        pipeline.ask("runtime environment RTE communication services")


def test_gemini_client_requires_real_key():
    with pytest.raises(ConfigurationError):
        GeminiClient(api_key="")
    with pytest.raises(ConfigurationError):
        GeminiClient(api_key="your_gemini_api_key_here")