"""Structured, provider-agnostic trace for one production RAG execution.

This module owns the generic retrieval -> prompt -> generation path and its
diagnostic data. Evaluation and UI layers consume the returned ``RAGTrace``
without re-running any backend step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from multimodal_rag.paths import (
    INGESTION_ARTIFACTS_DIR,
    LEGACY_INGESTION_ARTIFACTS_DIR,
)
from multimodal_rag.rag.embedding.embedder import EmbeddingConfig
from multimodal_rag.rag.generation.answer_generator import GenerationConfig, generate_answer
from multimodal_rag.rag.generation.citation import resolve_citations
from multimodal_rag.rag.generation.prompt_builder import ConversationTurn, build_prompt
from multimodal_rag.rag.retrieval.retriever_2 import RetrieverConfig, retrieve


@dataclass
class RetrievedItemTrace:
    rank: int
    raw_faiss_score: float
    chunk_id: str
    document_id: str
    document_name: str
    page_numbers: list[int]
    section_title: str | None
    chunk_text: str
    metadata: dict[str, Any] | None = None
    metadata_note: str | None = None
    combined_rerank_score: None = None


@dataclass
class RAGTrace:
    original_question: str
    generated_answer: str = ""
    retrieved_items: list[RetrievedItemTrace] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    uncited_sources: list[dict[str, Any]] = field(default_factory=list)
    retriever: str = "multimodal_rag.rag.retrieval.retriever_2"
    embedding_model: str | None = None
    generation_model: str | None = None
    configured_top_k: int = 0
    actual_retrieved_count: int = 0
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    rag_latency_ms: float | None = None
    generation_prompt_tokens: int | None = None
    generation_completion_tokens: int | None = None
    generation_total_tokens: int | None = None
    estimated_generation_cost: float | None = None


def load_chunk_metadata(
    roots: tuple[Path, ...] = (INGESTION_ARTIFACTS_DIR, LEGACY_INGESTION_ARTIFACTS_DIR),
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Read existing ingestion metadata by chunk ID without writing anything."""
    metadata_by_id: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("chunks.json")):
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read chunk metadata from {path}: {exc}") from exc
            if not isinstance(records, list):
                raise RuntimeError(f"Chunk metadata file must contain a JSON list: {path}")

            for record in records:
                if not isinstance(record, dict):
                    continue
                metadata = record.get("metadata")
                if not isinstance(metadata, dict) or metadata.get("chunk_id") is None:
                    continue
                chunk_id = str(metadata["chunk_id"])
                prior = metadata_by_id.get(chunk_id)
                if prior is not None and prior != metadata:
                    ambiguous.add(chunk_id)
                    continue
                metadata_by_id[chunk_id] = dict(metadata)

    return metadata_by_id, ambiguous


def _citation_dict(citation: Any) -> dict[str, Any]:
    return {
        "marker": citation.marker,
        "source_file": citation.source_file,
        "page_numbers": list(citation.page_numbers),
        "chunk_id": citation.chunk_id,
        "section_title": citation.section_title,
    }


def run_rag_trace(
    question: str,
    *,
    index,
    id_map,
    top_k: int,
    embedding_config: EmbeddingConfig | None = None,
    generation_config: GenerationConfig | None = None,
    conversation_history: list[ConversationTurn] | None = None,
    max_history_turns: int = 5,
    metadata_by_id: dict[str, dict[str, Any]] | None = None,
    ambiguous_metadata: set[str] | None = None,
    retrieve_fn: Callable[..., list[Any]] = retrieve,
    build_prompt_fn: Callable[..., Any] = build_prompt,
    generate_answer_fn: Callable[..., str] = generate_answer,
    resolve_citations_fn: Callable[..., Any] = resolve_citations,
) -> RAGTrace:
    """Execute retrieval, prompt construction, generation, and citation resolution once."""
    embedding_config = embedding_config or EmbeddingConfig()
    generation_config = generation_config or GenerationConfig()
    if metadata_by_id is None:
        metadata_by_id, loaded_ambiguous = load_chunk_metadata()
        if ambiguous_metadata is None:
            ambiguous_metadata = loaded_ambiguous
    if ambiguous_metadata is None:
        ambiguous_metadata = set()

    trace = RAGTrace(
        original_question=question,
        embedding_model=embedding_config.model_name,
        generation_model=generation_config.model_name,
        configured_top_k=top_k,
    )

    import time

    rag_start = time.perf_counter()
    retrieval_start = time.perf_counter()
    chunks = retrieve_fn(
        question,
        index,
        id_map,
        embedding_config=embedding_config,
        retriever_config=RetrieverConfig(top_k=top_k),
    )
    trace.retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
    trace.actual_retrieved_count = len(chunks)

    for rank, chunk in enumerate(chunks, start=1):
        metadata = metadata_by_id.get(chunk.chunk_id)
        metadata_note = None
        if chunk.chunk_id in ambiguous_metadata:
            metadata = None
            metadata_note = "unavailable: conflicting metadata records for this chunk ID"
        elif metadata is None:
            metadata_note = "unavailable: chunk ID was not found in existing chunks.json files"

        trace.retrieved_items.append(
            RetrievedItemTrace(
                rank=rank,
                raw_faiss_score=float(chunk.score),
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.source_file,
                page_numbers=list(chunk.page_numbers),
                section_title=chunk.section_title,
                chunk_text=chunk.chunk_text,
                metadata=metadata,
                metadata_note=metadata_note,
            )
        )

    if chunks:
        built = build_prompt_fn(
            question,
            chunks,
            conversation_history=conversation_history,
            max_history_turns=max_history_turns,
        )
        generation_start = time.perf_counter()
        raw_answer = generate_answer_fn(built.prompt_text, generation_config)
        trace.generation_latency_ms = (time.perf_counter() - generation_start) * 1000
        cited_answer = resolve_citations_fn(raw_answer, built.source_map)
        trace.generated_answer = cited_answer.answer_text
        trace.citations = [_citation_dict(citation) for citation in cited_answer.citations]
        trace.uncited_sources = [
            _citation_dict(citation) for citation in cited_answer.uncited_sources
        ] if hasattr(cited_answer, "uncited_sources") else []
    else:
        trace.generation_latency_ms = 0.0
        trace.generated_answer = "I couldn't find anything relevant to that in the documents."

    trace.rag_latency_ms = (time.perf_counter() - rag_start) * 1000
    return trace
