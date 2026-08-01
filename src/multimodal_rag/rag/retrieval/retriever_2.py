"""
Retriever Module (RAG Stage 4)
=================================

Turns a user's text query into ranked, relevant chunks: embeds the query
using the SAME model/config as Stage 2 (embedder.py), then searches the
FAISS index built in Stage 3 (faiss_index.py). Deliberately thin - all
the real logic (embedding, index search) already exists; this module's
job is just correct composition + result shaping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from multimodal_rag.rag.embedding.embedder import EmbeddingConfig, _embed_texts
from multimodal_rag.rag.indexing.faiss_index import IndexedChunkRef, search

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "what", "are", "is", "was", "were", "a", "an", "the", "of", "in", "on",
    "for", "to", "and", "or", "with", "as", "by", "at", "from", "be", "been",
    "it", "its", "do", "does", "did", "have", "has", "had", "can", "could",
    "will", "would", "should", "this", "that", "these", "those",
}
# Standard English stopwords - excluded so overlap scoring reflects real
# topic/polarity words (e.g. "long-term", "technology") rather than
# near-universal function words that would overlap with almost any chunk.


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\-]+", text.lower()) if t not in _STOPWORDS}


def _lexical_overlap(query_tokens: set[str], chunk_text: str) -> float:
    """
    Fraction of query_tokens also present in chunk_text (0..1). MiniLM
    captures topic similarity well but underweights lexical/polarity
    differences (e.g. "short-term" vs "long-term" score as near-
    identical topically) - this recovers that signal cheaply, as a
    post-FAISS reranking pass, without touching the embedding model.
    """
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokenize(chunk_text)) / len(query_tokens)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_file: str
    page_numbers: list[int]
    section_title: str | None
    chunk_text: str
    score: float
    combined_rerank_score: float | None = field(default=None, compare=False)


@dataclass
class RetrieverConfig:
    top_k: int = 5
    min_score: float = 0.0
    # Results below this cosine-similarity score are dropped - default 0
    # keeps all top_k results; raise it to filter out weak matches when a
    # query has no genuinely relevant chunks in the corpus.
    lexical_rerank_weight: float = 0.15
    # Blends a lexical token-overlap score into FAISS's cosine ranking:
    # final_score = cosine_score + lexical_rerank_weight * overlap_fraction.
    # 0.15 is the minimum weight (derived from real scored examples, not
    # guessed) that corrects cases where MiniLM ranks a topically-similar
    # but lexically/polarity-wrong chunk above the correct one - e.g. for
    # "long-term technology implications", a "Measurement" chunk (cosine
    # 0.3799, overlap 2/3 query terms) was outranking the correct
    # "Technology" chunk (cosine 0.3361, overlap 3/3 query terms).
    # Minimum flip weight = (0.3799-0.3361)/(1.0-0.667) ≈ 0.1315; 0.15
    # adds a small margin. Set to 0.0 to disable reranking entirely.


def retrieve(
    query: str,
    index,
    id_map: dict[int, IndexedChunkRef],
    embedding_config: EmbeddingConfig | None = None,
    retriever_config: RetrieverConfig | None = None,
    embed_fn=None,
) -> list[RetrievedChunk]:
    """
    Embed `query` and return its top-K most similar chunks from the
    index. `embed_fn` is injectable (same seam as embedder.py) for
    testing without the real model.
    """
    embedding_config = embedding_config or EmbeddingConfig()
    retriever_config = retriever_config or RetrieverConfig()
    embed_fn = embed_fn or _embed_texts

    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    query_vector = embed_fn([query.strip()], embedding_config)[0]
    raw_results = search(index, id_map, query_vector, top_k=retriever_config.top_k)

    query_tokens = _tokenize(query) if retriever_config.lexical_rerank_weight else set()
    scored_results = [
        (
            ref,
            score,
            score
            + retriever_config.lexical_rerank_weight
            * _lexical_overlap(query_tokens, ref.chunk_text),
        )
        for ref, score in raw_results
    ]
    if retriever_config.lexical_rerank_weight:
        scored_results = sorted(scored_results, key=lambda result: result[2], reverse=True)

    results = [
        RetrievedChunk(
            chunk_id=ref.chunk_id, document_id=ref.document_id, source_file=ref.source_file,
            page_numbers=ref.page_numbers, section_title=ref.section_title,
            chunk_text=ref.chunk_text, score=score,
            combined_rerank_score=combined_score,
        )
        for ref, score, combined_score in scored_results
        if score >= retriever_config.min_score
    ]
    logger.info("Retrieved %d chunk(s) for query (top_k=%d)", len(results), retriever_config.top_k)
    return results
