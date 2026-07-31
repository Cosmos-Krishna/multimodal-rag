"""
Embedding Generator Module (RAG Stage 2)
===========================================

First module of the RAG system proper, sitting immediately after
ingestion. Consumes the `chunks.json` files ingestion (Modules 1-10)
already produces per document, and turns each chunk's text into a
dense vector embedding, ready for FAISS indexing (Stage 3 - not built
yet).

This module does NOT modify, import internals from, or depend on
anything in `ingestion/` beyond reading its already-finalized JSON
output - kept deliberately decoupled, per the "don't rewrite existing
modules" requirement. It reads a public, stable file format
(`chunks.json`), not ingestion's internal Python objects.

Design (consistent with the ingestion pipeline's established patterns):
- Config-driven (EmbeddingConfig), model name defaults to the
  already-locked MiniLM choice from the architecture phase - no new
  design decision being made here.
- The embedding model is loaded lazily, once, as a module-level
  singleton (same pattern as ocr_extractor.py's RapidOCR engine) -
  model load has real overhead, don't pay it per document.
- Never silently drops a chunk: if a chunk's text is empty/unembeddable,
  it's skipped with a logged reason and reported in the returned
  EmbeddingResult's `skipped` list, not just absent with no explanation.

HONEST LIMITATION (confirmed, not assumed): sentence-transformers
downloads model weights from huggingface.co on first use. This sandbox's
network allowlist does not include that host (confirmed via a real
failed load attempt during development, same restriction that has
affected Docling and Gemini elsewhere in this project) - so live
embedding generation could not be executed end-to-end here. Every piece
of this module's OWN logic (chunk loading, batching, skip-handling,
output writing) is tested below against a fake embedding function
standing in for the real model call - the seam is `_embed_texts`, the
one function that actually calls the model. In an environment with
normal internet access, this works with zero code changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Keep Hugging Face model resolution offline by default. Explicit user
# environment values remain authoritative via setdefault().
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Matches the MiniLM embedding model already locked during the
# architecture phase (see final_engineering_decisions.md) - kept
# unchanged per that decision, not re-litigated here.


class EmbeddingError(Exception):
    """Base exception for embedding generation failures."""


class EmbeddingModelUnavailableError(EmbeddingError):
    """Raised when the embedding model's weights could not be loaded
    (e.g. blocked Hugging Face access, no cached weights) - an
    operational/environment problem, distinguished from a genuine bug."""


@dataclass
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    # "auto" | "cpu" | "cuda" - passed straight to SentenceTransformer.
    # "auto" lets sentence-transformers pick, matching this project's
    # established "use GPU opportunistically, never require it" stance
    # (same philosophy as OCRConfig/LayoutSegmenterConfig's accelerator options).
    batch_size: int = 32
    normalize_embeddings: bool = True
    # True by default: normalized vectors + FAISS's IndexFlatIP (inner
    # product) together implement cosine similarity, the standard,
    # well-understood similarity metric for sentence embeddings. This
    # choice is recorded here because Stage 3 (FAISS indexing) MUST
    # build its index type consistently with it - noted so that
    # dependency is explicit rather than implicit tribal knowledge.
    min_chunk_chars: int = 3
    # Chunks shorter than this after stripping are skipped rather than
    # embedded - an empty or near-empty chunk produces a low-information
    # embedding that mostly adds retrieval noise, not signal.
    # NOTE: left unchanged/untouched by the heading-only filter below -
    # this still governs "is there any text at all", nothing more.

    heading_only_char_threshold: int = 15
    # Separate, narrower check than min_chunk_chars: a chunk can be
    # longer than min_chunk_chars in raw length yet still be almost
    # entirely headings/page artifacts (e.g. "Long-term implications /
    # Technology / Knowledge Institute"), which otherwise get an
    # artificially strong embedding similarity from having almost no
    # real content to dilute the topic words. Chunks whose EFFECTIVE
    # body length (see _effective_body_length) falls below this are
    # skipped as heading-only, regardless of raw length. Deliberately
    # does NOT change min_chunk_chars behavior for legitimate small
    # chunks - this only catches the heading-artifact case.


@dataclass
class EmbeddedChunk:
    chunk_id: str
    document_id: str
    source_file: str
    page_numbers: list[int]
    section_title: str | None
    chunk_text: str
    embedding_index: int
    # Position of this chunk's vector within the returned embeddings
    # array - the join key between EmbeddingResult.embeddings[i] and
    # EmbeddingResult.chunks[i]. Kept explicit (not just relying on list
    # order) because Stage 3 will persist these to disk separately from
    # the raw vector array, where list-order-as-identity is fragile.


@dataclass
class SkippedChunk:
    chunk_id: str
    reason: str


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray  # shape (N, embedding_dim), float32
    chunks: list[EmbeddedChunk]  # len N, chunks[i] corresponds to embeddings[i]
    skipped: list[SkippedChunk]
    model_name: str
    embedding_dim: int


# --------------------------------------------------------------------------
# Heading-only / page-artifact detection
# --------------------------------------------------------------------------

_HEADING_LINE_RE = re.compile(r"^[A-Z][\w\s\-&]{0,40}$")
# A line counts as "heading-like" if it starts capitalized, is short
# (<=~6 words judged below), and carries no terminal sentence
# punctuation - matches patterns like "Long-term implications" or
# "Knowledge Institute", not real body sentences.


def _effective_body_length(text: str) -> int:
    """
    Length of `text` with heading-like lines stripped out. Used only to
    detect chunks that are almost entirely headings/page artifacts (a
    narrower, separate check from min_chunk_chars - see
    EmbeddingConfig.heading_only_char_threshold). Such chunks otherwise
    pass the normal length check while carrying almost no real content,
    letting them dominate cosine similarity with topic words alone.
    """
    body_lines = [
        line for line in text.splitlines()
        if not (
            _HEADING_LINE_RE.match(line.strip())
            and len(line.split()) <= 6
            and not line.strip().endswith((".", "!", "?", ":", ";", ","))
        )
    ]
    return len(" ".join(body_lines).strip())


# --------------------------------------------------------------------------
# Loading chunks.json (ingestion's public output format)
# --------------------------------------------------------------------------

def load_chunks_json(chunks_json_path: str | Path) -> list[dict]:
    """
    Load a document's chunks.json, exactly as Module 9's Output Writer
    produces it: `[{"chunk_text": ..., "metadata": {...}}, ...]`.

    Raises FileNotFoundError / json.JSONDecodeError directly (not wrapped)
    - a missing or malformed chunks.json is a whole-document problem for
    THIS stage, analogous to how ingestion's own PDFLoadError works: fail
    loudly and immediately, don't guess at partial content.
    """
    path = Path(chunks_json_path)
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"'{path}' does not contain a JSON list of chunks (got {type(data).__name__})")
    return data


# --------------------------------------------------------------------------
# Model loading (lazy singleton, same pattern as ocr_extractor.py)
# --------------------------------------------------------------------------

_model = None
_model_name_loaded: str | None = None


def _get_model(config: EmbeddingConfig):
    global _model, _model_name_loaded
    if _model is not None and _model_name_loaded == config.model_name:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbeddingModelUnavailableError(
            f"sentence-transformers is not installed: {e}"
        ) from e

    device = None if config.device == "auto" else config.device
    try:
        logger.info("Loading embedding model '%s' (first use)...", config.model_name)
        _model = SentenceTransformer(
            config.model_name,
            device=device,
            local_files_only=True,
        )
        _model_name_loaded = config.model_name
    except Exception as e:
        message = str(e)
        if "huggingface" in message.lower() or "connect" in message.lower():
            raise EmbeddingModelUnavailableError(
                f"Could not load embedding model '{config.model_name}' "
                f"(network/model-cache issue): {e}"
            ) from e
        raise EmbeddingModelUnavailableError(f"Could not load embedding model: {e}") from e

    return _model


def _embed_texts(texts: list[str], config: EmbeddingConfig) -> np.ndarray:
    """
    The one function that actually calls the model - deliberately
    isolated as a single seam so every OTHER piece of this module's logic
    (chunk loading, filtering, batching orchestration, result assembly,
    output writing) can be tested against a fake/injected version of just
    this function, without needing the real model to load. See module
    docstring's HONEST LIMITATION section.
    """
    model = _get_model(config)
    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def embed_chunks(
    chunk_records: list[dict],
    config: EmbeddingConfig | None = None,
    embed_fn=None,
) -> EmbeddingResult:
    """
    Embed a list of chunk records (as loaded via `load_chunks_json`).

    `embed_fn` is an injectable dependency defaulting to `_embed_texts`
    (the real model call) - the same seam-injection pattern already used
    for Docling's `segment_fn` in the orchestrator, for the same reason:
    the real model call can't run in this sandbox, but every other piece
    of logic here can and is tested against an injected fake.
    """
    config = config or EmbeddingConfig()
    embed_fn = embed_fn or _embed_texts

    texts_to_embed: list[str] = []
    embedded_chunk_stubs: list[EmbeddedChunk] = []
    skipped: list[SkippedChunk] = []

    for record in chunk_records:
        text = (record.get("chunk_text") or "").strip()
        metadata = record.get("metadata", {})
        chunk_id = metadata.get("chunk_id", "<unknown>")

        if len(text) < config.min_chunk_chars:
            skipped.append(SkippedChunk(
                chunk_id=chunk_id,
                reason=f"chunk_text below min_chunk_chars ({len(text)} < {config.min_chunk_chars})",
            ))
            continue

        effective_len = _effective_body_length(text)
        if effective_len < config.heading_only_char_threshold:
            skipped.append(SkippedChunk(
                chunk_id=chunk_id,
                reason=(
                    f"heading-only/page-artifact chunk (effective body "
                    f"length {effective_len} < {config.heading_only_char_threshold}, "
                    f"raw length {len(text)})"
                ),
            ))
            continue

        texts_to_embed.append(text)
        embedded_chunk_stubs.append(EmbeddedChunk(
            chunk_id=chunk_id,
            document_id=metadata.get("document_id", ""),
            source_file=metadata.get("source_file", ""),
            page_numbers=metadata.get("page_numbers", []),
            section_title=metadata.get("section_title"),
            chunk_text=text,
            embedding_index=-1,  # filled in below once final count is known
        ))

    if not texts_to_embed:
        logger.warning("No embeddable chunks found (%d skipped) - returning empty result", len(skipped))
        return EmbeddingResult(
            embeddings=np.zeros((0, 0), dtype=np.float32), chunks=[], skipped=skipped,
            model_name=config.model_name, embedding_dim=0,
        )

    embeddings = embed_fn(texts_to_embed, config)

    if embeddings.shape[0] != len(embedded_chunk_stubs):
        raise EmbeddingError(
            f"Embedding count mismatch: got {embeddings.shape[0]} vectors for "
            f"{len(embedded_chunk_stubs)} chunks - refusing to guess an alignment."
        )

    for i, chunk in enumerate(embedded_chunk_stubs):
        chunk.embedding_index = i

    logger.info(
        "Embedded %d chunks (dim=%d, model=%s), skipped %d",
        len(embedded_chunk_stubs), embeddings.shape[1], config.model_name, len(skipped),
    )

    return EmbeddingResult(
        embeddings=embeddings, chunks=embedded_chunk_stubs, skipped=skipped,
        model_name=config.model_name, embedding_dim=embeddings.shape[1],
    )


def embed_document(
    chunks_json_path: str | Path, config: EmbeddingConfig | None = None, embed_fn=None,
) -> EmbeddingResult:
    """Convenience wrapper: load a document's chunks.json and embed it in one call."""
    records = load_chunks_json(chunks_json_path)
    return embed_chunks(records, config, embed_fn)


# --------------------------------------------------------------------------
# Output writing
# --------------------------------------------------------------------------

def write_embeddings(result: EmbeddingResult, output_dir: str | Path) -> tuple[Path, Path]:
    """
    Write an EmbeddingResult to disk, into the SAME per-document output
    folder ingestion already uses (alongside chunks.json, metadata.json,
    etc) - consistent with the existing project convention of one
    self-contained folder per document.

    Writes:
    - embeddings.npy: the raw (N, dim) float32 vector array - compact,
      fast to load, standard format for feeding into FAISS.
    - embeddings_metadata.json: chunk_id/document_id/page_numbers/
      section_title/embedding_index for every embedded chunk, PLUS the
      skipped list with reasons - human-inspectable, and this is what
      Stage 3 (FAISS indexing) will use to map FAISS result positions
      back to real chunk identity and text.

    Returns (embeddings_npy_path, embeddings_metadata_json_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / "embeddings.npy"
    np.save(npy_path, result.embeddings)

    metadata = {
        "model_name": result.model_name,
        "embedding_dim": result.embedding_dim,
        "total_embedded": len(result.chunks),
        "total_skipped": len(result.skipped),
        "chunks": [
            {
                "chunk_id": c.chunk_id, "document_id": c.document_id,
                "source_file": c.source_file, "page_numbers": c.page_numbers,
                "section_title": c.section_title, "embedding_index": c.embedding_index,
                "chunk_text": c.chunk_text,
            }
            for c in result.chunks
        ],
        "skipped": [{"chunk_id": s.chunk_id, "reason": s.reason} for s in result.skipped],
    }
    metadata_path = output_dir / "embeddings_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info("Wrote embeddings for %d chunks to %s", len(result.chunks), output_dir)
    return npy_path, metadata_path
