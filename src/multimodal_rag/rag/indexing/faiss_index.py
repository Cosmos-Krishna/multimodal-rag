"""
FAISS Vector Indexing Module (RAG Stage 3)
==============================================

Aggregates every document's embeddings (Stage 2 output: embeddings.npy +
embeddings_metadata.json, one pair per document folder under
data/artifacts/ingestion/)
into a single FAISS index for corpus-wide search.

Uses IndexFlatIP (inner product) - correct choice ONLY because Stage 2's
EmbeddingConfig defaults to normalize_embeddings=True, so inner product
of normalized vectors == cosine similarity. If that default ever changes,
this index type must change with it.

FAISS indexes only store integer ids, not our string chunk_ids - so this
module assigns sequential int64 ids at build time and persists the
mapping (id -> chunk_id/document_id/page/text) in a separate JSON file,
the join key between a raw FAISS search result and real chunk identity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class IndexError_(Exception):
    """Base exception for indexing failures. Named with trailing
    underscore to avoid shadowing the builtin IndexError."""


class IndexNotFoundError(IndexError_):
    """Raised when attempting to load an index that doesn't exist on disk."""


class EmptyIndexError(IndexError_):
    """Raised when attempting to build an index from zero embeddings -
    an empty index is a meaningless artifact, better to fail loudly."""


@dataclass
class IndexConfig:
    index_filename: str = "faiss_index.bin"
    id_map_filename: str = "id_map.json"


@dataclass
class IndexedChunkRef:
    faiss_id: int
    chunk_id: str
    document_id: str
    source_file: str
    page_numbers: list[int]
    section_title: str | None
    chunk_text: str


def _discover_embedding_files(output_dir: str | Path) -> list[tuple[Path, Path]]:
    """Find every (embeddings.npy, embeddings_metadata.json) pair under
    output_dir's document subfolders. Skips a document folder cleanly
    (with a log line) if only one of the pair is present, rather than
    crashing the whole build over one incomplete document."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise EmptyIndexError(f"'{output_dir}' does not exist or is not a directory - nothing to index.")
    pairs: list[tuple[Path, Path]] = []
    for doc_dir in sorted(output_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        npy_path = doc_dir / "embeddings.npy"
        meta_path = doc_dir / "embeddings_metadata.json"
        if npy_path.exists() and meta_path.exists():
            pairs.append((npy_path, meta_path))
        elif npy_path.exists() or meta_path.exists():
            logger.warning(
                "Skipping '%s': found only one of embeddings.npy / embeddings_metadata.json", doc_dir
            )
    return pairs


def build_index_from_output_dir(
    output_dir: str | Path, config: IndexConfig | None = None,
) -> tuple[faiss.Index, list[IndexedChunkRef]]:
    """
    Scan every document folder under output_dir for Stage 2's embeddings,
    concatenate them, and build one FAISS index across the whole corpus.

    Raises EmptyIndexError if no embeddable chunks are found anywhere -
    building and persisting a zero-vector index would silently produce a
    retriever that always returns nothing, with no clear signal why.
    """
    config = config or IndexConfig()
    pairs = _discover_embedding_files(output_dir)

    all_vectors: list[np.ndarray] = []
    all_refs: list[IndexedChunkRef] = []
    next_id = 0

    for npy_path, meta_path in pairs:
        vectors = np.load(npy_path)
        metadata = json.loads(meta_path.read_text())
        chunks = metadata.get("chunks", [])

        if vectors.shape[0] != len(chunks):
            logger.warning(
                "Skipping '%s': vector count (%d) doesn't match chunk metadata count (%d)",
                npy_path.parent, vectors.shape[0], len(chunks),
            )
            continue

        for i, chunk_meta in enumerate(chunks):
            all_refs.append(IndexedChunkRef(
                faiss_id=next_id,
                chunk_id=chunk_meta["chunk_id"],
                document_id=chunk_meta["document_id"],
                source_file=chunk_meta["source_file"],
                page_numbers=chunk_meta.get("page_numbers", []),
                section_title=chunk_meta.get("section_title"),
                chunk_text=chunk_meta["chunk_text"],
            ))
            next_id += 1
        all_vectors.append(vectors)

    if not all_vectors:
        raise EmptyIndexError(f"No usable embeddings found under '{output_dir}' - nothing to index.")

    matrix = np.concatenate(all_vectors, axis=0).astype(np.float32)
    dim = matrix.shape[1]

    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base_index)
    index.add_with_ids(matrix, np.arange(len(all_refs), dtype=np.int64))

    logger.info("Built FAISS index: %d vectors, dim=%d, from %d document(s)", len(all_refs), dim, len(pairs))
    return index, all_refs


def save_index(
    index: faiss.Index, refs: list[IndexedChunkRef], index_dir: str | Path,
    config: IndexConfig | None = None,
) -> tuple[Path, Path]:
    config = config or IndexConfig()
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / config.index_filename
    id_map_path = index_dir / config.id_map_filename

    faiss.write_index(index, str(index_path))
    id_map = {str(r.faiss_id): {
        "chunk_id": r.chunk_id, "document_id": r.document_id, "source_file": r.source_file,
        "page_numbers": r.page_numbers, "section_title": r.section_title, "chunk_text": r.chunk_text,
    } for r in refs}
    id_map_path.write_text(json.dumps(id_map, indent=2, default=str))

    logger.info("Saved index (%d vectors) to %s", len(refs), index_path)
    return index_path, id_map_path


def load_index(
    index_dir: str | Path, config: IndexConfig | None = None,
) -> tuple[faiss.Index, dict[int, IndexedChunkRef]]:
    config = config or IndexConfig()
    index_dir = Path(index_dir)
    index_path = index_dir / config.index_filename
    id_map_path = index_dir / config.id_map_filename

    if not index_path.exists() or not id_map_path.exists():
        raise IndexNotFoundError(f"No index found at '{index_dir}' (expected {config.index_filename} + {config.id_map_filename})")

    index = faiss.read_index(str(index_path))
    raw_id_map = json.loads(id_map_path.read_text())
    id_map = {
        int(faiss_id): IndexedChunkRef(faiss_id=int(faiss_id), **fields)
        for faiss_id, fields in raw_id_map.items()
    }
    return index, id_map


def search(
    index: faiss.Index, id_map: dict[int, IndexedChunkRef], query_vector: np.ndarray, top_k: int = 5,
) -> list[tuple[IndexedChunkRef, float]]:
    """
    Raw vector search - takes an already-embedded query vector (embedding
    query text is the Retriever's job, Stage 4, not this module's).
    Returns (chunk_ref, similarity_score) pairs, best match first.
    """
    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    scores, ids = index.search(query_vector, top_k)

    results: list[tuple[IndexedChunkRef, float]] = []
    for score, faiss_id in zip(scores[0], ids[0]):
        if faiss_id == -1:  # FAISS pads with -1 when fewer than top_k results exist
            continue
        ref = id_map.get(int(faiss_id))
        if ref is None:
            logger.warning("FAISS returned id %d not present in id_map - skipping", faiss_id)
            continue
        results.append((ref, float(score)))
    return results
