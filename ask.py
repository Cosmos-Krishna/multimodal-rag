#!/usr/bin/env python3
"""
ask.py - Ask a question against the ingested, indexed document corpus.

Usage:
    python ask.py "What are the main risk factors?"
    python ask.py "..." --top-k 8 --index-dir index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerationUnavailableError,
    GenerationConfig,
    generate_answer,
)
from rag.generation.citation import resolve_citations
from rag.generation.prompt_builder import build_prompt
from rag.indexing.faiss_index import IndexNotFoundError, load_index
from rag.retrieval.retriever_2 import RetrieverConfig, retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question against the ingested document corpus.")
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument("--index-dir", default="index", help="Directory containing the FAISS index (default: index/)")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve (default: 8)")
    args = parser.parse_args()

    try:
        index, id_map = load_index(args.index_dir)
    except IndexNotFoundError as e:
        print(f"Error: {e}\nRun build_index.py first.")
        return 1

    # chunks = retrieve(args.query, index, id_map, retriever_config=RetrieverConfig(top_k=args.top_k))
    chunks = retrieve(args.query, index, id_map, retriever_config=RetrieverConfig(top_k=8))
    if not chunks:
        print("No relevant content found for that query.")
        return 0

    print("\n================ RETRIEVED CHUNKS ================\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print("Score:", chunk.score)
        print("Page:", chunk.page_numbers)
        print(chunk.chunk_text[:800])
        print("-" * 80)

    built = build_prompt(args.query, chunks)

    try:
        raw_answer = generate_answer(built.prompt_text, GenerationConfig())
    except AnswerGenerationUnavailableError as e:
        print(f"Error: {e}\nSet GEMINI_API_KEY to enable answer generation.")
        return 1
    except AnswerGenerationError as e:
        print(f"Error: {e}")
        return 1

    result = resolve_citations(raw_answer, built.source_map)

    print(f"\n{result.answer_text}\n")
    if result.citations:
        print("Sources:")
        for c in result.citations:
            pages = ", ".join(str(p) for p in c.page_numbers) or "unknown"
            print(f"  [{c.marker}] {c.source_file}, page(s): {pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
