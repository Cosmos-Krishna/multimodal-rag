"""Deterministic tests for the shared generic RAG trace."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimodal_rag.rag.generation.answer_generator import GenerationResult
from multimodal_rag.rag.indexing.faiss_index import IndexedChunkRef
from multimodal_rag.rag.retrieval.retriever_2 import (
    RetrievedChunk,
    RetrieverConfig,
    retrieve as retrieve_chunks,
)
from multimodal_rag.rag.trace import run_rag_trace


class SharedRAGTraceTests(unittest.TestCase):
    def test_trace_executes_each_backend_step_once_and_keeps_diagnostics(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_file="guide.pdf",
            page_numbers=[4],
            section_title="Overview",
            chunk_text="Complete chunk text",
            score=0.91,
            combined_rerank_score=0.97,
        )
        citation = SimpleNamespace(
            marker="S1",
            source_file="guide.pdf",
            page_numbers=[4],
            chunk_id="chunk-1",
            section_title="Overview",
        )
        retrieve = Mock(return_value=[chunk])
        build_prompt = Mock(return_value=SimpleNamespace(prompt_text="prompt", source_map={"S1": chunk}))
        generate = Mock(return_value="Answer [S1]")
        resolve = Mock(
            return_value=SimpleNamespace(
                answer_text="Answer [S1]",
                citations=[citation],
                uncited_sources=[],
            )
        )

        trace = run_rag_trace(
            "Question",
            index=object(),
            id_map={},
            top_k=5,
            metadata_by_id={"chunk-1": {"layout_type": "table"}},
            retrieve_fn=retrieve,
            build_prompt_fn=build_prompt,
            generate_answer_fn=generate,
            resolve_citations_fn=resolve,
        )

        retrieve.assert_called_once()
        build_prompt.assert_called_once()
        generate.assert_called_once()
        resolve.assert_called_once()
        self.assertEqual(trace.configured_top_k, 5)
        self.assertEqual(trace.actual_retrieved_count, 1)
        self.assertEqual(trace.retrieved_items[0].raw_faiss_score, 0.91)
        self.assertEqual(trace.retrieved_items[0].combined_rerank_score, 0.97)
        self.assertEqual(trace.retrieved_items[0].metadata["layout_type"], "table")
        self.assertEqual(trace.citations[0]["marker"], "S1")
        self.assertIsNotNone(trace.retrieval_latency_ms)
        self.assertIsNotNone(trace.generation_latency_ms)
        self.assertIsNotNone(trace.rag_latency_ms)

    def test_structured_generation_usage_is_copied_without_an_extra_call(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_file="guide.pdf",
            page_numbers=[4],
            section_title="Overview",
            chunk_text="Complete chunk text",
            score=0.91,
            combined_rerank_score=0.97,
        )
        generate = Mock(
            return_value=GenerationResult(
                text="Answer",
                prompt_tokens=12,
                completion_tokens=7,
                total_tokens=19,
            )
        )
        trace = run_rag_trace(
            "Question",
            index=object(),
            id_map={},
            top_k=5,
            metadata_by_id={},
            retrieve_fn=Mock(return_value=[chunk]),
            build_prompt_fn=Mock(
                return_value=SimpleNamespace(prompt_text="prompt", source_map={})
            ),
            generate_answer_fn=generate,
            resolve_citations_fn=Mock(
                return_value=SimpleNamespace(
                    answer_text="Answer", citations=[], uncited_sources=[]
                )
            ),
        )

        generate.assert_called_once()
        self.assertEqual(trace.generation_prompt_tokens, 12)
        self.assertEqual(trace.generation_completion_tokens, 7)
        self.assertEqual(trace.generation_total_tokens, 19)


class RetrieverScoreTelemetryTests(unittest.TestCase):
    @staticmethod
    def _ref(faiss_id: int, chunk_id: str, text: str) -> IndexedChunkRef:
        return IndexedChunkRef(
            faiss_id=faiss_id,
            chunk_id=chunk_id,
            document_id="doc",
            source_file="guide.pdf",
            page_numbers=[faiss_id + 1],
            section_title=None,
            chunk_text=text,
        )

    def test_ranking_matches_existing_formula_and_raw_scores_are_unchanged(self) -> None:
        less_overlap = self._ref(0, "less-overlap", "alpha")
        full_overlap = self._ref(1, "full-overlap", "alpha beta")
        raw_results = [(less_overlap, 0.80), (full_overlap, 0.75)]

        with patch(
            "multimodal_rag.rag.retrieval.retriever_2.search",
            return_value=raw_results,
        ):
            results = retrieve_chunks(
                "alpha beta",
                object(),
                {},
                retriever_config=RetrieverConfig(top_k=2, lexical_rerank_weight=0.15),
                embed_fn=lambda texts, config: [[0.0]],
            )

        expected_legacy_order = sorted(
            raw_results,
            key=lambda pair: pair[1]
            + 0.15 * (len(set("alpha beta".split()) & set(pair[0].chunk_text.split())) / 2),
            reverse=True,
        )
        self.assertEqual(
            [result.chunk_id for result in results],
            [ref.chunk_id for ref, _score in expected_legacy_order],
        )
        self.assertEqual(
            {result.chunk_id: result.score for result in results},
            {ref.chunk_id: score for ref, score in raw_results},
        )
        self.assertAlmostEqual(results[0].combined_rerank_score, 0.90)
        self.assertAlmostEqual(results[1].combined_rerank_score, 0.875)

    def test_disabled_reranking_preserves_order_and_uses_raw_combined_score(self) -> None:
        first = self._ref(0, "first", "alpha")
        second = self._ref(1, "second", "alpha beta")
        with patch(
            "multimodal_rag.rag.retrieval.retriever_2.search",
            return_value=[(first, 0.80), (second, 0.75)],
        ):
            results = retrieve_chunks(
                "alpha beta",
                object(),
                {},
                retriever_config=RetrieverConfig(top_k=2, lexical_rerank_weight=0.0),
                embed_fn=lambda texts, config: [[0.0]],
            )

        self.assertEqual([result.chunk_id for result in results], ["first", "second"])
        self.assertEqual(
            [result.combined_rerank_score for result in results], [0.80, 0.75]
        )

    def test_legacy_retrieved_chunk_construction_still_works(self) -> None:
        chunk = RetrievedChunk("id", "doc", "file.pdf", [1], None, "text", 0.5)
        self.assertEqual(chunk.score, 0.5)
        self.assertIsNone(chunk.combined_rerank_score)


if __name__ == "__main__":
    unittest.main()
