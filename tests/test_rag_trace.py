"""Deterministic tests for the shared generic RAG trace."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
        self.assertEqual(trace.retrieved_items[0].metadata["layout_type"], "table")
        self.assertEqual(trace.citations[0]["marker"], "S1")
        self.assertIsNotNone(trace.retrieval_latency_ms)
        self.assertIsNotNone(trace.generation_latency_ms)
        self.assertIsNotNone(trace.rag_latency_ms)


if __name__ == "__main__":
    unittest.main()
