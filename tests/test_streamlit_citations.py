"""Deterministic tests for Streamlit citation wiring."""

from __future__ import annotations

import sys
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimodal_rag.rag.generation.citation import Citation
from multimodal_rag.rag.trace import RAGTrace
from multimodal_rag.ui import streamlit_app as ui


class StreamlitCitationTests(unittest.TestCase):
    def test_chat_messages_use_streamlit_default_avatar(self) -> None:
        self.assertNotIn("avatar=", inspect.getsource(ui.main))

    def test_answer_query_resolves_citations_without_duplicate_backend_calls(self) -> None:
        ui.st.session_state.history_turns = []

        trace = RAGTrace(
            original_question="Question",
            generated_answer="Grounded answer [S1]",
            actual_retrieved_count=1,
        )
        with (
            patch.object(ui, "_load_index", return_value=(object(), {1: object()})),
            patch.object(ui, "run_rag_trace", return_value=trace) as run_trace,
        ):
            result = ui._answer_query("Question")

        run_trace.assert_called_once()
        self.assertIs(result, trace)

    def test_source_cards_show_metadata_without_full_chunk_text(self) -> None:
        citation = Citation(
            marker="S1",
            source_file="guide.pdf",
            page_numbers=[4, 5],
            chunk_id="chunk-1",
            section_title="Overview",
        )
        expander = Mock()
        expander.__enter__ = Mock(return_value=expander)
        expander.__exit__ = Mock(return_value=None)

        with (
            patch.object(ui.st, "expander", return_value=expander) as expander_factory,
            patch.object(ui.st, "markdown") as markdown,
            patch.object(ui.st, "caption") as caption,
        ):
            ui._render_source_cards([citation])

        expander_factory.assert_called_once_with("[S1]  guide.pdf · page 4, 5")
        markdown.assert_called_once_with("**Sources**")
        rendered = " ".join(str(call.args[0]) for call in caption.call_args_list)
        self.assertIn("guide.pdf", rendered)
        self.assertIn("Overview", rendered)
        self.assertNotIn("Complete source text", rendered)

    def test_source_metadata_fallback_is_available_without_citation_markers(self) -> None:
        item = SimpleNamespace(
            document_name="guide.pdf",
            page_numbers=[6, 18],
            section_title="Overview",
        )
        records = ui._source_records([], [item])
        self.assertEqual(
            [(record["source_file"], record["page_numbers"]) for record in records],
            [("guide.pdf", [6]), ("guide.pdf", [18])],
        )


if __name__ == "__main__":
    unittest.main()
