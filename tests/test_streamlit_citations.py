"""Deterministic tests for Streamlit Chat, sources, and trace presentation."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimodal_rag.rag.generation.citation import Citation
from multimodal_rag.rag.trace import RAGTrace, RetrievedItemTrace
from multimodal_rag.ui import streamlit_app as ui


def _context_mock():
    context = Mock()
    context.__enter__ = Mock(return_value=context)
    context.__exit__ = Mock(return_value=None)
    return context


class StreamlitCitationTests(unittest.TestCase):
    def test_chat_messages_use_streamlit_default_avatar(self) -> None:
        chat_source = inspect.getsource(ui._render_chat_workspace)
        self.assertNotIn("avatar=", chat_source)
        self.assertNotIn("_render_source_cards", chat_source)
        self.assertIn('"citations"', chat_source)
        self.assertIn('"trace"', chat_source)

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
        container = _context_mock()
        popover = _context_mock()

        with (
            patch.object(ui.st, "container", return_value=container),
            patch.object(ui.st, "popover", return_value=popover) as popover_factory,
            patch.object(ui.st, "markdown") as markdown,
            patch.object(ui.st, "caption") as caption,
        ):
            ui._render_source_cards([citation])

        popover_factory.assert_called_once_with("[S1] guide · p.4, 5")
        rendered_markdown = " ".join(str(item.args[0]) for item in markdown.call_args_list)
        self.assertIn("Sources", rendered_markdown)
        rendered = " ".join(str(item.args[0]) for item in caption.call_args_list)
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

    def test_feedback_is_stored_only_on_the_session_message(self) -> None:
        message = {"role": "assistant", "content": "Answer"}
        with (
            patch.object(ui.st, "feedback", return_value=1) as feedback,
            patch.object(ui.st, "caption"),
        ):
            ui._render_feedback(message, 3)

        feedback.assert_called_once_with(
            "thumbs", key="assistant_feedback_3", default=None
        )
        self.assertEqual(message["feedback"], 1)

    def test_score_formatting_does_not_change_the_trace_value(self) -> None:
        original = 0.8308675007386641
        self.assertEqual(ui._format_score(original), "0.831")
        self.assertEqual(original, 0.8308675007386641)

    def test_complete_chunk_text_is_behind_a_collapsed_expander(self) -> None:
        item = RetrievedItemTrace(
            rank=1,
            raw_faiss_score=0.735123456789,
            combined_rerank_score=0.8308675007386641,
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="guide.pdf",
            page_numbers=[21],
            section_title="Enterprise AI",
            chunk_text="Complete evidence text",
        )
        trace = RAGTrace(original_question="Question", retrieved_items=[item])
        container = _context_mock()
        expander = _context_mock()
        columns = [Mock(), Mock()]

        with (
            patch.object(ui.st, "container", return_value=container),
            patch.object(ui.st, "columns", return_value=columns),
            patch.object(ui.st, "expander", return_value=expander) as expanders,
            patch.object(ui.st, "caption"),
            patch.object(ui.st, "text") as text,
            patch.object(ui, "_render_metadata_fields"),
            patch.object(ui.st, "markdown"),
        ):
            ui._render_trace_evidence(trace)

        self.assertIn(call("View evidence", expanded=False), expanders.call_args_list)
        text.assert_called_once_with("Complete evidence text")


if __name__ == "__main__":
    unittest.main()
