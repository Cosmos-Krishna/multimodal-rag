"""Deterministic tests for Streamlit navigation and Developer Lab evaluation."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimodal_rag.ui import streamlit_app as ui


ITEM = {
    "id": 1,
    "question": "What is Enterprise AI?",
    "ground_truth": "Reference answer",
    "metadata": {"question_type": "definition", "difficulty": "easy"},
}
ITEM_2 = {**ITEM, "id": 2, "question": "What is retrieval?"}


def _status_context():
    status = Mock()
    status.__enter__ = Mock(return_value=status)
    status.__exit__ = Mock(return_value=None)
    return status


def _button_columns(run_value: bool, clear_value: bool):
    run_column = Mock()
    clear_column = Mock()
    run_column.button.return_value = run_value
    clear_column.button.return_value = clear_value
    return [run_column, clear_column]


class StreamlitEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        ui.st.session_state.evaluation_result = None
        ui.st.session_state.evaluation_error = None
        ui.st.session_state.evaluation_selected_id = None
        ui.st.session_state.evaluation_previous_id = None
        ui.st.session_state.messages = []
        ui.st.session_state.history_turns = []
        ui.st.session_state.playground_trace = None
        ui.st.session_state.playground_error = None
        ui.st.session_state.playground_pending_query = None

    def test_user_workspace_suggestions_are_exactly_the_approved_three(self) -> None:
        self.assertEqual(
            ui.SAMPLE_QUESTIONS,
            [
                "What are the key elements of robust AI governance?",
                "What distinguishes a strong data foundation from a weak data foundation?",
                "What are the three main advantages of a strong MVP approach?",
            ],
        )

    def test_developer_workspace_suggestions_are_exactly_the_approved_three(self) -> None:
        self.assertEqual(
            ui.DEVELOPER_SAMPLE_QUESTIONS,
            [
                "What are the five dimensions of AI readiness?",
                "Explain the data foundation evaluation checklist.",
                "What are the short-term and long-term implications of data?",
            ],
        )

    def test_selection_does_not_execute_evaluation(self) -> None:
        with (
            patch.object(ui, "load_ground_truth", return_value=[ITEM]),
            patch.object(ui, "evaluate_ground_truth_item") as evaluate,
            patch.object(ui.st, "selectbox", return_value="1"),
            patch.object(ui.st, "columns", return_value=_button_columns(False, False)),
        ):
            ui._render_evaluation_workspace()

        evaluate.assert_not_called()
        self.assertIsNone(ui.st.session_state.evaluation_result)

    def test_top_level_ground_truth_metadata_is_displayed(self) -> None:
        item = {
            "id": 1,
            "question": "What is Enterprise AI?",
            "ground_truth": "Reference answer",
            "question_type": "definition",
            "difficulty": "easy",
        }
        meta_columns = [Mock(), Mock()]
        with (
            patch.object(ui, "load_ground_truth", return_value=[item]),
            patch.object(ui.st, "selectbox", return_value="1"),
            patch.object(
                ui.st,
                "columns",
                side_effect=[meta_columns, _button_columns(False, False)],
            ),
        ):
            ui._render_evaluation_workspace()

        meta_columns[0].caption.assert_called_once_with("Question type: definition")
        meta_columns[1].caption.assert_called_once_with("Difficulty: easy")

    def test_explicit_run_executes_once_and_stores_result(self) -> None:
        result = SimpleNamespace(status="success")
        with (
            patch.object(ui, "load_ground_truth", return_value=[ITEM]),
            patch.object(ui, "evaluate_ground_truth_item", return_value=result) as evaluate,
            patch.object(ui.st, "selectbox", return_value="1"),
            patch.object(ui.st, "columns", return_value=_button_columns(True, False)),
            patch.object(ui.st, "status", return_value=_status_context()),
            patch.object(ui, "_render_evaluation_result"),
        ):
            ui._render_evaluation_workspace()

        evaluate.assert_called_once_with(ITEM)
        self.assertIs(ui.st.session_state.evaluation_result, result)

    def test_clear_removes_only_evaluation_result(self) -> None:
        result = SimpleNamespace(status="success")
        ui.st.session_state.evaluation_result = result
        ui.st.session_state.messages = [{"role": "user", "content": "Keep me"}]
        with (
            patch.object(ui, "load_ground_truth", return_value=[ITEM]),
            patch.object(ui, "evaluate_ground_truth_item") as evaluate,
            patch.object(ui.st, "selectbox", return_value="1"),
            patch.object(ui.st, "columns", return_value=_button_columns(False, True)),
            patch.object(ui.st, "rerun"),
        ):
            ui._render_evaluation_workspace()

        evaluate.assert_not_called()
        self.assertIsNone(ui.st.session_state.evaluation_result)
        self.assertEqual(ui.st.session_state.messages[0]["content"], "Keep me")

    def test_changing_id_clears_evaluation_only(self) -> None:
        result = SimpleNamespace(status="success")
        ui.st.session_state.evaluation_result = result
        ui.st.session_state.evaluation_error = "old error"
        ui.st.session_state.evaluation_previous_id = "1"
        ui.st.session_state.messages = [{"role": "user", "content": "Keep me"}]
        with (
            patch.object(ui, "load_ground_truth", return_value=[ITEM, ITEM_2]),
            patch.object(ui, "select_ground_truth_item", return_value=ITEM_2),
            patch.object(ui.st, "selectbox", return_value="2"),
            patch.object(ui.st, "columns", return_value=_button_columns(False, False)),
        ):
            ui._render_evaluation_workspace()

        self.assertIsNone(ui.st.session_state.evaluation_result)
        self.assertIsNone(ui.st.session_state.evaluation_error)
        self.assertEqual(ui.st.session_state.messages[0]["content"], "Keep me")

    def test_navigation_separates_user_workspace_and_developer_lab(self) -> None:
        def page_factory(callable_page, **kwargs):
            return SimpleNamespace(callable=callable_page, **kwargs)

        with patch.object(ui.st, "Page", side_effect=page_factory):
            pages = ui._navigation_pages()

        self.assertEqual(list(pages), ["USER WORKSPACE", "DEVELOPER LAB"])
        self.assertEqual([page.title for page in pages["USER WORKSPACE"]], ["Chat"])
        self.assertEqual(
            [page.title for page in pages["DEVELOPER LAB"]],
            ["Playground / Trace", "Evaluation"],
        )
        self.assertIs(pages["USER WORKSPACE"][0].callable, ui._render_chat_workspace)
        self.assertIs(pages["DEVELOPER LAB"][0].callable, ui._render_playground_workspace)
        self.assertIs(pages["DEVELOPER LAB"][1].callable, ui._render_evaluation_workspace)

    def test_evaluation_sidebar_hides_chat_controls(self) -> None:
        sidebar = Mock()
        sidebar.__enter__ = Mock(return_value=sidebar)
        sidebar.__exit__ = Mock(return_value=None)
        with (
            patch.object(ui.st, "sidebar", sidebar),
            patch.object(ui.st, "button", return_value=False) as button,
            patch.object(ui.st, "caption"),
            patch.object(ui.st, "markdown"),
            patch.object(ui.st, "divider"),
            patch.object(ui, "_load_index", side_effect=ui.IndexNotFoundError("missing")),
        ):
            ui._render_sidebar("Evaluation")

        button.assert_not_called()

    def test_chat_sidebar_uses_runtime_index_counts(self) -> None:
        sidebar = Mock()
        sidebar.__enter__ = Mock(return_value=sidebar)
        sidebar.__exit__ = Mock(return_value=None)
        refs = {
            1: SimpleNamespace(document_id="doc-a", page_numbers=[1]),
            2: SimpleNamespace(document_id="doc-a", page_numbers=[2]),
            3: SimpleNamespace(document_id="doc-b", page_numbers=[1]),
        }
        index = SimpleNamespace(ntotal=3)
        with (
            patch.object(ui.st, "sidebar", sidebar),
            patch.object(ui, "_load_index", return_value=(index, refs)),
            patch.object(ui.st, "button", return_value=False) as button,
            patch.object(ui.st, "markdown"),
            patch.object(ui.st, "caption") as caption,
            patch.object(ui.st, "divider"),
        ):
            ui._render_sidebar("Chat")

        labels = [str(item.args[0]) for item in button.call_args_list if item.args]
        self.assertTrue(any("New chat" in label for label in labels))
        rendered = " ".join(str(item.args[0]) for item in caption.call_args_list)
        self.assertIn("**2** documents", rendered)
        self.assertIn("**3** chunks", rendered)
        self.assertIn("3 indexed pages", rendered)
        self.assertNotIn("111", rendered)

    def test_timing_and_metric_formatting(self) -> None:
        self.assertEqual(ui._format_duration_ms(12.345), "12.3 ms")
        self.assertEqual(ui._format_duration_ms(1000), "1.00 s")
        self.assertEqual(ui._format_duration_ms(None), "Unavailable")
        self.assertEqual(ui._metric_text(0.123456), "0.123")
        self.assertEqual(ui._metric_text(None), "Unavailable")
        self.assertAlmostEqual(ui._relative_duration(250, 1000), 0.25)
        self.assertIsNone(ui._relative_duration(None, 1000))
        self.assertEqual(ui._metric_progress(1.2), 1.0)
        self.assertEqual(ui._metric_progress(-0.2), 0.0)

    def test_metric_bar_uses_reported_value_without_labels(self) -> None:
        columns = [Mock(), Mock(), Mock()]
        with patch.object(ui.st, "columns", return_value=columns):
            ui._render_metric_bar("Faithfulness", 0.9294)

        columns[1].progress.assert_called_once_with(0.9294)
        columns[2].markdown.assert_called_once_with("`0.929`")

    def test_unavailable_metric_does_not_render_a_progress_value(self) -> None:
        columns = [Mock(), Mock(), Mock()]
        with patch.object(ui.st, "columns", return_value=columns):
            ui._render_metric_bar("Faithfulness", None)

        columns[1].progress.assert_not_called()
        columns[1].caption.assert_called_once_with("Unavailable")
        columns[2].markdown.assert_called_once_with("`Unavailable`")

    def test_playground_with_no_trace_does_not_call_any_backend(self) -> None:
        suggestion_columns = [Mock(), Mock(), Mock()]
        input_columns = [Mock(), Mock()]
        for column in suggestion_columns + input_columns:
            column.button.return_value = False
        input_columns[0].text_input.return_value = ""
        with (
            patch.object(ui, "_answer_query") as answer,
            patch.object(ui, "evaluate_ground_truth_item") as evaluate,
            patch.object(ui.st, "markdown"),
            patch.object(ui.st, "info"),
            patch.object(ui.st, "columns", side_effect=[suggestion_columns, input_columns]),
        ):
            ui._render_playground_workspace()

        answer.assert_not_called()
        evaluate.assert_not_called()

    def test_developer_question_runs_one_trace_and_stores_it(self) -> None:
        trace = SimpleNamespace(
            original_question="Test question",
            generated_answer="Test answer",
        )
        suggestion_columns = [Mock(), Mock(), Mock()]
        input_columns = [Mock(), Mock()]
        for column in suggestion_columns:
            column.button.return_value = False
        input_columns[0].text_input.return_value = "Test question"
        input_columns[1].button.return_value = True
        status = _status_context()
        with (
            patch.object(ui, "_answer_query", return_value=trace) as answer,
            patch.object(ui.st, "columns", side_effect=[suggestion_columns, input_columns]),
            patch.object(ui.st, "status", return_value=status),
            patch.object(ui.st, "selectbox", return_value="Latest developer run"),
            patch.object(ui, "_render_developer_trace") as render_trace,
            patch.object(ui.st, "markdown"),
        ):
            ui._render_playground_workspace()

        answer.assert_called_once_with("Test question")
        self.assertIs(ui.st.session_state.playground_trace, trace)
        render_trace.assert_called_once_with(trace)

    def test_previous_chat_trace_remains_selectable(self) -> None:
        trace = SimpleNamespace(
            original_question="Previous question",
            generated_answer="Previous answer",
        )
        ui.st.session_state.messages = [
            {"role": "assistant", "content": "Previous answer", "trace": trace}
        ]
        suggestion_columns = [Mock(), Mock(), Mock()]
        input_columns = [Mock(), Mock()]
        for column in suggestion_columns + input_columns:
            column.button.return_value = False
        input_columns[0].text_input.return_value = ""
        with (
            patch.object(ui.st, "columns", side_effect=[suggestion_columns, input_columns]),
            patch.object(ui.st, "selectbox", return_value="Chat · Previous question"),
            patch.object(ui, "_render_developer_trace") as render_trace,
            patch.object(ui.st, "markdown"),
        ):
            ui._render_playground_workspace()

        render_trace.assert_called_once_with(trace)

    def test_citation_diagnostics_are_developer_only(self) -> None:
        chat_source = inspect.getsource(ui._render_chat_workspace)
        playground_source = inspect.getsource(ui._render_playground_workspace)
        self.assertNotIn("_render_developer_trace", chat_source)
        self.assertIn("_render_developer_trace", playground_source)
        self.assertNotIn("evaluate_ground_truth_item", chat_source)
        self.assertNotIn("_render_citation_diagnostics", playground_source)

    def test_playground_uses_compact_summary_without_redundant_model_card(self) -> None:
        source = inspect.getsource(ui._render_developer_trace)
        self.assertIn("st.columns(4)", source)
        self.assertNotIn('metric("Model"', source)


if __name__ == "__main__":
    unittest.main()
