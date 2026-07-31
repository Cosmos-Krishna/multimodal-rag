"""Deterministic tests for the display-only Streamlit evaluation workspace."""

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

    def test_evaluation_sidebar_hides_chat_controls(self) -> None:
        sidebar = Mock()
        sidebar.__enter__ = Mock(return_value=sidebar)
        sidebar.__exit__ = Mock(return_value=None)
        with (
            patch.object(ui.st, "sidebar", sidebar),
            patch.object(ui, "load_ground_truth", return_value=[ITEM]),
            patch.object(ui.st, "button", return_value=False) as button,
            patch.object(ui.st, "metric"),
            patch.object(ui.st, "caption"),
            patch.object(ui.st, "warning"),
            patch.object(ui.st, "markdown"),
        ):
            ui._render_sidebar("Evaluation")

        labels = [str(call.args[0]) for call in button.call_args_list if call.args]
        self.assertFalse(any("New chat" in label for label in labels))
        self.assertFalse(any("Suggested" in label for label in labels))
        self.assertFalse(any("Developer" in label for label in labels))

    def test_chat_sidebar_keeps_chat_controls(self) -> None:
        sidebar = Mock()
        sidebar.__enter__ = Mock(return_value=sidebar)
        sidebar.__exit__ = Mock(return_value=None)
        with (
            patch.object(ui.st, "sidebar", sidebar),
            patch.object(ui, "_load_index", side_effect=ui.IndexNotFoundError("missing")),
            patch.object(ui.st, "button", return_value=False) as button,
            patch.object(ui.st, "toggle", return_value=False) as toggle,
            patch.object(ui.st, "markdown"),
            patch.object(ui.st, "caption"),
            patch.object(ui.st, "warning"),
            patch.object(ui.st, "divider"),
            patch.object(ui.st, "expander"),
        ):
            ui._render_sidebar("Chat")

        labels = [str(call.args[0]) for call in button.call_args_list if call.args]
        self.assertTrue(any("New chat" in label for label in labels))
        self.assertTrue(any("risk factors" in label for label in labels))
        toggle.assert_called_once()

    def test_timing_and_metric_formatting(self) -> None:
        self.assertEqual(ui._format_duration_ms(12.345), "12.3 ms")
        self.assertEqual(ui._format_duration_ms(1000), "1.00 s")
        self.assertEqual(ui._format_duration_ms(None), "Unavailable")
        self.assertEqual(ui._metric_text(0.123456), "0.123")
        self.assertEqual(ui._metric_text(None), "Unavailable")

    def test_citation_diagnostics_are_developer_only(self) -> None:
        source = inspect.getsource(ui.main)
        self.assertIn("if st.session_state.developer_mode and msg.get(\"trace\") is not None", source)


if __name__ == "__main__":
    unittest.main()
