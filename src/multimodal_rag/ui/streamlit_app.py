#!/usr/bin/env python3
"""Maintained Streamlit UI for the multimodal RAG system.

Run with:
    python -m streamlit run src/multimodal_rag/ui/streamlit_app.py

The UI is split into a clean User Workspace and a read-only Developer Lab.
Backend behavior remains shared with the existing RAG and question-evaluation
modules. Page changes and trace inspection never execute provider calls.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multimodal_rag.evaluation.question_runner import (
    QuestionEvaluationTrace,
    SelectionError,
    evaluate_ground_truth_item,
    load_ground_truth,
    select_ground_truth_item,
)
from multimodal_rag.paths import (
    INDEX_DIR as DEFAULT_INDEX_DIR,
    LEGACY_INDEX_DIR,
    prefer_new_path,
)
from multimodal_rag.rag.generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerationUnavailableError,
)
from multimodal_rag.rag.generation.prompt_builder import ConversationTurn
from multimodal_rag.rag.indexing.faiss_index import IndexNotFoundError, load_index
from multimodal_rag.rag.retrieval.retriever_2 import retrieve  # compatibility export
from multimodal_rag.rag.trace import RAGTrace, run_rag_trace


INDEX_DIR = str(prefer_new_path(DEFAULT_INDEX_DIR, LEGACY_INDEX_DIR))
TOP_K = 5
MAX_HISTORY_TURNS = 5

SAMPLE_QUESTIONS = [
    "What are the key elements of robust AI governance?",
    "What distinguishes a strong data foundation from a weak data foundation?",
    "What are the three main advantages of a strong MVP approach?",
]

DEVELOPER_SAMPLE_QUESTIONS = [
    "What are the five dimensions of AI readiness?",
    "Explain the data foundation evaluation checklist.",
    "What are the short-term and long-term implications of data?",
]


st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --rag-accent: #4f46e5;
        --rag-accent-soft: #eef2ff;
        --rag-border: #e5e7eb;
        --rag-muted: #64748b;
        --rag-ink: #182230;
        --rag-user: #f1f5f9;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2.1rem;
        padding-bottom: 7rem;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--rag-border);
        background: #fafbfc;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    .rag-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 0 0 1.1rem;
    }
    .rag-brand-mark {
        width: 2.15rem;
        height: 2.15rem;
        display: grid;
        place-items: center;
        border-radius: 0.62rem;
        background: var(--rag-accent);
        color: white;
        font-weight: 800;
        font-size: 0.98rem;
    }
    .rag-brand-title { font-size: 1rem; font-weight: 760; line-height: 1.1; }
    .rag-brand-subtitle { color: var(--rag-muted); font-size: 0.72rem; margin-top: 0.18rem; }
    .rag-hero { text-align: center; padding: 4.5rem 0 1.5rem; }
    .rag-hero h1 {
        color: var(--rag-ink);
        font-size: clamp(2rem, 4vw, 2.85rem);
        letter-spacing: -0.042em;
        margin: 0 0 0.65rem;
    }
    .rag-hero p {
        color: var(--rag-muted);
        font-size: 1.02rem;
        line-height: 1.65;
        max-width: 590px;
        margin: 0 auto;
    }
    .rag-kicker {
        color: var(--rag-accent);
        font-size: 0.73rem;
        font-weight: 750;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .rag-page-intro { margin-bottom: 1.6rem; }
    .rag-page-intro h1 {
        color: var(--rag-ink);
        font-size: 2rem;
        letter-spacing: -0.035em;
        margin-bottom: 0.35rem;
    }
    .rag-page-intro p { color: var(--rag-muted); margin: 0; }
    .rag-message-label {
        color: var(--rag-muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.055em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    [data-testid="stChatMessage"] {
        border: 0;
        background: transparent;
        padding: 0.75rem 0.2rem;
        margin: 0.5rem 0;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--rag-user);
        border-radius: 0.85rem;
        padding: 0.85rem 1rem;
        margin-left: min(12%, 5rem);
    }
    [data-testid="stChatMessage"] p { line-height: 1.7; }
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 { letter-spacing: -0.02em; }
    .rag-section-label {
        color: var(--rag-muted);
        font-size: 0.73rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.3rem 0 0.45rem;
    }
    .rag-evidence-title { font-weight: 720; color: var(--rag-ink); }
    .rag-evidence-subtitle { color: var(--rag-muted); font-size: 0.84rem; margin-top: 0.15rem; }
    .rag-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    [data-testid="stPopover"] button {
        border-color: var(--rag-border);
        background: #ffffff;
        color: #334155;
        box-shadow: none;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--rag-border);
        border-radius: 0.75rem;
        padding: 0.75rem 0.85rem;
    }
    .stProgress > div > div > div > div { background-color: var(--rag-accent); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _load_index():
    return load_index(INDEX_DIR)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history_turns" not in st.session_state:
        st.session_state.history_turns = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "evaluation_selected_id" not in st.session_state:
        st.session_state.evaluation_selected_id = None
    if "evaluation_error" not in st.session_state:
        st.session_state.evaluation_error = None
    if "evaluation_previous_id" not in st.session_state:
        st.session_state.evaluation_previous_id = None
    if "playground_trace" not in st.session_state:
        st.session_state.playground_trace = None
    if "playground_error" not in st.session_state:
        st.session_state.playground_error = None
    if "playground_pending_query" not in st.session_state:
        st.session_state.playground_pending_query = None

    st.session_state.history_turns = st.session_state.history_turns[-MAX_HISTORY_TURNS:]
    if st.session_state.pending_query is not None and not isinstance(
        st.session_state.pending_query, str
    ):
        st.session_state.pending_query = None


def _new_chat() -> None:
    st.session_state.messages = []
    st.session_state.history_turns = []
    st.session_state.pending_query = None
    for key in list(st.session_state):
        if str(key).startswith("assistant_feedback_"):
            del st.session_state[key]


def _answer_query(query: str) -> RAGTrace:
    """Run the existing one-retrieval, one-generation Chat path."""
    index, id_map = _load_index()
    return run_rag_trace(
        query,
        index=index,
        id_map=id_map,
        top_k=TOP_K,
        conversation_history=st.session_state.history_turns[-MAX_HISTORY_TURNS:],
        max_history_turns=MAX_HISTORY_TURNS,
    )


def _index_stats(index, id_map) -> dict[str, int]:
    """Derive display-only counts from the loaded index and ID map."""
    refs = list(id_map.values())
    return {
        "chunks": int(getattr(index, "ntotal", len(refs))),
        "documents": len({ref.document_id for ref in refs}),
        "pages": len(
            {
                (ref.document_id, page)
                for ref in refs
                for page in (ref.page_numbers or [])
            }
        ),
    }


def _citation_field(citation, field: str):
    return citation.get(field) if isinstance(citation, dict) else getattr(citation, field)


def _source_records(citations: list[object], retrieved_items: list[object] | None = None):
    if citations:
        return [
            {
                "marker": _citation_field(citation, "marker"),
                "source_file": _citation_field(citation, "source_file"),
                "page_numbers": _citation_field(citation, "page_numbers"),
                "section_title": _citation_field(citation, "section_title"),
            }
            for citation in citations
        ]

    records = []
    seen = set()
    for item in retrieved_items or []:
        for page in item.page_numbers or [None]:
            key = (item.document_name, page)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "marker": None,
                    "source_file": item.document_name,
                    "page_numbers": [] if page is None else [page],
                    "section_title": item.section_title,
                }
            )
    return records


def _render_source_cards(
    citations: list[object], retrieved_items: list[object] | None = None
) -> None:
    """Render compact source controls using real citation or fallback metadata."""
    records = _source_records(citations, retrieved_items)
    if not records:
        return

    st.markdown('<div class="rag-section-label">Sources</div>', unsafe_allow_html=True)
    with st.container(horizontal=True, gap="small"):
        for source in records:
            marker = source["marker"]
            source_file = str(source["source_file"] or "Document")
            page_numbers = source["page_numbers"] or []
            section_title = source["section_title"]
            pages = ", ".join(str(page) for page in page_numbers) or "Unavailable"
            document_label = Path(source_file).stem or source_file
            marker_label = f"[{marker}] " if marker else ""
            page_label = f"p.{pages}" if pages != "Unavailable" else "page unavailable"
            with st.popover(f"{marker_label}{document_label} · {page_label}"):
                st.caption(f"**Document:** {source_file}")
                st.caption(f"**Page:** {pages}")
                st.caption(f"**Section:** {section_title or 'Unavailable'}")


def _render_feedback(message: dict[str, object], message_index: int) -> None:
    """Collect explicitly session-only feedback without backend persistence."""
    feedback = st.feedback(
        "thumbs",
        key=f"assistant_feedback_{message_index}",
        default=message.get("feedback"),
    )
    if feedback is not None:
        message["feedback"] = int(feedback)
    st.caption("Feedback is kept in this session only.")


def _display_telemetry(value, suffix: str = "") -> str:
    return "Unavailable" if value is None else f"{value}{suffix}"


def _format_duration_ms(value) -> str:
    """Format timings without changing the underlying trace values."""
    if value is None:
        return "Unavailable"
    milliseconds = float(value)
    if milliseconds < 1000:
        return f"{milliseconds:.1f} ms"
    return f"{milliseconds / 1000:.2f} s"


def _format_score(value, precision: int = 3) -> str:
    if value is None:
        return "Unavailable"
    number = float(value)
    if not math.isfinite(number):
        return "Unavailable"
    return f"{number:.{precision}f}"


def _format_tokens(value) -> str:
    return "Unavailable" if value is None else f"{int(value):,}"


def _relative_duration(value, total) -> float | None:
    """Return a display-only stage/total ratio for a neutral latency bar."""
    if value is None or total is None:
        return None
    total_value = float(total)
    value_number = float(value)
    if total_value <= 0 or not math.isfinite(total_value) or not math.isfinite(value_number):
        return None
    return min(max(value_number / total_value, 0.0), 1.0)


def _metric_progress(value) -> float | None:
    """Clamp only the progress-bar display; never mutate the metric value."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return min(max(number, 0.0), 1.0)


def _metric_text(value) -> str:
    return _format_score(value, precision=3)


def _render_metadata_fields(metadata: dict[str, object] | None) -> None:
    """Render structured metadata fields without a raw JSON dump."""
    if not metadata:
        st.caption("Metadata: unavailable")
        return

    labels = {
        "source_file": "Document",
        "document_name": "Document",
        "page_numbers": "Page",
        "section_title": "Section",
        "chunk_id": "Chunk ID",
        "extraction_method": "Extraction method",
        "validation_status": "Validation status",
        "pipeline_version": "Pipeline version",
        "layout_type": "Layout type",
        "table_reference": "Table reference",
        "image_reference": "Image reference",
        "ocr_confidence": "OCR confidence",
        "source_region_ids": "Source region IDs",
    }
    preferred = list(labels)
    ordered_keys = preferred + sorted(key for key in metadata if key not in labels)
    for key in ordered_keys:
        if key not in metadata:
            continue
        value = metadata[key]
        label = labels.get(key, key.replace("_", " ").title())
        if isinstance(value, dict):
            scalar_items = [
                f"{nested_key}: {nested_value}"
                for nested_key, nested_value in value.items()
                if not isinstance(nested_value, (dict, list, tuple, set))
            ]
            value = ", ".join(scalar_items) if scalar_items else "Available in source artifact"
        elif isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value) if value else "Unavailable"
        st.caption(f"**{label}:** {value if value not in (None, '') else 'Unavailable'}")


def _render_latency_visualization(trace: RAGTrace) -> None:
    st.markdown("### RAG performance")
    st.caption("Bars show measured stage time relative to complete RAG time; they are not quality ratings.")
    total = trace.rag_latency_ms
    for label, value in (
        ("Retrieval", trace.retrieval_latency_ms),
        ("Generation", trace.generation_latency_ms),
    ):
        label_column, bar_column, value_column = st.columns([1.2, 4, 1.1], vertical_alignment="center")
        label_column.markdown(f"**{label}**")
        ratio = _relative_duration(value, total)
        if ratio is None:
            bar_column.caption("Unavailable")
        else:
            bar_column.progress(ratio)
        value_column.markdown(f"`{_format_duration_ms(value)}`")
    total_columns = st.columns([1.2, 4, 1.1], vertical_alignment="center")
    total_columns[0].markdown("**Total**")
    if total is None:
        total_columns[1].caption("Unavailable")
    else:
        total_columns[1].progress(1.0)
    total_columns[2].markdown(f"`{_format_duration_ms(total)}`")


def _render_token_usage(trace: RAGTrace) -> None:
    st.markdown("### Token usage")
    columns = st.columns(3)
    columns[0].metric("Prompt", _format_tokens(trace.generation_prompt_tokens))
    columns[1].metric("Completion", _format_tokens(trace.generation_completion_tokens))
    columns[2].metric("Total", _format_tokens(trace.generation_total_tokens))
    st.caption(
        "Estimated generation cost: "
        f"{_display_telemetry(trace.estimated_generation_cost)}"
    )


def _render_trace_evidence(trace: RAGTrace, *, show_heading: bool = True) -> None:
    if show_heading:
        st.markdown("### Retrieved evidence")
    if not trace.retrieved_items:
        st.caption("No chunks retrieved.")
        return

    for item in trace.retrieved_items:
        pages = ", ".join(str(page) for page in item.page_numbers) or "Unavailable"
        section = item.section_title or "Section unavailable"
        with st.container(border=True):
            title_column, score_column = st.columns([4, 1.4], vertical_alignment="top")
            title_column.markdown(
                f'<div class="rag-evidence-title">#{item.rank} · Page {pages}</div>'
                f'<div class="rag-evidence-subtitle">{section}</div>',
                unsafe_allow_html=True,
            )
            score_column.markdown(
                f"**Rerank** `{_format_score(item.combined_rerank_score)}`  \n"
                f"**Raw** `{_format_score(item.raw_faiss_score)}`"
            )
            with st.expander("View evidence", expanded=False):
                st.caption(f"Document: {item.document_name}")
                st.caption(f"Chunk ID: `{item.chunk_id}`")
                st.caption(f"Page: {pages}")
                st.caption(f"Section: {item.section_title or 'Unavailable'}")
                st.text(item.chunk_text)
                with st.expander("Advanced details", expanded=False):
                    st.caption(
                        "Raw FAISS similarity: "
                        f"{_display_telemetry(item.raw_faiss_score)}"
                    )
                    st.caption(
                        "Combined rerank score: "
                        f"{_display_telemetry(item.combined_rerank_score)}"
                    )
                    st.caption(f"BM25 score: {_display_telemetry(item.bm25_score)}")
                    st.caption(f"RRF score: {_display_telemetry(item.rrf_score)}")
                    st.caption(
                        "Cross-encoder score: "
                        f"{_display_telemetry(item.cross_encoder_score)}"
                    )
                    _render_metadata_fields(item.metadata)
                    if item.metadata is None and item.metadata_note:
                        st.caption(item.metadata_note)


def _render_citation_diagnostics(trace: RAGTrace) -> None:
    with st.expander("Citation diagnostics", expanded=False):
        if trace.citations:
            for citation in trace.citations:
                st.caption(
                    f"[{citation['marker']}] {citation['source_file']} · "
                    f"page {citation['page_numbers'] or 'unavailable'} · "
                    f"section {citation['section_title'] or 'unavailable'}"
                )
        else:
            st.caption("Source metadata is available, but no citation markers were resolved.")


def _render_developer_trace(trace: RAGTrace) -> None:
    """Render an existing trace without executing any RAG stage."""
    summary = st.columns(4)
    summary[0].metric("Top-k", trace.configured_top_k)
    summary[1].metric("RAG latency", _format_duration_ms(trace.rag_latency_ms))
    summary[2].metric("Total tokens", _format_tokens(trace.generation_total_tokens))
    summary[3].metric("Retrieved", trace.actual_retrieved_count)

    _render_latency_visualization(trace)
    _render_token_usage(trace)

    with st.expander("Pipeline and model details", expanded=False):
        st.caption(f"Retriever: `{trace.retriever}`")
        st.caption(f"Embedding model: `{_display_telemetry(trace.embedding_model)}`")
        st.caption(f"Generation model: `{_display_telemetry(trace.generation_model)}`")
        st.caption(f"Configured top-k: {trace.configured_top_k}")
        st.caption(f"Actual retrieved chunks: {trace.actual_retrieved_count}")

    _render_trace_evidence(trace)


def _render_metric_bar(label: str, value) -> None:
    label_column, bar_column, value_column = st.columns(
        [1.8, 4, 0.8], vertical_alignment="center"
    )
    label_column.markdown(f"**{label}**")
    progress_value = _metric_progress(value)
    if progress_value is None:
        bar_column.caption("Unavailable")
    else:
        bar_column.progress(progress_value)
    value_column.markdown(f"`{_metric_text(value)}`")


def _render_evaluation_result(trace: QuestionEvaluationTrace) -> None:
    st.markdown("## Evaluation result")
    st.caption(f"Ground-truth ID: `{trace.ground_truth_id}`")
    if trace.status != "success":
        st.error("Evaluation did not complete successfully.")
        st.caption("One or more evaluation stages returned no result.")
        return

    reference_column, generated_column = st.columns(2, gap="large")
    with reference_column:
        st.markdown("### Reference answer")
        st.markdown(trace.reference_answer or "Unavailable")
    with generated_column:
        st.markdown("### Generated answer")
        st.markdown(trace.rag.generated_answer or "Unavailable")

    st.divider()
    st.markdown("### RAGAS metrics")
    for label, metric_key in (
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall", "context_recall"),
        ("Answer Correctness", "answer_correctness"),
    ):
        _render_metric_bar(label, trace.metrics.get(metric_key))

    composite_column, _ = st.columns([1, 3])
    composite_column.metric("Composite score", _metric_text(trace.composite_score))
    st.caption("Metric bars show the reported 0–1 values without quality labels or inferred thresholds.")

    st.markdown("### Execution timing")
    st.markdown("**RAG execution**")
    rag_columns = st.columns(3)
    rag_columns[0].metric("Retrieval", _format_duration_ms(trace.rag.retrieval_latency_ms))
    rag_columns[1].metric("Generation", _format_duration_ms(trace.rag.generation_latency_ms))
    rag_columns[2].metric("Complete RAG", _format_duration_ms(trace.rag.rag_latency_ms))
    evaluation_columns = st.columns(2)
    evaluation_columns[0].metric("Evaluation", _format_duration_ms(trace.evaluation_latency_ms))
    evaluation_columns[1].metric("Total", _format_duration_ms(trace.total_latency_ms))

    with st.expander("Evaluator and token details", expanded=False):
        st.caption(f"Evaluator provider: {_display_telemetry(trace.evaluator_provider)}")
        st.caption(f"Evaluator model: {_display_telemetry(trace.evaluator_model)}")
        st.caption(f"Generation model: {_display_telemetry(trace.rag.generation_model)}")
        st.caption(f"Embedding model: {_display_telemetry(trace.rag.embedding_model)}")
        st.caption(f"Prompt tokens: {_format_tokens(trace.evaluator_prompt_tokens)}")
        st.caption(f"Completion tokens: {_format_tokens(trace.evaluator_completion_tokens)}")
        st.caption(f"Total tokens: {_format_tokens(trace.evaluator_total_tokens)}")
        st.caption(
            "Estimated evaluator cost: "
            f"{_display_telemetry(trace.estimated_evaluator_cost)}"
        )

    _render_source_cards(trace.rag.citations, trace.rag.retrieved_items)
    with st.expander("Retrieved evidence", expanded=False):
        _render_trace_evidence(trace.rag, show_heading=False)


def _render_evaluation_workspace() -> None:
    st.markdown(
        """
        <div class="rag-page-intro">
            <div class="rag-kicker">Developer Lab</div>
            <h1>Evaluation bench</h1>
            <p>Run one selected ground-truth question without changing batch results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        items = load_ground_truth()
    except (OSError, ValueError):
        st.error("The ground-truth dataset is unavailable.")
        return

    options = [str(item["id"]) for item in items]
    if st.session_state.evaluation_selected_id not in options:
        st.session_state.evaluation_selected_id = options[0]
    selected_id = st.selectbox(
        "Ground-truth ID",
        options,
        key="evaluation_selected_id",
        format_func=lambda value: f"ID {value}",
    )
    previous_id = st.session_state.get("evaluation_previous_id")
    if previous_id is not None and selected_id != previous_id:
        st.session_state.evaluation_result = None
        st.session_state.evaluation_error = None
    st.session_state.evaluation_previous_id = selected_id

    try:
        selected = select_ground_truth_item(items, ground_truth_id=selected_id)
    except SelectionError as exc:
        st.error(str(exc))
        return

    metadata = selected.get("metadata") or selected
    st.markdown("### Canonical question")
    st.markdown(str(selected["question"]))
    meta_columns = st.columns(2)
    meta_columns[0].caption(f"Question type: {metadata.get('question_type', 'Unavailable')}")
    meta_columns[1].caption(f"Difficulty: {metadata.get('difficulty', 'Unavailable')}")
    st.markdown("### Reference answer")
    st.markdown(str(selected["ground_truth"]))

    st.warning("Running an evaluation calls Gemini and the configured RAGAS evaluator and may consume quota.")
    run_column, clear_column = st.columns([2, 1])
    run_clicked = run_column.button("Run evaluation", type="primary", use_container_width=True)
    if clear_column.button("Clear result", use_container_width=True):
        st.session_state.evaluation_result = None
        st.session_state.evaluation_error = None
        st.rerun()

    if run_clicked:
        st.session_state.evaluation_error = None
        with st.status("Running RAG and evaluation…", expanded=False) as status:
            try:
                result = evaluate_ground_truth_item(selected)
                st.session_state.evaluation_result = result
                status.update(
                    label=(
                        "Evaluation complete"
                        if result.status == "success"
                        else "Evaluation finished with errors"
                    ),
                    state="complete" if result.status == "success" else "error",
                )
            except Exception:
                st.session_state.evaluation_result = None
                st.session_state.evaluation_error = (
                    "Evaluation could not complete. Check provider availability and configuration."
                )
                status.update(label="Evaluation failed", state="error")

    if st.session_state.evaluation_error:
        st.error(st.session_state.evaluation_error)
    if st.session_state.evaluation_result is not None:
        _render_evaluation_result(st.session_state.evaluation_result)


def _render_navigation_brand() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="rag-brand">
                <div class="rag-brand-mark">R</div>
                <div>
                    <div class="rag-brand-title">Multimodal RAG</div>
                    <div class="rag-brand-subtitle">Grounded document intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_sidebar(page_title: str) -> None:
    """Render page-aware actions and factual index-derived knowledge counts."""
    with st.sidebar:
        if page_title == "Chat":
            if st.button("New chat", use_container_width=True, type="primary"):
                _new_chat()
                st.rerun()

        st.divider()
        st.markdown("#### Knowledge")
        try:
            index, id_map = _load_index()
            stats = _index_stats(index, id_map)
        except (IndexNotFoundError, OSError, ValueError):
            st.caption("Knowledge-base counts are unavailable.")
        else:
            document_label = "document" if stats["documents"] == 1 else "documents"
            chunk_label = "chunk" if stats["chunks"] == 1 else "chunks"
            st.caption(f"**{stats['documents']:,}** {document_label}")
            st.caption(f"**{stats['chunks']:,}** {chunk_label}")
            st.caption(f"{stats['pages']:,} indexed pages")


def _render_chat_empty_state() -> None:
    st.markdown(
        """
        <div class="rag-hero">
            <div class="rag-kicker">User workspace</div>
            <h1>Multimodal RAG</h1>
            <p>Ask questions grounded in your indexed knowledge base.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    question_columns = st.columns(3)
    for column, question in zip(question_columns, SAMPLE_QUESTIONS):
        if column.button(question, key=f"sample_{question}", use_container_width=True):
            st.session_state.pending_query = question
            st.rerun()


def _render_chat_workspace() -> None:
    st.markdown(
        '<style>[data-testid="stMainBlockContainer"] { max-width: 900px; }</style>',
        unsafe_allow_html=True,
    )
    if not st.session_state.messages:
        _render_chat_empty_state()
    else:
        st.markdown(
            """
            <div class="rag-page-intro">
                <div class="rag-kicker">User workspace</div>
                <h1>Chat</h1>
                <p>Answers are grounded in the current indexed knowledge base. History lasts for this session.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(
                f'<div class="rag-message-label">'
                f'{"You" if message["role"] == "user" else "Assistant"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_feedback(message, message_index)

    user_input = st.chat_input("Ask your question…")
    if not user_input and st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown('<div class="rag-message-label">You</div>', unsafe_allow_html=True)
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown('<div class="rag-message-label">Assistant</div>', unsafe_allow_html=True)
        rag_trace = None
        with st.status(
            "Retrieving evidence and generating a grounded answer…", expanded=False
        ) as status:
            try:
                rag_trace = _answer_query(user_input)
                answer = rag_trace.generated_answer
                status.update(label="Answer ready", state="complete")
            except IndexNotFoundError:
                answer = (
                    "No document index is available. Build the index and restart the application."
                )
                status.update(label="Knowledge base unavailable", state="error")
            except AnswerGenerationUnavailableError:
                answer = "The answer generation model is not configured."
                status.update(label="Generation model unavailable", state="error")
            except AnswerGenerationError:
                answer = "The generation provider could not complete this answer. Please try again."
                status.update(label="Generation failed", state="error")

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "citations": rag_trace.citations if rag_trace is not None else [],
            "uncited_sources": rag_trace.uncited_sources if rag_trace is not None else [],
            "trace": rag_trace,
        }
        st.markdown(answer)
        _render_feedback(assistant_message, len(st.session_state.messages))

    st.session_state.messages.append(assistant_message)
    st.session_state.history_turns.append(ConversationTurn(user_input, answer))


def _traced_messages() -> list[tuple[int, dict[str, object]]]:
    return [
        (index, message)
        for index, message in enumerate(st.session_state.messages)
        if message.get("role") == "assistant" and message.get("trace") is not None
    ]


def _render_playground_workspace() -> None:
    st.markdown(
        """
        <div class="rag-page-intro">
            <div class="rag-kicker">Developer Lab</div>
            <h1>Playground / Trace</h1>
            <p>Run controlled RAG questions or inspect a stored execution trace without invoking RAGAS.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Suggested test questions")
    question_columns = st.columns(3)
    for column, question in zip(question_columns, DEVELOPER_SAMPLE_QUESTIONS):
        if column.button(
            question,
            key=f"playground_sample_{question}",
            use_container_width=True,
        ):
            st.session_state.playground_pending_query = question
            st.rerun()

    input_columns = st.columns([5, 1])
    developer_question = input_columns[0].text_input(
        "Ask a test question",
        key="playground_question_input",
        placeholder="Type a question for the RAG trace…",
    )
    run_clicked = input_columns[1].button(
        "Run",
        key="playground_run_question",
        use_container_width=True,
        type="primary",
    )

    pending_query = st.session_state.playground_pending_query
    if pending_query:
        developer_question = pending_query
        st.session_state.playground_pending_query = None
        run_clicked = True

    if run_clicked:
        if not developer_question or not developer_question.strip():
            st.session_state.playground_error = "Enter a question before running the trace."
        else:
            st.session_state.playground_error = None
            with st.status("Running one RAG trace…", expanded=False) as status:
                try:
                    st.session_state.playground_trace = _answer_query(developer_question.strip())
                    status.update(label="Trace complete", state="complete")
                except (
                    IndexNotFoundError,
                    AnswerGenerationUnavailableError,
                    AnswerGenerationError,
                ):
                    st.session_state.playground_trace = None
                    st.session_state.playground_error = (
                        "The RAG trace could not complete. Check the index and generation configuration."
                    )
                    status.update(label="Trace failed", state="error")

    if st.session_state.playground_error:
        st.error(st.session_state.playground_error)

    available_traces: list[tuple[str, RAGTrace]] = []
    if st.session_state.playground_trace is not None:
        available_traces.append(("Latest developer run", st.session_state.playground_trace))

    traced = _traced_messages()
    available_traces.extend(
        (f"Chat · {message['trace'].original_question}", message["trace"])
        for _message_index, message in traced
    )
    if not available_traces:
        st.info("No trace is available yet. Run a developer question or ask one in Chat.")
        return

    selected_label = st.selectbox(
        "Previous traces",
        [label for label, _trace in available_traces],
        index=0 if st.session_state.playground_trace is not None else len(available_traces) - 1,
        key="playground_selected_trace",
    )
    trace = dict(available_traces)[selected_label]

    st.markdown("### Question")
    st.markdown(trace.original_question)
    st.markdown("### Generated answer")
    st.markdown(trace.generated_answer)
    st.divider()
    _render_developer_trace(trace)


def _navigation_pages():
    return {
        "USER WORKSPACE": [
            st.Page(
                _render_chat_workspace,
                title="Chat",
                icon=":material/chat_bubble_outline:",
                url_path="chat",
                default=True,
            )
        ],
        "DEVELOPER LAB": [
            st.Page(
                _render_playground_workspace,
                title="Playground / Trace",
                icon=":material/science:",
                url_path="playground",
            ),
            st.Page(
                _render_evaluation_workspace,
                title="Evaluation",
                icon=":material/fact_check:",
                url_path="evaluation",
            ),
        ],
    }


def main() -> None:
    _init_session_state()
    _render_navigation_brand()
    page = st.navigation(_navigation_pages(), position="sidebar", expanded=True)
    _render_sidebar(page.title)
    page.run()


if __name__ == "__main__":
    main()
