#!/usr/bin/env python3
"""
Maintained Streamlit chat UI for the RAG system.

Run with:
    python -m streamlit run src/multimodal_rag/ui/streamlit_app.py

Backend usage matches the CLI ask path: load_index -> retrieve -> build_prompt
-> generate_answer. This file only adds a UI layer and short-term
conversation memory (last few turns, used ONLY when building the prompt -
retrieval still runs on the latest user message alone, per
rag/retrieval/retriever_2.py, which is untouched).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multimodal_rag.evaluation.question_runner import (
    CONFIGURED_TOP_K,
    QuestionEvaluationTrace,
    SelectionError,
    evaluate_ground_truth_item,
    load_ground_truth,
    select_ground_truth_item,
)
from multimodal_rag.rag.generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerationUnavailableError,
)
from multimodal_rag.rag.generation.prompt_builder import ConversationTurn
from multimodal_rag.rag.indexing.faiss_index import IndexNotFoundError, load_index
from multimodal_rag.rag.trace import RAGTrace, run_rag_trace
from multimodal_rag.rag.retrieval.retriever_2 import retrieve  # compatibility export
from multimodal_rag.paths import INDEX_DIR as DEFAULT_INDEX_DIR, LEGACY_INDEX_DIR, prefer_new_path

INDEX_DIR = str(prefer_new_path(DEFAULT_INDEX_DIR, LEGACY_INDEX_DIR))
TOP_K = 5
MAX_HISTORY_TURNS = 5

SAMPLE_QUESTIONS = [
    "What are the main risk factors mentioned in the documents?",
    "Summarize the key findings.",
    "What tables or figures are discussed?",
]

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --rag-accent: #5b4bdb; --rag-muted: #667085; --rag-ink: #172033; }
    .block-container { max-width: 1180px; padding-top: 2.25rem; padding-bottom: 7rem; }
    section[data-testid="stSidebar"] { border-right: 1px solid #e4e7ec; background: #fbfcfe; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .rag-brand { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.4rem; }
    .rag-brand-mark { width: 2.35rem; height: 2.35rem; display: grid; place-items: center;
        border-radius: 0.75rem; background: linear-gradient(135deg, #5b4bdb, #3b82f6);
        color: white; font-weight: 800; font-size: 1.1rem; }
    .rag-brand-title { font-size: 1.05rem; font-weight: 750; line-height: 1.1; }
    .rag-brand-subtitle { color: var(--rag-muted); font-size: 0.73rem; margin-top: 0.2rem; }
    .rag-hero { padding: 2.5rem 0 1.4rem; }
    .rag-eyebrow { color: var(--rag-accent); font-size: 0.76rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; }
    .rag-hero h1 { font-size: clamp(2.1rem, 4vw, 3.35rem); letter-spacing: -0.045em; margin: 0.4rem 0 0.65rem; }
    .rag-hero p { color: var(--rag-muted); font-size: 1.05rem; max-width: 680px; }
    .rag-empty-card { border: 1px solid #e4e7ec; border-radius: 1rem;
        padding: 1.1rem 1.2rem; min-height: 8rem; background: #ffffff; }
    .rag-empty-card strong { display: block; margin-bottom: 0.45rem; }
    .rag-empty-card span { color: var(--rag-muted); font-size: 0.9rem; line-height: 1.4; }
    .rag-message-label { color: var(--rag-muted); font-size: 0.75rem; font-weight: 650;
        letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.25rem; }
    [data-testid="stChatMessage"] { border: 1px solid #eaecf0; background: #ffffff;
        border-radius: 1rem; padding: 1rem 1.2rem; margin: 0.8rem 0; }
    [data-testid="stChatMessage"] p { line-height: 1.65; }
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 { letter-spacing: -0.02em; }
    .rag-source-label { margin-top: 0.9rem; color: var(--rag-muted); font-size: 0.82rem; }
    .rag-status { border-radius: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _load_index():
    return load_index(INDEX_DIR)


def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]
    if "history_turns" not in st.session_state:
        st.session_state.history_turns = []  # list[ConversationTurn], for prompt memory only
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "developer_mode" not in st.session_state:
        st.session_state.developer_mode = False
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "evaluation_selected_id" not in st.session_state:
        st.session_state.evaluation_selected_id = None
    if "evaluation_error" not in st.session_state:
        st.session_state.evaluation_error = None
    if "evaluation_previous_id" not in st.session_state:
        st.session_state.evaluation_previous_id = None
    # Keep only the bounded prompt history even if a session survives a code reload.
    st.session_state.history_turns = st.session_state.history_turns[-MAX_HISTORY_TURNS:]
    if st.session_state.pending_query is not None and not isinstance(
        st.session_state.pending_query, str
    ):
        st.session_state.pending_query = None


def _new_chat():
    st.session_state.messages = []
    st.session_state.history_turns = []
    st.session_state.pending_query = None


def _answer_query(query: str) -> RAGTrace:
    """Retrieval (latest query only) -> prompt (with memory) -> generation.
    Every backend call here is identical in shape to the CLI ask path; the only
    addition is passing conversation_history into build_prompt."""
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
    """Return display-only knowledge-base statistics from the loaded index."""
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
    """Render compact citation cards, falling back to retrieved source metadata."""
    records = _source_records(citations, retrieved_items)
    if not records:
        return

    st.markdown("**Sources**")
    for source in records:
        marker = source["marker"]
        source_file = source["source_file"]
        page_numbers = source["page_numbers"]
        section_title = source["section_title"]
        pages = ", ".join(str(page) for page in page_numbers) or "Unavailable"
        label = f"[{marker}]  " if marker else ""
        with st.expander(f"{label}{source_file} · page {pages}"):
            st.caption(f"**Document:** {source_file}")
            st.caption(f"**Page:** {pages}")
            st.caption(f"**Section:** {section_title or 'Unavailable'}")


def _display_telemetry(value, suffix: str = "") -> str:
    return "Unavailable" if value is None else f"{value}{suffix}"


def _format_duration_ms(value) -> str:
    """Format UI timings without changing the underlying trace values."""
    if value is None:
        return "Unavailable"
    milliseconds = float(value)
    if milliseconds < 1000:
        return f"{milliseconds:.1f} ms"
    return f"{milliseconds / 1000:.2f} s"


def _render_metadata_fields(metadata: dict[str, object] | None) -> None:
    """Render useful metadata fields without dumping raw JSON."""
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
    }
    for key, value in metadata.items():
        label = labels.get(key, key.replace("_", " ").title())
        if isinstance(value, (dict, list, tuple)):
            value = ", ".join(str(item) for item in value) if value else "Unavailable"
        st.caption(f"**{label}:** {value if value not in (None, '') else 'Unavailable'}")


def _render_developer_trace(trace: RAGTrace) -> None:
    """Render technical diagnostics only when Developer Mode is enabled."""
    with st.expander("Developer trace", expanded=False):
        st.markdown("#### Pipeline")
        st.caption(f"Retriever: `{trace.retriever}`")
        st.caption(f"Embedding model: `{_display_telemetry(trace.embedding_model)}`")
        st.caption(f"Generation model: `{_display_telemetry(trace.generation_model)}`")
        st.caption(f"Configured top-k: {trace.configured_top_k}")
        st.caption(f"Actual retrieved chunks: {trace.actual_retrieved_count}")

        st.markdown("#### Performance")
        st.caption(f"Retrieval latency: {_format_duration_ms(trace.retrieval_latency_ms)}")
        st.caption(f"Generation latency: {_format_duration_ms(trace.generation_latency_ms)}")
        st.caption(f"Complete RAG latency: {_format_duration_ms(trace.rag_latency_ms)}")

        st.markdown("#### Token usage")
        st.caption(
            f"Prompt tokens: {_display_telemetry(trace.generation_prompt_tokens)}"
        )
        st.caption(
            f"Completion tokens: {_display_telemetry(trace.generation_completion_tokens)}"
        )
        st.caption(f"Total tokens: {_display_telemetry(trace.generation_total_tokens)}")
        st.caption(f"Estimated cost: {_display_telemetry(trace.estimated_generation_cost)}")

        st.markdown("#### Retrieved evidence")
        if not trace.retrieved_items:
            st.caption("No chunks retrieved.")
        for item in trace.retrieved_items:
            pages = ", ".join(str(page) for page in item.page_numbers) or "Unavailable"
            with st.expander(
                f"Rank {item.rank} · {item.document_name} · score {item.raw_faiss_score:.6f}"
            ):
                st.caption(f"Chunk ID: `{item.chunk_id}`")
                st.caption(f"Document: {item.document_name}")
                st.caption(f"Page: {pages}")
                st.caption(f"Section: {item.section_title or 'Unavailable'}")
                st.caption("Raw FAISS similarity score: " f"{item.raw_faiss_score:.17g}")
                st.caption(
                    "Combined rerank score: "
                    f"{_display_telemetry(item.combined_rerank_score)}"
                )
                with st.expander("Advanced metadata", expanded=False):
                    _render_metadata_fields(item.metadata)
                    if item.metadata is None and item.metadata_note:
                        st.caption(item.metadata_note)
                with st.expander("Complete chunk text", expanded=False):
                    st.text(item.chunk_text)

        st.markdown("#### Citation diagnostics")
        if trace.citations:
            for citation in trace.citations:
                st.caption(
                    f"[{citation['marker']}] {citation['source_file']} · "
                    f"page {citation['page_numbers'] or 'unavailable'} · "
                    f"section {citation['section_title'] or 'unavailable'}"
                )
        else:
            st.caption("Source metadata is available, but no citation markers were resolved.")


def _metric_text(value) -> str:
    return "Unavailable" if value is None else f"{float(value):.3f}"


def _render_evaluation_result(trace: QuestionEvaluationTrace) -> None:
    st.markdown("### Evaluation result")
    if trace.status != "success":
        st.error("Evaluation did not complete successfully.")
        for error in trace.errors:
            st.caption(error)
        return

    st.caption(f"Ground-truth ID: `{trace.ground_truth_id}`")
    st.markdown("**Reference answer**")
    st.info(trace.reference_answer)
    st.markdown("**Generated answer**")
    st.markdown(trace.rag.generated_answer or "Unavailable")

    st.markdown("#### RAGAS metrics")
    metric_values = [
        ("Faithfulness", trace.metrics.get("faithfulness")),
        ("Answer Relevancy", trace.metrics.get("answer_relevancy")),
        ("Context Precision", trace.metrics.get("context_precision")),
        ("Context Recall", trace.metrics.get("context_recall")),
        ("Answer Correctness", trace.metrics.get("answer_correctness")),
        ("Composite Score", trace.composite_score),
    ]
    for row in (metric_values[:3], metric_values[3:]):
        for column, (label, value) in zip(st.columns(3), row):
            column.metric(label, _metric_text(value))

    st.markdown("#### Timing")
    timing_columns = st.columns(5)
    for column, (label, value) in zip(
        timing_columns,
        [
            ("Retrieval", trace.rag.retrieval_latency_ms),
            ("Generation", trace.rag.generation_latency_ms),
            ("Complete RAG", trace.rag.rag_latency_ms),
            ("Evaluation", trace.evaluation_latency_ms),
            ("Total", trace.total_latency_ms),
        ],
    ):
        column.metric(label, _format_duration_ms(value))

    with st.expander("Models and evaluator", expanded=False):
        st.caption(f"Evaluator provider: {_display_telemetry(trace.evaluator_provider)}")
        st.caption(f"Evaluator model: {_display_telemetry(trace.evaluator_model)}")
        st.caption(f"Generation model: {_display_telemetry(trace.rag.generation_model)}")
        st.caption(f"Embedding model: {_display_telemetry(trace.rag.embedding_model)}")
        st.caption(f"Evaluator prompt tokens: {_display_telemetry(trace.evaluator_prompt_tokens)}")
        st.caption(f"Evaluator completion tokens: {_display_telemetry(trace.evaluator_completion_tokens)}")
        st.caption(f"Evaluator total tokens: {_display_telemetry(trace.evaluator_total_tokens)}")
        st.caption(f"Estimated evaluator cost: {_display_telemetry(trace.estimated_evaluator_cost)}")

    with st.expander("Retrieved chunks", expanded=False):
        for item in trace.rag.retrieved_items:
            pages = ", ".join(str(page) for page in item.page_numbers) or "Unavailable"
            with st.expander(
                f"Rank {item.rank} · {item.document_name} · score {item.raw_faiss_score:.6f}"
            ):
                st.caption(f"Document: {item.document_name}")
                st.caption(f"Page: {pages}")
                st.caption(f"Section: {item.section_title or 'Unavailable'}")
                st.caption(f"Chunk ID: `{item.chunk_id}`")
                st.caption(f"Raw FAISS score: {item.raw_faiss_score:.17g}")
                st.caption(
                    "Combined rerank score: "
                    f"{_display_telemetry(item.combined_rerank_score)}"
                )
                with st.expander("Metadata", expanded=False):
                    _render_metadata_fields(item.metadata)
                    if item.metadata is None and item.metadata_note:
                        st.caption(item.metadata_note)
                with st.expander("Complete chunk text", expanded=False):
                    st.text(item.chunk_text)

    _render_source_cards(trace.rag.citations, trace.rag.retrieved_items)


def _render_evaluation_workspace() -> None:
    st.markdown("## Evaluation workspace")
    st.caption("Run one selected ground-truth question without changing batch evaluation results.")

    try:
        items = load_ground_truth()
    except (OSError, ValueError) as exc:
        st.error(f"Ground-truth dataset unavailable: {exc}")
        return

    options = [str(item["id"]) for item in items]
    if st.session_state.evaluation_selected_id not in options:
        st.session_state.evaluation_selected_id = options[0]
    selected_id = st.selectbox(
        "Ground-truth question",
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

    st.markdown("**Canonical question**")
    st.info(str(selected["question"]))
    metadata = selected.get("metadata") or selected
    meta_columns = st.columns(2)
    meta_columns[0].caption(f"Question type: {metadata.get('question_type', 'Unavailable')}")
    meta_columns[1].caption(f"Difficulty: {metadata.get('difficulty', 'Unavailable')}")

    st.warning("This runs Gemini, Groq, and RAGAS and may consume provider quota.")
    run_column, clear_column = st.columns([2, 1])
    run_clicked = run_column.button("Run evaluation", type="primary", use_container_width=True)
    if clear_column.button("Clear result", use_container_width=True):
        st.session_state.evaluation_result = None
        st.session_state.evaluation_error = None
        st.rerun()

    if run_clicked:
        st.session_state.evaluation_error = None
        with st.status("Running retrieval, generation, and RAGAS evaluation…", expanded=False) as status:
            try:
                result = evaluate_ground_truth_item(selected)
                st.session_state.evaluation_result = result
                status.update(
                    label="Evaluation complete" if result.status == "success" else "Evaluation finished with errors",
                    state="complete" if result.status == "success" else "error",
                )
            except Exception as exc:
                st.session_state.evaluation_result = None
                st.session_state.evaluation_error = f"{type(exc).__name__}: {exc}"
                status.update(label="Evaluation failed", state="error")

    if st.session_state.evaluation_error:
        st.error(st.session_state.evaluation_error)
    if st.session_state.evaluation_result is not None:
        _render_evaluation_result(st.session_state.evaluation_result)


def _render_sidebar(workspace: str = "Chat"):
    with st.sidebar:
        if workspace == "Evaluation":
            st.markdown(
                '<div class="rag-brand"><div class="rag-brand-mark">R</div>'
                '<div><div class="rag-brand-title">Multimodal RAG</div>'
                '<div class="rag-brand-subtitle">Evaluation workspace</div></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Evaluation controls")
            try:
                st.metric("Ground-truth questions", f"{len(load_ground_truth()):,}")
            except (OSError, ValueError):
                st.caption("Ground-truth questions: Unavailable")
            from multimodal_rag.evaluation import runner as evaluation_runner

            evaluator_model = (
                evaluation_runner.OLLAMA_MODEL_NAME
                if evaluation_runner.EVALUATOR_PROVIDER == "ollama"
                else evaluation_runner.GROQ_MODEL_NAME
            )
            st.caption(f"Evaluator: `{evaluation_runner.EVALUATOR_PROVIDER}` · `{evaluator_model}`")
            st.caption(f"Evaluation retrieval: top {CONFIGURED_TOP_K} chunks")
            st.warning("Evaluation calls Gemini, Groq, and RAGAS and may consume quota.")
            selected_id = st.session_state.get("evaluation_selected_id")
            st.caption(f"Selected ID: {selected_id if selected_id is not None else 'None'}")
            if st.session_state.get("evaluation_result") is not None:
                if st.button("Clear evaluation result", use_container_width=True):
                    st.session_state.evaluation_result = None
                    st.session_state.evaluation_error = None
                    st.rerun()
            return

        st.markdown(
            """
            <div class="rag-brand">
                <div class="rag-brand-mark">R</div>
                <div>
                    <div class="rag-brand-title">Multimodal RAG</div>
                    <div class="rag-brand-subtitle">Your private knowledge workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋  New chat", use_container_width=True, type="primary"):
            _new_chat()
            st.rerun()

        st.session_state.developer_mode = st.toggle(
            "Developer Mode",
            value=st.session_state.get("developer_mode", False),
            help="Show retrieval, metadata, timing, and model diagnostics for each answer.",
        )

        st.divider()
        st.markdown("#### Suggested questions")
        st.caption("Start with one of these prompts or ask anything below.")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()

        st.divider()
        st.markdown("#### Knowledge base")
        try:
            index, id_map = _load_index()
            stats = _index_stats(index, id_map)
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Documents", f"{stats['documents']:,}")
            stat_col2.metric("Chunks", f"{stats['chunks']:,}")
            st.caption(f"{stats['pages']:,} indexed pages · top {TOP_K} chunks per question")
        except IndexNotFoundError:
            st.warning("No knowledge base is indexed yet.")
            st.caption("Run `python -m multimodal_rag.cli.build_index` after ingestion.")

        with st.expander("System details"):
            st.caption(f"Index: `{INDEX_DIR}`")
            st.caption("Retriever: semantic similarity")
            st.caption("Embeddings: all-MiniLM-L6-v2")


def main():
    _init_session_state()
    workspace = st.radio(
        "Workspace",
        ("Chat", "Evaluation"),
        horizontal=True,
        label_visibility="collapsed",
        key="workspace",
    )
    _render_sidebar(workspace)
    if workspace == "Evaluation":
        _render_evaluation_workspace()
        return

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="rag-hero">
                <div class="rag-eyebrow">Research, without the busywork</div>
                <h1>Ask your knowledge base.</h1>
                <p>Search your indexed documents and get a grounded answer with the context of your conversation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        empty_cols = st.columns(3)
        empty_cards = [
            ("Find the signal", "Ask for a fact, definition, or detail hidden in your documents."),
            ("Connect the dots", "Compare themes, summarize findings, or explore related ideas."),
            ("Stay in flow", "Your recent conversation is remembered while you investigate."),
        ]
        for column, (title, text) in zip(empty_cols, empty_cards):
            with column:
                st.markdown(
                    f'<div class="rag-empty-card"><strong>{title}</strong><span>{text}</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown("## Your research workspace")
        st.caption("Continue the conversation or start a fresh thread from the sidebar.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(
                f'<div class="rag-message-label">{"You" if msg["role"] == "user" else "RAG assistant"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(msg["content"])
            stored_trace = msg.get("trace")
            _render_source_cards(
                msg.get("citations", []),
                stored_trace.retrieved_items if stored_trace is not None else None,
            )
            if st.session_state.developer_mode and msg.get("trace") is not None:
                _render_developer_trace(msg["trace"])

    user_input = st.chat_input("Ask a question about your documents...")
    if not user_input and st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown('<div class="rag-message-label">You</div>', unsafe_allow_html=True)
            st.markdown(user_input)

        with st.chat_message("assistant"):
            st.markdown('<div class="rag-message-label">RAG assistant</div>', unsafe_allow_html=True)
            rag_trace = None
            with st.status("Searching your knowledge base…", expanded=False) as status:
                try:
                    rag_trace = _answer_query(user_input)
                    answer = rag_trace.generated_answer
                    status.update(label="Answer ready", state="complete")
                except IndexNotFoundError:
                    answer = (
                        "No document index found. Run `python -m multimodal_rag.cli.build_index` after "
                        "ingesting your PDFs, then restart this app."
                    )
                    status.update(label="Knowledge base unavailable", state="error")
                except AnswerGenerationUnavailableError:
                    answer = (
                        "The answer generation model isn't configured yet "
                        "(missing GEMINI_API_KEY)."
                    )
                    status.update(label="Generation model unavailable", state="error")
                except AnswerGenerationError as e:
                    answer = f"Sorry, something went wrong generating an answer: {e}"
                    status.update(label="Generation failed", state="error")
            st.markdown(answer)
            _render_source_cards(
                rag_trace.citations if rag_trace is not None else [],
                rag_trace.retrieved_items if rag_trace is not None else None,
            )
            if st.session_state.developer_mode and rag_trace is not None:
                _render_developer_trace(rag_trace)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": rag_trace.citations if rag_trace is not None else [],
                "uncited_sources": rag_trace.uncited_sources if rag_trace is not None else [],
                "trace": rag_trace,
            }
        )
        st.session_state.history_turns.append(ConversationTurn(user_input, answer))


if __name__ == "__main__":
    main()
