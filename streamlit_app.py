#!/usr/bin/env python3
"""
streamlit_app.py - Chat UI for the RAG system.

Run with:
    streamlit run streamlit_app.py

Backend usage is unchanged from ask.py: load_index -> retrieve -> build_prompt
-> generate_answer. This file only adds a UI layer and short-term
conversation memory (last few turns, used ONLY when building the prompt -
retrieval still runs on the latest user message alone, per
rag/retrieval/retriever.py, which is untouched).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerationUnavailableError,
    GenerationConfig,
    generate_answer,
)
from rag.generation.prompt_builder import ConversationTurn, build_prompt
from rag.indexing.faiss_index import IndexNotFoundError, load_index
from rag.retrieval.retriever_2 import RetrieverConfig, retrieve

INDEX_DIR = "index"
TOP_K = 5
MAX_HISTORY_TURNS = 5

SAMPLE_QUESTIONS = [
    "What are the main risk factors mentioned in the documents?",
    "Summarize the key findings.",
    "What tables or figures are discussed?",
]

st.set_page_config(page_title="Document Q&A", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    section[data-testid="stSidebar"] { border-right: 1px solid #2A2D36; }
    .sample-question-caption { color: #9A9DA6; font-size: 0.85rem; margin-bottom: 0.25rem; }
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


def _new_chat():
    st.session_state.messages = []
    st.session_state.history_turns = []
    st.session_state.pending_query = None


def _answer_query(query: str) -> str:
    """Retrieval (latest query only) -> prompt (with memory) -> generation.
    Every backend call here is identical in shape to ask.py; the only
    addition is passing conversation_history into build_prompt."""
    index, id_map = _load_index()

    chunks = retrieve(query, index, id_map, retriever_config=RetrieverConfig(top_k=TOP_K))
    if not chunks:
        return "I couldn't find anything relevant to that in the documents."

    built = build_prompt(
        query,
        chunks,
        conversation_history=st.session_state.history_turns[-MAX_HISTORY_TURNS:],
        max_history_turns=MAX_HISTORY_TURNS,
    )
    answer = generate_answer(built.prompt_text, GenerationConfig())
    return answer


def _render_sidebar():
    with st.sidebar:
        st.markdown("## 💬 Document Q&A")
        st.caption("Ask questions about your ingested documents.")

        if st.button("🆕 New Chat", use_container_width=True):
            _new_chat()
            st.rerun()

        st.divider()
        st.markdown("**Sample questions**")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()

        st.divider()
        st.caption(f"Index directory: `{INDEX_DIR}`")
        st.caption(f"Retrieving top {TOP_K} chunks per question")


def main():
    _init_session_state()
    _render_sidebar()

    st.title("Ask your documents")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your documents...")
    if not user_input and st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = _answer_query(user_input)
                except IndexNotFoundError:
                    answer = (
                        "No document index found. Run `python build_index.py` after "
                        "ingesting your PDFs, then restart this app."
                    )
                except AnswerGenerationUnavailableError:
                    answer = (
                        "The answer generation model isn't configured yet "
                        "(missing GEMINI_API_KEY)."
                    )
                except AnswerGenerationError as e:
                    answer = f"Sorry, something went wrong generating an answer: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.history_turns.append(ConversationTurn(user_input, answer))


if __name__ == "__main__":
    main()
