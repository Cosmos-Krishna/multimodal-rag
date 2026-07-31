#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# 1. Load environment variables at startup
load_dotenv()

# Force workspace root into sys.path to mimic ask.py's import resolution
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import your exact backend functions and configuration classes
try:
    from multimodal_rag.rag.generation.answer_generator import (
        AnswerGenerationError,
        AnswerGenerationUnavailableError,
        GenerationConfig,
        generate_answer,
    )
    from multimodal_rag.rag.generation.citation import resolve_citations
    from multimodal_rag.rag.generation.prompt_builder import build_prompt
    from multimodal_rag.rag.indexing.faiss_index import IndexNotFoundError, load_index
    from multimodal_rag.rag.retrieval.retriever_2 import RetrieverConfig, retrieve
    from multimodal_rag.paths import INDEX_DIR as DEFAULT_INDEX_DIR, LEGACY_INDEX_DIR, prefer_new_path
except ImportError as e:
    st.error(
        f"Backend Import Error: {e}\n"
        "Please ensure app.py is placed in your project root directory and the virtual environment is active."
    )

# 2. Page Configuration (Wide layout, Custom Title)
st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 3. CSS to Hide default Streamlit Menu, Header, Footer & adjust top padding
st.markdown(
    """
    <style>
    /* Hide the Streamlit Hamburger Menu, Header, and Footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}

    /* Adjust page margins slightly to compensate for hidden header */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom CSS adjustments for sidebar */
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid #1e293b;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    .sidebar-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        margin-bottom: 0.1rem;
    }
    .sidebar-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        background-color: #1e293b;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 4. Cache loading of the FAISS Index (Loads only once across runs)
@st.cache_resource(show_spinner="Loading Index...")
def get_cached_index(directory: str):
    return load_index(directory)

# 5. Initialization of Chat History & Safety Check Variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

index_dir = str(prefer_new_path(DEFAULT_INDEX_DIR, LEGACY_INDEX_DIR))
index_exists = os.path.exists(index_dir) and len(os.listdir(index_dir)) > 0
api_key_exists = "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"].strip() != ""

# 6. Sidebar Implementation
with st.sidebar:
    st.markdown('<div class="sidebar-header">System Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Project Name</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Multimodal RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Embedding Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">all-MiniLM-L6-v2</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">LLM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Gemini 2.5 Flash</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Vector Database</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">FAISS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Semantic Search</div>', unsafe_allow_html=True)

    st.markdown("---")
    # "New Chat" button to clear conversation session state
    if st.button("🗑 New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# 7. Main Page Titles
st.title("📄 Multimodal RAG Chatbot")
st.markdown("##### Ask questions over your indexed PDF documents.")
st.markdown("---")

# Safety Guards & Warnings
if not index_exists:
    st.warning(f"⚠️ **Index missing:** No vector database directory detected in `{index_dir}`. Please run `build_index.py` first to index your documents.")

if not api_key_exists:
    st.warning("⚠️ **API Key missing:** The environment variable `GEMINI_API_KEY` is not set. Please ensure it is exported in your environment or defined in `.env`.")

# 8. Render Current Chat Session via Native Elements
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 9. Handle User Chat Input
if user_input := st.chat_input("Ask a question..."):
    # Render user query instantly
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant answer
    with st.chat_message("assistant"):
        if not index_exists or not api_key_exists:
            assistant_response = "I cannot process your query. Please resolve the workspace warnings above."
            st.markdown(assistant_response)
        else:
            with st.spinner("Thinking..."):
                try:
                    # Flow matching ask.py with cached resource
                    index, id_map = get_cached_index(index_dir)
                    chunks = retrieve(user_input, index, id_map, retriever_config=RetrieverConfig(top_k=5))
                    if not chunks:
                        assistant_response = "No relevant content found for that query."
                    else:
                        built = build_prompt(user_input, chunks)
                        raw_answer = generate_answer(built.prompt_text, GenerationConfig())
                        result = resolve_citations(raw_answer, built.source_map)
                        # Extract exclusively clean, textual Markdown answer without citations
                        assistant_response = result.answer_text

                except IndexNotFoundError as e:
                    assistant_response = f"Database index error: {e}. Please build your vector database index."
                except AnswerGenerationUnavailableError as e:
                    assistant_response = f"LLM backend unavailable: {e}. Ensure GEMINI_API_KEY is configured properly."
                except AnswerGenerationError as e:
                    assistant_response = f"An error occurred during generative output extraction: {e}"
                except Exception as e:
                    assistant_response = f"An unexpected system error occurred: {str(e)}"

                st.markdown(assistant_response)
        # Commit to global session history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
