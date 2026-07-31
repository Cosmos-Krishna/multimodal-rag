"""Backward-compatible wrapper for the maintained Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multimodal_rag.ui.streamlit_app import main


main()
