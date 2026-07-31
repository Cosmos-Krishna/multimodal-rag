"""Backward-compatible wrapper for the legacy Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import multimodal_rag.ui.legacy_app  # noqa: F401
