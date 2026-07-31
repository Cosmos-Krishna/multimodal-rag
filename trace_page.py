"""Backward-compatible wrapper for the page diagnostic tool."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multimodal_rag.tools.diagnostics.trace_page import main


if __name__ == "__main__":
    main()
