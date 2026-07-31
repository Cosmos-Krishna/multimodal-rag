"""Backward-compatible wrapper for the index-building command."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multimodal_rag.cli.build_index import main


if __name__ == "__main__":
    raise SystemExit(main())
