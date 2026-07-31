"""Print-only compatibility entry point for one-question RAGAS evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multimodal_rag.evaluation.question_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
