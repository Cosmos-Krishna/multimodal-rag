"""Canonical repository paths for the modular project layout.

This module is intentionally side-effect free: importing it never creates
directories or changes files. Later migration phases will consume these
constants when their respective code and runtime data are moved.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
INGESTION_ARTIFACTS_DIR = ARTIFACTS_DIR / "ingestion"
INDEX_DIR = ARTIFACTS_DIR / "index"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LEGACY_INGESTION_ARTIFACTS_DIR = ARTIFACTS_DIR / "legacy-ingestion"
COMPARISONS_DIR = DATA_DIR / "comparisons"
MARKER_COMPARISON_DIR = COMPARISONS_DIR / "marker"
UNSTRUCTURED_COMPARISON_DIR = COMPARISONS_DIR / "unstructured"
EVALUATION_ARTIFACTS_DIR = DATA_DIR / "evaluation"
LOGS_DIR = DATA_DIR / "logs"

CONFIG_DIR = PROJECT_ROOT / "config"
EVALUATION_DATASET_DIR = PROJECT_ROOT / "evaluation" / "datasets"
GROUND_TRUTH_PATH = EVALUATION_DATASET_DIR / "ground_truth.json"
STREAMLIT_CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "config.toml"

# Legacy locations are retained only as temporary read/write fallbacks for
# users who have not moved their local runtime data yet.
LEGACY_INPUT_DIR = PROJECT_ROOT / "input"
LEGACY_OUTPUT_DIR = PROJECT_ROOT / "output"
LEGACY_INDEX_DIR = PROJECT_ROOT / "index"
LEGACY_LOGS_DIR = PROJECT_ROOT / "logs"
LEGACY_INGESTION_IMAGES_DIR = PROJECT_ROOT / "ingestion_output" / "images"
LEGACY_COMPARISONS_DIR = PROJECT_ROOT / "comparison_output"
LEGACY_MARKER_COMPARISON_DIR = LEGACY_COMPARISONS_DIR / "marker_output"
LEGACY_UNSTRUCTURED_COMPARISON_DIR = LEGACY_COMPARISONS_DIR / "unstructured_output"
LEGACY_EVALUATION_DIR = PROJECT_ROOT / "evaluation"
LEGACY_GROUND_TRUTH_PATH = LEGACY_EVALUATION_DIR / "ground_truth.json"


def prefer_new_path(new_path: Path, legacy_path: Path) -> Path:
    """Use the migrated location, falling back to a legacy location only
    when the new location is absent and the legacy location still exists."""
    if new_path.exists() or not legacy_path.exists():
        return new_path
    return legacy_path
