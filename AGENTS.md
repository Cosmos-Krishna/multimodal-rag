# Repository instructions for coding agents

This file contains stable project rules. Read `docs/PROJECT_STATUS.md` for the latest verified state and `docs/ARCHITECTURE_REFERENCE.md` for implementation detail. Source code remains authoritative.

## Canonical layout and environment

- Application source lives under `src/multimodal_rag/`; do not recreate root-level Python wrappers.
- Package discovery uses the `src/` layout in `pyproject.toml`.
- The canonical virtual environment is `.venv` and the canonical interpreter is `.venv\Scripts\python.exe`.
- Supported Python is 3.10+; the currently verified `.venv` uses Python 3.11.9.
- `comparison_env` is optional and isolated from the maintained application. Never substitute it for `.venv`.
- Use `git mv` for tracked moves. Preserve public names and compatibility unless a user explicitly approves breaking changes.

## Maintained entry points

```powershell
.\.venv\Scripts\python.exe -m multimodal_rag.cli.ingest --help
.\.venv\Scripts\python.exe -m multimodal_rag.cli.build_index --help
.\.venv\Scripts\python.exe -m multimodal_rag.cli.ask --help
.\.venv\Scripts\python.exe -m multimodal_rag.evaluation.question_runner --help
.\.venv\Scripts\python.exe -m multimodal_rag.evaluation.runner
.\.venv\Scripts\python.exe -m streamlit run src/multimodal_rag/ui/streamlit_app.py
```

Maintained optional tools live under `multimodal_rag.tools.comparison` and `multimodal_rag.tools.diagnostics`.

## Architecture invariants

- Ingestion flow: PyMuPDF load -> page pre-analysis -> Docling layout segmentation -> layout analysis -> routing -> native/OCR/conditional Vision extraction -> validation/cleaning -> structure-aware chunking -> artifacts.
- Measurement and policy are separate: analysis modules measure; `routing/routing_policy.py` decides; validators judge extraction quality.
- Embedding model identity is `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional normalized vectors.
- Embedding resolution is offline by default. `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` default to `1`, and `SentenceTransformer` uses `local_files_only=True`.
- FAISS uses `IndexIDMap2(IndexFlatIP)`; normalized inner product is treated as cosine similarity.
- The active retriever is `rag/retrieval/retriever_2.py`.
- Lexical tokens are lowercase `[a-z0-9-]+` terms minus the retriever's fixed stopword set.
- `lexical_overlap = |query_tokens intersect chunk_tokens| / |query_tokens|` (or `0` for an empty query-token set).
- `combined_rerank_score = raw_faiss_score + 0.15 * lexical_overlap`.
- Final order uses the combined score; `min_score` filtering uses the preserved raw FAISS score.
- Top-k is caller-specific: Streamlit Chat `5`; CLI ask default `8` and configurable; question-wise and batch evaluation `8`.
- Generation model is `gemini-3.1-flash-lite`.
- `run_rag_trace()` is shared by Streamlit Chat and question-wise evaluation. CLI ask and batch evaluation currently retain independent execution paths.
- RAGAS never runs during normal Chat.

## Evaluation invariants

- Ground truth is `evaluation/datasets/ground_truth.json` (currently 25 items).
- Default evaluator: Ollama `qwen2.5:7b`.
- Optional evaluator: Groq `openai/gpt-oss-20b`, selected by `EVALUATOR_PROVIDER=groq`; `GROQ_MODEL` may override the Groq model.
- Groq Faithfulness uses a dedicated direct model clone with `max_tokens=4096`.
- Groq Answer Correctness uses a separate direct model clone with `max_tokens=4096`.
- Groq Answer Relevancy uses `strictness=1`; Context Precision and Context Recall retain shared evaluator defaults.
- Ollama metric construction remains unchanged and shared.
- Question-wise evaluation is print/display-only and must never read completed IDs or call batch persistence functions.
- Batch evaluation must continue to skip completed provider-specific IDs, append each completed row, and regenerate the active report.
- Composite score is the unweighted mean of available metric values.

## Protected files and data

Treat these as read-only unless the user explicitly requests a data operation:

- `.env` and all credentials
- `.venv/` and `comparison_env/`
- `evaluation/datasets/ground_truth.json`
- `data/artifacts/index/faiss_index.bin`
- `data/artifacts/index/id_map.json`
- `data/artifacts/`, ingestion outputs, images, and embedding arrays
- `data/evaluation/results_*.csv`
- `data/evaluation/evaluation_report_*.md`
- `data/comparisons/`, `data/logs/`, and all other runtime data

Do not rebuild indexes, regenerate evaluation outputs, rewrite ground truth, or move/delete runtime artifacts during ordinary source, test, UI, or documentation work. For work that could affect them, snapshot hashes, sizes, timestamps, and counts first and compare afterward.

## Provider-call safety

- Do not call Gemini, Groq, RAGAS, Ollama evaluation, or any paid/quota-backed provider unless the user explicitly authorizes the exact live run.
- Before an authorized live evaluation, identify the selected record and expected number of retrieval, generation, and evaluator invocations.
- Question-wise evaluation must remain print-only unless a separately approved save mode is designed outside batch artifacts.
- Never print API keys. Inspect environment variable names or presence only.
- Offline tests and help commands must not trigger provider initialization.

## Change boundaries

Do not casually change:

- Prompts, model identities, temperature, token caps, retry policy, or provider selection
- Chunk size/overlap, heading logic, metadata schema, OCR/Vision routing, or validation thresholds
- Embedding normalization/dimension/model, FAISS type, index format, or serialized artifacts
- Retrieval formula, lexical weight, top-k, filtering, or final ranking
- RAGAS metrics, batch resume/persistence semantics, or ground-truth content
- Streamlit Chat/Evaluation isolation or one-call-per-question behavior

If a task requires one of these changes, reproduce the issue, report evidence, propose the smallest change, and obtain explicit approval where requested.

## Testing expectations

Use `.venv\Scripts\python.exe` for every check. The baseline command is:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Also run focused tests for changed behavior, canonical module `--help` checks, import smoke tests, `git diff --check`, and an artifact-integrity comparison when protected files could be touched. Provider calls must be mocked in automated tests. Expected Streamlit bare-mode warnings are not application failures.

## Cleanup and Git rules

- Never delete `.venv`, `comparison_env`, `.env`, `data/`, FAISS files, evaluation artifacts, or ground truth during cleanup.
- Repository-local `__pycache__`, `*.pyc`, logs, and temporary files may be removed only after verifying their exact paths and that no source/data is included.
- Avoid destructive Git commands. Preserve unrelated user changes.
- Do not stage, commit, push, tag, or rewrite history unless the user explicitly asks for that action.
- Before any approved commit, audit staged paths for secrets, environments, caches, logs, runtime data, evaluation outputs, and FAISS artifacts.
