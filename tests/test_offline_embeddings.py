"""Focused offline-loading tests for the locked MiniLM embedding model."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_SHA256 = "416815f564438dd68fea9bf0af16b6229745be9bbadf2bebe4ca11cb8a64f15a"
ID_MAP_SHA256 = "879e2b734e38e57950ce8e1ddb1b119c6780d95fbb36babad4519210ede47c20"


class _FakeSentenceTransformer:
    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs):
        self.__class__.calls.append((args, kwargs))


class _FakeChat:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, _prompt):
        return "OK"


class _FakeEmbedding:
    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs):
        self.__class__.calls.append((args, kwargs))


class _IdentityWrapper:
    def __init__(self, value):
        self.value = value


def _fake_provider_modules() -> dict[str, types.ModuleType]:
    huggingface = types.ModuleType("langchain_huggingface")
    huggingface.HuggingFaceEmbeddings = _FakeEmbedding

    ollama = types.ModuleType("langchain_ollama")
    ollama.ChatOllama = _FakeChat

    groq = types.ModuleType("langchain_groq")
    groq.ChatGroq = _FakeChat

    ragas_llms = types.ModuleType("ragas.llms")
    ragas_llms.LangchainLLMWrapper = _IdentityWrapper
    ragas_embeddings = types.ModuleType("ragas.embeddings")
    ragas_embeddings.LangchainEmbeddingsWrapper = _IdentityWrapper

    groq_sdk = types.ModuleType("groq")
    for name in ("AuthenticationError", "RateLimitError", "APIConnectionError", "GroqError"):
        setattr(groq_sdk, name, type(name, (Exception,), {}))

    return {
        "langchain_huggingface": huggingface,
        "langchain_ollama": ollama,
        "langchain_groq": groq,
        "ragas.llms": ragas_llms,
        "ragas.embeddings": ragas_embeddings,
        "groq": groq_sdk,
    }


class OfflineEmbeddingTests(unittest.TestCase):
    def test_huggingface_offline_defaults_are_set_when_unset(self):
        env = os.environ.copy()
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import os; import multimodal_rag.rag.embedding.embedder; print(os.environ['HF_HUB_OFFLINE'], os.environ['TRANSFORMERS_OFFLINE'])"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "1 1")

    def test_huggingface_offline_explicit_values_are_preserved(self):
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "0"
        env["TRANSFORMERS_OFFLINE"] = "custom"
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import os; import multimodal_rag.rag.embedding.embedder; print(os.environ['HF_HUB_OFFLINE'], os.environ['TRANSFORMERS_OFFLINE'])"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "0 custom")

    def test_sentence_transformer_constructor_is_local_only(self):
        from multimodal_rag.rag.embedding import embedder

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = _FakeSentenceTransformer
        _FakeSentenceTransformer.calls.clear()
        old_model, old_name = embedder._model, embedder._model_name_loaded
        try:
            embedder._model = None
            embedder._model_name_loaded = None
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                embedder._get_model(embedder.EmbeddingConfig())
        finally:
            embedder._model, embedder._model_name_loaded = old_model, old_name

        self.assertEqual(len(_FakeSentenceTransformer.calls), 1)
        args, kwargs = _FakeSentenceTransformer.calls[0]
        self.assertEqual(args, (MODEL_NAME,))
        self.assertIsNone(kwargs["device"])
        self.assertTrue(kwargs["local_files_only"])

    def test_both_evaluator_embedding_constructors_are_local_only(self):
        from multimodal_rag.evaluation import runner

        modules = _fake_provider_modules()
        _FakeEmbedding.calls.clear()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            runner._build_ollama_llm_and_embeddings()
            runner._build_groq_llm_and_embeddings()

        self.assertEqual(len(_FakeEmbedding.calls), 2)
        for args, kwargs in _FakeEmbedding.calls:
            self.assertEqual(kwargs["model_name"], MODEL_NAME)
            self.assertEqual(kwargs["model_kwargs"], {"local_files_only": True})

    def test_cached_model_loads_and_encodes_without_hub_access(self):
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        code = (
            "from sentence_transformers import SentenceTransformer; "
            "m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', "
            "device='cpu', local_files_only=True); "
            "v=m.encode(['offline probe'], normalize_embeddings=True); "
            "assert tuple(v.shape)==(1,384); print(tuple(v.shape))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        self.assertIn("(1, 384)", result.stdout)

    def test_entry_points_keep_the_locked_embedding_configuration(self):
        from multimodal_rag.cli import ask as ask_cli
        from multimodal_rag.evaluation import runner
        from multimodal_rag.rag.embedding.embedder import EmbeddingConfig
        from multimodal_rag.ui import streamlit_app

        self.assertEqual(EmbeddingConfig().model_name, MODEL_NAME)
        self.assertEqual(ask_cli.retrieve.__module__, "multimodal_rag.rag.retrieval.retriever_2")
        self.assertEqual(streamlit_app.retrieve.__module__, "multimodal_rag.rag.retrieval.retriever_2")
        self.assertIn(MODEL_NAME, runner._build_ollama_llm_and_embeddings.__code__.co_consts)
        self.assertIn(MODEL_NAME, runner._build_groq_llm_and_embeddings.__code__.co_consts)

    def test_faiss_hashes_and_vector_counts_are_unchanged(self):
        import faiss

        index_dir = PROJECT_ROOT / "data" / "artifacts" / "index"
        faiss_path = index_dir / "faiss_index.bin"
        id_map_path = index_dir / "id_map.json"
        self.assertEqual(hashlib.sha256(faiss_path.read_bytes()).hexdigest(), FAISS_SHA256)
        self.assertEqual(hashlib.sha256(id_map_path.read_bytes()).hexdigest(), ID_MAP_SHA256)
        index = faiss.read_index(str(faiss_path))
        id_map = json.loads(id_map_path.read_text())
        self.assertEqual(index.ntotal, 111)
        self.assertEqual(index.d, 384)
        self.assertEqual(len(id_map), index.ntotal)


if __name__ == "__main__":
    unittest.main()
