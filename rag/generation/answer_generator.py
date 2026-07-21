"""
Answer Generation Module (RAG Stage 6)
==========================================

Calls Gemini with the prompt built in Stage 5 to produce the final
answer. Reuses the exact API-key/error-handling pattern already
established in ingestion/extractors/vision_describer.py - same
GEMINI_API_KEY env var, same "unavailable vs failed" exception split -
so this module behaves consistently with the rest of the project rather
than inventing a new convention.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AnswerGenerationError(Exception):
    """Raised when the Gemini call itself fails (bad response, API error, timeout)."""


class AnswerGenerationUnavailableError(AnswerGenerationError):
    """Raised when no API key is configured."""


@dataclass
class GenerationConfig:
    model_name: str = "gemini-3.1-flash-lite"
    temperature: float = 0.2
    # Low temperature by default: RAG answers should be grounded in the
    # provided sources, not creative - matches the same reasoning used
    # for the diagram-description prompt in vision_describer.py.


def _get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise AnswerGenerationUnavailableError(
            "GEMINI_API_KEY is not set in the environment - answer generation is unavailable."
        )
    return api_key


def generate_answer(prompt_text: str, config: GenerationConfig | None = None) -> str:
    config = config or GenerationConfig()
    api_key = _get_api_key()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.model_name,
            contents=prompt_text,
            config=types.GenerateContentConfig(temperature=config.temperature),
        )
        text = getattr(response, "text", None)
        if not text:
            raise AnswerGenerationError("Gemini returned a response with no text content.")
        return text.strip()
    except AnswerGenerationError:
        raise
    except Exception as e:
        raise AnswerGenerationError(f"Gemini answer generation failed: {e}") from e
