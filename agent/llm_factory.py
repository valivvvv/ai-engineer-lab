"""Builds LangChain ChatModel instances per provider.

Keeps vendor-specific construction details (env vars, package imports, kwargs)
out of the rest of the codebase. Callers receive a BaseChatModel and don't
need to care which vendor it's backed by.
"""
from __future__ import annotations

import os
from enum import StrEnum
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel


class Provider(StrEnum):
    OLLAMA = "ollama"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class LLMFactory:
    @staticmethod
    def create(
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Build a ChatModel for the requested provider.

        Resolution order: explicit args > env vars > sensible defaults.
        """
        resolved_provider = Provider(
            (provider or os.getenv("DEFAULT_PROVIDER", "ollama")).lower()
        )
        resolved_model = model or os.getenv("DEFAULT_MODEL")
        if not resolved_model:
            raise ValueError(
                "Model not specified and DEFAULT_MODEL is not set. "
                "Pass model= or set DEFAULT_MODEL in .env."
            )

        return _BUILDERS[resolved_provider](resolved_model, temperature, **kwargs)


def _build_ollama(model: str, temperature: float, **kwargs: Any) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model, base_url=base_url, temperature=temperature, **kwargs
    )


def _build_gemini(model: str, temperature: float, **kwargs: Any) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=_require_env("GOOGLE_API_KEY"),
        temperature=temperature,
        **kwargs,
    )


def _build_anthropic(model: str, temperature: float, **kwargs: Any) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        api_key=_require_env("ANTHROPIC_API_KEY"),
        temperature=temperature,
        **kwargs,
    )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set in .env")
    return value


_BUILDERS: dict[Provider, Callable[..., BaseChatModel]] = {
    Provider.OLLAMA: _build_ollama,
    Provider.GEMINI: _build_gemini,
    Provider.ANTHROPIC: _build_anthropic,
}
