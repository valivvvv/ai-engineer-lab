"""Single source of truth for which provider/model each role uses.

A "role" is a job the app needs an LLM for — the conversational agent versus
structured extraction — and each role has its own provider/model pair,
overridable via env vars. Resolution mirrors LLMFactory: explicit env var >
role default. LLMFactory owns *how* a client is built; this module owns
*which* provider/model each role gets.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from agent.llm_factory import LLMFactory


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model: str


AGENT = ModelChoice(provider="ollama", model="llama3.2:3b")
EXTRACTION = ModelChoice(provider="gemini", model="gemini-2.5-flash")


def build_model(
    choice: ModelChoice, temperature: float = 0.0, **kwargs
) -> BaseChatModel:
    """Build a ChatModel for the given role choice via LLMFactory."""
    return LLMFactory.create(
        provider=choice.provider,
        model=choice.model,
        temperature=temperature,
        **kwargs,
    )
