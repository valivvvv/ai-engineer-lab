"""QAAgent — the public chat interface.

Wires together the LLM factory, the prompt registry, and the tool registry,
and drives the ReAct loop on each chat() call. Maintains conversation history
across calls; clear_history() resets it back to the system prompt only.
"""
from __future__ import annotations

from typing import AsyncIterator, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from agent.llm_factory import LLMFactory
from agent.react import (
    Final,
    TextChunk,
    react_events,
    react_events_async,
    react_loop,
    react_loop_async,
)
from prompts.registry import get_prompt_registry
from tools import ToolWrapper


_DEFAULT_PROMPT_NAME = "qa_system"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_ITERATIONS = 5


class QAAgent:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        prompt_name: str = _DEFAULT_PROMPT_NAME,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        assistant_name: str = "QA Assistant",
        language: str | None = None,
    ) -> None:
        self._max_iterations = max_iterations

        tool_catalog = ToolWrapper.catalog()
        self._llm: BaseChatModel = LLMFactory.create(
            provider=provider,
            model=model,
            temperature=temperature,
        ).bind_tools(tool_catalog)

        system_prompt = get_prompt_registry().render(
            prompt_name,
            **self._build_render_variables(assistant_name, tool_catalog, language),
        )
        self._system_message = SystemMessage(content=system_prompt)
        self._history: list[BaseMessage] = [self._system_message]

    @staticmethod
    def _build_render_variables(
        assistant_name: str,
        tool_catalog: list[StructuredTool],
        language: str | None,
    ) -> dict[str, object]:
        variables: dict[str, object] = {
            "assistant_name": assistant_name,
            "tools": [
                {"name": tool.name, "description": tool.description}
                for tool in tool_catalog
            ],
        }
        if language is not None:
            variables["language"] = language
        return variables

    def chat(self, message: str) -> str:
        """Send a message, run the ReAct loop, return the final answer."""
        working_messages = self._history + [HumanMessage(content=message)]
        result = react_loop(
            self._llm,
            working_messages,
            max_iterations=self._max_iterations,
        )
        self._history = result.messages
        return result.answer

    def stream(self, message: str) -> Iterator[str]:
        """Send a message, run the ReAct loop, yield the final answer in chunks.

        Tools execute synchronously between LLM calls. Text content streams as
        it arrives; chunks during a tool-call decision phase typically have
        empty content, so nothing leaks during silent reasoning.
        """
        working_messages = self._history + [HumanMessage(content=message)]
        for event in react_events(
            self._llm,
            working_messages,
            max_iterations=self._max_iterations,
        ):
            if isinstance(event, TextChunk):
                yield event.text
            elif isinstance(event, Final):
                self._history = event.messages
                return

    async def chat_async(self, message: str) -> str:
        """Async variant of chat(); tools within a turn run in parallel."""
        working_messages = self._history + [HumanMessage(content=message)]
        result = await react_loop_async(
            self._llm,
            working_messages,
            max_iterations=self._max_iterations,
        )
        self._history = result.messages
        return result.answer

    async def stream_async(self, message: str) -> AsyncIterator[str]:
        """Async variant of stream(); tools within a turn run in parallel."""
        working_messages = self._history + [HumanMessage(content=message)]
        async for event in react_events_async(
            self._llm,
            working_messages,
            max_iterations=self._max_iterations,
        ):
            if isinstance(event, TextChunk):
                yield event.text
            elif isinstance(event, Final):
                self._history = event.messages
                return

    def clear_history(self) -> None:
        """Reset conversation, keeping only the system prompt."""
        self._history = [self._system_message]

    @property
    def history(self) -> list[BaseMessage]:
        """Read-only snapshot of the current conversation history."""
        return list(self._history)