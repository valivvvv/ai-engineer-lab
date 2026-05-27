"""ReAct loop — Think → Act → Observe → Repeat.

The core primitive is `react_events()`: a generator that drives the loop and
yields TextChunk events as the LLM streams text, then a single terminating
Final event with the answer plus the full updated message list.

Two consumer shapes are built on top of it:
  - react_loop()       — returns ReActResult, discards TextChunk events.
  - QAAgent.stream()   — yields text chunks live, captures Final to update
                          history.

Both share the same loop logic, so behavior cannot drift between them.

An async mirror (react_events_async / react_loop_async) exposes the same
contract while parallelizing tool execution within a turn via asyncio.gather.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage

from tools import ToolWrapper


# --- Event types yielded by the core generator ------------------------------


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text content streamed from the LLM."""
    text: str


@dataclass(frozen=True)
class Final:
    """The terminating event: final answer plus the full updated message list."""
    answer: str
    messages: list[BaseMessage]


ReActEvent = TextChunk | Final


# --- Public result type returned by the returning wrappers ------------------


@dataclass(frozen=True)
class ReActResult:
    answer: str
    messages: list[BaseMessage]


# --- Core generators (the single source of truth for the loop) --------------


def react_events(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    max_iterations: int = 5,
) -> Iterator[ReActEvent]:
    """Drive the Think→Act→Observe loop, yielding events as they happen.

    Always uses llm.stream() under the hood so callers can opt into text
    streaming by filtering TextChunk events; callers that don't care can
    ignore everything except the terminating Final.
    """
    working_messages: list[BaseMessage] = list(messages)

    for _ in range(max_iterations):
        accumulated_message: AIMessageChunk | None = None

        for chunk in llm.stream(working_messages):
            accumulated_message = (
                chunk if accumulated_message is None else accumulated_message + chunk
            )
            for text in _extract_text(chunk):
                yield TextChunk(text=text)

        assert accumulated_message is not None, (
            "LLM returned zero stream chunks — likely a provider error."
        )
        working_messages.append(accumulated_message)

        if not accumulated_message.tool_calls:
            yield Final(
                answer=_flatten_message_content(accumulated_message),
                messages=working_messages,
            )
            return

        for tool_call in accumulated_message.tool_calls:
            tool_result = ToolWrapper.call(tool_call["name"], tool_call["args"])
            working_messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            )

    yield Final(
        answer=_iteration_cap_message_for_user(),
        messages=working_messages,
    )


async def react_events_async(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    max_iterations: int = 5,
) -> AsyncIterator[ReActEvent]:
    """Async mirror of react_events; parallelizes tool execution per turn."""
    working_messages: list[BaseMessage] = list(messages)

    for _ in range(max_iterations):
        accumulated_message: AIMessageChunk | None = None

        async for chunk in llm.astream(working_messages):
            accumulated_message = (
                chunk if accumulated_message is None else accumulated_message + chunk
            )
            for text in _extract_text(chunk):
                yield TextChunk(text=text)

        assert accumulated_message is not None, (
            "LLM returned zero stream chunks — likely a provider error."
        )
        working_messages.append(accumulated_message)

        if not accumulated_message.tool_calls:
            yield Final(
                answer=_flatten_message_content(accumulated_message),
                messages=working_messages,
            )
            return

        tool_results = await asyncio.gather(*[
            asyncio.to_thread(ToolWrapper.call, tool_call["name"], tool_call["args"])
            for tool_call in accumulated_message.tool_calls
        ])
        for tool_call, tool_result in zip(
            accumulated_message.tool_calls, tool_results
        ):
            working_messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            )

    yield Final(
        answer=_iteration_cap_message_for_user(),
        messages=working_messages,
    )


# --- Returning wrappers (consumers of the generators) -----------------------


def react_loop(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    max_iterations: int = 5,
) -> ReActResult:
    """Drive the ReAct loop and return the final answer + updated messages.

    Thin consumer of react_events; discards TextChunk events.
    """
    for event in react_events(llm, messages, max_iterations):
        if isinstance(event, Final):
            return ReActResult(answer=event.answer, messages=event.messages)
    raise RuntimeError("react_events did not yield a Final event")


async def react_loop_async(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    max_iterations: int = 5,
) -> ReActResult:
    """Async variant — parallelizes tool execution within a turn."""
    async for event in react_events_async(llm, messages, max_iterations):
        if isinstance(event, Final):
            return ReActResult(answer=event.answer, messages=event.messages)
    raise RuntimeError("react_events_async did not yield a Final event")


# --- Internal helpers -------------------------------------------------------


def _flatten_message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def _extract_text(message: BaseMessage) -> Iterator[str]:
    content = message.content
    if isinstance(content, str):
        if content:
            yield content
        return
    for block in content:
        if isinstance(block, dict):
            text = block.get("text", "")
            if text:
                yield text


def _iteration_cap_message_for_user() -> str:
    return (
        "I couldn't complete this request. Try rephrasing it, or breaking it "
        "into smaller steps."
    )