"""Tool Registry — Pydantic-validated tools registered via @register_tool.

The decorator inspects each function's signature, extracts its Pydantic params
model and docstring, and stores them in TOOL_REGISTRY. ToolWrapper is the
public surface: call() for runtime execution, catalog() for LangChain binding.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError


_MIN_DESCRIPTION_LENGTH = 15


@dataclass(frozen=True)
class ToolEntry:
    name: str
    description: str
    params_model: type[BaseModel]
    func: Callable[[BaseModel], str]


TOOL_REGISTRY: dict[str, ToolEntry] = {}


def register_tool(func: Callable[[BaseModel], str]) -> Callable[[BaseModel], str]:
    """Register a tool function.

    Contract: the function takes exactly one parameter typed as a Pydantic
    BaseModel subclass, and has a docstring of at least 15 characters (used
    verbatim as the LLM-facing description).
    """
    name = func.__name__
    description = inspect.getdoc(func) or ""
    if len(description) < _MIN_DESCRIPTION_LENGTH:
        raise ValueError(
            f"Tool '{name}' needs a docstring of at least "
            f"{_MIN_DESCRIPTION_LENGTH} characters (used as the LLM-facing "
            f"description). Got: {len(description)}."
        )

    params = list(inspect.signature(func).parameters.values())
    if len(params) != 1:
        raise ValueError(
            f"Tool '{name}' must take exactly one parameter (got {len(params)})."
        )
    # get_type_hints() resolves string annotations produced by
    # `from __future__ import annotations`, returning the actual class.
    params_model = get_type_hints(func).get(params[0].name)
    if not (isinstance(params_model, type) and issubclass(params_model, BaseModel)):
        raise ValueError(
            f"Tool '{name}' parameter must be annotated with a Pydantic "
            f"BaseModel subclass (got {params_model!r})."
        )

    TOOL_REGISTRY[name] = ToolEntry(
        name=name,
        description=description,
        params_model=params_model,
        func=func,
    )
    return func


class ToolWrapper:
    """Adapter between the registry and its two consumers: the ReAct loop and
    LangChain. Errors are returned as strings (never raised) so the LLM can
    see them and self-correct.
    """

    @staticmethod
    def call(name: str, args: dict[str, Any]) -> str:
        entry = TOOL_REGISTRY.get(name)
        if entry is None:
            return f"Tool not found: '{name}'. Available: {sorted(TOOL_REGISTRY)}"
        try:
            params = entry.params_model(**args)
        except ValidationError as exc:
            return f"Invalid arguments for '{name}': {exc}"
        try:
            return str(entry.func(params))
        except Exception as exc:
            return f"Tool '{name}' raised {type(exc).__name__}: {exc}"

    @staticmethod
    def catalog() -> list[StructuredTool]:
        """Return LangChain StructuredTools ready for llm.bind_tools()."""
        return [_to_structured_tool(entry) for entry in TOOL_REGISTRY.values()]


def _to_structured_tool(entry: ToolEntry) -> StructuredTool:
    # LangChain calls the tool's func with kwargs taken from the LLM's JSON
    # response. Our registered functions take a single Pydantic model, so we
    # wrap them here. This keeps tool authors writing the cleaner Pydantic
    # style without imposing the kwargs convention on them.
    def kwargs_adapter(**kwargs: Any) -> str:
        return str(entry.func(entry.params_model(**kwargs)))

    kwargs_adapter.__name__ = entry.name
    kwargs_adapter.__doc__ = entry.description

    return StructuredTool.from_function(
        func=kwargs_adapter,
        name=entry.name,
        description=entry.description,
        args_schema=entry.params_model,
    )