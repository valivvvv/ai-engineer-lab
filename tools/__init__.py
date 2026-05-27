"""Tools package.

Importing this module triggers registration of every tool defined in submodules
via the @register_tool decorator. Add a new tool by creating a submodule here
and importing it below — no central manifest to keep in sync.
"""
from .registry import TOOL_REGISTRY, ToolWrapper, register_tool


def _register_builtin_tools() -> None:
    from . import calculator, current_datetime, web_search
    _ = calculator, current_datetime, web_search


_register_builtin_tools()

__all__ = ["TOOL_REGISTRY", "ToolWrapper", "register_tool"]