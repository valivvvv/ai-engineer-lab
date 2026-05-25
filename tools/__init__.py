"""Tools package.

Importing this module triggers registration of every tool defined in submodules
via the @register_tool decorator. Add a new tool by creating a submodule here
and importing it below — no central manifest to keep in sync.
"""
from .registry import TOOL_REGISTRY, ToolWrapper, register_tool

# Side-effect imports: each module's @register_tool decorator runs at import
# time, populating TOOL_REGISTRY. Order doesn't matter.
from . import calculator  # noqa: F401
from . import current_datetime  # noqa: F401
from . import web_search  # noqa: F401

__all__ = ["TOOL_REGISTRY", "ToolWrapper", "register_tool"]