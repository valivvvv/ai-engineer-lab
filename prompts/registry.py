"""Prompt Registry — versioned YAML prompts rendered with Jinja2.

Prompts are treated as configuration: edit a .yaml file, reload the registry,
no Python changes required. Each template is loaded once and cached as an
immutable PromptTemplate; render() applies runtime variables via Jinja2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined


_DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"
_FAIL_ON_MISSING_VARIABLES = StrictUndefined


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    prompt: str
    description: str = ""
    variables: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
    """Loads and renders YAML prompt templates from a directory tree."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        self._templates_dir = templates_dir or _DEFAULT_TEMPLATES_DIR
        self._jinja = Environment(undefined=_FAIL_ON_MISSING_VARIABLES)
        self._templates: dict[str, PromptTemplate] = {}
        self.reload()

    def reload(self) -> None:
        """Re-scan the templates directory. Useful after editing a .yaml file."""
        self._templates.clear()
        for path in self._templates_dir.rglob("*.yaml"):
            template = self._load_one(path)
            self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:
            raise KeyError(
                f"Prompt '{name}' not found. Available: {self.list_templates()}"
            ) from exc

    def render(self, name: str, **variables: Any) -> str:
        template = self.get(name)
        return self._jinja.from_string(template.prompt).render(**variables)

    def list_templates(self) -> list[str]:
        return sorted(self._templates)

    @staticmethod
    def _load_one(path: Path) -> PromptTemplate:
        data = yaml.safe_load(path.read_text())
        return PromptTemplate(
            name=data["name"],
            version=data["version"],
            prompt=data["prompt"],
            description=data.get("description", ""),
            variables=tuple(data.get("variables", [])),
            metadata=data.get("metadata", {}),
        )


_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Lazy singleton accessor — loads templates on first call, then caches."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
