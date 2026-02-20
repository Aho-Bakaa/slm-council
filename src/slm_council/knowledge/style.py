"""Style profiling — ABCs for learning user code style.

A StyleExtractor analyses a codebase to build a StyleProfile that agents
can reference for consistent code generation.

Future implementations:
- AST-based style analysis (indentation, naming, docstring format)
- Import ordering preferences
- Architecture pattern detection (MVC, hexagonal, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StyleProfile:
    """Captured style preferences from a user's codebase."""

    naming_convention: str = ""  # snake_case, camelCase, PascalCase
    docstring_style: str = ""  # google, numpy, sphinx, none
    indent_style: str = ""  # spaces, tabs
    indent_size: int = 4
    max_line_length: int = 88
    import_style: str = ""  # absolute, relative
    type_hints: bool = True
    patterns: list[str] = field(default_factory=list)  # detected architecture patterns
    raw_observations: dict[str, Any] = field(default_factory=dict)


class StyleExtractor(ABC):
    """Abstract base for code style analysis."""

    @abstractmethod
    async def analyse(self, file_paths: list[str]) -> StyleProfile:
        """Analyse source files and return a style profile."""
        ...

    @abstractmethod
    def to_prompt_context(self, profile: StyleProfile) -> str:
        """Convert a style profile into a prompt string for agents."""
        ...


class NullStyleExtractor(StyleExtractor):
    """No-op implementation when style extraction is not configured."""

    async def analyse(self, file_paths: list[str]) -> StyleProfile:
        return StyleProfile()

    def to_prompt_context(self, profile: StyleProfile) -> str:
        return ""
