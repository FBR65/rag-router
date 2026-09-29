"""DTOs und Protocol fuer RAG-Backends (P5)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchHit:
    """Ein gefundener Chunk mit Score-Detail je Modus."""

    rag_key: str
    doc_id: str
    text: str
    score: float
    scores: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class RagBackend(Protocol):
    """Ein durchsuchbares RAG (eine Wissensbasis)."""

    def search(
        self, question: str, top_k: int, modes: list[str] | None = None
    ) -> list[SearchHit]: ...

    def index_texts(self, texts: list[str], ids: list[str] | None = None) -> int: ...
