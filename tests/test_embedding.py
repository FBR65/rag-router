"""Tests fuer den bge-m3-Embedder (Unit: Injektion, kein echtes Modell)."""

import numpy as np
import pytest

from rag_router.embedding import BgeM3Embedder


class StubFlagModel:
    """FlagModel-Nachbau: encode(texts) -> deterministische Vektoren."""

    def __init__(self, texts_seen: list | None = None) -> None:
        self.texts_seen = texts_seen if texts_seen is not None else []

    def encode(self, texts, batch_size=None, **kw):
        self.texts_seen.extend(texts)
        return np.array(
            [[float(len(t) % 7), 0.1, 0.2, 0.3] for t in texts],
            dtype=np.float32,
        )


class TestBgeM3Embedder:
    def test_embed_delegates_and_casts(self) -> None:
        stub = StubFlagModel()
        embedder = BgeM3Embedder.from_stub(stub)
        vectors = embedder.embed(["kurz", "etwas laenger"])
        assert len(vectors) == 2
        assert vectors[0][0] == 4.0  # len('kurz') % 7 = 4
        assert vectors[1][0] == 6.0  # len('etwas laenger') % 7 = 6
        assert stub.texts_seen == ["kurz", "etwas laenger"]

    def test_batch_size_reached(self) -> None:
        stub = StubFlagModel()
        embedder = BgeM3Embedder.from_stub(stub, batch_size=2)
        embedder.embed(["a", "b", "c"])
        assert stub.texts_seen == ["a", "b", "c"]

    def test_requires_settings_or_stub(self) -> None:
        with pytest.raises(ValueError):
            BgeM3Embedder(model_name=None, stub=None)  # type: ignore[arg-type]
