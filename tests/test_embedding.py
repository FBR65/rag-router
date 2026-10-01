"""Tests fuer den bge-m3-Embedder (Unit: Injektion, kein echtes Modell)."""

import numpy as np
import pytest

from rag_router.embedding import BgeM3Embedder, HttpEmbedder


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


class _Embedding:
    def __init__(self, index: int, values: list[float]) -> None:
        self.index = index
        self.embedding = values


class _Response:
    def __init__(self, data: list[_Embedding]) -> None:
        self.data = data


class _EmbeddingsApi:
    def __init__(self, seen: list[dict]) -> None:
        self._seen = seen

    def create(self, *, model: str, input: list[str]) -> _Response:
        self._seen.append({"model": model, "input": list(input)})
        # Absichtlich verdreht: der Client muss nach `index` sortieren.
        return _Response(
            [
                _Embedding(1, [1.0, 1.5]),
                _Embedding(0, [0.0, 0.5]),
            ]
        )


class _EmbeddingClient:
    def __init__(self) -> None:
        self.seen: list[dict] = []
        self.embeddings = _EmbeddingsApi(self.seen)


class TestHttpEmbedder:
    """bge-m3 ueber llama-swap: nur die Client-Anbindung, kein Netz."""

    def test_embed_posts_and_sorts_by_index(self) -> None:
        client = _EmbeddingClient()
        embedder = HttpEmbedder.from_stub(client)
        vectors = embedder.embed(["eins", "zwei"])
        assert vectors == [[0.0, 0.5], [1.0, 1.5]]
        assert client.seen == [{"model": "bge-m3-gguf", "input": ["eins", "zwei"]}]

    def test_embed_casts_to_float(self) -> None:
        client = _EmbeddingClient()
        embedder = HttpEmbedder.from_stub(client)
        assert all(isinstance(value, float) for value in embedder.embed(["x"])[0])

    def test_from_settings_needs_base_url(self) -> None:
        with pytest.raises(TypeError):
            HttpEmbedder.from_settings(model="bge-m3-gguf")  # type: ignore[call-arg]
