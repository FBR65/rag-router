"""bge-m3 Embedding-Modul (P8-Grundlage). Ein Modell fuer dense + sparse."""

from __future__ import annotations

from typing import Any


class BgeM3Embedder:
    """Dense-Embedder auf FlagEmbedding (bge-m3).

    Ein Modell liefert dense Vektoren (1024 d); die sparse lexical_weights
    des Modells brauchte AurumVector als CSR — fuer dieses Release nutzt das
    LanceDB-Hybrid FTS statt sparse-Overlays (AurumVector-Option bleibt
    moeglich, FTS-Semantik ist identisch).
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        stub: Any = None,
    ) -> None:
        if model_name is None and stub is None:
            raise ValueError("model_name oder stub erforderlich")
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = stub
        if self._model is None:
            assert isinstance(model_name, str), "model_name oder stub erforderlich"
            self._model = self._load(model_name, device)

    @classmethod
    def from_stub(cls, stub: Any, *, batch_size: int = 16) -> BgeM3Embedder:
        """Aus injiziertem Stub (Tests)."""
        return cls(stub=stub, batch_size=batch_size)

    @staticmethod
    def _load(model_name: str | None, device: str) -> Any:
        from FlagEmbedding import BGEM3FlagModel

        return BGEM3FlagModel(model_name, use_fp16=False, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Texte -> dense Vektoren (list[list[float]])."""
        output = self._model.encode(texts, batch_size=self._batch_size)
        import numpy as np

        dense = output.get("dense_vecs") if isinstance(output, dict) else output
        return [row.tolist() for row in np.asarray(dense)]


class HttpEmbedder:
    """bge-m3 ueber einen OpenAI-kompatiblen Endpunkt (`/v1/embeddings`).

    Nutzt das Modell im llama-swap-Container (llama-server mit `--embeddings`),
    statt torch/FlagEmbedding im Prozess zu laden. Gleiches Modell, gleiche
    Vektor-Dimension (1024); die API liefert bereits normierte Vektoren.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "bge-m3-gguf",
        api_key: str = "not-needed",
        timeout: float = 300.0,
        client: Any = None,
    ) -> None:
        self._model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
            )
        self._client = client

    @classmethod
    def from_settings(
        cls, *, base_url: str, model: str, api_key: str = "not-needed"
    ) -> HttpEmbedder:
        return cls(base_url=base_url, model=model, api_key=api_key)

    @classmethod
    def from_stub(cls, stub: Any, *, model: str = "bge-m3-gguf") -> HttpEmbedder:
        """Aus injiziertem OpenAI-Stub (Tests)."""
        return cls(base_url="http://stub.invalid/v1", model=model, client=stub)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        data = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in data]
