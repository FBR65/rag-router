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
