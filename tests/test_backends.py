"""Tests fuer das LanceDB-Backend (P5) — mit Fake-Embedder, kein Modell."""

import pytest

from rag_router.backends.base import RagBackend, SearchHit
from rag_router.backends.lancedb import LanceDbBackend, registry

FAKE_VECTORS = {
    "Urlaubsantrag": {
        "text": "Urlaubsantrag: 30 Tage im Jahr, Rest verfaellt im Maerz.",
        "vector": [1.0, 0.0, 0.0, 0.0],
    },
    "Rueckgabe": {
        "text": "Rueckgabe: Artikel koennen 14 Tage unbenutzt zurueckgegeben werden.",
        "vector": [0.0, 1.0, 0.0, 0.0],
    },
    "API-Limit": {
        "text": "API-Limit: 600 Requests pro Minute Standardplan.",
        "vector": [0.0, 0.0, 1.0, 0.0],
    },
}


class FakeEmbedder:
    """Deterministischer Embedder: Frage->Vektor ueber aehnlichste Textzeile.

    Nicht realistisch, aber deterministisch und testet die Backlogik.
    """

    def __init__(self) -> None:
        self._next = 0

    def embed(self, texts: list[str]):
        out = []
        for text in texts:
            for entry in FAKE_VECTORS.values():
                if entry["text"] == text:
                    out.append(entry["vector"])
                    break
            else:
                # Unbekannter Text (Frage): aehnlichste Zeile anhand Wortueberlappung
                words = set(text.lower().split())
                best = None
                best_overlap = -1
                for entry in FAKE_VECTORS.values():
                    overlap = len(words & set(entry["text"].lower().split()))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best = entry["vector"]
                out.append(best or [0.0, 0.0, 0.0, 1.0])
        return out


def make_backend(tmp_path, mode: str) -> LanceDbBackend:
    """Backend auf einem tmp-LanceDB mit Fake-Embedder und 3 Dokumenten."""
    backend = LanceDbBackend(
        db_path=tmp_path / "probe.lance",
        table="policy_chunks",
        embedder=FakeEmbedder(),
        retriever=mode,
        rrf_k=60,
        vector_dim=4,  # Fake-Embedder liefert 4-dim
    )
    backend.index_texts(
        [entry["text"] for entry in FAKE_VECTORS.values()],
        ids=list(FAKE_VECTORS.keys()),
    )
    return backend


@pytest.fixture()
def hybrid(tmp_path):
    return make_backend(tmp_path, "hybrid")


class TestIndexing:
    def test_index_puts_rows(self, tmp_path) -> None:
        backend = make_backend(tmp_path, "dense")
        assert len(backend) == 3

    def test_index_is_idempotent_by_id(self, tmp_path) -> None:
        backend = make_backend(tmp_path, "dense")
        before = len(backend)
        backend.index_texts(
            [FAKE_VECTORS["Urlaubsantrag"]["text"]], ids=["Urlaubsantrag"]
        )
        assert len(backend) == before, (
            "gleiche id darf verdoppeln"
        )  # bewusst: id-basiert upsert

    def test_rag_backend_protocol(self, tmp_path) -> None:
        backend = make_backend(tmp_path, "hybrid")
        assert isinstance(backend, RagBackend)


class TestSearch:
    def test_dense_exact_match(self, tmp_path) -> None:
        backend = make_backend(tmp_path, "dense")
        hits = backend.search(
            "Rueckgabe: Artikel koennen 14 Tage unbenutzt zurueckgegeben werden.",
            top_k=2,
        )
        assert hits[0].doc_id == "Rueckgabe"
        assert hits[0].rag_key == ""
        assert isinstance(hits[0], SearchHit)

    def test_hybrid_returns_sorted(self, hybrid) -> None:
        hits = hybrid.search("Urlaubsantrag Restverfall", top_k=3)
        assert len(hits) <= 3
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True), "sortiert absteigend"
        assert hits[0].doc_id in FAKE_VECTORS

    def test_fts_keyword_hit(self, tmp_path) -> None:
        backend = make_backend(tmp_path, "fts")
        hits = backend.search("API-Limit Requests", top_k=3)
        assert hits[0].doc_id == "API-Limit"

    def test_top_k_limit_respected(self, hybrid) -> None:
        hits = hybrid.search("Urlaubsantrag", top_k=1)
        assert len(hits) == 1

    def test_hybrid_scores_detail(self, hybrid) -> None:
        hits = hybrid.search("Urlaubsantrag", top_k=2)
        assert hits[0].scores, "Score-Detail je Modus sollte gefuellt sein"

    def test_empty_table_search(self, tmp_path) -> None:
        backend = LanceDbBackend(
            db_path=tmp_path / "empty.lance",
            table="t",
            embedder=FakeEmbedder(),
            retriever="hybrid",
        )
        assert backend.search("irgendwas", top_k=3) == []


class TestRegistry:
    def test_lancedb_registered(self) -> None:
        assert "lancedb" in registry

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(KeyError):
            registry["nichtda"]  # type: ignore[index]
