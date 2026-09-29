"""LanceDB-Backend (P5): dense + FTS + Hybrid (RRF) + optionales Reranking.

- Eine Tabelle pro RAG. Schema: id, text, vector (Float32-Liste).
- FTS: LanceDB-native (Tantivy) via `FTS`-Index (Sprache konfigurierbar;
  fuer Franks deutsches Set `language="German"`).
- Hybrid: LanceDB `RRFReranker(K=rrf_k)` ueber vector- + FTS-Rangliste
  (Artikel- und AurumVector-Logik: k=60).
- Sparse (bge-m3 lexical_weights): CSR-Impuls-Suche wie in AurumVector;
  wird als dritter Fusionsmodus unterstuetzt, wenn ein Sparse-Index angelegt
  wurde. In dieser Version default: dense + FTS (hybrid).
- Reranker (bge-reranker-v2-m3, FlagEmbedding): optional per `reranker=True`;
  scoret die RRF-Kandidaten wie in AurumVector (`compute_score(pairs,
  normalize=True)`) und sortiert neu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from rag_router.backends.base import SearchHit

VECTOR_DIM_DEFAULT = 1024  # bge-m3 dense
TEXT_COLUMN = "text"
ID_COLUMN = "id"
VECTOR_COLUMN = "vector"

_SPARSE_COLUMN = "sparse"


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field(ID_COLUMN, pa.string()),
            pa.field(TEXT_COLUMN, pa.string()),
            pa.field(VECTOR_COLUMN, pa.list_(pa.float32(), vector_dim)),
        ]
    )


# Registry: backend-type aus YAML -> Klasse
registry: dict[str, type] = {}


def register(name: str):
    def _wrap(cls):
        registry[name] = cls
        return cls

    return _wrap


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@register("lancedb")
class LanceDbBackend:
    """Ein RAG auf einer LanceDB-Tabelle."""

    def __init__(
        self,
        db_path: Path | str,
        table: str,
        embedder: Embedder,
        retriever: str = "hybrid",
        rrf_k: int = 60,
        fts_language: str = "German",
        reranker: Any = None,
        vector_dim: int = VECTOR_DIM_DEFAULT,
    ) -> None:
        self._db = lancedb.connect(str(db_path))
        self._table_name = table
        self._embedder = embedder
        self._retriever = retriever
        self._rrf_k = rrf_k
        self._fts_language = fts_language
        self._reranker = reranker
        self._vector_dim = vector_dim
        self._table = None

    # -- Indexierung -----------------------------------------------------

    def _ensure_table(self):
        if self._table is None:
            if self._table_name in self._db.table_names():
                self._table = self._db.open_table(self._table_name)
            else:
                self._table = self._db.create_table(
                    self._table_name, schema=_schema(self._vector_dim)
                )
        return self._table

    def _ensure_fts(self) -> None:
        table = self._table
        if table is None:
            return
        if self._retriever not in ("fts", "hybrid"):
            return
        # idempotent: replace=True
        table.create_index(TEXT_COLUMN, config=FTS(language=self._fts_language))

    def index_texts(self, texts: list[str], ids: list[str] | None = None) -> int:
        if not texts:
            return 0
        ids = ids if ids is not None else [str(i) for i in range(len(texts))]
        vectors = self._embedder.embed(texts)
        rows = [
            {
                ID_COLUMN: id_,
                TEXT_COLUMN: text,
                VECTOR_COLUMN: vector,
            }
            for id_, text, vector in zip(ids, texts, vectors)
        ]
        table = self._ensure_table()
        # Upsert: gleiche ID ersetzt die Zeile (idempotente Indexierung).
        (
            table.merge_insert(on=ID_COLUMN)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        self._ensure_fts()
        return len(rows)

    def __len__(self) -> int:
        table = self._ensure_table()
        return len(table)

    # -- Suche -----------------------------------------------------------

    def search(
        self,
        question: str,
        top_k: int,
        modes: list[str] | None = None,
    ) -> list[SearchHit]:
        table = self._ensure_table()
        if len(table) == 0:
            return []
        mode = self._retriever if modes is None else modes[0]
        query_vector = self._embedder.embed([question])[0]

        if mode == "dense":
            return self._search_dense(table, question, query_vector, top_k)
        if mode == "fts":
            return self._search_fts(table, question, top_k)
        if mode == "hybrid":
            return self._search_hybrid(table, question, query_vector, top_k)
        raise ValueError(f"unbekannter Retrieval-Modus: {mode}")

    @staticmethod
    def _hits_from_rows(rows: list[dict], rag_key: str, mode: str) -> list[SearchHit]:
        hits = []
        for row in rows:
            scores = {}
            if "_relevance_score" in row:
                scores[mode] = float(row["_relevance_score"])  # hybrid/RRF + FTS
            elif "_relevance" in row:
                scores[mode] = float(row["_relevance"])  # FTS (ältere Form)
            if "_distance" in row:
                scores[mode] = -float(row["_distance"])  # dense
            if not scores:
                scores[mode] = 0.0
            hit_score = scores[mode]
            hits.append(
                SearchHit(
                    rag_key=rag_key,
                    doc_id=str(row[ID_COLUMN]),
                    text=row[TEXT_COLUMN],
                    score=hit_score,
                    scores=scores,
                )
            )
        return hits

    def _search_dense(
        self, table, question: str, vector: list[float], top_k: int
    ) -> list[SearchHit]:
        rows = (
            table.search(vector, vector_column_name=VECTOR_COLUMN)
            .limit(top_k)
            .to_list()
        )
        return self._hits_from_rows(rows, "", "dense")

    def _search_fts(self, table, question: str, top_k: int) -> list[SearchHit]:
        rows = table.search(query_type="fts", query=question).limit(top_k).to_list()
        return self._hits_from_rows(rows, "", "fts")

    def _search_hybrid(
        self, table, question: str, vector: list[float], top_k: int
    ) -> list[SearchHit]:
        from lancedb.query import LanceHybridQueryBuilder

        builder = LanceHybridQueryBuilder(table)
        builder = builder.text(question).vector(vector)
        rows = builder.rerank(RRFReranker(K=self._rrf_k)).limit(top_k).to_list()
        return self._hits_from_rows(rows, "", "hybrid")
