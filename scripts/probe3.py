"""Probe 3: Hybrid-Row-Keys + merge_insert-API."""

import tempfile

import lancedb
import pyarrow as pa

db = lancedb.connect(tempfile.mkdtemp() + "/p3.lance")
schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 4)),
    ]
)
t = db.create_table("t", schema=schema)
t.add(
    [
        {
            "id": "1",
            "text": "Urlaubsantrag: 30 Tage im Jahr.",
            "vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "id": "2",
            "text": "Rueckgabe: 14 Tage Frist.",
            "vector": [0.0, 1.0, 0.0, 0.0],
        },
        {
            "id": "3",
            "text": "API-Limit 600 pro Minute.",
            "vector": [0.0, 0.0, 1.0, 0.0],
        },
    ]
)
t.create_index(
    "text", config=__import__("lancedb.index", fromlist=["FTS"]).FTS(language="German")
)

from lancedb.query import LanceHybridQueryBuilder
from lancedb.rerankers import RRFReranker

rows = (
    LanceHybridQueryBuilder(t)
    .text("Urlaubsantrag")
    .vector([1.0, 0.0, 0.0, 0.0])
    .rerank(RRFReranker(K=60))
    .limit(3)
    .to_list()
)
print("hybrid keys:", list(rows[0].keys()))
for r in rows:
    print(
        {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in r.items()
            if k != "vector"
        }
    )

print("merge_insert:", hasattr(t, "merge_insert"))
import inspect

if hasattr(t, "merge_insert"):
    print(inspect.signature(t.merge_insert))
    from lancedb.merge import LanceMergeInsertBuilder

    print([m for m in dir(LanceMergeInsertBuilder) if not m.startswith("_")])
