"""Probe: lancedb 0.39 API — FTS, hybrid, RRF, sparse; ohne Model-Download."""

import tempfile

import lancedb
import pyarrow as pa

print("lancedb", lancedb.__version__)

db_dir = tempfile.mkdtemp() + "/probe.lance"
db = lancedb.connect(db_dir)

rows = [
    {
        "id": "1",
        "text": "Urlaubsantrag: 30 Tage im Jahr, Rest verfaellt im Maerz.",
        "vector": [0.1, 0.2, 0.3, 0.4],
    },
    {
        "id": "2",
        "text": "Rueckgabe: Artikel koennen 14 Tage unbenutzt zurueckgegeben werden.",
        "vector": [0.9, 0.1, 0.4, 0.2],
    },
    {
        "id": "3",
        "text": "Sturm Elin richtete Schaden an der Kueste an.",
        "vector": [0.2, 0.8, 0.1, 0.5],
    },
]
schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 4)),
    ]
)
table = db.create_table("probe", data=rows, schema=schema)
print("table created:", table.name)

# --- Vector search ---
res = table.search([0.1, 0.2, 0.3, 0.4], vector_column_name="vector").limit(2).to_list()
print("vector top1:", res[0]["id"], round(res[0].get("_distance", -1), 3))

# --- FTS index ---
try:
    table.create_index("FTS", config=lancedb.index.FtsIndexConfig())
    print("FTS index created")
except Exception as e:
    print("FTS create_index failed:", type(e).__name__, e)

for qtype in ("fts", "hybrid"):
    try:
        res = (
            table.search(query_type=qtype, query="Rueckgabe 14 Tage").limit(3).to_list()
        )
        print(
            f"{qtype} ok:",
            [(r.get("id"), r.get("_relevance") or r.get("_distance")) for r in res],
        )
    except Exception as e:
        print(f"{qtype} failed:", type(e).__name__, e)

# --- RRF reranker ---
try:
    from lancedb.rerankers import RRFReranker

    res = (
        table.search()
        .text("Rueckgabe 14 Tage")
        .vector([0.9, 0.1, 0.4, 0.2])
        .rerank(RRFReranker(k=60))
        .limit(2)
        .to_list()
    )
    print("hybrid+RRF ok:", [r["id"] for r in res])
except Exception as e:
    print("hybrid+RRF failed:", type(e).__name__, str(e)[:200])

# --- sparse? ---
print("has SparseVector:", hasattr(lancedb, "SparseVector"))
try:
    from lancedb.vectors import SparseVector  # noqa: F401

    print("sparse importable from lancedb.vectors")
except Exception as e:
    print("sparse import failed:", type(e).__name__)

# --- schema introspection / add ---
try:
    table.add(
        [
            {
                "id": "4",
                "text": "API-Limit: 600 Requests pro Minute.",
                "vector": [0.5, 0.5, 0.5, 0.1],
            }
        ]
    )
    print("add ok, rows:", len(table))
except Exception as e:
    print("add failed:", type(e).__name__, str(e)[:200])
