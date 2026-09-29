"""Probe 2: exakte Signaturen fuer FTS + RRF in lancedb 0.39 (ohne Rateversuch)."""

import inspect

from lancedb.rerankers import RRFReranker

print("== RRFReranker ==")
print("sig:", inspect.signature(RRFReranker.__init__))
print("doc:", (RRFReranker.__doc__ or "")[:500])

from lancedb.table import Table

print("== Table.create_index ==")
print("sig:", inspect.signature(Table.create_index))
print("doc:", (Table.create_index.__doc__ or "")[:1500])

print("== Table.search ==")
print("sig:", inspect.signature(Table.search))

print("== QueryBuilder surface ==")
from lancedb.query import LanceQueryBuilder

qb_methods = [n for n in dir(LanceQueryBuilder) if not n.startswith("_")]
print("query builder methods:", qb_methods)

print("== tokenizer options ==")
import lancedb.index as li

print("index module:", [n for n in dir(li) if not n.startswith("_")])
