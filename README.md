# rag-router

Generischer RAG-Router: verteilt Fragen an eines von N frei konfigurierbaren
RAGs oder an „No Retrieval" (kein Suchen). Routing-Logik nach dem Artikel
„A RAG Router Built on Laya" (vishalmysore/layaAsRagJudge): zwei getrennte
Entscheidungen mit Wahrscheinlichkeiten — Skip-Schwelle (p ≥ 0.60 → nicht
suchen), KB-Wahl renormalisiert mit Fan-out (Führung < 0.55 → Top-2), danach
Answer Check (p(answered) ≥ 0.50).

- RAGs ausschließlich per YAML definiert (neues RAG = neuer Eintrag, kein Code)
- LanceDB pro RAG (diese Version), Retrieval dense/sparse/fts/hybrid,
  bge-m3 für dense + sparse, Hybrid-Fusion per RRF, optionaler Reranker
  (bge-reranker-v2-m3) aktiv
- Zwei Decision-Backends: Laya (convaiinnovations/laya-multilingual) und
  OpenAI-kompatible Endpunkte (Logprob-Abgleich), per Konfig umschaltbar
- Einbindung als Python-Library (`RagRouter.route()`); CLI nur für Tests

Status: Aufbau (Phase P1, Scaffold).