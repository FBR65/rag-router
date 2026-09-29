"""CLI (P7): dunner Client der Library fuer manuelle Tests.

Befehle: route, ask, index. JSON-Ausgabe, Exit-Codes 0/1/2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-router", description="RAG-Router CLI (manuelles Testen)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, extra in (
        ("route", {}),
        ("ask", {"--top-k": int}),
    ):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("question")
        if name == "ask":
            cmd.add_argument(
                "--top-k", type=int, default=None, help="Override fuer Top-K"
            )
    index_parser = sub.add_parser("index")
    index_parser.add_argument("--config", required=True)
    index_parser.add_argument("--rag", required=True)
    index_parser.add_argument("--texts-file", required=True)
    return parser


def _load(config_path: str, **kwargs: Any) -> Any:
    """Injection-Hook (Tests ueberschreiben die statische Methode)."""
    from rag_router.config import load_config
    from rag_router.pipeline import RagRouter

    config = load_config(Path(config_path))
    return RagRouter.from_config(config, **kwargs)


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        router = _load(args.config)
        if args.command == "route":
            decision = router.route(args.question)
            _emit(
                {
                    "question": args.question,
                    "routes": decision.routes,
                    "reason": decision.reason,
                    "distribution": dict(decision.distribution),
                }
            )
        elif args.command == "ask":
            result = router.route_and_fetch(args.question, top_k=args.top_k)
            payload = {
                "question": args.question,
                "routes": result.decision.routes,
                "reason": result.decision.reason,
                "final": result.final,
                "hits": [
                    {
                        "rag_key": hit.rag_key,
                        "doc_id": hit.doc_id,
                        "text": hit.text,
                        "score": hit.score,
                    }
                    for hit in result.hits
                ],
            }
            if result.check is not None:
                payload["p_answered"] = result.check.p_answered
            _emit(payload)
        elif args.command == "index":
            count = router.index_texts_from_file(args.rag, args.texts_file)
            _emit({"indexed": count})
    except Exception as error:  # noqa: BLE001 — CLI-Fehlergrenze
        message = str(error)
        _emit({"error": message})
        if "nicht gefunden" in message or "fehlt" in message:
            return 2
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
