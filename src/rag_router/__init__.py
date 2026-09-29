"""rag-router: generischer RAG-Router auf Laya-Basis."""

__version__ = "0.1.0"


def main(argv: list[str] | None = None) -> int:
    """CLI-Entry (siehe rag_router.cli)."""
    from rag_router.cli import main as cli_main

    return cli_main(argv)
