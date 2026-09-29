"""RED fuer P1: das Paket kann importiert und gebaut werden."""

import rag_router


def test_version_exists() -> None:
    """Das Paket meldet eine Version."""
    assert isinstance(rag_router.__version__, str)
    assert rag_router.__version__ == "0.1.0"
