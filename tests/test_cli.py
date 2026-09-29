"""Tests fuer die CLI (P7): route, ask, index mit fakes ueber entry-points."""

import pytest

from rag_router.cli import build_parser, main


class TestParser:
    def test_route_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["route", "--config", "c.yaml", "Wie viele Tage?"])
        assert args.command == "route"
        assert args.config == "c.yaml"
        assert args.question == "Wie viele Tage?"

    def test_ask_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["ask", "--config", "c.yaml", "Frage"])
        assert args.command == "ask"
        assert args.top_k is None

    def test_ask_top_k(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["ask", "--config", "c.yaml", "--top-k", "2", "Frage"])
        assert args.top_k == 2

    def test_no_command_is_error(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestMain:
    def test_unknown_config_exits_clean_error(self, capsys) -> None:
        code = main(["route", "--config", "/nicht/da.yaml", "F"])
        assert code == 2
        out = capsys.readouterr().out
        assert "nicht gefunden" in out

    def test_json_output_route(self, monkeypatch, tmp_path, capsys) -> None:
        # Injiziere fakes ueber den Injection-Hook der CLI
        from rag_router.config import load_config

        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(
            "router:\n  decision_backend: laya\nrags:\n  policy:\n"
            '    description: "Richtlinien"\n    backend:\n      db: ./d\n      table: t\n',
            encoding="utf-8",
        )

        class FakeDecision:
            def decide(self, question, rag_descriptions):
                from rag_router.decision.base import RouteDist

                return RouteDist(probabilities={"policy": 0.1, "none": 0.9})

        def fake_loader(config, **kw):
            from rag_router.pipeline import RagRouter as RealRagRouter

            class FakeBackend:
                def index_texts(self, texts, ids=None):
                    return len(texts)

                def search(self, question, top_k, modes=None):
                    return []

            return RealRagRouter.injected(
                config=config,
                decision=FakeDecision(),
                backends={"policy": FakeBackend()},
                checker=None,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(
            "rag_router.cli._load",
            lambda config_path, **kw: fake_loader(load_config(cfg_path)),
        )

        code = main(["route", "--config", str(cfg_path), "Danke!"])
        assert code == 0
        out = capsys.readouterr().out
        assert '"reason": "skip"' in out
        assert '"none"' in out
        assert '"distribution"' in out
        assert '"final"' not in out  # route ohne Fetch
