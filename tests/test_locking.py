"""Tests fuer Kalibrierungs-Nebenlaeufigkeit (P12/S10): dateibasiertes Lock."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rag_router.calibration import CalibrationProfile
from rag_router.locking import ensure_cached_calibration, file_lock


def profile(tag: str = "p") -> CalibrationProfile:
    return CalibrationProfile(
        decision_route="slm",
        answer_backend="slm",
        skip=0.8,
        fanout=0.5,
        answer=0.5,
        cascade_lo=0.4,
        cascade_hi=0.6,
        created_at=tag,
        sources={"tag": tag},
    )


class TestFileLock:
    def test_acquire_and_release(self, tmp_path) -> None:
        lock = tmp_path / "c.json.lock"
        with file_lock(lock, timeout=1.0) as acquired:
            assert acquired is True
            assert lock.exists()
        assert not lock.exists()

    def test_second_acquire_fails_while_held(self, tmp_path) -> None:
        lock = tmp_path / "c.json.lock"
        with file_lock(lock, timeout=1.0) as first:
            assert first is True
            with file_lock(lock, timeout=0.2, poll=0.02) as second:
                assert second is False, "gehaltenes Lock darf nicht doppelt vergeben"

    def test_stale_lock_taken_over(self, tmp_path) -> None:
        lock = tmp_path / "c.json.lock"
        lock.write_text("999999", encoding="utf-8")  # tote PID
        old = time.time() - 3600
        import os

        os.utime(lock, (old, old))
        with file_lock(lock, stale_after=10.0, timeout=1.0) as acquired:
            assert acquired is True

    def test_old_lock_with_live_pid_taken_over(self, tmp_path) -> None:
        """Haengender Prozess: lebende PID, aber Lock aelter als stale_after."""
        import os

        lock = tmp_path / "c.json.lock"
        lock.write_text(str(os.getpid()), encoding="utf-8")  # lebende PID
        old = time.time() - 3600
        os.utime(lock, (old, old))
        with file_lock(lock, stale_after=10.0, timeout=1.0) as acquired:
            assert acquired is True, "altes Lock muss per stale_after uebernommen werden"

    def test_live_pid_fresh_lock_not_taken_over(self, tmp_path) -> None:
        """Lebende PID + frisches Lock: nicht uebernehmen (kein Doppel-Lauf)."""
        import os

        lock = tmp_path / "c.json.lock"
        lock.write_text(str(os.getpid()), encoding="utf-8")
        with file_lock(lock, stale_after=3600.0, timeout=0.2, poll=0.05) as acquired:
            assert acquired is False

    def test_timeout_returns_false_small(self, tmp_path) -> None:
        lock = tmp_path / "c.json.lock"
        lock.write_text(str(__import__("os").getpid()), encoding="utf-8")
        start = time.monotonic()
        with file_lock(lock, stale_after=3600.0, timeout=0.3, poll=0.05) as acquired:
            assert acquired is False
        assert time.monotonic() - start < 2.0, "darf nicht blockieren"


class TestEnsureCachedCalibration:
    def test_existing_cache_skips_run(self, tmp_path) -> None:
        cache = tmp_path / "c.json"
        cache.write_text(
            __import__("json").dumps(profile("cached").to_dict()), encoding="utf-8"
        )
        calls = {"run": 0}

        def run():
            calls["run"] += 1
            return profile("fresh")

        result = ensure_cached_calibration(cache, run)
        assert calls["run"] == 0
        assert result is not None and result.created_at == "cached"

    def test_runs_once_and_writes_cache(self, tmp_path) -> None:
        cache = tmp_path / "c.json"
        calls = {"run": 0}
        result = ensure_cached_calibration(cache, lambda: (calls.__setitem__("run", calls["run"] + 1) or profile("fresh")))
        assert calls["run"] == 1
        assert result is not None and result.created_at == "fresh"
        assert cache.is_file()

    def test_concurrent_second_process_waits(self, tmp_path) -> None:
        """Zwei Prozesse, kalter Cache: genau eine Kalibrierung (S10)."""
        cache = tmp_path / "c.json"
        run_calls = {"n": 0}
        started = threading.Event()
        results: dict[str, CalibrationProfile | None] = {}

        def slow_run():
            started.set()
            time.sleep(0.3)
            run_calls["n"] += 1
            return profile("winner")

        def first():
            results["first"] = ensure_cached_calibration(cache, slow_run, wait=5.0)

        def second():
            # startet erst, wenn der erste die Messung begonnen hat
            started.wait(timeout=2.0)
            time.sleep(0.05)
            results["second"] = ensure_cached_calibration(
                cache, lambda: profile("loser"), wait=5.0
            )

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert run_calls["n"] == 1, f"nur eine Kalibrierung erwartet, war {run_calls['n']}"
        assert results["first"] is not None and results["first"].created_at == "winner"
        assert results["second"] is not None and results["second"].created_at == "winner"

    def test_run_failure_falls_back(self, tmp_path) -> None:
        cache = tmp_path / "c.json"
        result = ensure_cached_calibration(cache, lambda: None)
        assert result is None
        assert not cache.exists()

    def test_lock_unavailable_returns_none_without_blocking(self, tmp_path) -> None:
        cache = tmp_path / "c.json"
        lock = Path(str(cache) + ".lock")
        lock.write_text(str(__import__("os").getpid()), encoding="utf-8")
        start = time.monotonic()
        result = ensure_cached_calibration(
            cache, lambda: profile("x"), wait=0.3, poll=0.05
        )
        assert result is None
        assert time.monotonic() - start < 2.0
