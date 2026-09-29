"""Dateibasiertes Lock + kalibrierungs-Cache (P12/S10).

Ein einziger Prozess darf bei kaltem Cache kalibrieren; wartende Prozesse lesen
danach das Ergebnis. Nie blockierend in dem Sinn, dass der Router an der
Kalibrierung scheitert: ist das Lock nicht erreichbar, gibt es None und der
Aufrufer nutzt seinen dokumentierten Fallback.
"""

from __future__ import annotations

import errno
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from rag_router.calibration import CalibrationProfile

DEFAULT_STALE_AFTER = 1800.0
DEFAULT_WAIT = 120.0
DEFAULT_POLL = 0.1


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno == errno.EPERM  # existiert, aber fremder Nutzer
    return True


def _read_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


@contextmanager
def file_lock(
    path: Path,
    *,
    timeout: float = DEFAULT_WAIT,
    poll: float = DEFAULT_POLL,
    stale_after: float = DEFAULT_STALE_AFTER,
) -> Iterator[bool]:
    """Erwirbt ein Lock per atomarem O_CREAT|O_EXCL.

    Yields True bei Erfolg, False, wenn das Lock innerhalb von ``timeout``
    nicht erworben werden kann. Ein Lock, dessen PID nicht mehr lebt oder das
    aelter als ``stale_after`` ist, wird uebernommen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    acquired = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if (
                not _pid_alive(_read_pid(path))
                or time.time() - path.stat().st_mtime > stale_after
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                yield False
                return
            time.sleep(poll)
            continue
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        acquired = True
        break
    try:
        yield True
    finally:
        if acquired:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def ensure_cached_calibration(
    cache: Path,
    run: Callable[[], CalibrationProfile | None],
    *,
    wait: float = DEFAULT_WAIT,
    poll: float = DEFAULT_POLL,
    stale_after: float = DEFAULT_STALE_AFTER,
) -> CalibrationProfile | None:
    """Liefert das Profil: aus dem Cache oder genau einer Messung.

    - Cache vorhanden: lesen, nicht messen.
    - Sonst: Lock erwerben. Wer es hat, misst und schreibt. Wer wartet, liest
      danach das Ergebnis (und misst nicht selbst).
    - Lock nicht erreichbar: `run()` wird NICHT aufgerufen; None zurueck, damit
      der Aufrufer seinen Fallback nimmt (nie blockieren).
    """
    cache = Path(cache)

    def _load() -> CalibrationProfile | None:
        if not cache.is_file():
            return None
        try:
            return CalibrationProfile.from_dict(
                json.loads(cache.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, KeyError):
            return None

    cached = _load()
    if cached is not None:
        return cached

    lock = Path(str(cache) + ".lock")
    with file_lock(lock, timeout=wait, poll=poll, stale_after=stale_after) as acquired:
        if not acquired:
            return None
        # Zwischen Pruefung und Lock kann ein anderer Prozess geschrieben haben.
        cached = _load()
        if cached is not None:
            return cached
        profile = run()
        if profile is None:
            return None
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        return profile
