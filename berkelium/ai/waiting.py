"""Visible, fail-fast waiting for a model server (ADR-025).

``check(wait_s=...)`` used to poll silently for up to 30 minutes — a server that crashed in its first second
(e.g. a LoRA rank above --max-lora-rank) looked exactly like a slow download. ``make_waiter`` prints a status line
every ``every`` seconds (elapsed, last error, last server-log line) and aborts IMMEDIATELY if the server process
named in ``pidfile`` has exited, printing the tail of its log.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .gateway import GatewayError


def tail(path: str | os.PathLike | None, n: int = 40) -> str:
    if not path or not Path(path).exists():
        return ""
    lines = Path(path).read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def pid_alive(pidfile: str | os.PathLike | None) -> bool | None:
    """True/False if the pidfile names a live/dead process; None if unknown (no pidfile yet)."""
    if not pidfile or not Path(pidfile).exists():
        return None
    try:
        pid = int(Path(pidfile).read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def make_waiter(url: str, pidfile=None, log=None, every: float = 30.0, out=None):
    out = out or sys.stderr
    last = [-1e9]

    def on_wait(elapsed: float, err: Exception) -> None:
        if pid_alive(pidfile) is False:
            raise GatewayError(f"model server exited while starting (see {log or 'its log'}).\n"
                               f"--- last log lines ---\n{tail(log, 40)}")
        if elapsed - last[0] >= every:
            last[0] = elapsed
            line = tail(log, 1).strip()
            msg = str(err).splitlines()[0][:120]
            print(f"[{elapsed / 60:.1f} min] waiting for {url}: {msg}" + (f" | log: {line[-160:]}" if line else ""),
                  file=out, flush=True)
    return on_wait
