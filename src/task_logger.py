"""Structured (JSON Lines) task logging."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


class TaskLogger:
    """Writes one JSON object per line to logs/<run_id>.jsonl and echoes a short
    human-readable line to the console."""

    def __init__(self, log_dir: str | Path, run_id: str | None = None, echo: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"{self.run_id}.jsonl"
        self.echo = echo
        self._fh = self.path.open("w", encoding="utf-8")
        self.events: list[dict] = []

    def log(self, event: str, sim_time: float, attempt: int, **fields) -> dict:
        rec = {"event": event, "sim_time": round(sim_time, 3), "attempt": attempt,
               "wall_time": round(time.time(), 3), **fields}
        self._fh.write(json.dumps(rec, default=_jsonable) + "\n")
        self._fh.flush()
        self.events.append(rec)
        if self.echo:
            self._print(rec)
        return rec

    def _print(self, rec: dict) -> None:
        ev, a, t = rec["event"], rec["attempt"], rec["sim_time"]
        head = f"[t={t:6.2f}s a{a}]"
        if ev == "state":
            print(f"{head} STATE {rec['state']:<24} {rec.get('detail', '')}")
        elif ev == "verification":
            r = rec["result"]
            mark = "PASS" if r["success"] else "FAIL"
            fails = [k for k, v in r["checks"].items() if not v]
            extra = f" failed_checks={fails}" if fails else ""
            print(f"{head} VERIFY {rec['kind']:<16} {mark}: {r['reason']}{extra}")
        elif ev == "recovery":
            print(f"{head} RECOVERY failure={rec['failure']} -> action={rec['action']} "
                  f"(source={rec['source']}) {rec.get('note', '')}")
        elif ev == "outcome":
            print(f"{head} OUTCOME {rec['outcome']}: {rec['reason']}")
        elif ev == "fault_injected":
            print(f"{head} FAULT  {rec['fault']}: {rec['detail']}")
        elif ev in ("task_start", "task_end"):
            pass
        else:
            print(f"{head} {ev.upper()} {rec.get('detail', '')}")

    def write_summary(self, summary: dict) -> Path:
        path = self.log_dir / f"{self.run_id}_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
        return path

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
