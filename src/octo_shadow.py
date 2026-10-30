"""Octo SHADOW mode: record what Octo would have been asked, never act on it.

TrustLoop (this laptop env) and Octo (isolated Python 3.10 env, usually Colab)
run in different environments, so shadow mode is a 3-step, file-based pipeline:

  1. export  (here)  : during normal TrustLoop runs, render the overview camera at
                       256x256 at fixed decision points and append one row per frame
                       to <dir>/manifest.jsonl (instruction, run/attempt/state, image
                       path, previous frame). Task outcomes go to <dir>/outcomes.jsonl.
  2. infer   (Colab) : octo_colab/octo_inference.py --manifest <dir>/manifest.jsonl
                       loads Octo ONCE and writes <dir>/octo_predictions.jsonl.
  3. merge   (here)  : python src/octo_shadow.py merge <dir>
                       joins predictions + frames + verified task outcomes into
                       <dir>/octo_shadow_merged.jsonl and octo_shadow_summary.json.

The robot is always driven by the scripted controller. Octo's actions are in a
training dataset's convention (e.g. Bridge/WidowX EEF deltas), which has NOT
been mapped to this arm, so they are logged for analysis only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulation import write_png  # noqa: E402

FRAME_TAGS = ("pre_approach", "grasp_closed", "pre_release", "final")


class OctoShadowRecorder:
    def __init__(self, out_dir: str | Path, size: int = 256, camera: str = "overview"):
        self.dir = Path(out_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.size, self.camera = size, camera
        self.manifest = self.dir / "manifest.jsonl"
        self.outcomes = self.dir / "outcomes.jsonl"
        self._renderer = None
        self._model = None
        self._prev: dict[str, str] = {}      # run_id -> previous frame rel path
        self.frames_written = 0
        self.disabled_reason: str | None = None

    def _render(self, sim) -> np.ndarray:
        import mujoco
        if self._renderer is None or self._model is not sim.model:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(sim.model, height=self.size, width=self.size)
            self._model = sim.model
        self._renderer.update_scene(sim.data, camera=self.camera)
        return self._renderer.render()

    def hook(self, task, tag: str) -> None:
        """frame_hook for TrustLoopTask. Rendering does not change the physics state."""
        if self.disabled_reason:
            return
        try:
            img = self._render(task.sim)
        except Exception as exc:
            self.disabled_reason = f"offscreen rendering unavailable: {exc!r}"
            task.log("warning", detail=f"Octo shadow disabled: {self.disabled_reason}")
            return
        run_id = task.log_.run_id
        frame_id = f"{run_id}_a{task.attempt}_{tag}"
        rel = f"frames/{frame_id}.png"
        write_png(self.dir / rel, img)
        row = {
            "frame_id": frame_id, "run_id": run_id, "mode": task.mode, "fault": task.fault,
            "attempt": task.attempt, "tag": tag, "state": task.state,
            "sim_time": round(task.sim.time, 3), "instruction": task.instruction,
            "image": rel, "prev_image": self._prev.get(run_id),
            "camera": self.camera, "image_size": [self.size, self.size],
        }
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self._prev[run_id] = rel
        self.frames_written += 1
        task.log("octo_shadow_frame", detail=rel, frame_id=frame_id, tag=tag)

    def record_outcome(self, summary: dict) -> None:
        keys = ("run_id", "mode", "fault_injected", "outcome", "verified_success", "false_success",
                "controller_reported_success", "attempts_used", "recovered", "failure_reasons", "reason")
        with self.outcomes.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: summary.get(k) for k in keys}, default=str) + "\n")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def merge(shadow_dir: str | Path, predictions: str | Path | None = None) -> dict:
    d = Path(shadow_dir)
    frames = _read_jsonl(d / "manifest.jsonl")
    outcomes = {o["run_id"]: o for o in _read_jsonl(d / "outcomes.jsonl")}
    pred_path = Path(predictions) if predictions else d / "octo_predictions.jsonl"
    preds = {p["frame_id"]: p for p in _read_jsonl(pred_path)}

    merged, latencies_warm, errors = [], [], 0
    for f in frames:
        p = preds.get(f["frame_id"])
        rec = {**f, "task_outcome": outcomes.get(f["run_id"]), "octo": None}
        if p is not None:
            rec["octo"] = {k: p.get(k) for k in (
                "stats_key", "seed", "window_size", "actions", "actions_normalized",
                "normalization_mask", "inference_seconds", "jit_compiled_this_call", "error",
                "checkpoint")}
            if p.get("error"):
                errors += 1
            elif not p.get("jit_compiled_this_call") and p.get("inference_seconds") is not None:
                latencies_warm.append(float(p["inference_seconds"]))
        merged.append(rec)

    out = d / "octo_shadow_merged.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")
    n_pred = sum(1 for r in merged if r["octo"] and not r["octo"].get("error"))
    summary = {
        "shadow_dir": str(d),
        "control_source": "scripted controller (Octo predictions NOT executed)",
        "frames": len(frames),
        "frames_with_prediction": n_pred,
        "frames_missing_prediction": sum(1 for r in merged if r["octo"] is None),
        "prediction_errors": errors,
        "runs": len({f["run_id"] for f in frames}),
        "runs_with_outcome": len([r for r in {f["run_id"] for f in frames} if r in outcomes]),
        "warm_latency_s": None if not latencies_warm else {
            "n": len(latencies_warm), "mean": round(statistics.mean(latencies_warm), 4),
            "median": round(statistics.median(latencies_warm), 4),
            "max": round(max(latencies_warm), 4)},
        "action_shape": next((list(np.shape(r["octo"]["actions"])) for r in merged
                              if r["octo"] and r["octo"].get("actions") is not None), None),
        "merged_file": str(out),
    }
    (d / "octo_shadow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Octo shadow-mode utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="join Octo predictions with frames and task outcomes")
    m.add_argument("shadow_dir")
    m.add_argument("--predictions", help="default: <shadow_dir>/octo_predictions.jsonl")
    args = ap.parse_args(argv)
    if args.cmd == "merge":
        print(json.dumps(merge(args.shadow_dir, args.predictions), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
