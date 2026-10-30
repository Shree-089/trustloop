"""Reproducible TrustLoop experiments.

Experiments (same controller, same seeded cube poses, same injected faults):
  baseline        scripted controller; success = controller completed (verifier = audit only)
  verify_only     verifier gates; no recovery (1 attempt)
  verify_recover  verifier gates + bounded deterministic recovery (<=3 attempts)
  octo_shadow     verify_recover + Octo observation export (predictions added later, see octo_shadow.py)
  nemotron        verify_recover + Nemotron advisor (needs NEBIUS_API_KEY + NEMOTRON_MODEL;
                  if unavailable the experiment is SKIPPED and the reason recorded, never faked)

Every number in the output is computed from the runs it just executed.

  python src/experiments.py --trials 10 --seed 0
  python src/experiments.py --experiments baseline,verify_recover --conditions none,grasp_miss --trials 5
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulation import PROJECT_ROOT, Simulation  # noqa: E402
from task_logger import TaskLogger  # noqa: E402
from task_manager import DEFAULT_INSTRUCTION, FAULTS, TrustLoopTask  # noqa: E402

EXPERIMENTS = {
    "baseline":       {"mode": "baseline"},
    "verify_only":    {"mode": "verify_only"},
    "verify_recover": {"mode": "verify_recover"},
    "octo_shadow":    {"mode": "verify_recover", "octo_shadow": True},
    "nemotron":       {"mode": "verify_recover", "advisor": "nemotron"},
}
DEFAULT_CONDITIONS = ["none", "grasp_miss", "weak_grip", "place_offset", "rim", "unrecoverable"]

CSV_FIELDS = ["experiment", "mode", "condition", "trial", "seed", "cube_x", "cube_y", "cube_yaw",
              "outcome", "controller_reported_success", "verified_success", "false_success",
              "attempts_used", "first_attempt_verified", "recovery_actions", "recovered",
              "failure_reasons", "advisor_calls", "advisor_ok", "advisor_fallbacks",
              "advisor_latency_mean_s", "octo_frames", "sim_time_s", "wall_time_s", "reason", "run_id"]


def cube_pose(seed: int, trial: int) -> tuple[list[float], float]:
    rng = np.random.default_rng([seed, trial])
    return ([float(rng.uniform(0.32, 0.48)), float(rng.uniform(-0.30, -0.12)), 0.42],
            float(rng.uniform(-np.pi / 4, np.pi / 4)))


def _rate(rows, key) -> float | None:
    return None if not rows else round(sum(bool(r[key]) for r in rows) / len(rows), 3)


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    first_fail = [r for r in rows if not r["first_attempt_verified"]]
    lat = [r["advisor_latency_mean_s"] for r in rows if r["advisor_latency_mean_s"] is not None]
    return {
        "n": len(rows),
        "reported_success_rate": _rate(rows, "outcome_success"),
        "verified_success_rate": _rate(rows, "verified_success"),
        "false_success_count": sum(bool(r["false_success"]) for r in rows),
        "controller_reported_success_rate": _rate(rows, "controller_reported_success"),
        "mean_attempts": round(statistics.mean(r["attempts_used"] for r in rows), 3),
        "first_attempt_failures": len(first_fail),
        "runs_with_recovery_actions": sum(r["recovery_actions"] > 0 for r in rows),
        "recovered": sum(bool(r["recovered"]) for r in rows),
        "recovery_success_rate": (round(sum(bool(r["recovered"]) for r in first_fail) / len(first_fail), 3)
                                  if first_fail else None),
        "advisor_calls": sum(r["advisor_calls"] for r in rows),
        "advisor_ok": sum(r["advisor_ok"] for r in rows),
        "advisor_fallbacks": sum(r["advisor_fallbacks"] for r in rows),
        "advisor_latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
        "mean_sim_time_s": round(statistics.mean(r["sim_time_s"] for r in rows), 2),
        "mean_wall_time_s": round(statistics.mean(r["wall_time_s"] for r in rows), 3),
        "failure_reasons": dict(sorted(
            {f: sum(f in r["failure_reasons"].split("|") for r in rows)
             for r0 in rows for f in r0["failure_reasons"].split("|") if f}.items())),
    }


def make_advisor():
    from llm_advisor import AdvisorError, NemotronAdvisor
    try:
        return NemotronAdvisor(), None
    except AdvisorError as exc:
        return None, f"{exc.category}: {exc}"


def run(experiments, conditions, trials, seed, out_dir, echo=False) -> dict:
    out = Path(out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    sim = Simulation()
    rows, skipped = [], {}
    advisor = None
    recorder = None
    for exp in experiments:
        cfg = EXPERIMENTS[exp]
        exp_advisor = None
        if cfg.get("advisor") == "nemotron":
            if advisor is None:
                advisor, err = make_advisor()
                if advisor is None:
                    skipped[exp] = f"Nemotron unavailable -> experiment skipped ({err})"
                    print(f"[{exp}] SKIPPED: {skipped[exp]}")
                    continue
            exp_advisor = advisor
        if cfg.get("octo_shadow"):
            from octo_shadow import OctoShadowRecorder
            recorder = OctoShadowRecorder(out / "octo_shadow")
        for cond in conditions:
            for trial in range(trials):
                pos, yaw = cube_pose(seed, trial)
                sim.reset(cube_pos=pos, cube_yaw=yaw)
                run_id = f"{exp}__{cond}__t{trial:02d}"
                logger = TaskLogger(out / "logs", run_id=run_id, echo=echo)
                hook = recorder.hook if cfg.get("octo_shadow") else None
                task = TrustLoopTask(sim, logger, fault=cond, advisor=exp_advisor,
                                     mode=cfg["mode"], frame_hook=hook,
                                     instruction=DEFAULT_INSTRUCTION)
                s = task.run()
                logger.close()
                if cfg.get("octo_shadow"):
                    recorder.record_outcome(s)
                calls = s["advisor_calls"]
                lats = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
                row = {
                    "experiment": exp, "mode": s["mode"], "condition": cond, "trial": trial,
                    "seed": seed, "cube_x": round(pos[0], 4), "cube_y": round(pos[1], 4),
                    "cube_yaw": round(yaw, 4), "outcome": s["outcome"],
                    "outcome_success": s["outcome"] == "SUCCESS",
                    "controller_reported_success": s["controller_reported_success"],
                    "verified_success": s["verified_success"], "false_success": s["false_success"],
                    "attempts_used": s["attempts_used"],
                    "first_attempt_verified": bool(s["attempt_records"] and
                                                   s["attempt_records"][0]["verified_success"]),
                    "recovery_actions": s["recovery_actions"], "recovered": s["recovered"],
                    "failure_reasons": "|".join(s["failure_reasons"]),
                    "advisor_calls": len(calls),
                    "advisor_ok": sum(1 for r in s["recoveries"] if r["source"] == "nemotron"),
                    "advisor_fallbacks": sum(1 for r in s["recoveries"] if r["source"] == "fallback"),
                    "advisor_latency_mean_s": round(statistics.mean(lats), 3) if lats else None,
                    "octo_frames": sum(1 for e in logger.events if e["event"] == "octo_shadow_frame"),
                    "sim_time_s": s["sim_time_s"], "wall_time_s": s["wall_time_s"],
                    "reason": s["reason"], "run_id": s["run_id"],
                }
                rows.append(row)
                print(f"[{exp:<14}] {cond:<13} t{trial:02d}: {s['outcome']:<7} verified={s['verified_success']!s:<5} "
                      f"false_success={s['false_success']!s:<5} attempts={s['attempts_used']}")
        if recorder is not None:
            recorder.close()

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "instruction": DEFAULT_INSTRUCTION,
        "seed": seed, "trials_per_condition": trials, "conditions": conditions,
        "experiments_requested": experiments, "skipped": skipped,
        "per_experiment": {e: aggregate([r for r in rows if r["experiment"] == e])
                           for e in experiments if e not in skipped},
        "per_experiment_condition": {
            f"{e}/{c}": aggregate([r for r in rows if r["experiment"] == e and r["condition"] == c])
            for e in experiments if e not in skipped for c in conditions},
        "octo_shadow_dir": str(out / "octo_shadow") if "octo_shadow" in experiments else None,
        "results_csv": str(out / "results.csv"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def markdown(summary: dict) -> str:
    lines = [f"# TrustLoop experiment results",
             f"Instruction: \"{summary['instruction']}\"  |  seed {summary['seed']}  |  "
             f"{summary['trials_per_condition']} trial(s) per condition  |  conditions: "
             f"{', '.join(summary['conditions'])}", "",
             "| experiment | n | reported success | verified success | false successes | "
             "mean attempts | first-attempt failures | recovered | recovery success | advisor ok/fallback |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e, a in summary["per_experiment"].items():
        if not a.get("n"):
            continue
        lines.append(f"| {e} | {a['n']} | {a['reported_success_rate']:.0%} | {a['verified_success_rate']:.0%} | "
                     f"{a['false_success_count']} | {a['mean_attempts']} | {a['first_attempt_failures']} | "
                     f"{a['recovered']} | "
                     f"{'-' if a['recovery_success_rate'] is None else format(a['recovery_success_rate'], '.0%')} | "
                     f"{a['advisor_ok']}/{a['advisor_fallbacks']} |")
    for e, why in summary["skipped"].items():
        lines.append(f"| {e} | skipped: {why} |||||||||")
    lines += ["", "Per condition (verified success / false successes):", "",
              "| experiment / condition | n | verified | false successes | mean attempts |", "|---|---|---|---|---|"]
    for k, a in summary["per_experiment_condition"].items():
        if a.get("n"):
            lines.append(f"| {k} | {a['n']} | {a['verified_success_rate']:.0%} | {a['false_success_count']} | "
                         f"{a['mean_attempts']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run reproducible TrustLoop experiments")
    ap.add_argument("--experiments", default="baseline,verify_only,verify_recover",
                    help=f"comma list from {list(EXPERIMENTS)}")
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                    help=f"comma list of injected faults from {FAULTS}")
    ap.add_argument("--trials", type=int, default=5, help="trials per condition (seeded cube poses)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default logs/experiments/<timestamp>)")
    ap.add_argument("--echo", action="store_true", help="print every state transition")
    args = ap.parse_args(argv)
    exps = [e.strip() for e in args.experiments.split(",") if e.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for e in exps:
        if e not in EXPERIMENTS:
            ap.error(f"unknown experiment {e!r}")
    for c in conds:
        if c not in FAULTS:
            ap.error(f"unknown condition {c!r}")
    out = args.out or str(PROJECT_ROOT / "logs" / "experiments" / time.strftime("%Y%m%d_%H%M%S"))
    summary = run(exps, conds, args.trials, args.seed, out, echo=args.echo)
    print("\n" + markdown(summary))
    print(f"Results: {summary['results_csv']}\nSummary: {Path(out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
