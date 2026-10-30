"""Evaluation modes, controller-vs-verified success, bounded termination, experiment logging,
Octo shadow export/merge."""
import csv
import json

import pytest

from simulation import Simulation
from task_logger import TaskLogger
from task_manager import TrustLoopTask


def run(tmp_path, mode, fault="none", advisor=None, hook=None, run_id="r"):
    sim = Simulation()
    sim.reset()
    lg = TaskLogger(tmp_path, run_id=run_id, echo=False)
    s = TrustLoopTask(sim, lg, fault=fault, mode=mode, advisor=advisor, frame_hook=hook).run()
    lg.close()
    return s


def test_baseline_reports_false_success_that_verifier_catches(tmp_path):
    s = run(tmp_path, "baseline", "grasp_miss")
    assert s["outcome"] == "SUCCESS"                      # what the scripted controller claims
    assert s["controller_reported_success"] is True
    assert s["verified_success"] is False                 # physical audit disagrees
    assert s["false_success"] is True
    assert s["attempts_used"] == 1 and s["recoveries"] == []
    assert all(v["role"] == "audit" for v in s["verifications"])


def test_baseline_nominal_is_true_success(tmp_path):
    s = run(tmp_path, "baseline")
    assert s["outcome"] == "SUCCESS" and s["verified_success"] and not s["false_success"]


def test_verify_only_stops_after_one_attempt_without_recovery(tmp_path):
    s = run(tmp_path, "verify_only", "weak_grip")
    assert s["outcome"] == "FAILURE" and s["attempts_used"] == 1
    assert s["failure_reasons"] == ["grasp_slipped"] and not s["false_success"]
    events = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    rec = [e for e in events if e["event"] == "recovery"]
    assert len(rec) == 1 and rec[0]["source"] == "disabled"


def test_verify_recover_recovers_and_is_verified(tmp_path):
    s = run(tmp_path, "verify_recover", "weak_grip")
    assert s["outcome"] == "SUCCESS" and s["verified_success"] and s["recovered"]
    assert s["attempts_used"] == 2 and not s["false_success"]


class AlwaysBogus:
    name = "bogus"
    last_call = None

    def suggest(self, report, allowed):
        return "move_joint_3_to_pi"


class AlwaysAbort:
    name = "aborter"
    last_call = None

    def suggest(self, report, allowed):
        return "abort_task"


def test_bogus_advisor_cannot_break_recovery(tmp_path):
    s = run(tmp_path, "verify_recover", "place_offset", advisor=AlwaysBogus())
    assert s["outcome"] == "SUCCESS" and s["verified_success"]
    assert all(r["source"] in ("fallback", "deterministic") for r in s["recoveries"])


def test_unrecoverable_terminates_cleanly_within_budget(tmp_path):
    s = run(tmp_path, "verify_recover", "unrecoverable")
    assert s["outcome"] == "FAILURE" and s["attempts_used"] == 3 and not s["false_success"]
    s2 = run(tmp_path, "verify_recover", "grasp_miss", advisor=AlwaysAbort(), run_id="r2")
    assert s2["outcome"] == "FAILURE" and s2["attempts_used"] == 1


def test_experiment_runner_logs_results(tmp_path):
    from experiments import run as run_experiments
    summ = run_experiments(["baseline", "verify_recover"], ["none", "grasp_miss"], 1, 3, tmp_path)
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert len(rows) == 4
    by = {(r["experiment"], r["condition"]): r for r in rows}
    assert by[("baseline", "grasp_miss")]["false_success"] == "True"
    assert by[("verify_recover", "grasp_miss")]["verified_success"] == "True"
    assert by[("verify_recover", "grasp_miss")]["attempts_used"] == "2"
    # same seeded cube pose for every experiment
    assert by[("baseline", "none")]["cube_x"] == by[("verify_recover", "none")]["cube_x"]
    agg = json.loads((tmp_path / "summary.json").read_text())["per_experiment"]
    assert agg["baseline"]["false_success_count"] == 1
    assert agg["verify_recover"]["false_success_count"] == 0
    assert agg["verify_recover"]["recovery_success_rate"] == 1.0
    assert (tmp_path / "summary.md").exists() and summ["skipped"] == {}


def test_nemotron_experiment_skipped_not_faked_without_key(tmp_path, monkeypatch):
    from experiments import run as run_experiments
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    summ = run_experiments(["nemotron"], ["none"], 1, 0, tmp_path)
    assert "nemotron" in summ["skipped"] and "nemotron" not in summ["per_experiment"]


def _gl_available():
    try:
        import mujoco
        m = Simulation().model
        r = mujoco.Renderer(m, 64, 64)
        r.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gl_available(), reason="no offscreen OpenGL (set MUJOCO_GL=egl)")
def test_octo_shadow_export_and_merge(tmp_path):
    from octo_shadow import OctoShadowRecorder, merge
    rec = OctoShadowRecorder(tmp_path / "shadow")
    s = run(tmp_path, "verify_recover", "grasp_miss", hook=rec.hook)
    rec.record_outcome(s)
    rec.close()
    frames = [json.loads(l) for l in (tmp_path / "shadow" / "manifest.jsonl").read_text().splitlines()]
    tags = [f["tag"] for f in frames]
    assert tags[:2] == ["pre_approach", "grasp_closed"] and tags[-1] == "final"
    assert frames[1]["prev_image"] == frames[0]["image"]
    assert all((tmp_path / "shadow" / f["image"]).exists() for f in frames)
    assert s["outcome"] == "SUCCESS"            # shadow capture does not change control
    # fake Octo output in the format octo_inference.py --manifest writes
    preds = tmp_path / "shadow" / "octo_predictions.jsonl"
    with preds.open("w") as fh:
        for i, f in enumerate(frames[:-1]):
            fh.write(json.dumps({"frame_id": f["frame_id"], "error": None, "stats_key": "bridge_dataset",
                                 "actions": [[0.0] * 7] * 4, "inference_seconds": 0.1,
                                 "jit_compiled_this_call": i == 0}) + "\n")
    summ = merge(tmp_path / "shadow")
    assert summ["frames"] == len(frames) and summ["frames_with_prediction"] == len(frames) - 1
    assert summ["frames_missing_prediction"] == 1 and summ["action_shape"] == [4, 7]
    assert summ["warm_latency_s"]["n"] == len(frames) - 2
    merged = [json.loads(l) for l in (tmp_path / "shadow" / "octo_shadow_merged.jsonl").read_text().splitlines()]
    assert merged[0]["task_outcome"]["verified_success"] is True
