"""End-to-end TrustLoop runs, bounded recovery, logging, and advisor validation."""
import json

import numpy as np
import pytest

from recovery import MAX_ATTEMPTS, RecoveryPolicy
from simulation import Simulation
from task_logger import TaskLogger
from task_manager import TrustLoopTask


def run_task(tmp_path, fault="none", cube=None, yaw=0.0, advisor=None, run_id="t"):
    sim = Simulation()
    sim.reset(cube_pos=cube, cube_yaw=yaw)
    logger = TaskLogger(tmp_path, run_id=run_id, echo=False)
    summary = TrustLoopTask(sim, logger, fault=fault, advisor=advisor).run()
    logger.close()
    return sim, summary


def test_nominal_task_succeeds_with_verified_placement(tmp_path):
    sim, s = run_task(tmp_path)
    assert s["outcome"] == "SUCCESS" and s["attempts_used"] == 1
    assert s["final_placement_verification"]["success"]
    p = sim.cube_pos()
    assert abs(p[0] - 0.40) < 0.06 and abs(p[1] - 0.25) < 0.06 and p[2] < 0.45


@pytest.mark.parametrize("fault,failure", [
    ("grasp_miss", "grasp_missed"),
    ("weak_grip", "grasp_slipped"),
    ("place_offset", "placed_outside_container"),
    ("rim", "resting_on_rim"),
])
def test_injected_failure_is_detected_and_recovered(tmp_path, fault, failure):
    _, s = run_task(tmp_path, fault=fault)
    assert s["outcome"] == "SUCCESS"
    assert s["attempts_used"] == 2
    assert s["recoveries"][0]["failure"] == failure


def test_unrecoverable_failure_reports_failure_after_max_attempts(tmp_path):
    _, s = run_task(tmp_path, fault="unrecoverable")
    assert s["outcome"] == "FAILURE"
    assert s["attempts_used"] == MAX_ATTEMPTS
    assert len(s["recoveries"]) == MAX_ATTEMPTS
    assert s["recoveries"][-1]["action"] == "abort_task"


def test_machine_readable_log(tmp_path):
    _, s = run_task(tmp_path, fault="place_offset", run_id="logcheck")
    lines = (tmp_path / "logcheck.jsonl").read_text().splitlines()
    events = [json.loads(l) for l in lines]
    kinds = {e["event"] for e in events}
    assert {"task_start", "state", "verification", "recovery", "outcome"} <= kinds
    states = [e["state"] for e in events if e["event"] == "state"]
    for st in ("HOME", "APPROACH_CUBE", "OPEN_GRIPPER", "DESCEND_TO_CUBE", "CLOSE_GRIPPER",
               "VERIFY_GRASP", "LIFT_CUBE", "MOVE_ABOVE_CONTAINER", "DESCEND_INTO_CONTAINER",
               "RETREAT", "VERIFY_PLACEMENT", "RECOVERY", "SUCCESS"):
        assert st in states
    summary = json.loads((tmp_path / "logcheck_summary.json").read_text())
    assert summary["outcome"] == events[-1]["outcome"] == "SUCCESS"


def test_repeated_runs_from_fresh_state(tmp_path):
    rng = np.random.default_rng(42)
    for i in range(3):
        pos = [float(rng.uniform(0.33, 0.47)), float(rng.uniform(-0.28, -0.13)), 0.42]
        _, s = run_task(tmp_path, cube=pos, yaw=float(rng.uniform(-0.7, 0.7)), run_id=f"rep{i}")
        assert s["outcome"] == "SUCCESS", s["reason"]


# ------------------------------------------------------------- advisor guard
class FakeAdvisor:
    name = "fake"

    def __init__(self, reply=None, exc=None):
        self.reply, self.exc = reply, exc

    def suggest(self, report, allowed):
        if self.exc:
            raise self.exc
        return self.reply


def test_advisor_invalid_action_is_rejected():
    d = RecoveryPolicy(FakeAdvisor("set_joint_3_to_2.5_rad")).decide("grasp_missed", {}, 1)
    assert d.action == "retry_grasp" and d.source == "fallback"


def test_advisor_valid_action_is_used():
    d = RecoveryPolicy(FakeAdvisor("adjust_approach")).decide("grasp_missed", {}, 1)
    assert d.action == "adjust_approach" and d.source == "fake"


def test_advisor_error_falls_back():
    d = RecoveryPolicy(FakeAdvisor(exc=TimeoutError("no network"))).decide("grasp_slipped", {}, 1)
    assert d.action == "increase_grip_force" and d.source == "fallback"


def test_budget_exhausted_only_allows_abort():
    d = RecoveryPolicy(FakeAdvisor("retry_grasp")).decide("grasp_missed", {}, MAX_ATTEMPTS)
    assert d.allowed_actions == ["abort_task"] and d.action == "abort_task"


def test_llm_reply_parser():
    from llm_advisor import parse_action
    assert parse_action('Sure! {"action": "retry_grasp", "rationale": "x"}') == ("retry_grasp", "x")
    assert parse_action("I think you should retry") == (None, None)
