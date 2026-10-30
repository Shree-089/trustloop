"""Bounded recovery policy for TrustLoop.

A failure category (from the verifier) maps to a *restricted* list of
predefined recovery actions. The deterministic policy always picks from that
list. An optional advisor (e.g. Nemotron) may *suggest* an action, but:
  * it only receives structured verification evidence (build_evidence),
    never raw simulator internals;
  * its suggestion is validated against the same whitelist before anything is
    executed; invalid, failed or timed-out suggestions fall back to the
    deterministic choice;
  * it never sees or produces joint commands, only action names;
  * the attempt budget is enforced here, so it cannot extend retries.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Protocol

MAX_ATTEMPTS = 3

# failure category -> allowed actions (first entry = deterministic default)
ALLOWED_ACTIONS: dict[str, list[str]] = {
    "grasp_missed":             ["retry_grasp", "adjust_approach", "abort_task"],
    "grasp_partial":            ["adjust_approach", "retry_grasp", "abort_task"],
    "grasp_slipped":            ["increase_grip_force", "retry_grasp", "abort_task"],
    "dropped_in_transit":       ["increase_grip_force", "retry_grasp", "abort_task"],
    "placed_outside_container": ["repick_and_place", "abort_task"],
    "resting_on_rim":           ["repick_and_place", "abort_task"],
    "not_resting_on_floor":     ["wait_and_reverify", "repick_and_place", "abort_task"],
    "not_settled":              ["wait_and_reverify", "abort_task"],
    "still_in_gripper":         ["reopen_and_retreat", "abort_task"],
    "motion_failed":            ["return_home_and_retry", "abort_task"],
    "cube_unreachable":         ["abort_task"],
    "cube_fell_off_table":      ["abort_task"],
}
SUPPORTED_ACTIONS = sorted({a for acts in ALLOWED_ACTIONS.values() for a in acts})

# Actions that start a new pick-and-place attempt (consume the attempt budget).
NEW_ATTEMPT_ACTIONS = {"retry_grasp", "adjust_approach", "increase_grip_force",
                       "repick_and_place", "return_home_and_retry"}

# Only these verifier fields are ever sent to an external advisor.
EVIDENCE_MEASUREMENT_KEYS = {
    "gap", "cube_width", "cube_offset_tool_frame", "ee_rise", "cube_rise", "relative_drift",
    "cube_bottom_local", "cube_top_local", "floor_top_local", "rim_local",
    "interior_x_local", "interior_y_local", "cube_xy_extent_local",
    "lin_vel", "ang_vel", "ee_clearance", "contacts", "state", "reason",
}


def build_evidence(failure: str, report: dict | None) -> dict:
    """Structured, whitelisted verification evidence for an advisor."""
    report = report or {}
    meas = report.get("measurements") or {}
    if not meas:   # motion failures carry their info at the top level
        meas = {k: v for k, v in report.items() if k not in ("checks", "success", "failure")}
    return {
        "failure": failure,
        "verifier_reason": report.get("reason"),
        "checks": {k: bool(v) for k, v in (report.get("checks") or {}).items()},
        "measurements": {k: v for k, v in meas.items() if k in EVIDENCE_MEASUREMENT_KEYS},
    }


class Advisor(Protocol):
    name: str

    def suggest(self, failure_report: dict, allowed_actions: list[str]) -> str | None: ...


@dataclass
class RecoveryDecision:
    failure: str
    action: str
    source: str               # "deterministic" | advisor name | "fallback"
    allowed_actions: list[str]
    note: str = ""
    advisor_call: dict | None = field(default=None)   # status/latency of the advisor call, if any

    def to_dict(self) -> dict:
        return asdict(self)


class RecoveryPolicy:
    def __init__(self, advisor: Advisor | None = None):
        self.advisor = advisor
        self.history: list[str] = []   # failure categories seen this task

    def reset(self) -> None:
        self.history.clear()

    def deterministic_action(self, failure: str) -> str:
        allowed = ALLOWED_ACTIONS.get(failure, ["abort_task"])
        # Escalate: the same grasp failure twice in a row -> change the approach.
        if failure in ("grasp_missed",) and self.history[-2:-1] == [failure]:
            return "adjust_approach"
        if failure == "not_settled" and self.history.count("not_settled") > 2:
            return "abort_task"
        return allowed[0]

    def decide(self, failure: str, report: dict | None, attempts_used: int,
               max_attempts: int = MAX_ATTEMPTS) -> RecoveryDecision:
        self.history.append(failure)
        allowed = list(ALLOWED_ACTIONS.get(failure, ["abort_task"]))
        if attempts_used >= max_attempts:
            # Budget exhausted: only non-attempt-consuming actions (or abort) remain.
            allowed = [a for a in allowed if a not in NEW_ATTEMPT_ACTIONS] or ["abort_task"]
        default = self.deterministic_action(failure)
        if default not in allowed:
            default = allowed[0]

        if self.advisor is None:
            return RecoveryDecision(failure, default, "deterministic", allowed)
        if len(allowed) == 1:
            return RecoveryDecision(failure, default, "deterministic", allowed,
                                    "single allowed action; advisor not consulted")

        payload = {"failure": failure, "allowed_actions": allowed,
                   "attempts_used": attempts_used, "max_attempts": max_attempts,
                   "evidence": build_evidence(failure, report)}
        try:
            suggestion = self.advisor.suggest(payload, allowed)
        except Exception as exc:  # network errors, auth, rate limit, bad JSON ...
            return RecoveryDecision(failure, default, "fallback", allowed,
                                    f"advisor error: {type(exc).__name__}: {exc}",
                                    getattr(self.advisor, "last_call", None))
        call = getattr(self.advisor, "last_call", None)
        if suggestion in allowed:
            return RecoveryDecision(failure, suggestion, self.advisor.name, allowed, "", call)
        return RecoveryDecision(failure, default, "fallback", allowed,
                                f"advisor suggested invalid action {suggestion!r}; rejected", call)
