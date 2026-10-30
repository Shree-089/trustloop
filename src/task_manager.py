"""TrustLoop task manager: Execute -> Verify -> Recover -> Verify again.

Runs the pick-and-place state machine, asks the independent Verifier to check
the physical result at every critical point, and applies bounded recovery
(max 3 pick-and-place attempts per task). Success is only declared when the
placement verifier, reading the physics state, says so.
"""
from __future__ import annotations

import time

import numpy as np

from gripper import Gripper, NOMINAL_FORCE, MAX_FORCE, OPEN_POS
from recovery import MAX_ATTEMPTS, NEW_ATTEMPT_ACTIONS, RecoveryPolicy
from robot_controller import RobotController
from simulation import Simulation
from task_logger import TaskLogger
from verification import Verifier, VerificationResult

FAULTS = ("none", "grasp_miss", "weak_grip", "place_offset", "rim", "unrecoverable")

# Evaluation modes. The control sequence is identical in all of them.
#   baseline       : scripted controller only. Success = "controller completed all motions".
#                    The verifier still runs, but only as a post-hoc AUDIT (never gates, never recovers),
#                    so false successes can be counted.
#   verify_only    : verifier gates every stage; any failure ends the task (no recovery, 1 attempt).
#   verify_recover : verifier gates every stage + bounded recovery (max 3 attempts). The default.
MODES = ("baseline", "verify_only", "verify_recover")
DEFAULT_INSTRUCTION = "Pick up the red cube and place it in the blue container."

APPROACH_HEIGHT = 0.12     # above cube centre
TEST_LIFT = 0.03           # micro-lift used by VERIFY_GRASP
TRANSPORT_Z = 0.62         # world z of grasp site while carrying
RELEASE_CLEARANCE = 0.012  # cube bottom above bin floor at release
MAX_INPLACE_RECOVERIES = 3


class AttemptFailure(Exception):
    def __init__(self, failure: str, report: dict):
        super().__init__(failure)
        self.failure, self.report = failure, report


class TrustLoopTask:
    def __init__(self, sim: Simulation, logger: TaskLogger, fault: str = "none",
                 advisor=None, screenshots: bool = False, mode: str = "verify_recover",
                 frame_hook=None, instruction: str = DEFAULT_INSTRUCTION):
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        self.mode = mode
        self.gating = mode != "baseline"
        self.max_attempts = MAX_ATTEMPTS if mode == "verify_recover" else 1
        self.frame_hook = frame_hook          # optional callable(task, tag) -> observation capture (Octo shadow)
        self.instruction = instruction
        self.attempt_records: list[dict] = []  # per attempt: controller-reported vs verified
        self.sim, self.log_ = sim, logger
        self.fault = fault
        self.rc = RobotController(sim)
        self.gripper = Gripper(sim)
        self.verifier = Verifier(sim)
        self.policy = RecoveryPolicy(advisor)
        self.screenshots = screenshots
        self.attempt = 0
        self.state = "INIT"
        self.recoveries: list[dict] = []
        self.verifications: list[dict] = []
        # Per-attempt tunables that recovery actions adjust.
        self.grasp_z_offset = 0.0
        self.descend_time = 1.5
        self.cube_half = float(self.verifier.cube_half[2])

    # ------------------------------------------------------------ utilities
    def log(self, event: str, **kw) -> dict:
        return self.log_.log(event, self.sim.time, self.attempt, **kw)

    def enter(self, state: str, detail: str = "") -> None:
        self.state = state
        self.log("state", state=state, detail=detail)
        if not self.sim.viewer_alive():
            raise KeyboardInterrupt("viewer closed")

    def shot(self, name: str) -> None:
        if self.screenshots:
            p = self.sim.screenshot(f"{self.log_.run_id}_a{self.attempt}_{name}")
            if p:
                self.log("screenshot", detail=str(p), path=str(p))

    def record(self, kind: str, res: VerificationResult) -> VerificationResult:
        d = res.to_dict()
        role = "gate" if self.gating else "audit"   # audit = measured but not acted upon (baseline)
        self.verifications.append({"kind": kind, "attempt": self.attempt, "role": role, **d})
        self.log("verification", kind=kind, role=role, result=d)
        return res

    def capture(self, tag: str) -> None:
        if self.frame_hook is not None:
            try:
                self.frame_hook(self, tag)
            except Exception as exc:  # observation capture must never break control
                self.log("warning", detail=f"frame capture '{tag}' failed: {exc!r}")

    def fault_active(self, name: str) -> bool:
        if self.fault == "unrecoverable" and name == "grasp_miss":
            return True
        return self.fault == name and self.attempt == 1

    def move(self, state: str, pos, yaw: float, duration: float):
        r = self.rc.move_to(pos, yaw, duration)
        if not r.success:
            raise AttemptFailure("motion_failed", {"state": state, **r.to_dict()})
        return r

    def wait_for_settle(self, hold: float = 0.3, timeout: float = 3.0) -> bool:
        t0, stable_since = self.sim.time, None
        while self.sim.time - t0 < timeout:
            self.rc.hold(0.02)
            ok, _, _ = self.verifier.is_stable()
            if ok:
                stable_since = stable_since if stable_since is not None else self.sim.time
                if self.sim.time - stable_since >= hold:
                    return True
            else:
                stable_since = None
        return False

    # ---------------------------------------------------------- perception
    def observe_cube(self) -> tuple[np.ndarray, float]:
        """'Perception': the cube pose read from the simulator (oracle sensor)."""
        return self.sim.cube_pos(), self.sim.cube_yaw()

    # ------------------------------------------------------------- main run
    def run(self) -> dict:
        wall0 = time.time()
        self.log("task_start", detail="TrustLoop pick-and-place", instruction=self.instruction,
                 mode=self.mode, fault=self.fault, max_attempts=self.max_attempts,
                 advisor=getattr(self.policy.advisor, "name", None),
                 container=self.verifier.container_info(),
                 cube_initial=self.sim.cube_pos(), cube_yaw=self.sim.cube_yaw())
        self.rc.sync_command()
        self.gripper.set(OPEN_POS)
        self.enter("HOME")
        self.rc.go_home(1.0)
        self.shot("home")

        outcome, reason, final = "FAILURE", "unknown", None
        failure = None
        try:
            while True:
                if self.attempt >= self.max_attempts:
                    outcome, reason = "FAILURE", f"Exhausted {self.max_attempts} attempt(s)"
                    break
                self.attempt += 1
                self.log("attempt_start", detail=f"attempt {self.attempt}/{self.max_attempts}")
                rec = {"attempt": self.attempt, "controller_reported_success": False,
                       "verified_success": False, "failure": None}
                self.attempt_records.append(rec)
                try:
                    final = self.pick_and_place()
                    rec["controller_reported_success"] = True     # every commanded motion completed
                    failure, report = (None, None) if final.success else (final.failure, final.to_dict())
                except AttemptFailure as af:
                    failure, report = af.failure, af.report
                rec["verified_success"] = failure is None
                rec["failure"] = failure

                if self.mode == "baseline":
                    # The scripted baseline trusts its controller; the verifier result is audit-only.
                    if rec["controller_reported_success"]:
                        outcome, reason = "SUCCESS", "Controller completed all motions (unverified)"
                    else:
                        outcome, reason = "FAILURE", f"Controller error: {failure}"
                    break
                if failure is not None and self.mode == "verify_only":
                    self.log("recovery", failure=failure, action="none", source="disabled",
                             allowed_actions=[], note="recovery disabled in verify_only mode")
                    outcome, reason = "FAILURE", f"Verified failure '{failure}' (recovery disabled)"
                    self.safe_retreat()
                    break

                # Recover -> verify again loop (in-place actions do not start a new attempt)
                inplace = 0
                action = None
                while failure is not None:
                    self.enter("RECOVERY", f"failure={failure}")
                    dec = self.policy.decide(failure, report, self.attempt, self.max_attempts)
                    self.recoveries.append({"attempt": self.attempt, **dec.to_dict()})
                    self.log("recovery", failure=failure, action=dec.action, source=dec.source,
                             allowed_actions=dec.allowed_actions, note=dec.note,
                             advisor_call=dec.advisor_call)
                    action = dec.action
                    if action == "abort_task" or action in NEW_ATTEMPT_ACTIONS:
                        break
                    inplace += 1
                    if inplace > MAX_INPLACE_RECOVERIES:
                        action = "abort_task"
                        break
                    final = self.inplace_recovery(action)
                    failure, report = (None, None) if final.success else (final.failure, final.to_dict())
                    rec["verified_success"] = failure is None
                    rec["failure"] = failure

                if failure is None:
                    outcome, reason = "SUCCESS", final.reason
                    break
                if action == "abort_task":
                    outcome, reason = "FAILURE", f"Recovery aborted after failure '{failure}'"
                    self.safe_retreat()
                    break
                self.prepare_new_attempt(action)
        except KeyboardInterrupt as exc:
            outcome, reason = "FAILURE", f"Interrupted: {exc}"

        # Independent final audit of the physical state, whatever the mode said.
        audit = self.verifier.verify_placement()
        self.log("verification", kind="final_audit", role="audit", result=audit.to_dict())
        self.enter(outcome)
        self.shot("final")
        self.capture("final")
        controller_ok = bool(self.attempt_records and self.attempt_records[-1]["controller_reported_success"])
        verified = bool(audit.success)
        advisor_calls = [r["advisor_call"] for r in self.recoveries if r.get("advisor_call")]
        summary = {
            "run_id": self.log_.run_id,
            "instruction": self.instruction,
            "mode": self.mode,
            "outcome": outcome,                       # what the system REPORTED
            "reason": reason,
            "controller_reported_success": controller_ok,
            "verified_success": verified,             # independent physical audit at the end
            "false_success": outcome == "SUCCESS" and not verified,
            "attempts_used": self.attempt,
            "max_attempts": self.max_attempts,
            "recovery_actions": len([r for r in self.recoveries if r["action"] != "abort_task"]),
            "recovered": outcome == "SUCCESS" and verified and self.attempt > 1,
            "fault_injected": self.fault,
            "failure_reasons": [r["failure"] for r in self.attempt_records if r["failure"]],
            "attempt_records": self.attempt_records,
            "final_audit": audit.to_dict(),
            "final_placement_verification": final.to_dict() if final is not None else None,
            "recoveries": self.recoveries,
            "advisor_calls": advisor_calls,
            "verifications": self.verifications,
            "cube_final_pos": self.sim.cube_pos(),
            "sim_time_s": round(self.sim.time, 3),
            "wall_time_s": round(time.time() - wall0, 3),
            "log_file": str(self.log_.path),
        }
        self.log("outcome", outcome=outcome, reason=reason, attempts_used=self.attempt,
                 mode=self.mode, controller_reported_success=controller_ok,
                 verified_success=verified, false_success=summary["false_success"])
        summary["summary_file"] = str(self.log_.write_summary(summary))
        return summary

    # ----------------------------------------------------- one full attempt
    def pick_and_place(self) -> VerificationResult:
        cube_pos, cube_yaw = self.observe_cube()
        table_z = float(self.sim.data.geom_xpos[self.verifier.table_geom][2]
                        + self.sim.model.geom_size[self.verifier.table_geom][2])
        if cube_pos[2] < table_z - 0.05:
            raise AttemptFailure("cube_fell_off_table", {"cube_pos": cube_pos.tolist()})
        yaw = self.rc.nearest_grasp_yaw(cube_yaw)
        target = cube_pos.copy()

        if self.fault_active("grasp_miss"):
            tool_y = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
            target = target + 0.05 * tool_y
            self.log("fault_injected", fault="grasp_miss",
                     detail="perception offset 5 cm along the pad-width axis")
        if self.fault_active("weak_grip"):
            self.gripper.set_force_limit(0.1)
            self.log("fault_injected", fault="weak_grip", detail="finger force limited to 0.1 N")

        reach = self.rc.ik.solve(target + [0, 0, APPROACH_HEIGHT], yaw, self.rc.q_cmd, max_iters=300)
        if not reach.success:
            raise AttemptFailure("cube_unreachable", {"cube_pos": cube_pos.tolist()})

        # 2. APPROACH_CUBE
        self.capture("pre_approach")
        self.enter("APPROACH_CUBE", f"cube observed at {np.round(cube_pos, 3).tolist()}")
        self.move("APPROACH_CUBE", target + [0, 0, APPROACH_HEIGHT], yaw, 2.0)
        # 3. OPEN_GRIPPER
        self.enter("OPEN_GRIPPER", "pre-grasp")
        self.gripper.open()
        # 4. DESCEND_TO_CUBE
        self.enter("DESCEND_TO_CUBE", f"z offset {self.grasp_z_offset:+.3f}")
        self.move("DESCEND_TO_CUBE", target + [0, 0, self.grasp_z_offset], yaw, self.descend_time)
        # 5. CLOSE_GRIPPER
        self.enter("CLOSE_GRIPPER", f"force limit {self.gripper.force_limit:.2f} N")
        g = self.gripper.close()
        self.shot("grasp")

        # 6. VERIFY_GRASP: (a) closure evidence, (b) physical test lift
        self.enter("VERIFY_GRASP")
        self.capture("grasp_closed")
        res = self.record("grasp_closure", self.verifier.check_grasp_closure(g["gap"]))
        if not res.success and self.gating:
            raise AttemptFailure(res.failure, res.to_dict())
        ref = self.verifier.snapshot()
        grasp_offset = ref["cube_offset"].copy()
        self.move("VERIFY_GRASP", ref["ee_pos"] + [0, 0, TEST_LIFT], yaw, 0.8)
        self.rc.hold(0.2)
        ee_rise = float(self.sim.site_pos()[2] - ref["ee_pos"][2])
        res = self.record("grasp_lift_test", self.verifier.check_grasp_lift(ref, ee_rise))
        if not res.success and self.gating:
            raise AttemptFailure(res.failure, res.to_dict())

        # 7. LIFT_CUBE
        self.enter("LIFT_CUBE")
        lift_to = self.sim.site_pos().copy()
        lift_to[2] = TRANSPORT_Z
        self.move("LIFT_CUBE", lift_to, yaw, 1.2)
        self.carry_check("LIFT_CUBE", grasp_offset)

        # 8. MOVE_ABOVE_CONTAINER
        c = self.verifier.container_center_world()
        place_xy = c[:2].copy()
        cube_bottom_target = self.verifier.floor_top_world() + RELEASE_CLEARANCE
        if self.fault_active("place_offset"):
            place_xy += [0.0, -0.17]
            self.log("fault_injected", fault="place_offset",
                     detail="placement target shifted 17 cm (outside the bin)")
        if self.fault_active("rim"):
            wall_x = c[0] + 0.5 * (self.verifier.interior_x[1] + 0.08)  # +x wall centre line
            place_xy = np.array([wall_x, c[1]])
            cube_bottom_target = self.verifier.rim_world() + 0.004
            self.log("fault_injected", fault="rim",
                     detail="placement target over the +x wall top")
        place_yaw = self.rc.nearest_grasp_yaw(0.0)
        self.enter("MOVE_ABOVE_CONTAINER")
        self.move("MOVE_ABOVE_CONTAINER", [place_xy[0], place_xy[1], TRANSPORT_Z], place_yaw, 2.0)
        self.carry_check("MOVE_ABOVE_CONTAINER", grasp_offset)

        # 9. DESCEND_INTO_CONTAINER: use the *measured* cube-in-hand offset
        cube_center_target = cube_bottom_target + self.cube_half
        ee_z = cube_center_target - float(self.verifier.cube_offset_from_ee()[2])
        self.enter("DESCEND_INTO_CONTAINER")
        self.move("DESCEND_INTO_CONTAINER", [place_xy[0], place_xy[1], ee_z], place_yaw, 1.5)
        self.carry_check("DESCEND_INTO_CONTAINER", grasp_offset)

        # 10. OPEN_GRIPPER (release)
        self.capture("pre_release")
        self.enter("OPEN_GRIPPER", "release")
        self.gripper.open()
        if self.gripper.force_limit < NOMINAL_FORCE:
            self.gripper.set_force_limit(NOMINAL_FORCE)
        # 11. RETREAT
        self.enter("RETREAT")
        self.move("RETREAT", [place_xy[0], place_xy[1], self.verifier.rim_world() + 0.12], place_yaw, 1.0)

        # 12. VERIFY_PLACEMENT
        self.enter("VERIFY_PLACEMENT", "waiting for cube to settle")
        settled = self.wait_for_settle()
        self.shot("placed")
        res = self.record("placement", self.verifier.verify_placement())
        res.measurements["settle_wait_converged"] = settled
        return res

    def carry_check(self, state: str, grasp_offset) -> None:
        res = self.record("carry", self.verifier.check_carry(grasp_offset, state))
        if not res.success and self.gating:
            raise AttemptFailure(res.failure, res.to_dict())

    # ------------------------------------------------------------ recovery
    def safe_retreat(self) -> None:
        """Open the gripper, back straight up, and return to HOME."""
        self.gripper.open()
        up = self.sim.site_pos().copy()
        up[2] = max(up[2] + 0.10, TRANSPORT_Z)
        r = self.rc.move_to(up, self.rc.ee_yaw(), 1.0)
        if not r.success:
            self.log("warning", detail=f"retreat move failed: {r.reason}")
        self.rc.go_home(1.5)

    def prepare_new_attempt(self, action: str) -> None:
        if action == "adjust_approach":
            self.grasp_z_offset = max(self.grasp_z_offset - 0.003, -0.008)
            self.descend_time = min(self.descend_time * 1.5, 3.0)
        elif action == "increase_grip_force":
            new = NOMINAL_FORCE if self.gripper.force_limit < NOMINAL_FORCE else self.gripper.force_limit * 1.5
            self.gripper.set_force_limit(min(new, MAX_FORCE))
        self.log("recovery_action", detail=f"{action}: safe retreat, re-observe cube, retry "
                 f"(z_offset={self.grasp_z_offset:+.3f}, force={self.gripper.force_limit:.1f} N)")
        self.safe_retreat()

    def inplace_recovery(self, action: str) -> VerificationResult:
        if action == "wait_and_reverify":
            self.log("recovery_action", detail="waiting 1.5 s for the cube to settle")
            self.rc.hold(1.5)
            self.wait_for_settle()
        elif action == "reopen_and_retreat":
            self.log("recovery_action", detail="re-opening gripper and retreating higher")
            self.gripper.open()
            up = self.sim.site_pos().copy()
            up[2] += 0.08
            self.rc.move_to(up, self.rc.ee_yaw(), 1.0)
            self.wait_for_settle()
        self.enter("VERIFY_PLACEMENT", f"re-verify after {action}")
        return self.record("placement", self.verifier.verify_placement())
