"""End-effector motion control for the TrustLoop arm.

Moves are smooth Cartesian min-jerk trajectories of the grasp site, converted
to joint targets by warm-started damped-least-squares IK at 100 Hz and tracked
by MuJoCo position actuators (joint-space PD with force limits; gravity is
compensated in the model via body gravcomp). Every move has a timeout and
reports whether the site actually reached the target in the physics state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from kinematics import ARM_JOINTS, IKSolver, wrap_angle
from simulation import Simulation

CONTROL_SUBSTEPS = 5          # physics steps per control tick (0.002 s * 5 = 100 Hz)
MAX_JOINT_STEP = 0.03         # rad per control tick (≈3 rad/s) – rate limit on targets


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


@dataclass
class MoveResult:
    success: bool
    reason: str
    final_pos_err: float
    duration: float
    ik_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


class RobotController:
    def __init__(self, sim: Simulation):
        self.sim = sim
        self.ik = IKSolver(sim.model)
        m = sim.model
        self.act = np.array([m.actuator(f"a{i}").id for i in range(1, 7)])
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.q_cmd = sim.data.qpos[self.qadr].copy()
        self.home_q = sim.model.key_qpos[sim.home_key][self.qadr].copy()
        self.monitor = None   # optional callback(sim) invoked every control tick

    def sync_command(self) -> None:
        self.q_cmd = self.sim.data.qpos[self.qadr].copy()
        self.sim.data.ctrl[self.act] = self.q_cmd

    # ----------------------------------------------------------- read state
    def ee_pos(self) -> np.ndarray:
        return self.sim.site_pos("ee_site")

    def ee_yaw(self) -> float:
        r = self.sim.data.site("ee_site").xmat.reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def q(self) -> np.ndarray:
        return self.sim.data.qpos[self.qadr].copy()

    def qd(self) -> np.ndarray:
        return self.sim.data.qvel[self.dadr].copy()

    def nearest_grasp_yaw(self, object_yaw: float) -> float:
        """A cube is 4-fold symmetric about z and the gripper 2-fold, so pick the
        equivalent yaw closest to the current tool yaw (minimises wrist rotation)."""
        cur = self.ee_yaw()
        cands = [wrap_angle(object_yaw + k * np.pi / 2) for k in range(4)]
        return min(cands, key=lambda c: abs(wrap_angle(c - cur)))

    # ---------------------------------------------------------------- moves
    def _tick(self, q_target: np.ndarray) -> None:
        step = np.clip(q_target - self.q_cmd, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        self.q_cmd = np.clip(self.q_cmd + step, self.ik.lo, self.ik.hi)
        self.sim.data.ctrl[self.act] = self.q_cmd
        self.sim.step(CONTROL_SUBSTEPS)
        if self.monitor is not None:
            self.monitor(self.sim)

    def move_to(self, target_pos, target_yaw: float, duration: float,
                pos_tol: float = 0.004, settle_timeout: float = 1.5) -> MoveResult:
        """Min-jerk straight-line move of the grasp site to target_pos with the tool
        pointing down at target_yaw. Returns once converged or timed out."""
        target_pos = np.asarray(target_pos, dtype=float)
        start_pos = self.ee_pos()
        start_yaw = self.ee_yaw()
        dyaw = wrap_angle(target_yaw - start_yaw)
        t0 = self.sim.time

        # Reject unreachable goals up front instead of waiting for a timeout.
        goal = self.ik.solve(target_pos, target_yaw, self.q_cmd, max_iters=300)
        if not goal.success:
            return MoveResult(False, f"IK failed for target (pos_err={goal.pos_err:.4f})",
                              goal.pos_err, 0.0, False)

        q_ref = self.q_cmd.copy()
        ik_ok = True
        n_ticks = max(1, int(round(duration / (CONTROL_SUBSTEPS * self.sim.model.opt.timestep))))
        for k in range(1, n_ticks + 1):
            s = min_jerk(k / n_ticks)
            p = start_pos + s * (target_pos - start_pos)
            yaw = start_yaw + s * dyaw
            res = self.ik.solve(p, yaw, q_ref, max_iters=30)
            ik_ok &= res.success
            q_ref = res.q
            self._tick(q_ref)

        # Hold the final target until the physical end effector converges.
        q_goal = self.ik.solve(target_pos, target_yaw, q_ref, max_iters=300).q
        t_settle = self.sim.time
        while True:
            self._tick(q_goal)
            err = float(np.linalg.norm(self.ee_pos() - target_pos))
            if err < pos_tol and np.max(np.abs(self.qd())) < 0.05:
                return MoveResult(True, "reached", err, self.sim.time - t0, ik_ok)
            if self.sim.time - t_settle > settle_timeout:
                return MoveResult(False, "timeout before reaching target", err,
                                  self.sim.time - t0, ik_ok)

    def move_joints(self, q_target, duration: float, settle_timeout: float = 1.5,
                    tol: float = 0.02) -> MoveResult:
        """Smooth joint-space move (used for HOME / safe-pose returns)."""
        q_target = np.clip(np.asarray(q_target, dtype=float), self.ik.lo, self.ik.hi)
        q_start = self.q_cmd.copy()
        t0 = self.sim.time
        n_ticks = max(1, int(round(duration / (CONTROL_SUBSTEPS * self.sim.model.opt.timestep))))
        for k in range(1, n_ticks + 1):
            self._tick(q_start + min_jerk(k / n_ticks) * (q_target - q_start))
        t_settle = self.sim.time
        while True:
            self._tick(q_target)
            err = float(np.max(np.abs(self.q() - q_target)))
            if err < tol and np.max(np.abs(self.qd())) < 0.05:
                return MoveResult(True, "reached", err, self.sim.time - t0, True)
            if self.sim.time - t_settle > settle_timeout:
                return MoveResult(False, "timeout before reaching joint target", err,
                                  self.sim.time - t0, True)

    def go_home(self, duration: float = 2.0) -> MoveResult:
        return self.move_joints(self.home_q, duration)

    def hold(self, seconds: float) -> None:
        t0 = self.sim.time
        while self.sim.time - t0 < seconds:
            self._tick(self.q_cmd)
