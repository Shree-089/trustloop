"""Numerical inverse kinematics for the TrustLoop arm.

Damped-least-squares (Levenberg-Marquardt style) IK on the 6 arm joints,
using MuJoCo's own forward kinematics (mj_kinematics / mj_comPos) and site
Jacobian (mj_jacSite). IK runs on a private scratch MjData so it never
disturbs the physics state of the running simulation.

Target orientation is always "tool pointing straight down" plus a yaw angle
(rotation of the finger-closing axis about world z). That is all a top-down
pick-and-place needs, and it keeps the solver well-conditioned.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def topdown_rotation(yaw: float) -> np.ndarray:
    """Rotation matrix whose z-axis points down and whose x-axis
    (the finger-closing direction) has heading `yaw` in the world xy-plane."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def orientation_error(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    """Small-angle rotation error (world frame) taking r_cur towards r_tgt."""
    return 0.5 * (np.cross(r_cur[:, 0], r_tgt[:, 0])
                  + np.cross(r_cur[:, 1], r_tgt[:, 1])
                  + np.cross(r_cur[:, 2], r_tgt[:, 2]))


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    pos_err: float
    rot_err: float
    iterations: int


class IKSolver:
    def __init__(self, model: mujoco.MjModel, site: str = "ee_site"):
        self.model = model
        self.data = mujoco.MjData(model)  # scratch copy, never stepped
        self.site_id = model.site(site).id
        self.qpos_idx = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
        self.dof_idx = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
        self.lo = np.array([model.jnt_range[model.joint(j).id][0] for j in ARM_JOINTS])
        self.hi = np.array([model.jnt_range[model.joint(j).id][1] for j in ARM_JOINTS])
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site position and rotation matrix for arm configuration q."""
        self.data.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(self.model, self.data)
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def _tool_yaw(self, q) -> float:
        _, r = self.forward(q)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def _seed_wrist(self, q: np.ndarray, target_yaw: float) -> np.ndarray:
        """Pre-rotate the last joint so the tool yaw starts close to the target.
        Gradient steps alone can wind the wrist into its joint limit when the
        required rotation is large; this picks the in-limit wrist angle with the
        smallest remaining yaw error."""
        delta = wrap_angle(target_yaw - self._tool_yaw(q))
        if abs(delta) < 0.2:
            return q
        best, best_err = q, abs(delta)
        for cand in (q[5] - delta, q[5] + delta, q[5] - delta + 2 * np.pi, q[5] - delta - 2 * np.pi,
                     q[5] + delta + 2 * np.pi, q[5] + delta - 2 * np.pi):
            if not (self.lo[5] <= cand <= self.hi[5]):
                continue
            qc = q.copy()
            qc[5] = cand
            err = abs(wrap_angle(target_yaw - self._tool_yaw(qc)))
            if err < best_err:
                best, best_err = qc, err
        return best

    def solve(self, target_pos, target_yaw: float, q_init,
              max_iters: int = 200, pos_tol: float = 5e-4, rot_tol: float = 5e-3,
              damping: float = 0.05, max_step: float = 0.2, rot_weight: float = 0.5) -> IKResult:
        target_pos = np.asarray(target_pos, dtype=float)
        r_tgt = topdown_rotation(target_yaw)
        q = np.clip(np.asarray(q_init, dtype=float).copy(), self.lo, self.hi)
        # Seed the base so it faces the target (the arm is planar after j1), then
        # seed the wrist roll for the remaining yaw. Only applied for large changes,
        # so warm-started per-tick solves stay continuous.
        base_yaw = float(np.arctan2(target_pos[1], target_pos[0]))
        if abs(wrap_angle(base_yaw - q[0])) > 0.2 and self.lo[0] <= base_yaw <= self.hi[0]:
            q[0] = base_yaw
        q = self._seed_wrist(q, target_yaw)
        pos_err = rot_err = np.inf
        it = 0
        for it in range(1, max_iters + 1):
            self.data.qpos[self.qpos_idx] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            p = self.data.site_xpos[self.site_id]
            r = self.data.site_xmat[self.site_id].reshape(3, 3)
            e_p = target_pos - p
            e_r = orientation_error(r, r_tgt)
            pos_err, rot_err = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if pos_err < pos_tol and rot_err < rot_tol:
                return IKResult(q, True, pos_err, rot_err, it)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, self.dof_idx], rot_weight * self._jacr[:, self.dof_idx]])
            e = np.concatenate([e_p, rot_weight * e_r])
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        return IKResult(q, pos_err < 5 * pos_tol and rot_err < 5 * rot_tol, pos_err, rot_err, it)
