"""Parallel-jaw gripper interface (two position-controlled, force-limited slide fingers)."""
from __future__ import annotations

import numpy as np

from simulation import Simulation

OPEN_POS = 0.040      # per-finger opening (m): 8 cm gap
CLOSED_POS = 0.0      # command for "squeeze": fingers stop on the object
NOMINAL_FORCE = 12.0  # N per finger (actuator force limit)
MAX_FORCE = 20.0      # hard ceiling recovery may raise the force to


class Gripper:
    def __init__(self, sim: Simulation):
        self.sim = sim
        m = sim.model
        self.act = [m.actuator("finger_left_act").id, m.actuator("finger_right_act").id]
        self.qadr = [m.jnt_qposadr[m.joint(j).id] for j in ("finger_left_joint", "finger_right_joint")]
        self.dadr = [m.jnt_dofadr[m.joint(j).id] for j in ("finger_left_joint", "finger_right_joint")]
        self.pad_geoms = {m.geom("finger_left_pad").id, m.geom("finger_right_pad").id}
        self.left_pad = m.geom("finger_left_pad").id
        self.right_pad = m.geom("finger_right_pad").id
        self.command = OPEN_POS
        self.set_force_limit(NOMINAL_FORCE)

    def set_force_limit(self, force: float) -> None:
        force = float(np.clip(force, 0.0, MAX_FORCE))
        for a in self.act:
            self.sim.model.actuator_forcerange[a] = [-force, force]
        self.force_limit = force

    def set(self, pos: float) -> None:
        self.command = float(pos)
        for a in self.act:
            self.sim.data.ctrl[a] = self.command

    def gap(self) -> float:
        """Current distance between the inner faces of the pads (m)."""
        return float(sum(self.sim.data.qpos[i] for i in self.qadr))

    def finger_speed(self) -> float:
        return float(max(abs(self.sim.data.qvel[i]) for i in self.dadr))

    def actuate(self, pos: float, min_time: float = 0.3, timeout: float = 1.5,
                on_step=None) -> dict:
        """Command the fingers and step until they stop moving (or timeout)."""
        self.set(pos)
        t0 = self.sim.time
        settled = False
        while self.sim.time - t0 < timeout:
            self.sim.step(5)
            if on_step is not None:
                on_step()
            if self.sim.time - t0 > min_time and self.finger_speed() < 1e-3:
                settled = True
                break
        return {"settled": settled, "gap": self.gap(), "elapsed": self.sim.time - t0}

    def open(self, **kw) -> dict:
        return self.actuate(OPEN_POS, **kw)

    def close(self, **kw) -> dict:
        return self.actuate(CLOSED_POS, **kw)
