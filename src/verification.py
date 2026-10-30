"""TrustLoop independent verification engine.

Everything here reads the *physical* MuJoCo state (body poses, velocities,
contacts, finger joint positions). Nothing here looks at what the controller
intended or commanded. Container dimensions come from the model geoms, so if
the bin is moved or resized in the XML the verifier follows automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import product

import mujoco
import numpy as np

from simulation import Simulation

# Tolerances (metres / SI). Tuned to be strict but robust to contact jitter.
HORIZ_TOL = 0.002          # allowed overlap of cube footprint with wall inner face
FLOOR_TOL_BELOW = 0.003    # cube bottom may sink this far into floor (soft contact)
FLOOR_TOL_ABOVE = 0.006    # ...or hover this far above it and still count as resting
STABLE_LIN_VEL = 0.01      # m/s
STABLE_ANG_VEL = 0.10      # rad/s
RELEASE_CLEARANCE = 0.03   # min distance between grasp site and cube centre after release


@dataclass
class VerificationResult:
    success: bool
    reason: str
    failure: str | None
    checks: dict
    measurements: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _f(x) -> float:
    return round(float(x), 5)


class Verifier:
    def __init__(self, sim: Simulation):
        self.sim = sim
        m = sim.model
        self.cube_geom = m.geom("cube_geom").id
        self.cube_half = m.geom_size[self.cube_geom].copy()
        self.table_geom = m.geom("table_top").id
        self.left_pad = m.geom("finger_left_pad").id
        self.right_pad = m.geom("finger_right_pad").id
        self.container_body = m.body("container").id
        self.floor_geom = m.geom("bin_floor").id
        self.wall_geoms = [m.geom(n).id for n in ("bin_wall_px", "bin_wall_nx", "bin_wall_py", "bin_wall_ny")]
        # All geoms belonging to the robot (base subtree) – used for "released" check.
        base = m.body("base").id
        self.robot_geoms = {g for g in range(m.ngeom) if self._descends_from(m.geom_bodyid[g], base)}
        self._compute_container_geometry()

    def _descends_from(self, body: int, root: int) -> bool:
        m = self.sim.model
        while body != 0:
            if body == root:
                return True
            body = m.body_parentid[body]
        return False

    # ------------------------------------------------------------ geometry
    def _compute_container_geometry(self) -> None:
        """Interior box of the container, in the container body frame, from the geoms."""
        m = self.sim.model
        fp, fs = m.geom_pos[self.floor_geom], m.geom_size[self.floor_geom]
        self.floor_top = float(fp[2] + fs[2])
        inner_x, inner_y, rims = [], [], []
        for g in self.wall_geoms:
            p, s = m.geom_pos[g], m.geom_size[g]
            rims.append(p[2] + s[2])
            if s[0] < s[1]:   # thin in x -> an x-wall
                inner_x.append(p[0] - s[0] if p[0] > 0 else p[0] + s[0])
            else:             # thin in y -> a y-wall
                inner_y.append(p[1] - s[1] if p[1] > 0 else p[1] + s[1])
        self.interior_x = (min(inner_x), max(inner_x))
        self.interior_y = (min(inner_y), max(inner_y))
        self.rim = float(min(rims))

    def container_info(self) -> dict:
        d = self.sim.data
        return {
            "center_world": [_f(v) for v in d.xpos[self.container_body]],
            "interior_x_local": [_f(v) for v in self.interior_x],
            "interior_y_local": [_f(v) for v in self.interior_y],
            "floor_top_local": _f(self.floor_top),
            "rim_local": _f(self.rim),
        }

    def container_center_world(self) -> np.ndarray:
        return self.sim.data.xpos[self.container_body].copy()

    def floor_top_world(self) -> float:
        return float(self.sim.data.xpos[self.container_body][2] + self.floor_top)

    def rim_world(self) -> float:
        return float(self.sim.data.xpos[self.container_body][2] + self.rim)

    def cube_corners_world(self) -> np.ndarray:
        p, r = self.sim.cube_pos(), self.sim.cube_rot()
        h = self.cube_half
        return np.array([p + r @ (np.array(s) * h) for s in product((-1, 1), repeat=3)])

    def cube_corners_in_container(self) -> np.ndarray:
        d = self.sim.data
        pc = d.xpos[self.container_body]
        rc = d.xmat[self.container_body].reshape(3, 3)
        return (self.cube_corners_world() - pc) @ rc

    # ------------------------------------------------------------- contacts
    def contacts_with_cube(self) -> list[tuple[int, np.ndarray, float]]:
        """(other_geom, world contact position, normal force) for every cube contact."""
        out = []
        d, m = self.sim.data, self.sim.model
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.cube_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.cube_geom else c.geom1
            mujoco.mj_contactForce(m, d, i, f6)
            out.append((int(other), c.pos.copy(), float(abs(f6[0]))))
        return out

    def cube_touching(self, geoms) -> bool:
        geoms = set(geoms) if not isinstance(geoms, (set, frozenset)) else geoms
        return any(g in geoms for g, _, _ in self.contacts_with_cube())

    def cube_offset_from_ee(self) -> np.ndarray:
        return self.sim.cube_pos() - self.sim.site_pos("ee_site")

    # =========================================================== GRASP CHECKS
    def check_grasp_closure(self, gap: float) -> VerificationResult:
        """Right after the close command: is the cube physically pinched between both pads?"""
        cube_w = 2 * float(self.cube_half[0])
        left = self.cube_touching({self.left_pad})
        right = self.cube_touching({self.right_pad})
        # Gap must be consistent with an object between the pads: not ~0 (closed
        # on air) and not wide open. Range covers any cube orientation up to the diagonal.
        gap_ok = (cube_w - 0.004) <= gap <= (cube_w * 1.42 + 0.004)
        # Cube centre must lie between the pads in the tool frame.
        r_ee = self.sim.data.site("ee_site").xmat.reshape(3, 3)
        off_tool = r_ee.T @ self.cube_offset_from_ee()
        centred = abs(off_tool[0]) < 0.012 and abs(off_tool[1]) < 0.015 and abs(off_tool[2]) < 0.03
        checks = {"left_pad_contact": left, "right_pad_contact": right,
                  "gap_consistent_with_cube": gap_ok, "cube_between_fingers": centred}
        meas = {"gap": _f(gap), "cube_width": _f(cube_w),
                "cube_offset_tool_frame": [_f(v) for v in off_tool]}
        if all(checks.values()):
            return VerificationResult(True, "Cube pinched between both pads", None, checks, meas)
        if not left and not right:
            reason, failure = "No finger contact with cube – gripper closed on empty space", "grasp_missed"
        elif not (left and right):
            reason, failure = "Only one finger touches the cube", "grasp_partial"
        elif not gap_ok:
            reason, failure = "Finger gap inconsistent with cube width", "grasp_missed"
        else:
            reason, failure = "Cube not centred between the fingers", "grasp_partial"
        return VerificationResult(False, reason, failure, checks, meas)

    def check_grasp_lift(self, ref: dict, ee_rise: float) -> VerificationResult:
        """After a small test lift: did the cube actually come with the gripper?
        `ref` is a snapshot taken (by snapshot()) just before the test lift."""
        cube_rise = float(self.sim.cube_pos()[2] - ref["cube_pos"][2])
        drift = float(np.linalg.norm(self.cube_offset_from_ee() - ref["cube_offset"]))
        on_table = self.cube_touching({self.table_geom})
        pads = self.cube_touching({self.left_pad}) and self.cube_touching({self.right_pad})
        checks = {
            "cube_height_increased": cube_rise > 0.7 * ee_rise,
            "cube_follows_gripper": drift < 0.008,
            "lifted_off_table": not on_table,
            "still_pinched": pads,
        }
        meas = {"ee_rise": _f(ee_rise), "cube_rise": _f(cube_rise), "relative_drift": _f(drift)}
        if all(checks.values()):
            return VerificationResult(True, "Cube lifted together with gripper", None, checks, meas)
        if not checks["cube_height_increased"] or on_table:
            reason = "Cube stayed on the table while the gripper rose (slipped out)"
        else:
            reason = "Cube moved relative to the gripper during the test lift (slipping)"
        return VerificationResult(False, reason, "grasp_slipped", checks, meas)

    def check_carry(self, ref_offset: np.ndarray, phase: str) -> VerificationResult:
        """During transport: is the cube still held where it was grasped?"""
        drift = float(np.linalg.norm(self.cube_offset_from_ee() - ref_offset))
        pads = self.cube_touching({self.left_pad}) and self.cube_touching({self.right_pad})
        checks = {"still_pinched": pads, "cube_follows_gripper": drift < 0.012}
        meas = {"relative_drift": _f(drift), "cube_pos": [_f(v) for v in self.sim.cube_pos()]}
        if all(checks.values()):
            return VerificationResult(True, f"Cube held during {phase}", None, checks, meas)
        return VerificationResult(False, f"Cube lost or slipping during {phase}",
                                  "dropped_in_transit", checks, meas)

    def snapshot(self) -> dict:
        return {"cube_pos": self.sim.cube_pos(), "cube_offset": self.cube_offset_from_ee(),
                "ee_pos": self.sim.site_pos("ee_site")}

    # ======================================================= PLACEMENT CHECK
    def is_stable(self) -> tuple[bool, float, float]:
        v, w = self.sim.cube_vel()
        lv, av = float(np.linalg.norm(v)), float(np.linalg.norm(w))
        return (lv < STABLE_LIN_VEL and av < STABLE_ANG_VEL), lv, av

    def verify_placement(self) -> VerificationResult:
        corners = self.cube_corners_in_container()
        x0, x1 = self.interior_x
        y0, y1 = self.interior_y
        inside = bool(np.all(corners[:, 0] >= x0 - HORIZ_TOL) and np.all(corners[:, 0] <= x1 + HORIZ_TOL)
                      and np.all(corners[:, 1] >= y0 - HORIZ_TOL) and np.all(corners[:, 1] <= y1 + HORIZ_TOL))
        bottom, top = float(corners[:, 2].min()), float(corners[:, 2].max())
        below_rim = top <= self.rim + 0.002

        contacts = self.contacts_with_cube()
        touching_floor = any(g == self.floor_geom for g, _, _ in contacts)
        resting = touching_floor and (self.floor_top - FLOOR_TOL_BELOW <= bottom <= self.floor_top + FLOOR_TOL_ABOVE)

        # "On the rim": any wall contact at (near) rim height, or the cube's bottom
        # is up at rim level.
        rim_world = self.rim_world()
        wall_top_contact = any(g in self.wall_geoms and p[2] > rim_world - 0.006 for g, p, _ in contacts)
        not_on_rim = not wall_top_contact and bottom < self.rim - 0.01

        robot_contact = any(g in self.robot_geoms for g, _, _ in contacts)
        clearance = float(np.linalg.norm(self.cube_offset_from_ee()))
        released = (not robot_contact) and clearance > RELEASE_CLEARANCE

        stable, lv, av = self.is_stable()

        checks = {
            "inside_horizontal_bounds": inside,
            "below_rim": below_rim,
            "resting_on_floor": resting,
            "not_on_rim": not_on_rim,
            "released_from_gripper": released,
            "stable": stable,
        }
        meas = {
            "cube_pos_world": [_f(v) for v in self.sim.cube_pos()],
            "cube_bottom_local": _f(bottom), "cube_top_local": _f(top),
            "floor_top_local": _f(self.floor_top), "rim_local": _f(self.rim),
            "interior_x_local": [_f(v) for v in self.interior_x],
            "interior_y_local": [_f(v) for v in self.interior_y],
            "cube_xy_extent_local": [[_f(corners[:, 0].min()), _f(corners[:, 0].max())],
                                     [_f(corners[:, 1].min()), _f(corners[:, 1].max())]],
            "lin_vel": _f(lv), "ang_vel": _f(av), "ee_clearance": _f(clearance),
            "contacts": sorted({self.sim.model.geom(g).name or f"geom{g}" for g, _, _ in contacts}),
        }
        if all(checks.values()):
            return VerificationResult(True, "Cube settled inside container", None, checks, meas)

        # Failure categorisation (most specific first).
        table_z_local = float(self.sim.data.geom_xpos[self.table_geom][2]
                              + self.sim.model.geom_size[self.table_geom][2]
                              - self.sim.data.xpos[self.container_body][2])
        if not released:
            reason, failure = "Cube is still in contact with / carried by the gripper", "still_in_gripper"
        elif bottom < table_z_local - 0.05:
            reason, failure = "Cube fell off the table", "cube_fell_off_table"
        elif not not_on_rim:
            reason, failure = "Cube is resting on the container rim/wall top", "resting_on_rim"
        elif not inside:
            reason, failure = "Cube landed outside the container", "placed_outside_container"
        elif not stable:
            reason, failure = "Cube has not settled yet", "not_settled"
        else:
            reason, failure = "Cube inside container but not resting flat on the floor", "not_resting_on_floor"
        return VerificationResult(False, reason, failure, checks, meas)
