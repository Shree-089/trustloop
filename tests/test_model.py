"""Stage 1-2 checks: model loads, is physically stable, arm moves, gripper actuates, cube can be lifted."""
import mujoco
import numpy as np

from conftest import settle
from gripper import Gripper
from kinematics import IKSolver
from robot_controller import RobotController
from simulation import DEFAULT_SCENE, Simulation


def test_model_loads_with_required_elements():
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    for body in ("base", "hand", "finger_left", "finger_right", "cube", "container", "table"):
        assert m.body(body).id >= 0
    for geom in ("cube_geom", "bin_floor", "bin_wall_px", "bin_wall_nx", "bin_wall_py",
                 "bin_wall_ny", "finger_left_pad", "finger_right_pad", "table_top"):
        g = m.geom(geom).id
        assert m.geom_contype[g] or m.geom_conaffinity[g], f"{geom} has no collision"
    assert m.nu == 8                      # 6 arm + 2 finger actuators
    assert sum(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE) == 6
    assert abs(m.body_mass[m.body("cube").id] - 0.05) < 1e-9


def test_scene_is_physically_stable(sim):
    p0 = sim.cube_pos()
    settle(sim, 3.0)
    assert np.all(np.isfinite(sim.data.qpos))
    assert np.linalg.norm(sim.cube_pos() - p0) < 1e-3     # cube rests on the table
    assert abs(sim.cube_pos()[2] - 0.42) < 2e-3


def test_cube_falls_into_container_and_stays():
    s = Simulation()
    s.reset(cube_pos=[0.40, 0.25, 0.60])  # drop it from above the bin
    settle(s, 2.0)
    p = s.cube_pos()
    assert abs(p[0] - 0.40) < 0.06 and abs(p[1] - 0.25) < 0.06
    assert abs(p[2] - 0.43) < 3e-3        # resting on the bin floor (floor top 0.41)


def test_ik_accuracy(sim):
    ik = IKSolver(sim.model)
    q0 = sim.data.qpos[ik.qpos_idx].copy()
    for target in ([0.40, -0.20, 0.45], [0.40, 0.25, 0.55], [0.30, 0.0, 0.60]):
        r = ik.solve(target, 0.3, q0, max_iters=300)
        assert r.success, r
        p, R = ik.forward(r.q)
        assert np.linalg.norm(p - target) < 1e-3
        assert R[2, 2] < -0.99           # tool points down
        assert np.all(r.q >= ik.lo - 1e-9) and np.all(r.q <= ik.hi + 1e-9)


def test_arm_moves_toward_cube(sim):
    rc = RobotController(sim)
    rc.sync_command()
    target = sim.cube_pos() + [0, 0, 0.12]
    d0 = np.linalg.norm(rc.ee_pos() - target)
    res = rc.move_to(target, rc.nearest_grasp_yaw(sim.cube_yaw()), 2.0)
    assert res.success, res
    assert np.linalg.norm(rc.ee_pos() - target) < 0.005 < d0


def test_unreachable_target_is_rejected_quickly(sim):
    rc = RobotController(sim)
    rc.sync_command()
    t0 = sim.time
    res = rc.move_to([2.0, 0.0, 0.5], 0.0, 2.0)
    assert not res.success and not res.ik_ok
    assert sim.time - t0 < 0.1


def test_gripper_opens_and_closes(sim):
    g = Gripper(sim)
    assert g.close()["gap"] < 0.003
    assert abs(g.open()["gap"] - 0.08) < 0.002


def test_gripper_physically_lifts_cube(sim):
    rc, g = RobotController(sim), Gripper(sim)
    rc.sync_command()
    c = sim.cube_pos()
    yaw = rc.nearest_grasp_yaw(sim.cube_yaw())
    assert rc.move_to(c + [0, 0, 0.1], yaw, 1.5).success
    g.open()
    assert rc.move_to(c, yaw, 1.0).success
    g.close()
    assert rc.move_to(c + [0, 0, 0.15], yaw, 1.5).success
    rc.hold(0.5)
    assert sim.cube_pos()[2] > c[2] + 0.14            # cube really went up with the gripper
