"""Placement verification must judge the physical state, not the plan."""
import numpy as np

from conftest import settle
from gripper import Gripper
from robot_controller import RobotController
from simulation import Simulation
from verification import Verifier

BIN = np.array([0.40, 0.25])
FLOOR_TOP = 0.41


def _place_cube_and_settle(pos, yaw=0.0, seconds=1.0):
    s = Simulation()
    s.reset(cube_pos=pos, cube_yaw=yaw)
    settle(s, seconds)
    return s, Verifier(s)


def test_cube_outside_container_is_not_success():
    """Required test: cube deliberately starts outside the container."""
    s, v = _place_cube_and_settle([0.40, -0.20, 0.42])
    res = v.verify_placement()
    assert not res.success
    assert res.failure == "placed_outside_container"
    assert not res.checks["inside_horizontal_bounds"]
    assert not res.checks["resting_on_floor"]


def test_cube_just_next_to_container_is_not_success():
    # Touching the outside of the -y wall: close to the target, but not in it.
    s, v = _place_cube_and_settle([0.40, 0.25 - 0.08 - 0.021, 0.42])
    res = v.verify_placement()
    assert not res.success and res.failure == "placed_outside_container"


def test_cube_inside_container_is_success():
    s, v = _place_cube_and_settle([0.41, 0.24, FLOOR_TOP + 0.02], yaw=0.4)
    res = v.verify_placement()
    assert res.success, res.to_dict()
    assert all(res.checks.values())


def test_cube_balanced_on_rim_is_not_success():
    wall_x = 0.40 + 0.075
    s, v = _place_cube_and_settle([wall_x, 0.25, 0.49 + 0.02])
    assert s.cube_pos()[2] > 0.50                          # it really is balancing up there
    res = v.verify_placement()
    assert not res.success
    assert res.failure == "resting_on_rim"
    assert not res.checks["not_on_rim"] and not res.checks["below_rim"]


def test_falling_cube_is_not_yet_success():
    s = Simulation()
    s.reset(cube_pos=[0.40, 0.25, 0.70], settle_steps=0)
    s.step(60)                                             # mid-fall, inside bounds
    res = Verifier(s).verify_placement()
    assert not res.success
    assert not res.checks["stable"] and not res.checks["resting_on_floor"]


def test_cube_still_in_gripper_is_not_success():
    """Cube lowered into the bin but never released."""
    s = Simulation()
    s.reset()
    rc, g, v = RobotController(s), Gripper(s), Verifier(s)
    rc.sync_command()
    c = s.cube_pos()
    yaw = rc.nearest_grasp_yaw(s.cube_yaw())
    rc.move_to(c + [0, 0, 0.1], yaw, 1.5); g.open()
    rc.move_to(c, yaw, 1.0); g.close()
    rc.move_to([BIN[0], BIN[1], 0.62], yaw, 2.0)
    rc.move_to([BIN[0], BIN[1], FLOOR_TOP + 0.022], yaw, 1.5)
    rc.hold(0.5)
    res = v.verify_placement()
    assert not res.success
    assert res.failure == "still_in_gripper"
    assert not res.checks["released_from_gripper"]


def test_verifier_uses_model_geometry_not_hardcoded_target():
    """Move the container in the model: the same cube pose flips from pass to fail."""
    s = Simulation()
    cid = s.model.body("container").id
    s.model.body_pos[cid] = [0.40, 0.0, 0.40]             # bin moved 25 cm
    s.reset(cube_pos=[0.40, 0.0, 0.43])
    settle(s, 1.0)
    v = Verifier(s)
    assert v.verify_placement().success
    # Cube sitting exactly where the bin USED to be (original target) now fails.
    s.reset(cube_pos=[0.40, 0.25, 0.42])
    settle(s, 1.0)
    assert not v.verify_placement().success
