"""Grasp verification must distinguish a real grasp from controlled failures."""
import numpy as np
import pytest

from gripper import Gripper
from robot_controller import RobotController
from verification import Verifier


def _grasp(sim, offset=(0, 0, 0), force=None):
    rc, g, v = RobotController(sim), Gripper(sim), Verifier(sim)
    rc.sync_command()
    if force is not None:
        g.set_force_limit(force)
    c = sim.cube_pos() + np.asarray(offset)
    yaw = rc.nearest_grasp_yaw(sim.cube_yaw())
    assert rc.move_to(c + [0, 0, 0.12], yaw, 1.5).success
    g.open()
    assert rc.move_to(c, yaw, 1.0).success
    closed = g.close()
    return rc, g, v, yaw, closed


def _test_lift(rc, v, yaw):
    ref = v.snapshot()
    rc.move_to(ref["ee_pos"] + [0, 0, 0.03], yaw, 0.8)
    rc.hold(0.2)
    return v.check_grasp_lift(ref, float(rc.ee_pos()[2] - ref["ee_pos"][2]))


def test_successful_grasp_passes_both_checks(sim):
    rc, g, v, yaw, closed = _grasp(sim)
    res = v.check_grasp_closure(closed["gap"])
    assert res.success, res.to_dict()
    lift = _test_lift(rc, v, yaw)
    assert lift.success, lift.to_dict()
    assert all(lift.checks.values())


def test_closing_on_empty_space_is_detected(sim):
    """Controlled failure: perception error of 5 cm -> fingers close beside the cube."""
    rc, g, v, yaw, closed = _grasp(sim, offset=(0, 0.05, 0))
    res = v.check_grasp_closure(closed["gap"])
    assert not res.success
    assert res.failure == "grasp_missed"
    assert not res.checks["left_pad_contact"] and not res.checks["right_pad_contact"]


def test_weak_grip_slip_is_detected_by_lift_test(sim):
    """Controlled failure: grip force too low. The fingers DO touch the cube, so a
    naive 'contact = grasped' check would pass; the physical test lift must not."""
    rc, g, v, yaw, closed = _grasp(sim, force=0.1)
    closure = v.check_grasp_closure(closed["gap"])
    assert closure.success                     # contacts look fine...
    lift = _test_lift(rc, v, yaw)
    assert not lift.success                    # ...but the cube does not come along
    assert lift.failure == "grasp_slipped"
    assert not lift.checks["cube_height_increased"]


def test_issuing_close_command_alone_is_not_success(sim):
    """Close the gripper far from the cube: the command completes, verification fails."""
    v, g = Verifier(sim), Gripper(sim)
    closed = g.close()
    res = v.check_grasp_closure(closed["gap"])
    assert not res.success and res.failure == "grasp_missed"


@pytest.mark.parametrize("yaw_deg", [-40, 0, 25])
def test_grasp_verification_with_rotated_cube(yaw_deg):
    from simulation import Simulation
    s = Simulation()
    s.reset(cube_pos=[0.38, -0.18, 0.42], cube_yaw=np.radians(yaw_deg))
    rc, g, v, yaw, closed = _grasp(s)
    assert v.check_grasp_closure(closed["gap"]).success
    assert _test_lift(rc, v, yaw).success
