import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(viewer=False)
    s.reset()
    yield s
    s.close()


def settle(sim, seconds=0.6):
    n = int(seconds / sim.model.opt.timestep)
    sim.step(n)
