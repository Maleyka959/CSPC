"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


# TODO 1: test_rejects_negative_rate
#   Check that calling simulate(...) with a negative lam raises a ValueError.
#   Which pytest tool checks that an error is raised?


# TODO 2: test_matches_law
#   Check that the simulation's AVERAGE over many seeds is close to the
#   physical law  N0 * exp(-lam * t).
#   Which pytest tool compares floating-point values with a tolerance?

import pytest
import numpy as np

# 1. Test for negative rate raising ValueError
def test_negative_rate_raises_value_error():
    with pytest.raises(ValueError):
        simulate(N0=1000, lam=-0.5, dt=0.05, steps=200)

# 2. Test average/final count matches N0 * exp(-lam * total_time)
def test_simulation_average_close_to_theoretical():
    N0 = 10000
    lam = 0.05
    dt = 0.05
    steps = 200
    total_time = steps * dt  # total simulation time = 10.0
    
    expected = N0 * np.exp(-lam * total_time)
    
    # Get final atom count from returned array
    counts = simulate(N0=N0, lam=lam, dt=dt, steps=steps)
    actual_remaining = counts[-1]
    
    assert actual_remaining == pytest.approx(expected, rel=0.05)