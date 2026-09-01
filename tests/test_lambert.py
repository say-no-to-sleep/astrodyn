import numpy as np
from astrodyn.states import StateVector
from astrodyn.lambert import solve_lambert
from astrodyn.constants import MU_EARTH
import astrodyn.propagate

def test_values():
    """Test against known calculated values
    """
    r1_vec = np.array([5000, 10000, 2100])
    r2_vec = np.array([-14600, 2500, 7000])
    dt = 3600
    mu = MU_EARTH

    (before_state, after_state) = solve_lambert(r1_vec, r2_vec, dt, mu, True)

    expected_v1_vec = np.array([-5.9925, 1.9254, 3.2456])
    expected_v2_vec = np.array([-3.3125, -4.1966, -0.38529])

    np.testing.assert_allclose(before_state.v_vec, expected_v1_vec, atol=1e-2)
    np.testing.assert_allclose(after_state.v_vec, expected_v2_vec, atol=1e-2)


def test_lambert_against_propagator():
    """Test lambert against known initial state, and propagated final state
    """
    r1_vec = np.array([6778, 0, 0])
    v1_vec = np.array([0, 7.669, 0])
    mu = MU_EARTH

    before_state = StateVector(r1_vec, v1_vec)
    after_state = astrodyn.propagate.propagate_universal(before_state, 2700, mu)

    r2_vec = after_state.r_vec
    v2_vec = after_state.v_vec

    # Run the lambert on known numbers

    (before_calculated, after_calculated) = solve_lambert(r1_vec, r2_vec, 2700, mu, True)

    calculated_v1_vec = before_calculated.v_vec
    calculated_v2_vec = after_calculated.v_vec

    np.testing.assert_allclose(v1_vec, calculated_v1_vec, atol=1e-10)
    np.testing.assert_allclose(v2_vec, calculated_v2_vec, atol=1e-10)