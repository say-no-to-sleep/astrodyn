import numpy as np
from numpy import linalg as LA
import astrodyn
from astrodyn.states import StateVector
import astrodyn.elements
import astrodyn.propagate
from astrodyn.constants import MU_EARTH

def test_energy_conservation():
    r_vec = np.array([6778, 0, 0])
    v_vec = np.array([0, 7.669, 0])
    mu = MU_EARTH

    r = LA.norm(r_vec)
    v = LA.norm(v_vec)

    epsilon_before = v**2/2 - mu/r

    before_state = StateVector(r_vec, v_vec)
    after_state = astrodyn.propagate.propagate_universal(before_state, 2700, mu)

    r_after = LA.norm(after_state.r_vec)
    v_after = LA.norm(after_state.v_vec)

    epsilon_after = v_after**2/2 - mu/r_after

    np.testing.assert_allclose(epsilon_after, epsilon_before, atol=1e-10)


def test_ellptical_period():
    r_vec = np.array([8000, 0, 0])
    v_vec = np.array([0, 6.0, 0])
    mu = MU_EARTH

    before_state = StateVector(r_vec, v_vec)

    # Obtain period using the classical orbit elements 
    elements = astrodyn.elements.state_2_elements(before_state, mu)
    period = elements.period(mu)

    # Assert it is not None so propagate_universal works
    assert period is not None
    
    after_state = astrodyn.propagate.propagate_universal(before_state, period, mu)

    np.testing.assert_allclose(r_vec, after_state.r_vec, atol=1e-10)
    np.testing.assert_allclose(v_vec, after_state.v_vec, atol=1e-10)

def test_concrete_propagation():
    r_vec = np.array([7000, -12124, 0])
    v_vec = np.array([2.6679,4.6210,0])
    mu = MU_EARTH

    before_state = StateVector(r_vec, v_vec)
    after_state = astrodyn.propagate.propagate_universal(before_state, 3600, mu)

    desired_r_vec = np.array([-3296.8, 7413.9, 0])
    desired_v_vec = np.array([-8.2977, -0.96309, 0])

    np.testing.assert_allclose(desired_r_vec, after_state.r_vec, atol=1)
    np.testing.assert_allclose(desired_v_vec, after_state.v_vec, atol=1e-3)