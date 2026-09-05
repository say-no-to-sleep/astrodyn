import numpy as np
import pytest
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


@pytest.mark.parametrize("speed", [12.0, 14.0])
def test_hyperbolic_lambert_against_propagator(speed):
    # The 14 km/s case overshoots y = 0 during bracket expansion.
    r1_vec = np.array([6778.0, 0, 0])
    initial = StateVector(r1_vec, np.array([0, speed, 0]))
    dt = 2700
    final = astrodyn.propagate.propagate_universal(initial, dt, MU_EARTH)

    before, after = solve_lambert(r1_vec, final.r_vec, dt, MU_EARTH)

    np.testing.assert_allclose(before.v_vec, initial.v_vec, rtol=0, atol=1e-10)
    np.testing.assert_allclose(after.v_vec, final.v_vec, rtol=0, atol=1e-10)


@pytest.mark.parametrize("dt_offset", [-1e-8, 1e-8])
def test_lambert_near_parabolic(dt_offset):
    # Barker's equation: a parabolic orbit from periapsis to true anomaly pi/2.
    radius = 6778.0
    r1_vec = np.array([radius, 0, 0])
    r2_vec = np.array([0, 2 * radius, 0])
    parabolic_dt = (4 / 3) * np.sqrt(2 * radius**3 / MU_EARTH)
    dt = parabolic_dt * (1 + dt_offset)

    before, after = solve_lambert(r1_vec, r2_vec, dt, MU_EARTH)
    propagated = astrodyn.propagate.propagate_universal(before, dt, MU_EARTH)

    energy = np.dot(before.v_vec, before.v_vec) / 2 - MU_EARTH / radius
    assert energy * dt_offset < 0
    assert abs(energy) < 1e-5
    np.testing.assert_allclose(propagated.r_vec, r2_vec, rtol=0, atol=1e-5)
    np.testing.assert_allclose(propagated.v_vec, after.v_vec, rtol=0, atol=1e-8)


@pytest.mark.parametrize("residual", [np.nan, np.inf, -np.inf])
def test_lambert_rejects_nonfinite_initial_residual(monkeypatch, residual):
    monkeypatch.setattr("astrodyn.lambert._F_dF", lambda *args: (residual, 1.0))

    with pytest.raises(ArithmeticError, match="Invalid initial Lambert residual"):
        solve_lambert(np.array([6778, 0, 0]), np.array([0, 6778, 0]), 2700, MU_EARTH)
