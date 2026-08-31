import numpy as np
import astrodyn.frames
from astrodyn.constants import MU_EARTH

def test_zero_angle_perifocal_to_eci_and_back():
    """Test known value with 0 angles to eci and back
    """
    r = np.array([-3670, -3870, 4400])
    v = np.array([4.7, -7.4, 1])

    eci = astrodyn.frames.perifocal_to_eci(r, v, 0, 0, 0)

    recovered = astrodyn.frames.eci_to_perifocal(eci.r_vec, eci.v_vec, 0, 0, 0)

    np.testing.assert_allclose(recovered.r_vec, r, atol=1e-10)
    np.testing.assert_allclose(recovered.v_vec, v, atol=1e-10)



def test_perifocal_to_eci_and_back():
    """Test known value with plausible angles to eci and back
    """
    r = np.array([-3670, -3870, 4400])
    v = np.array([4.7, -7.4, 1])

    eci = astrodyn.frames.perifocal_to_eci(r, v, np.deg2rad(40), np.deg2rad(30), np.deg2rad(60))

    recovered = astrodyn.frames.eci_to_perifocal(eci.r_vec, eci.v_vec, np.deg2rad(40), np.deg2rad(30), np.deg2rad(60))

    np.testing.assert_allclose(recovered.r_vec, r, atol=1e-10)
    np.testing.assert_allclose(recovered.v_vec, v, atol=1e-10)



def test_eci_to_perifocal_known_value():
    """Test known r, v, and hand calculate Q to compare calculated value to expected value
    """
    r = np.array([1, 0, 0])
    v = np.array([0, 1, 0])

    eci = astrodyn.frames.eci_to_perifocal(r, v, 0, np.deg2rad(90), 0)

    np.testing.assert_allclose(eci.r_vec, np.array([1, 0, 0]), atol = 1e-10)
    np.testing.assert_allclose(eci.v_vec, np.array([0, 0, -1]), atol = 1e-10)