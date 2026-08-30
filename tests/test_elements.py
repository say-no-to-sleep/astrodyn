import numpy as np
import astrodyn
import astrodyn.states
import astrodyn.elements
from astrodyn.constants import MU_EARTH

def test_round_trip_state_to_elements_and_back():
    r = np.array([-6045, -3490, 2500])
    v = np.array([-3.457, 6.618, 2.533])
    state = astrodyn.states.StateVector(r, v)

    elements = astrodyn.elements.state_2_elements(state, MU_EARTH)
    recovered = astrodyn.elements.elements_2_state(elements, MU_EARTH)

    np.testing.assert_allclose(recovered.r_vec, r, atol=1e-6)
    np.testing.assert_allclose(recovered.v_vec, v, atol=1e-6)