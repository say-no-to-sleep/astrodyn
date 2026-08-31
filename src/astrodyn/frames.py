import numpy as np
from astrodyn.states import StateVector
from astrodyn.states import ClassicalElements

def _perifocal_eci_conversion(r: np.ndarray, v: np.ndarray, Omega: float, i: float, omega: float, direction: str) -> StateVector:
    # Build Q

    ## Build R3(omega)
    R3_omega = np.array([[np.cos(omega), np.sin(omega), 0],
                         [-np.sin(omega), np.cos(omega), 0],
                         [0, 0, 1]])
    

    ## Build R1(i)
    R1_i = np.array([[1, 0, 0],
                        [0, np.cos(i), np.sin(i)],
                        [0, -np.sin(i), np.cos(i)]])

    ## Build R3(Omega)
    R3_Omega = np.array([[np.cos(Omega), np.sin(Omega), 0],
                         [-np.sin(Omega), np.cos(Omega), 0],
                         [0, 0, 1]])

    Q = np.matmul(np.matmul(R3_omega, R1_i), R3_Omega)

    if direction == "pf_to_eci":
        Q = Q.T
    elif direction == "eci_to_pf":
        Q = Q
    else:
        raise ValueError('Invalid direction "%s", choose between "pf_to_eci" and "eci_to_pf"', direction)

    r_x = np.matmul(Q, r)
    v_x = np.matmul(Q, v)

    state = StateVector(r_x, v_x)
    return state

def perifocal_to_eci(r_pf: np.ndarray, v_pf: np.ndarray, Omega: float, i: float, omega: float) -> StateVector:
    return _perifocal_eci_conversion(r_pf, v_pf, Omega, i, omega, "pf_to_eci")

def eci_to_perifocal(r_eci: np.ndarray, v_eci: np.ndarray, Omega: float, i: float, omega: float) -> StateVector:
    return _perifocal_eci_conversion(r_eci, v_eci, Omega, i, omega, "eci_to_pf")