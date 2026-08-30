import numpy as np
from astrodyn.states import StateVector
from astrodyn.states import ClassicalElements

def perifocal_to_eci(r_pf: np.ndarray, v_pf: np.ndarray, Omega: float, i: float, omega: float) -> StateVector:
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

    r_x = np.matmul(Q.T, r_pf)
    v_x = np.matmul(Q.T, v_pf)

    state = StateVector(r_x, v_x)
    return state