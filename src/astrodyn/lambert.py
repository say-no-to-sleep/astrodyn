import numpy as np 
from numpy import linalg as LA
from astrodyn.stumpff import stumpff_s
from astrodyn.stumpff import stumpff_c
from astrodyn.states import StateVector

def _y(z: float, r1: float, r2: float, A: float) -> float:
    """Calculate y, the helper function

    Args:
        z (float): Universal Variable Parameter 
        r1 (float): Initial position
        r2 (float): Final position
        A (float): Constant used in calculation

    Returns:
        float: The result, y
    """
    return float(r1 + r2 + A * (z * stumpff_s(z) - 1) / np.sqrt(stumpff_c(z)))

def _F_dF(z: float, r1: float, r2: float, A: float, sqrt_mu_dt: float) -> tuple:
    """Calculates the value, F, and its derivative dF

    Args:
        z (float): Universal Variable Parameter
        r1 (float): Initial position
        r2 (float): Final position
        A (float): Constant used in calculation
        sqrt_mu_dt (float): Short for sqrt(mu)*dt, saves calculation time

    Returns:
        tuple: (F, dF)
    """
    S = stumpff_s(z)
    C = stumpff_c(z)

    y = _y(z, r1, r2, A)
    y_sqrt = np.sqrt(y)

    shared = (y / C)**(3/2)

    F = shared * S + A * y_sqrt - sqrt_mu_dt

    if np.abs(z) < 1e-6:
        # Case where z = 0
        dF = np.sqrt(2)/40 * y**(3/2) + A/8 * (np.sqrt(y) + A * np.sqrt(1/(2 * y)))
    else:
        # Normal case
        dF = (shared * (1/(2*z) * (C - (3 * S) / (2 * C)) + 3/4 * S**2/C) 
        + A/8 * (3 * S/C * y_sqrt + A * (np.sqrt(C)/y_sqrt))) 

    return (F, dF)

def solve_lambert(r1_vec: np.ndarray, r2_vec: np.ndarray, dt: float, mu: float, prograde=True) -> tuple:
    """Solves lambert's equation for initial and final velocities

    Args:
        r1_vec (np.ndarray): Initial position
        r2_vec (np.ndarray): Final position
        dt (float): Time passed since epoch
        mu (float): Universal Gravitational Parameter
        prograde (bool, optional): Decide if the orbit is prograde or retrograde. Affects calculation. Defaults to True.

    Raises:
        ValueError: Inputs are degenerate, causing dtheta to be close to 0 or pi
        ArithmeticError: Newton's method did not converge

    Returns:
        tuple: Tuple of the before and after StateVectors
    """
    # Calculate prerequisites
    r1 = LA.norm(r1_vec)
    r2 = LA.norm(r2_vec)

    sqrt_mu_dt = np.sqrt(mu) * dt

    # Calculate delta theta

    # Calculate the cross
    r1_r2_cross = np.cross(r1_vec, r2_vec)
    # Determine sign
    sign = r1_r2_cross[2] >= 0
    # Decide which of the two calculation for delta theta to use
    first_one = not (sign ^ prograde)

    if first_one: # First one is selected
        dtheta = np.arccos(np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1, 1))
    else: # Second one is selected
        dtheta = 2 * np.pi - np.arccos(np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1, 1)) 

    if np.abs(dtheta) < 1e-6 or np.abs(np.pi - dtheta) < 1e-6:
        raise ValueError("Inputs are degenerate, check your numbers")

    A = np.sin(dtheta) * np.sqrt((r1 * r2) / (1 - np.cos(dtheta)))

    # Initial guess for z
    z = 0
    iterate = 0

    while True:
        if iterate > 200:
            # did not converge
            raise ArithmeticError("Did not converge")
        
        # Calculate F and its derivative
        (F, dF) = _F_dF(z, r1, r2, A, sqrt_mu_dt)

        # Next guess for z
        z = z - F/dF

        if np.abs(F/dF) < 1e-8:
            # converged successfully
            break

        # increment count
        iterate += 1

    # Find the actual y
    y = _y(z, r1, r2, A)

    # Find f, g, and g dot
    f = 1 - y/r1
    g = A * np.sqrt(y/mu)
    g_dot = 1 - y/r2

    # Calculate v1, v2
    v1_vec = 1/g * (r2_vec - f * r1_vec)
    v2_vec = 1/g * (g_dot * r2_vec - r1_vec)

    before_state = StateVector(r1_vec, v1_vec)
    after_state = StateVector(r2_vec, v2_vec)

    return (before_state, after_state)
