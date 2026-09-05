import numpy as np 
from numpy import linalg as LA
from astrodyn.stumpff import stumpff_s
from astrodyn.stumpff import stumpff_c
from astrodyn.states import StateVector
from collections.abc import Callable

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

def bracketed_newton(evaluate: Callable[[float], tuple[float, float]], z_low: float, z_high: float, residual_tol: float) -> float:
    """Runs a bracketed newton for more numerical stableness.
    Requires valid endpoints with F(z_low) < 0 < F(z_high) and a continuous interval between.

    Args:
        evaluate (Callable[[float], tuple[float, float]]): runs function _F_dF, returns (F, dF)
        z_low (float): Lower bound of bracket
        z_high (float): Upper bound of bracket
        residual_tol (float): How close to zero it must be to accept solution

    Raises:
        ArithmeticError: A bracket endpoint has a non-finite residual.
        ValueError: Endpoints do not satisfy F(z_low) < 0 < F(z_high).
        ArithmeticError: A residual inside the bracket is non-finite.
        ArithmeticError: The solver does not converge within 200 iterations.

    Returns:
        float: Root estimate z with abs(F(z)) <= residual_tol.
    """
    F_low, _ = evaluate(z_low)
    F_high, _ = evaluate(z_high)

    # Non finite
    if not (np.isfinite(F_low) and np.isfinite(F_high)):
        raise ArithmeticError("Non finite bracket residuals")

    if abs(F_low) <= residual_tol: return z_low
    if abs(F_high) <= residual_tol: return z_high

    if not F_low < 0 < F_high:
        raise ValueError("The endpoints bracket does not contain the root")

    z = (z_low + z_high) / 2

    # 200 steps
    for i in range(200):
        F, dF = evaluate(z)

        if not np.isfinite(F):
            raise ArithmeticError("Residual is invalid inside bracket")

        # Check the equation before taking a step
        if abs(F) <= residual_tol: return z

        # Keep the half containing the root
        if F < 0: z_low = z
        else: z_high = z

        z_next = (z_low + z_high) / 2

        if np.isfinite(dF) and dF != 0:
            candidate = z - F/dF

            margin = 0.1 * (z_high - z_low)
            if np.isfinite(candidate) and z_low + margin < candidate < z_high - margin:
                z_next = candidate

        z = z_next

    raise ArithmeticError("Lambert did not converge")
    

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

    def evaluate(z): return _F_dF(z, r1, r2, A, sqrt_mu_dt)

    (F, dF) = evaluate(z)
    if F < 0:
        # Found z_low
        z_low = 0.0
        # Initial value for search
        z_high = 0.0
        # Search upwards until F is above 0
        # Then we have found z_high
        for i in range(100):
            z_high = (z_high + 4*np.pi**2) / 2
            F_high, _ = evaluate(z_high)

            if not np.isfinite(F_high):
                raise ArithmeticError("Invalid evaluation when attempting to bracket")

            if F_high > 0: break
            z_low = z_high
        else:
            raise ArithmeticError("Could not bracket the root")

        # Run bracketed Newton
        z = bracketed_newton(evaluate, z_low, z_high, residual_tol=1e-10 * sqrt_mu_dt)
    elif F > 0:
        # Found z_high
        z_high = 0.0
        # Set initial value for search
        z_low = 0.0
        for i in range(100):
            z_low = (z_low - z) / 2
            F_low, _ = evaluate(z_low)

            if not np.isfinite(F_low):
                raise ArithmeticError("Invalid evaluation when attempting to bracket")

            if F_low < 0: break
            z_high = z_low
        else:
            raise ArithmeticError("Could not bracket the root")

        z = bracketed_newton(evaluate, z_low, z_high, residual_tol=1e-10 * sqrt_mu_dt)
    elif F == 0:
        pass

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
