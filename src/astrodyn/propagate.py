import numpy as np
from numpy import linalg as LA
from astrodyn.stumpff import stumpff_c
from astrodyn.stumpff import stumpff_s
from astrodyn.states import StateVector

def _solve_universal_kepler(dt: float, r0: float, vr0: float, alpha: float, mu: float) -> float:
    """Solve the universal Kepler equation for the universal anomaly chi via Newton iteration.

    TODO: Implement Halley's method with second derivative

    Args:
        dt (float): Elapsed time since epoch [s]
        r0 (float): Initial orbit radius [km]
        vr0 (float): Initial radial velocity [km/s]
        alpha (float): Reciprocal of semimajor axis [1/km]
        mu (float): Gravitational parameter [km^3/s^2]

    Raises:
        ArithmeticError: If Newton iteration does not converge within 50 steps.

    Returns:
        float: Universal anomaly chi [km^(1/2)].
    """

    # Implement Newton Iteration

    # Initial guess
    chi = np.sqrt(mu) * np.abs(alpha) * dt

    # Keep iteration count. If it exceeds 50, we failed to converge.
    iterations = 0

    while True:
        # Calculate the substitution variable z = a chi^2
        z = alpha * chi**2
        S = stumpff_s(z)
        C = stumpff_c(z)

        # Calculate f and df for iteration
        f = ((r0 * vr0) / np.sqrt(mu)) * chi**2 * C + (1 - alpha * r0) * chi**3 * S + r0 * chi - np.sqrt(mu) * dt
        df = ((r0 * vr0) / np.sqrt(mu)) * chi * (1 - alpha * chi**2 * S) + (1 - alpha * r0) * chi**2 * C + r0

        # Converged
        if np.abs(f/df) < 1e-8:
            break

        # Did not converge
        if iterations > 50:
            raise ArithmeticError("Newton iteration failed to converge, check numbers.")

        # Calculate value for next iteration
        chi = chi - (f/df)
        iterations += 1

    return chi

def propagate_universal(before_state: StateVector, dt: float, mu: float) -> StateVector:
    """Propagate a two-body state vector forward by dt using universal variables and Lagrange coefficients

    Args:
        before_state (StateVector): Initial position and velocity vector states
        dt (float): Elapsed time since epoch [s]
        mu (float): Gravitational parameter [km^3/s^2]

    Returns:
        StateVector: Propagated position and velocity vectors
    """

    r0_vec = before_state.r_vec
    v0_vec = before_state.v_vec

    # Compute the variables needed
    r0 = LA.norm(r0_vec)
    v0 = LA.norm(v0_vec)
    vr0 = (np.dot(r0_vec, v0_vec)/r0)

    # Compute the reciprocal of semimajor axis
    alpha = 2/r0 - (v0**2)/mu

    # Solve the universal kepler equation for chi
    chi = _solve_universal_kepler(dt, r0, vr0, alpha, mu)

    # Calculate the substitution variable and stumpff 
    z = alpha * chi**2
    C = stumpff_c(z)
    S = stumpff_s(z)

    # Compute f and g
    f = 1 - (chi**2)/r0 * C
    g = dt - chi**3/np.sqrt(mu) * S

    # Compute r
    r_vec = f * r0_vec + g * v0_vec
    r = LA.norm(r_vec)

    # Compute f dot and g dot
    f_dot = np.sqrt(mu)/(r * r0) * (alpha * chi**3 * S - chi)
    g_dot = 1 - chi**2/r * C

    # Compute v
    v_vec = f_dot * r0_vec + g_dot * v0_vec

    # Lagrange identity check
    assert abs(f * g_dot - f_dot * g - 1.0) < 1e-6

    # Return the current state
    state = StateVector(r_vec, v_vec)
    return state