import numpy as np
from numpy import linalg as LA
from astrodyn.frames import perifocal_to_eci
from astrodyn.states import StateVector
from astrodyn.states import ClassicalElements

circular_tolerance = 1e-8
equatorial_tolerance = 1e-8

def state_2_elements(state: StateVector, mu: float) -> ClassicalElements:
    """Convert the current state to classical orbit elements

    Args:
        state_vector (StateVector): Contains the positional vector and velocity vector
        mu (float): Universal Gravitational Parameter

    Returns:
        ClassicalElements: Classical Orbit Elements
    """
    r_vec = state.r_vec
    v_vec = state.v_vec

    # Get magnitude of r and v
    r = LA.norm(r_vec)
    v = LA.norm(v_vec)

    # Calculate v_r
    # This number being positive would be moving away from periapsis
    v_r = np.dot(r_vec, v_vec) /r

    # Calculate the specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    # Return as standard python type
    h = LA.norm(h_vec).item()

    # Calculate the inclination of the orbit
    # Clamping acos
    i = np.arccos(np.clip((h_vec[2]) / h, -1, 1)).item()

    # Calculate the Node Vector
    # Here we cross K hat with h, which result in (-h_y, h_x, 0)
    # N = np.cross(np.array([0,0,1]), h)
    N_vec = np.array([-h_vec[1], h_vec[0], 0])
    N = LA.norm(N_vec)

    # Calculate the eccentricity (shape)
    e_vec = (1/mu) * ((v*v - mu/r) * r_vec - r * v_r * v_vec)
    e = LA.norm(e_vec).item()

    # Circular cases

    is_circular = e < circular_tolerance
    is_equatorial = (N / h) < equatorial_tolerance

    if is_circular and is_equatorial:
        # Nothing except position direction is useful
        Omega = 0.0
        omega = 0.0
        true_longitude = resolve_arccos(np.arccos(np.clip(r_vec[0]/r, -1, 1)), r_vec[1])
        theta = true_longitude
    elif is_circular and not is_equatorial:
        Omega = resolve_arccos(np.arccos(np.clip(N_vec[0]/N, -1, 1)), N_vec[1])
        omega = 0.0
        argument_of_latitude = resolve_arccos(np.arccos(np.clip(np.dot(N_vec, r_vec) / (N * r), -1, 1)), r_vec[2])
        theta = argument_of_latitude
    elif not is_circular and is_equatorial:
        Omega = 0.0
        longitude_of_periapsis = resolve_arccos(np.arccos(np.clip(e_vec[0] / e, -1, 1)), e_vec[1])
        omega = longitude_of_periapsis
        theta = resolve_arccos(np.arccos(np.clip( np.dot(e_vec, r_vec) / (e * r) , -1, 1)), v_r)
    else:
        Omega = resolve_arccos(np.arccos(np.clip(N_vec[0]/N, -1, 1)), N_vec[1])
        omega = resolve_arccos(np.arccos(np.clip((np.dot(N_vec, e_vec))/ (N * e), -1, 1)), e_vec[2])
        theta = resolve_arccos(np.arccos(np.clip( np.dot(e_vec, r_vec) / (e * r) , -1, 1)), v_r)

    elements = ClassicalElements(h, e, i, Omega, omega, theta)
    return elements

def elements_2_state(elements: ClassicalElements, mu: float) -> StateVector:

    h = elements.h
    e = elements.e
    theta = elements.theta

    # Build position in perifocal frame
    r_x_bar = (h*h / mu) / (1 + e * np.cos(theta)) * np.array([np.cos(theta), np.sin(theta), 0])

    # Build velocity in perifocal frame
    v_x_bar = (mu / h) * np.array([-np.sin(theta), e + np.cos(theta), 0])

    return perifocal_to_eci(r_x_bar, v_x_bar, elements.Omega, elements.i, elements.omega)


# === HELPERS ===

def resolve_arccos(angle: float, disambiguator: float) -> float:
    """resolve arccos angle to (0 to 2pi) with disambiguator

    arccos returns [0, pi], it can't tell an angle apart from its reflection about the reference direction. 
    A second quantity, whose sign differs between the two halves, resolves it. 
    Below zero means the angle lies in the lower half turn, so we take 2 pi - angle.

    Returns:
        angle: angle in radian on [0, 2pi)
    """
    if disambiguator < 0:
        return 2 * np.pi - angle
    else:
        return angle