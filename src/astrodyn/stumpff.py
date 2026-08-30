import numpy as np


def stumpff_c(z):
    """stumpff C function, c_2(z)

    Args:
        z (float or array_like): Universal variable parameter (alpha * chi^2)

    Returns:
        float or ndarray: Stumpff c_2(z) values
    """

    # convert z to numpy array immediately.
    z = np.asarray(z, dtype=float)

    # Using np so it works with numpy arrays.

    # z is small case
    taylor_value = 1/2 - z/24 + (z*z)/720

    # Using absolute value so we can calculate both cases.
    # The wrong case gets discarded at np.where.
    z_abs = np.abs(z)
    # Replace small values with 1 to avoid division by zero.
    z_abs = np.where(z_abs < 1e-6, 1.0, z_abs)

    # z > 0 case, ellipse
    z_ellipse =  (1 - np.cos(np.sqrt(z_abs))) / z_abs
    # z < 0 case, hyperbola
    z_hyperbola = (np.cosh(np.sqrt(z_abs)) - 1) / z_abs

    # return the final 
    return np.where(np.abs(z) < 1e-6, taylor_value, np.where(z > 0, z_ellipse, z_hyperbola))

def stumpff_s(z):
    """stumpff S function, c_3(z)

    Args:
        z (float or array_like): Universal variable parameter (alpha * chi^2)
    
    Returns:
        float or ndarray: Stumpff c_3(z) values
    """

    z = np.asarray(z, dtype=float)

    taylor_value = 1/6 - z/120 + (z*z)/5040

    z_sqrt = np.sqrt(np.where(np.abs(z) < 1e-6, 1.0, np.abs(z)))
    z_ellipse = (z_sqrt - np.sin(z_sqrt)) / np.power(z_sqrt, 3)
    z_hyperbola = (np.sinh(z_sqrt) - z_sqrt) / np.power(z_sqrt, 3)

    return np.where(np.abs(z) < 1e-6, taylor_value, np.where(z > 0, z_ellipse, z_hyperbola))
