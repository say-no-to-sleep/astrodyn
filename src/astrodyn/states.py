import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

# Tolerance for orbits near Parabola
PARABOLIC_TOLERANCE = 1e-8

class StateVector(NamedTuple):
    r_vec: np.ndarray
    v_vec: np.ndarray

@dataclass(frozen=True)
class ClassicalElements:
    h: float
    e: float
    i: float
    Omega: float
    omega: float
    theta: float

    # Calculate the semimajor axis, a
    def semimajor_axis(self, mu: float) -> float:
        if abs(1.0 - self.e) < PARABOLIC_TOLERANCE:
            return float("inf")
        return self.h**2 / (mu * (1-self.e**2))

    # Calculate the Period (only for elliptical), T
    def period(self, mu: float) -> float | None:
        if self.e >= (1.0 - PARABOLIC_TOLERANCE):
            return None
        return (2 * np.pi * self.semimajor_axis(mu)**(3/2)) / np.sqrt(mu)

    # Calculate the periapsis radius, r_p
    def periapsis_radius(self, mu: float) -> float:
        return self.h**2 / (mu * (1 + self.e))

    # Calculate the apoapsis radius, r_a
    def apoapsis_radius(self, mu: float) -> float | None:
        if self.e >= (1.0 - PARABOLIC_TOLERANCE):
            return None
        return self.h**2 / (mu * (1 - self.e))

    # Calculate the specific energy, epsilon
    def specific_energy(self, mu: float) -> float:
        # Update: use identity relating specific orbital energy 
        # directly to angular momentum and eccentricity
        return -(mu**2 / (2 * self.h**2)) * (1 - self.e**2)
    