import spiceypy as spice
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from astrodyn.lambert import solve_lambert
from astrodyn.constants import MU_SUN
from astrodyn.constants import MU_EARTH
from astrodyn.constants import MU_MARS 
from astrodyn.states import StateVector

# Load kernel pool

# Ephemeris data. Time -> Location
spice.furnsh("data/de442.bsp")

# Convert calendar dates to ephemeris (continuous) count
spice.furnsh("data/naif0012.tls")

# Targeting the optimal transfer window that SR-1 Freedom (2028) is using
departure_start = float(spice.str2et("2028-05-01"))
departure_end = float(spice.str2et("2029-09-01"))
arrival_start = float(spice.str2et("2028-12-01"))
arrival_end = float(spice.str2et("2030-05-01"))

departure_times = np.arange(departure_start, departure_end, 86400)
arrival_times = np.arange(arrival_start, arrival_end, 86400)

departure_vs_arrival = np.zeros((len(departure_times), len(arrival_times)))

for i in range(0, len(departure_times)):
    for j in range(0, len(arrival_times)):
        # Departure and arrival time
        departure_ephemeris = departure_times[i]
        arrival_ephemeris = arrival_times[j]

        # Calculate delta t (time to get there)
        dt = arrival_ephemeris - departure_ephemeris

        # Get Earth and Mars states relative to the sun in sphemeris time frame using SPK, easy reader (spkezr)
        # Omitting one way light travel time
        earth_state, _ = spice.spkezr("EARTH BARYCENTER", departure_ephemeris, "ECLIPJ2000", "NONE", "SUN")
        mars_state, _ = spice.spkezr("MARS BARYCENTER", arrival_ephemeris, "ECLIPJ2000", "NONE", "SUN")
        earth_state = np.array(earth_state)
        mars_state = np.array(mars_state)

        # Get Earth and Mars's location and velocity
        earth_r = earth_state[:3]
        earth_v = earth_state[3:]
        mars_r = mars_state[:3]
        mars_v = mars_state[3:]

        # Important numbers
        mu = MU_SUN
        mu_earth = MU_EARTH
        mu_mars = MU_MARS
        r_leo = 6778
        r_lmo = 3696

        # Solving for lambert
        # Use try-except, some may not converge...
        try:
            (before_lambert, after_lambert) = solve_lambert(earth_r, mars_r, dt, mu, True)

            # Speed of spacecraft relative to the planet
            v_inf_dep = LA.norm(before_lambert.v_vec - earth_v)
            v_inf_arr = LA.norm(after_lambert.v_vec - mars_v)

            # Convert v_inf into delta V from low orbits for each planet
            dv_dep = np.sqrt(v_inf_dep**2 + (2 * mu_earth)/r_leo) - np.sqrt(mu_earth/r_leo)
            dv_arr = np.sqrt(v_inf_arr**2 + (2 * mu_mars)/r_lmo) - np.sqrt(mu_mars/r_lmo)

            total_dv = dv_dep + dv_arr
            departure_vs_arrival[i, j] = total_dv
        except:
            departure_vs_arrival[i, j] = np.nan

# Sanity check
print(np.nanmin(departure_vs_arrival), np.nanmax(departure_vs_arrival))

# Some values are converging but blowing up... Capping
departure_vs_arrival[departure_vs_arrival > 30] = np.nan

# Plot it in matplotlib

plt.contourf(spice.et2datetime(arrival_times), spice.et2datetime(departure_times), departure_vs_arrival, levels = 20)
plt.colorbar(label="Total Delta V in km/s")
plt.xlabel("Arrival date")
plt.ylabel("Departure Date")
plt.title("Earth to Mars Porkchop Plot (Oct 2028 - Apr 2029)")
plt.show()