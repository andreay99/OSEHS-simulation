"""
OSEHS Simulation Configuration

Edit this file to change orbital parameters, swarm size, simulation length, and dropout events.
Everything else flows from these settings.
"""

import numpy as np
from sim.orbital import AU

# -----------------------------------------------------------------------
# Orbital Parameters
# -----------------------------------------------------------------------
# Replace these with real values from KesUraNu's orbital dynamics analysis.

ORBITAL_RADIUS = 0.6722 * AU    # semimajor axis (m) — from KesUraNu's mission profile
ECCENTRICITY = 0.0760           # eccentricity — from KesUraNu's mission profile
INCLINATION = np.radians(5)     # inclination (radians) — placeholder (not provided)
ARGUMENT_OF_PERIGEE = 0.0       # argument of perigee (rad)

# -----------------------------------------------------------------------
# Swarm Configuration
# -----------------------------------------------------------------------

N_UNITS = 12                     # number of solar collector panels

MIN_SAFE_DIST = 0.005 * AU       # collision avoidance threshold (~750,000 km)
                                  # If two units closer than this, they nudge apart

MIN_ANGULAR_SEP = np.radians(10) # shadow avoidance threshold (radians)
                                  # If angular separation < 10°, panels spread out
                                  # to avoid mutual shadowing

# -----------------------------------------------------------------------
# Simulation Parameters
# -----------------------------------------------------------------------

DT = 3600 * 24                    # time step (seconds) — 1 day per step
MAX_STEPS = 365 * 3               # simulation length (steps) — 3 simulated years

# Dropout Events (optional)
# Format: {step: unit_id} — unit_id fails at that simulation step
# Remove/modify as needed for different fault-tolerance scenarios
DROPOUT_EVENTS = {
    100: 2,   # Day 100: Unit 2 fails
    200: 7,   # Day 200: Unit 7 fails
    300: 5,   # Day 300: Unit 5 fails
}

# -----------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------
# - Solar irradiance at 1 AU: 1361 W/m² (built into sim/unit.py)
# - Sun's gravitational parameter: 1.327e20 m³/s² (built into sim/orbital.py)
# - All angles in radians (use np.radians(degrees) for conversion)
# - 1 AU = 1.496e11 meters
#
# Example: To test with a 0.5 AU circular orbit at 0° inclination:
#   ORBITAL_RADIUS = 0.5 * AU
#   ECCENTRICITY = 0.0
#   INCLINATION = 0.0
