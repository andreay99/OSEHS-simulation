"""
Example orbital configurations for different scenarios.

Copy-paste orbital parameters from KesUraNu into config.py to test different missions.
"""

import numpy as np
from sim.orbital import AU

# -----------------------------------------------------------------------
# Example 1: Earth orbit (LEO-ish at 1 AU for testing)
# -----------------------------------------------------------------------
EARTH_ORBIT = {
    'ORBITAL_RADIUS': 1.0 * AU,
    'ECCENTRICITY': 0.0,
    'INCLINATION': np.radians(5),
    'ARGUMENT_OF_PERIGEE': 0.0,
}

# -----------------------------------------------------------------------
# Example 2: Closer to Sun for higher irradiance (0.5 AU)
# -----------------------------------------------------------------------
CLOSE_ORBIT = {
    'ORBITAL_RADIUS': 0.5 * AU,
    'ECCENTRICITY': 0.1,
    'INCLINATION': np.radians(2),
    'ARGUMENT_OF_PERIGEE': np.radians(45),
}

# -----------------------------------------------------------------------
# Example 3: Further from Sun (1.5 AU, lower irradiance but less thermal stress)
# -----------------------------------------------------------------------
FAR_ORBIT = {
    'ORBITAL_RADIUS': 1.5 * AU,
    'ECCENTRICITY': 0.0,
    'INCLINATION': np.radians(10),
    'ARGUMENT_OF_PERIGEE': 0.0,
}

# -----------------------------------------------------------------------
# How to use
# -----------------------------------------------------------------------
"""
To test a scenario:

1. Pick an example from above (or paste KesUraNu's data)
2. Update config.py with those parameters:

   config.py:
   -----------
   ORBITAL_RADIUS = 0.5 * AU       # from example
   ECCENTRICITY = 0.1
   INCLINATION = np.radians(2)
   ARGUMENT_OF_PERIGEE = np.radians(45)

3. Run the simulation:
   python3 main.py

4. Observe KPP metrics in the real-time GUI
"""
