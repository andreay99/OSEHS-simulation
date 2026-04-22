import numpy as np
from sim.orbital import coe_to_eci, propagate_nu, MU_SUN, AU

SOLAR_IRRADIANCE_1AU = 1361.0  # W/m^2 at 1 AU


class SwarmUnit:
    """
    A single autonomous solar collection panel in the OSEHS swarm.
    State is stored as Classical Orbital Elements (COE).
    """

    def __init__(self, uid, a, e, i, omega, raan, nu):
        self.uid = uid
        self.a = a          # semimajor axis (m)
        self.e = e          # eccentricity
        self.i = i          # inclination (rad)
        self.omega = omega  # argument of perigee (rad)
        self.raan = raan    # right ascension of ascending node (rad)
        self.nu = nu        # true anomaly (rad)

        self.alive = True
        self.energy_collected = 0.0  # W·s accumulated

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def step(self, dt):
        if not self.alive:
            return
        self.nu = propagate_nu(self.a, self.e, self.nu, dt)
        self._harvest_energy(dt)

    def _harvest_energy(self, dt):
        r = np.linalg.norm(self.position())
        irradiance = SOLAR_IRRADIANCE_1AU * (AU / r) ** 2
        self.energy_collected += irradiance * dt  # W·s per m^2

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def position(self):
        r, _ = coe_to_eci(self.a, self.e, self.i, self.omega, self.raan, self.nu)
        return r

    def velocity(self):
        _, v = coe_to_eci(self.a, self.e, self.i, self.omega, self.raan, self.nu)
        return v

    def state_packet(self):
        """Lightweight broadcast packet sent to neighbors."""
        return {
            'id':       self.uid,
            'position': self.position(),
            'velocity': self.velocity(),
            'nu':       self.nu,
            'raan':     self.raan,
            'alive':    self.alive,
        }

    # ------------------------------------------------------------------
    # Fault simulation
    # ------------------------------------------------------------------

    def kill(self):
        self.alive = False
