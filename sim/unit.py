import numpy as np
from sim.orbital import (
    coe_to_eci, propagate_nu, apply_srp_perturbation,
    panel_power_output, MU_SUN, AU,
    PANEL_AREA_M2, PANEL_EFFICIENCY
)

# ------------------------------------------------------------------
# Battery / energy storage model
# ------------------------------------------------------------------
BATTERY_CAPACITY_WH  = 500_000.0   # Wh — large energy storage buffer
                                    # (think supercapacitor bank / flywheel equivalent)
                                    # sized so charge/drain cycles smoothly over
                                    # ~100 day orbital arcs near aphelion
BATTERY_CAPACITY_WS  = BATTERY_CAPACITY_WH * 3600.0

# Draw set just above mid-orbit power so battery:
#   charges during inner arc (closer to Sun, ~3026 W/panel)
#   drains during outer arc (near aphelion, ~2614 W/panel)
# Net drain near apo: ~206 W/panel → drains 500,000 Wh over ~100 days ✓
BASELINE_DRAW_W = 2820.0           # W continuous draw per unit

# Comms range — units only exchange state packets within this distance
# Pearl necklace comms range must cover worst-case adjacent spacing.
# With e=0.076 orbit and 9 surviving units (after 3 dropouts), max adjacent
# distance across the full elliptical orbit reaches ~0.495 AU at aphelion.
# Use 0.52 AU (5% margin) so coordination never breaks down.
COMMS_RANGE_M = 0.52 * AU       # ~78 million km (deep-space laser/RF link)


class SwarmUnit:
    """
    A single autonomous solar collection panel in the OSEHS swarm.

    Realistic upgrades (Phase 3):
    - Solar radiation pressure perturbation shifts orbital elements over time
    - Real panel power output: irradiance × area × efficiency
    - Battery model: charges during sunlit arc, limited by capacity
    - Comms range: only broadcasts to neighbors within COMMS_RANGE_M
    - Tracks orbital drift from nominal semimajor axis
    """

    def __init__(self, uid, a, e, i, omega, raan, nu):
        self.uid = uid

        # Classical orbital elements
        self.a     = a          # semimajor axis (m)
        self.e     = e          # eccentricity
        self.i     = i          # inclination (rad)
        self.omega = omega      # argument of perigee (rad)
        self.raan  = raan       # right ascension of ascending node (rad)
        self.nu    = nu         # true anomaly (rad)

        # Nominal semimajor axis (set once at init, never changed)
        self.a_nominal = a

        self.alive = True

        # Energy / power tracking
        self.battery_Ws       = BATTERY_CAPACITY_WS * 0.5  # start at 50% charge
        self.energy_collected = 0.0   # cumulative W·s delivered (not stored)
        self.current_power_W  = 0.0   # instantaneous electrical output this step

        # State flags
        self.shadowed         = False
        self.comms_isolated   = False  # True if no neighbors in comms range

        # Drift tracking — cumulative peak drift from nominal
        self.a_drift_history  = []    # (day, drift_m) log
        self.peak_drift_km    = 0.0   # monotonically increasing peak drift

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def step(self, dt, day=0):
        if not self.alive:
            return

        # 1. Propagate true anomaly (Kepler)
        self.nu = propagate_nu(self.a, self.e, self.nu, dt)

        # 2. Apply solar radiation pressure perturbation to orbital elements
        self.a, self.e, self.raan = apply_srp_perturbation(
            self.a, self.e, self.i, self.omega, self.raan, self.nu, dt
        )

        # 3. Harvest energy with real power model
        self._harvest_energy(dt)

        # 4. Log orbital drift — track cumulative peak so it never resets
        drift = abs(self.a - self.a_nominal)
        self.peak_drift_km = max(self.peak_drift_km, drift / 1e3)
        self.a_drift_history.append((day, drift))

    def _harvest_energy(self, dt):
        """
        Real power output: irradiance × panel area × cell efficiency.
        Charges battery up to capacity; excess is delivered power.
        """
        r_vec = self.position()

        if self.shadowed:
            # No solar input when shadowed — battery only
            self.current_power_W = 0.0
        else:
            self.current_power_W = panel_power_output(r_vec)

        # Energy generated this step (W·s)
        energy_in  = self.current_power_W * dt

        # Subtract baseline system power draw (comms, sensors, computer)
        energy_out = BASELINE_DRAW_W * dt

        # Net energy: positive = charging, negative = draining
        net = energy_in - energy_out

        # Update battery, clamped to [0, capacity]
        self.battery_Ws = np.clip(self.battery_Ws + net, 0.0, BATTERY_CAPACITY_WS)

        # Delivered energy = surplus above what battery + draw needed
        delivered = max(0.0, energy_in - energy_out - (BATTERY_CAPACITY_WS - self.battery_Ws))
        self.energy_collected += delivered

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def position(self):
        r, _ = coe_to_eci(self.a, self.e, self.i, self.omega, self.raan, self.nu)
        return r

    def velocity(self):
        _, v = coe_to_eci(self.a, self.e, self.i, self.omega, self.raan, self.nu)
        return v

    def battery_pct(self):
        return 100.0 * self.battery_Ws / BATTERY_CAPACITY_WS

    def orbital_drift_km(self):
        """Cumulative peak drift of semimajor axis from nominal, in km.
        Monotonically increases — never resets — so the chart shows real accumulation."""
        return self.peak_drift_km

    def state_packet(self):
        """
        Lightweight broadcast packet sent to neighbors.
        Now includes battery level, power output, and orbital drift.
        """
        return {
            'id':              self.uid,
            'position':        self.position(),
            'velocity':        self.velocity(),
            'nu':              self.nu,
            'raan':            self.raan,
            'alive':           self.alive,
            'battery_pct':     self.battery_pct(),
            'power_W':         self.current_power_W,
            'drift_km':        self.orbital_drift_km(),
        }

    def can_hear(self, other_pos):
        """Return True if other_pos is within comms range."""
        return np.linalg.norm(self.position() - other_pos) <= COMMS_RANGE_M

    # ------------------------------------------------------------------
    # Fault simulation
    # ------------------------------------------------------------------

    def kill(self):
        self.alive = False
