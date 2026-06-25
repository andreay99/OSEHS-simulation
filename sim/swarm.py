import numpy as np
from sim.unit import SwarmUnit, COMMS_RANGE_M
from sim.orbital import AU

try:
    from config import MIN_SAFE_DIST, MIN_ANGULAR_SEP
except ImportError:
    MIN_SAFE_DIST    = 0.005 * AU
    MIN_ANGULAR_SEP  = np.radians(10.0)

RAAN_NUDGE = np.radians(0.05)   # RAAN correction per step for shadow avoidance
NU_NUDGE   = 0.001              # true anomaly nudge for collision avoidance


class Swarm:
    """
    OSEHS swarm in pearl-necklace formation.

    Realistic Phase 3 upgrades:
    - Comms-range-limited broadcast: units only share state with neighbors
      within COMMS_RANGE_M — no magic global broadcast
    - Battery-aware health reporting
    - Orbital drift tracking per unit
    - Power output in real watts
    """

    def __init__(self, n_units, a=AU, e=0.0, i=0.0, omega=0.0):
        self.n_units = n_units
        self.units   = self._init_pearl_necklace(n_units, a, e, i, omega)
        self.t       = 0.0
        self.metrics = []
        self._day    = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_pearl_necklace(self, n, a, e, i, omega):
        units     = []
        raan_step = 2.0 * np.pi / n
        for uid in range(n):
            raan = uid * raan_step
            unit = SwarmUnit(uid, a, e, i, omega, raan, nu=0.0)
            units.append(unit)
        return units

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, dt):
        alive = self.alive_units()

        # Build comms-range-limited packet map:
        # each unit only receives packets from neighbors it can hear
        all_packets = [u.state_packet() for u in alive]

        for unit in alive:
            # Filter packets to only those within comms range
            visible = [
                pkt for pkt in all_packets
                if pkt['id'] != unit.uid and unit.can_hear(pkt['position'])
            ]
            unit.comms_isolated = (len(visible) == 0)

            unit.step(dt, day=self._day)

            self._apply_collision_avoidance(unit, visible)
            self._apply_shadow_avoidance(unit, visible)

        self._log_metrics()
        self.t    += dt
        self._day += 1

    def _broadcast(self):
        return [u.state_packet() for u in self.alive_units()]

    # ------------------------------------------------------------------
    # Collision avoidance
    # ------------------------------------------------------------------

    def _apply_collision_avoidance(self, unit, packets):
        """
        If two units come within MIN_SAFE_DIST, nudge true anomaly outward.
        Only runs if the other unit is within comms range.
        """
        pos = unit.position()
        for pkt in packets:
            if not pkt['alive']:
                continue
            dist = np.linalg.norm(pos - pkt['position'])
            if dist < MIN_SAFE_DIST:
                if unit.nu > pkt['nu']:
                    unit.nu = (unit.nu + NU_NUDGE) % (2.0 * np.pi)
                else:
                    unit.nu = (unit.nu - NU_NUDGE) % (2.0 * np.pi)

    # ------------------------------------------------------------------
    # Shadow avoidance
    # ------------------------------------------------------------------

    def _apply_shadow_avoidance(self, unit, packets):
        """
        Check angular separation between this unit and visible neighbors.
        If within MIN_ANGULAR_SEP, autonomously nudge RAAN to spread out.
        """
        r_self     = unit.position()
        r_self_hat = r_self / np.linalg.norm(r_self)
        unit.shadowed = False

        for pkt in packets:
            if not pkt['alive']:
                continue
            r_other     = pkt['position']
            r_other_hat = r_other / np.linalg.norm(r_other)

            cos_ang = np.clip(np.dot(r_self_hat, r_other_hat), -1.0, 1.0)
            angle   = np.arccos(cos_ang)

            if angle < MIN_ANGULAR_SEP:
                unit.shadowed = True
                raan_diff = unit.raan - pkt['raan']
                raan_diff = (raan_diff + np.pi) % (2.0 * np.pi) - np.pi
                if raan_diff >= 0:
                    unit.raan = (unit.raan + RAAN_NUDGE) % (2.0 * np.pi)
                else:
                    unit.raan = (unit.raan - RAAN_NUDGE) % (2.0 * np.pi)

    def _angular_sep(self, r_a, r_b):
        a_hat = r_a / np.linalg.norm(r_a)
        b_hat = r_b / np.linalg.norm(r_b)
        return np.arccos(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0))

    # ------------------------------------------------------------------
    # Fault tolerance + AUTO-rebalance
    # ------------------------------------------------------------------

    def simulate_dropout(self, unit_id):
        """Kill a unit, then automatically rebalance the remaining swarm."""
        self.units[unit_id].kill()
        n_alive = len(self.alive_units())
        print(f"[t={self.t/86400:.1f}d] Unit {unit_id} dropped out. "
              f"{n_alive}/{self.n_units} alive "
              f"({100*n_alive/self.n_units:.1f}% functional) — auto-rebalancing RAAN...")
        self.rebalance_raan()

    def rebalance_raan(self):
        """
        Redistribute RAAN evenly across surviving units.
        Called automatically on every dropout — no ground command needed.
        """
        alive     = self.alive_units()
        n         = len(alive)
        if n == 0:
            return
        raan_step = 2.0 * np.pi / n
        for idx, unit in enumerate(alive):
            unit.raan = idx * raan_step

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def _log_metrics(self):
        alive  = self.alive_units()
        n_alive = len(alive)

        collision_violations = self._count_collision_violations(alive)
        shadow_violations    = sum(1 for u in alive if getattr(u, 'shadowed', False))
        comms_isolated       = sum(1 for u in alive if getattr(u, 'comms_isolated', False))

        raans = sorted(u.raan for u in alive)
        if len(raans) > 1:
            gaps        = np.diff(raans + [raans[0] + 2.0 * np.pi])
            spacing_std = float(np.std(gaps))
        else:
            spacing_std = 0.0

        total_energy   = sum(u.energy_collected for u in alive)
        total_power_W  = sum(u.current_power_W for u in alive)
        avg_battery    = np.mean([u.battery_pct() for u in alive]) if alive else 0.0
        max_drift_km   = max((u.orbital_drift_km() for u in alive), default=0.0)

        self.metrics.append({
            't':                    self.t,
            'alive':                n_alive,
            'pct_functional':       100.0 * n_alive / self.n_units,
            'collision_violations': collision_violations,
            'shadow_violations':    shadow_violations,
            'comms_isolated':       comms_isolated,
            'spacing_std_rad':      spacing_std,
            'total_energy_Ws':      total_energy,
            'total_power_W':        total_power_W,
            'avg_battery_pct':      avg_battery,
            'max_drift_km':         max_drift_km,
        })

    def _count_collision_violations(self, alive):
        violations = 0
        for idx, u1 in enumerate(alive):
            p1 = u1.position()
            for u2 in alive[idx + 1:]:
                if np.linalg.norm(p1 - u2.position()) < MIN_SAFE_DIST:
                    violations += 1
        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def alive_units(self):
        return [u for u in self.units if u.alive]

    def failed_units(self):
        return [u for u in self.units if not u.alive]
