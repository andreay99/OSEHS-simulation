"""
OSEHS Swarm Mesh Intelligence
------------------------------
Models the autonomous communication mesh between swarm units.

Roles:
  HARVESTER  — maximizes solar collection, stays in sun-optimal position
  RELAYER    — routes data/power packets between harvesters and receiver

Autonomous behaviors:
  - Each unit broadcasts a health packet to up to 6 nearest neighbors
  - If a harvester fails, the nearest relayer autonomously switches role
  - If a relayer fails, mesh re-routes through next available relayer
  - Shadow detection triggers RAAN adjustment (logged as an event)
  - All decisions made locally — no ground command needed
"""

import numpy as np

HARVESTER = 'harvester'
RELAYER   = 'relayer'

# Max neighbors each unit reports to (6-neighbor mesh)
MAX_NEIGHBORS = 6


class MeshNetwork:
    """
    Manages the autonomous communication mesh across the swarm.
    Assigns roles, tracks neighbor links, handles role switching on failure.
    """

    def __init__(self, units):
        self.units      = units
        self.n          = len(units)
        self.roles      = {}       # uid -> HARVESTER or RELAYER
        self.neighbors  = {}       # uid -> [uid, ...] (up to 6 nearest alive)
        self.health_log = []       # list of health packet dicts per step
        self.events     = []       # autonomous decisions log
        self.mesh_lines = []       # (pos_a, pos_b) pairs for visualization

        self._assign_initial_roles()

    # ------------------------------------------------------------------
    # Role assignment
    # ------------------------------------------------------------------

    def _assign_initial_roles(self):
        """
        Alternate harvester / relayer around the pearl necklace.
        Units 0,2,4,6,8,10  → HARVESTER
        Units 1,3,5,7,9,11  → RELAYER
        """
        for u in self.units:
            self.roles[u.uid] = HARVESTER if u.uid % 2 == 0 else RELAYER

    def get_role(self, uid):
        return self.roles.get(uid, HARVESTER)

    def role_color(self, uid):
        """Return display color for this unit's role."""
        r = self.get_role(uid)
        if r == HARVESTER:
            return '#00ddff'   # cyan
        else:
            return '#00ff88'   # green

    def role_glow(self, uid):
        return '#00aadd' if self.get_role(uid) == HARVESTER else '#00cc66'

    # ------------------------------------------------------------------
    # Neighbor discovery (runs each step)
    # ------------------------------------------------------------------

    def update_neighbors(self, alive_units, comms_range_m):
        """
        For each alive unit, find up to MAX_NEIGHBORS nearest alive neighbors
        within comms range. Builds the mesh link list for visualization.
        """
        self.neighbors  = {}
        self.mesh_lines = []

        positions = {u.uid: u.position() for u in alive_units}

        for u in alive_units:
            pos = positions[u.uid]
            dists = []
            for other in alive_units:
                if other.uid == u.uid:
                    continue
                d = np.linalg.norm(pos - positions[other.uid])
                if d <= comms_range_m:
                    dists.append((d, other.uid))

            dists.sort()
            self.neighbors[u.uid] = [uid for _, uid in dists[:MAX_NEIGHBORS]]

        # Build mesh lines (only draw each link once)
        drawn = set()
        for uid, nbrs in self.neighbors.items():
            for nid in nbrs:
                key = tuple(sorted([uid, nid]))
                if key not in drawn:
                    drawn.add(key)
                    self.mesh_lines.append((positions[uid], positions[nid]))

    # ------------------------------------------------------------------
    # Health packet broadcast
    # ------------------------------------------------------------------

    def broadcast_health(self, alive_units, day):
        """
        Each unit compiles a self-report health packet and sends it to
        its 6 nearest neighbors. Packets are logged for visualization.
        """
        packets = []
        for u in alive_units:
            pkt = {
                'day':        day,
                'from':       u.uid,
                'role':       self.get_role(u.uid),
                'to':         self.neighbors.get(u.uid, []),
                'battery':    round(u.battery_pct(), 1),
                'power_W':    round(u.current_power_W, 1),
                'drift_km':   round(u.orbital_drift_km(), 3),
                'shadowed':   getattr(u, 'shadowed', False),
                'alive':      True,
            }
            packets.append(pkt)

        self.health_log.append(packets)
        return packets

    # ------------------------------------------------------------------
    # Autonomous role switching on failure
    # ------------------------------------------------------------------

    def handle_dropout(self, failed_uid, alive_units, day):
        """
        When a unit fails:
        - If it was a HARVESTER → find nearest alive RELAYER and switch it to HARVESTER
          (maximize energy collection by replacing lost harvester)
        - If it was a RELAYER   → find nearest alive RELAYER to absorb the routing load
          (mesh re-routes automatically, no role change needed unless mesh is thin)

        All decisions logged as autonomous events.
        """
        failed_role = self.roles.get(failed_uid, HARVESTER)
        failed_pos  = None

        # Get position of failed unit from last known state
        for u in self.units:
            if u.uid == failed_uid:
                failed_pos = u.position()
                break

        if failed_role == HARVESTER:
            # Switch nearest alive RELAYER → HARVESTER
            best_uid  = None
            best_dist = np.inf
            for u in alive_units:
                if self.roles[u.uid] == RELAYER:
                    d = np.linalg.norm(u.position() - failed_pos) if failed_pos is not None else np.inf
                    if d < best_dist:
                        best_dist = d
                        best_uid  = u.uid

            if best_uid is not None:
                self.roles[best_uid] = HARVESTER
                self.events.append({
                    'day':    day,
                    'type':   'ROLE_SWITCH',
                    'msg':    f'Unit {best_uid} switched RELAYER→HARVESTER to replace failed Unit {failed_uid}',
                    'uid':    best_uid,
                })
                print(f'[Mesh] Day {day}: Unit {best_uid} RELAYER→HARVESTER '
                      f'(replacing failed harvester {failed_uid})')
            else:
                self.events.append({
                    'day':  day,
                    'type': 'MESH_DEGRADED',
                    'msg':  f'No relayer available to replace failed harvester {failed_uid}',
                    'uid':  failed_uid,
                })

        else:
            # Relayer failed — mesh re-routes through remaining relayers
            n_relayers = sum(1 for u in alive_units if self.roles[u.uid] == RELAYER)
            self.events.append({
                'day':  day,
                'type': 'MESH_REROUTE',
                'msg':  f'Relayer {failed_uid} failed — mesh re-routed. {n_relayers} relayers remain.',
                'uid':  failed_uid,
            })
            print(f'[Mesh] Day {day}: Relayer {failed_uid} failed — '
                  f'mesh re-routed through {n_relayers} remaining relayers')

    # ------------------------------------------------------------------
    # Shadow event logging
    # ------------------------------------------------------------------

    def log_shadow_event(self, uid, day):
        self.events.append({
            'day':  day,
            'type': 'SHADOW_AVOID',
            'msg':  f'Unit {uid} detected shadow risk — autonomously adjusting RAAN',
            'uid':  uid,
        })

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def role_counts(self, alive_units):
        h = sum(1 for u in alive_units if self.roles.get(u.uid) == HARVESTER)
        r = sum(1 for u in alive_units if self.roles.get(u.uid) == RELAYER)
        return h, r

    def mesh_connectivity(self, alive_units):
        """Fraction of alive units that have at least one neighbor link."""
        if not alive_units:
            return 0.0
        connected = sum(1 for u in alive_units if self.neighbors.get(u.uid))
        return connected / len(alive_units)
