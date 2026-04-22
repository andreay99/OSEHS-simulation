"""
OSEHS Swarm Simulation — Star Maker Team
NASA ORBIT Challenge, Phase 2

Runs a real-time 3D visualization of the pearl-necklace solar swarm
with live KPP metrics. Press Ctrl-C or close the window to stop.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # faster backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sim.swarm import Swarm
from sim.orbital import AU
from config import (
    N_UNITS, DT, MAX_STEPS, DROPOUT_EVENTS,
    ORBITAL_RADIUS, ECCENTRICITY, INCLINATION, ARGUMENT_OF_PERIGEE
)

swarm = Swarm(
    n_units=N_UNITS,
    a=ORBITAL_RADIUS,
    e=ECCENTRICITY,
    i=INCLINATION,
    omega=ARGUMENT_OF_PERIGEE
)

# Precompute orbit traces to avoid expensive COE→ECI in animation loop
def _precompute_traces(units):
    traces = {}
    from sim.orbital import coe_to_eci
    for u in units:
        nus = np.linspace(0, 2 * np.pi, 80)
        xs, ys, zs = [], [], []
        for nu in nus:
            r, _ = coe_to_eci(u.a, u.e, u.i, u.omega, u.raan, nu)
            xs.append(r[0] / AU); ys.append(r[1] / AU); zs.append(r[2] / AU)
        traces[u.uid] = (xs, ys, zs)
    return traces

orbit_traces = _precompute_traces(swarm.units)

# -----------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('#0a0a1a')

ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.set_facecolor('#0a0a1a')

ax_m = fig.add_subplot(2, 2, 2)
ax_e = fig.add_subplot(2, 2, 4)
for ax in (ax_m, ax_e):
    ax.set_facecolor('#0d0d2b')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333366')

fig.suptitle('OSEHS — Orbital Solar Energy Harvesting Swarm', color='white', fontsize=13)

# Setup 3D plot with static elements
ax3d.set_xlim(-1.5, 1.5)
ax3d.set_ylim(-1.5, 1.5)
ax3d.set_zlim(-0.3, 0.3)
ax3d.set_xlabel('X (AU)', color='white', fontsize=8)
ax3d.set_ylabel('Y (AU)', color='white', fontsize=8)
ax3d.set_zlabel('Z (AU)', color='white', fontsize=8)
ax3d.tick_params(colors='white', labelsize=7)

# Draw sun once
ax3d.scatter([0], [0], [0], color='yellow', s=300, zorder=5)

# Draw orbit traces once
for u in swarm.units:
    xs, ys, zs = orbit_traces[u.uid]
    ax3d.plot(xs, ys, zs, color='cyan', alpha=0.05, linewidth=0.4)

title_text = ax3d.set_title('', color='white', fontsize=10)

# -----------------------------------------------------------------------
# Animation update
# -----------------------------------------------------------------------

def update(frame):
    global title_text

    # Trigger dropouts
    if frame in DROPOUT_EVENTS:
        uid = DROPOUT_EVENTS[frame]
        swarm.simulate_dropout(uid)

    swarm.step(DT)

    # Update title only
    title_text.set_text(f'Day {frame}  |  {len(swarm.alive_units())}/{N_UNITS} units')

    # -- Update unit positions (no cla, just scatter) --
    # Clear just the scatter/failed markers (keep orbits and sun)
    for artist in ax3d.collections[:]:
        if len(ax3d.collections) > 0:
            artist.remove()

    alive = swarm.alive_units()
    if alive:
        normal   = [u for u in alive if not getattr(u, 'shadowed', False)]
        shadowed = [u for u in alive if getattr(u, 'shadowed', False)]

        if normal:
            pos_n = np.array([u.position() / AU for u in normal])
            ax3d.scatter(pos_n[:, 0], pos_n[:, 1], pos_n[:, 2],
                         c='cyan', s=50, zorder=4)
        if shadowed:
            pos_s = np.array([u.position() / AU for u in shadowed])
            ax3d.scatter(pos_s[:, 0], pos_s[:, 1], pos_s[:, 2],
                         c='orange', s=60, marker='^', zorder=5)

    failed = swarm.failed_units()
    if failed:
        pos_f = np.array([u.position() / AU for u in failed])
        ax3d.scatter(pos_f[:, 0], pos_f[:, 1], pos_f[:, 2],
                     c='red', s=50, marker='x', zorder=4)

    # -- Update metrics (every 5 frames to reduce lag) --
    if frame % 5 == 0 and swarm.metrics:
        m = swarm.metrics
        days = [x['t'] / 86400 for x in m]
        pct  = [x['pct_functional'] for x in m]
        energy = [x['total_energy_Ws'] / 1e12 for x in m]
        shadow_viols = [x['shadow_violations'] for x in m]

        ax_m.clear()
        ax_m.set_facecolor('#0d0d2b')
        ax_m.plot(days, pct, color='lime', linewidth=1.2)
        ax_m.axhline(94, color='orange', linestyle='--', linewidth=0.8)
        ax_m.set_ylabel('Swarm Functional %', color='white', fontsize=8)
        ax_m.set_ylim(0, 105)
        ax_m.tick_params(colors='white', labelsize=7)
        ax_m.set_title('Swarm Health', color='white', fontsize=9)

        ax_e.clear()
        ax_e.set_facecolor('#0d0d2b')
        ax_e.plot(days, energy, color='gold', linewidth=1.2)
        ax_e.set_xlabel('Simulation Day', color='white', fontsize=8)
        ax_e.set_ylabel('Energy', color='white', fontsize=8)
        ax_e.tick_params(colors='white', labelsize=7)
        ax_e.set_title('Energy Collection', color='white', fontsize=9)


ani = animation.FuncAnimation(
    fig, update,
    frames=MAX_STEPS,
    interval=50,
    repeat=False,
)

plt.show()
