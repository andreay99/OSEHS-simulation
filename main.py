"""
OSEHS Swarm Simulation — Star Maker Team
NASA ORBIT Challenge, Phase 3

Real-time 3D visualization of the pearl-necklace solar swarm
orbiting KesUraNu (Ranu Baylor) at 0.6722 AU.

Physics modelled:
  - Keplerian propagation (Newton-Raphson Kepler solver)
  - Solar radiation pressure perturbation (Gauss variational equations)
  - Real panel power: irradiance × 3.46 m² × 29% GaAs efficiency
  - 500 Wh battery per unit, charges/drains with orbital distance
  - Comms-range-limited swarm coordination (0.52 AU laser link)
  - Shadow avoidance: autonomous RAAN nudge if angular sep < 10°
  - Collision avoidance: true anomaly nudge if spacing < 750,000 km
  - Auto-rebalance RAAN on unit dropout (no ground command needed)

Controls:
  Space   — pause / resume
  S       — save screenshot (while paused)
  = / -   — speed up / slow down animation
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.swarm import Swarm
from sim.orbital import (AU, coe_to_eci, PANEL_AREA_M2,
                         PANEL_EFFICIENCY, solar_irradiance_at)
from config import (
    N_UNITS, DT, MAX_STEPS, DROPOUT_EVENTS,
    ORBITAL_RADIUS, ECCENTRICITY, INCLINATION, ARGUMENT_OF_PERIGEE
)

# -----------------------------------------------------------------------
# Panel geometry — Jennifer's CAD: 2 m flat-to-flat regular hexagon
# Panels face the Sun (normal = -r_hat, i.e. sunward)
# -----------------------------------------------------------------------
# Panel rendering uses scatter markers — reliable in matplotlib 3D at any angle
# Each unit: bright core dot + outer glow ring = looks like a glowing solar panel
PANEL_CORE_SIZE  = 120   # pt² for the bright center dot
PANEL_GLOW_SIZE  = 380   # pt² for the outer glow halo


# -----------------------------------------------------------------------
# Build swarm
# -----------------------------------------------------------------------
swarm = Swarm(
    n_units=N_UNITS,
    a=ORBITAL_RADIUS,
    e=ECCENTRICITY,
    i=INCLINATION,
    omega=ARGUMENT_OF_PERIGEE
)

# ONE shared orbit ring — pearl necklace means all units share the same ellipse
# (RAAN spacing just sets their position around it, not the shape)
def _ellipse_trace(a, e, i, omega, raan=0.0, n=300):
    nus = np.linspace(0, 2 * np.pi, n)
    pts = [coe_to_eci(a, e, i, omega, raan, nu)[0] / AU for nu in nus]
    pts = np.array(pts)
    return pts[:, 0], pts[:, 1], pts[:, 2]

u0 = swarm.units[0]
shared_trace = _ellipse_trace(u0.a, u0.e, u0.i, u0.omega)

# Perihelion / aphelion distances
R_PERI = ORBITAL_RADIUS * (1 - ECCENTRICITY) / AU   # AU
R_APO  = ORBITAL_RADIUS * (1 + ECCENTRICITY) / AU   # AU

# Power reference values
_r_peri = np.array([ORBITAL_RADIUS * (1 - ECCENTRICITY), 0.0, 0.0])
_r_apo  = np.array([ORBITAL_RADIUS * (1 + ECCENTRICITY), 0.0, 0.0])
POWER_PERI_KW = solar_irradiance_at(_r_peri) * PANEL_AREA_M2 * PANEL_EFFICIENCY * N_UNITS / 1000
POWER_APO_KW  = solar_irradiance_at(_r_apo)  * PANEL_AREA_M2 * PANEL_EFFICIENCY * N_UNITS / 1000
POWER_NOM_KW  = (POWER_PERI_KW + POWER_APO_KW) / 2

# Orbital period
T_DAYS = 2 * np.pi * np.sqrt(ORBITAL_RADIUS**3 / 1.32712440018e20) / 86400

PLOT_LIM = R_APO * 1.25   # AU, with padding

# -----------------------------------------------------------------------
# Figure layout — dark space background
# -----------------------------------------------------------------------
BG       = '#03030d'
PANEL_BG = '#080818'

fig = plt.figure(figsize=(18, 9), facecolor=BG)
fig.patch.set_facecolor(BG)

ax3d       = fig.add_axes([0.00, 0.02, 0.54, 0.94], projection='3d')
ax_health  = fig.add_axes([0.57, 0.755, 0.41, 0.175], facecolor=PANEL_BG)
ax_power   = fig.add_axes([0.57, 0.550, 0.41, 0.175], facecolor=PANEL_BG)
ax_battery = fig.add_axes([0.57, 0.345, 0.41, 0.175], facecolor=PANEL_BG)
ax_drift   = fig.add_axes([0.57, 0.090, 0.41, 0.175], facecolor=PANEL_BG)

for ax in (ax_health, ax_power, ax_battery, ax_drift):
    ax.tick_params(colors='#7777aa', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#1a1a3a')

# Annotation row below charts
fig.text(0.775, 0.960,
         f'Orbit: a={ORBITAL_RADIUS/AU:.4f} AU  e={ECCENTRICITY:.4f}  '
         f'i={np.degrees(INCLINATION):.1f}°  T={T_DAYS:.1f} d  '
         f'r_peri={R_PERI:.3f} AU  r_apo={R_APO:.3f} AU',
         color='#6666aa', fontsize=7, ha='center', va='top')

fig.text(0.5, 0.995,
         'OSEHS — Orbital Solar Energy Harvesting Swarm  |  Phase 3  |  Star Maker Team',
         color='white', fontsize=12, ha='center', va='top', fontweight='bold')

# -----------------------------------------------------------------------
# 3D axes — invisible panes, very dark grid
# -----------------------------------------------------------------------
ax3d.set_facecolor(BG)
ax3d.set_xlim(-PLOT_LIM, PLOT_LIM)
ax3d.set_ylim(-PLOT_LIM, PLOT_LIM)
ax3d.set_zlim(-PLOT_LIM * 0.08, PLOT_LIM * 0.08)

for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor('#0a0a1a')

ax3d.grid(True, color='#0c0c22', linewidth=0.3)
ax3d.set_xlabel('X (AU)', color='#44447a', fontsize=7, labelpad=0)
ax3d.set_ylabel('Y (AU)', color='#44447a', fontsize=7, labelpad=0)
ax3d.set_zlabel('Z (AU)', color='#44447a', fontsize=7, labelpad=0)
ax3d.tick_params(colors='#33335a', labelsize=5)

# Camera: 35° elevation gives nice 3D perspective of the orbit ring
# Billboard panels always face the camera so they look like hexagons from any angle
ax3d.view_init(elev=35, azim=-60)

# Draw static elements once
def _draw_static():
    # Sun glow — 5 layers for a realistic corona look
    for s, c, a in [
        (500,   '#ffffff', 1.00),
        (1800,  '#fffacc', 0.60),
        (4500,  '#ffee88', 0.30),
        (9000,  '#ffcc22', 0.14),
        (18000, '#ff8800', 0.06),
    ]:
        ax3d.scatter([0], [0], [0], s=s, color=c, alpha=a, zorder=10, depthshade=False)
    ax3d.text(0.02, 0.02, 0.04, '☀ Sun', color='#ffee88',
              fontsize=7, fontweight='bold', zorder=12)

    # Perihelion / aphelion markers
    ax3d.scatter([R_PERI], [0], [0], s=18, color='#ff6644',
                 marker='D', zorder=8, alpha=0.7)
    ax3d.text(R_PERI + 0.02, 0, 0.02, f'Peri\n{R_PERI:.3f} AU',
              color='#ff6644', fontsize=5.5)
    ax3d.scatter([-R_APO], [0], [0], s=18, color='#4488ff',
                 marker='D', zorder=8, alpha=0.7)
    ax3d.text(-R_APO - 0.12, 0, 0.02, f'Apo\n{R_APO:.3f} AU',
              color='#4488ff', fontsize=5.5)

_draw_static()

# Draw ONE shared orbit ring — bright enough to see clearly
xs, ys, zs = shared_trace
orbit_line, = ax3d.plot(xs, ys, zs, color='#4466ff', alpha=0.55,
                        linewidth=1.2, zorder=2)

# Compact legend
legend_data = [
    ('#00ccff', '■ Active'),
    ('#ffaa00', '■ Shadowed'),
    ('#cc44ff', '■ Isolated'),
    ('#ff3333', '✕ Failed'),
]
for idx, (col, lbl) in enumerate(legend_data):
    ax3d.text2D(0.02, 0.13 - idx * 0.038, lbl,
                color=col, fontsize=7, transform=ax3d.transAxes,
                fontfamily='monospace')

title_text = ax3d.set_title('', color='#ddddff', fontsize=9, pad=3)

# -----------------------------------------------------------------------
# Pause / screenshot / speed
# -----------------------------------------------------------------------
paused         = False
_current_frame = [0]
_interval      = [40]

def on_key(event):
    global paused
    if event.key == ' ':
        paused = not paused
        fig.canvas.draw_idle()
    elif event.key == 's' and paused:
        fname = f'osehs_day{_current_frame[0]:04d}.png'
        fig.savefig(fname, dpi=180, bbox_inches='tight', facecolor=BG)
        print(f'Saved → {fname}')
    elif event.key in ('=', '+'):
        _interval[0] = max(10, _interval[0] - 15)
        ani.event_source.interval = _interval[0]
        print(f'Speed up → interval {_interval[0]} ms')
    elif event.key in ('-', '_'):
        _interval[0] = min(600, _interval[0] + 15)
        ani.event_source.interval = _interval[0]
        print(f'Slow down → interval {_interval[0]} ms')

fig.canvas.mpl_connect('key_press_event', on_key)

# -----------------------------------------------------------------------
# Force 3D panes to stay dark (matplotlib resets them each frame)
# -----------------------------------------------------------------------
def _force_dark_3d():
    ax3d.set_facecolor(BG)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#080818')
    ax3d.xaxis._axinfo['grid']['color'] = '#0a0a1e'
    ax3d.yaxis._axinfo['grid']['color'] = '#0a0a1e'
    ax3d.zaxis._axinfo['grid']['color'] = '#0a0a1e'
    ax3d.tick_params(colors='#22224a', labelsize=5)

# -----------------------------------------------------------------------
# Helper: style a metric subplot
# -----------------------------------------------------------------------
def _style(ax, title, ylabel, xlim=None):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color='#aaaadd', fontsize=7.5, pad=2, loc='left')
    ax.set_ylabel(ylabel, color='#7777aa', fontsize=6.5)
    ax.tick_params(colors='#7777aa', labelsize=6)
    for sp in ax.spines.values():
        sp.set_color('#1a1a3a')
    if xlim:
        ax.set_xlim(xlim)

# -----------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------
def update(frame):
    if paused:
        return

    _current_frame[0] = frame
    _force_dark_3d()

    # Slow near dropout events
    near = any(abs(frame - d) <= 12 for d in DROPOUT_EVENTS)
    ani.event_source.interval = 300 if near else _interval[0]

    if frame in DROPOUT_EVENTS:
        swarm.simulate_dropout(DROPOUT_EVENTS[frame])

    swarm.step(DT)

    alive    = swarm.alive_units()
    n_shadow = sum(1 for u in alive if getattr(u, 'shadowed', False))
    n_iso    = sum(1 for u in alive if getattr(u, 'comms_isolated', False))
    avg_batt = np.mean([u.battery_pct() for u in alive]) if alive else 0
    tot_kw   = sum(u.current_power_W for u in alive) / 1000

    title_text.set_text(
        f'Day {frame:4d}  │  {len(alive)}/{N_UNITS} active  │  '
        f'{tot_kw:.1f} kW  │  Batt {avg_batt:.0f}%'
        + (f'  │  ⚠ {n_shadow} shadowed' if n_shadow else '')
        + (f'  │  ⚠ {n_iso} isolated'    if n_iso    else '')
    )

    # -- Rebuild 3D panel geometry --
    for artist in list(ax3d.collections):
        artist.remove()

    # Re-draw Sun each frame (cleared with collections)
    for s, c, a in [
        (500,   '#ffffff', 1.00),
        (1800,  '#fffacc', 0.60),
        (4500,  '#ffee88', 0.30),
        (9000,  '#ffcc22', 0.14),
        (18000, '#ff8800', 0.06),
    ]:
        ax3d.scatter([0], [0], [0], s=s, color=c, alpha=a, zorder=10, depthshade=False)

    # Separate units by state
    pos_ok, pos_sh, pos_iso = [], [], []
    for u in alive:
        p = u.position() / AU
        if getattr(u, 'shadowed', False):
            pos_sh.append(p)
        elif getattr(u, 'comms_isolated', False):
            pos_iso.append(p)
        else:
            pos_ok.append(p)

    def _draw_units(positions, core_col, glow_col):
        if not positions:
            return
        pts = np.array(positions)
        # Outer glow halo
        ax3d.scatter(pts[:,0], pts[:,1], pts[:,2],
                     s=PANEL_GLOW_SIZE, c=glow_col, alpha=0.25,
                     zorder=4, depthshade=False, linewidths=0)
        # Bright core
        ax3d.scatter(pts[:,0], pts[:,1], pts[:,2],
                     s=PANEL_CORE_SIZE, c=core_col, alpha=0.95,
                     zorder=5, depthshade=False, linewidths=0,
                     marker='h')   # 'h' = regular hexagon marker in matplotlib

    _draw_units(pos_ok,  '#00ddff', '#00aadd')
    _draw_units(pos_sh,  '#ffbb00', '#dd8800')
    _draw_units(pos_iso, '#dd55ff', '#aa22cc')

    dead = swarm.failed_units()
    if dead:
        pf = np.array([u.position() / AU for u in dead])
        ax3d.scatter(pf[:,0], pf[:,1], pf[:,2],
                     c='#ff3333', s=140, marker='x', zorder=6,
                     linewidths=2.5, depthshade=False)

    # Perihelion / aphelion dots (persist after collection clear)
    ax3d.scatter([R_PERI], [0], [0], s=18, color='#ff6644', marker='D',
                 zorder=8, alpha=0.7, depthshade=False)
    ax3d.scatter([-R_APO], [0], [0], s=18, color='#4488ff', marker='D',
                 zorder=8, alpha=0.7, depthshade=False)

    # -- Metric charts (every 4 frames to keep animation smooth) --
    if frame % 4 == 0 and len(swarm.metrics) > 1:
        m    = swarm.metrics
        days = [x['t'] / 86400 for x in m]
        xl   = (0, MAX_STEPS)

        # — Swarm Health —
        pct = [x['pct_functional'] for x in m]
        ax_health.clear()
        ax_health.fill_between(days, pct, 75,
                               where=[p >= 75 for p in pct],
                               alpha=0.20, color='#00ff88', interpolate=True)
        ax_health.fill_between(days, pct, 75,
                               where=[p < 75 for p in pct],
                               alpha=0.30, color='#ff4444', interpolate=True)
        ax_health.plot(days, pct, color='#00ff88', linewidth=1.4)
        ax_health.axhline(75, color='#ffaa00', linestyle='--',
                          linewidth=0.8, label='75% KPP threshold')
        ax_health.set_ylim(0, 105)
        ax_health.legend(fontsize=5.5, facecolor=PANEL_BG,
                         labelcolor='#ffaa00', loc='lower left',
                         framealpha=0.7)
        _style(ax_health, 'Swarm Health', '%', xl)

        # — Total Power Output —
        pw = [x['total_power_W'] / 1000 for x in m]
        pw_ymax = max(pw) * 1.15 if pw else POWER_PERI_KW * 1.2
        ax_power.clear()
        ax_power.fill_between(days, pw, alpha=0.18, color='#ffdd00')
        ax_power.plot(days, pw, color='#ffdd00', linewidth=1.4)
        ax_power.axhline(POWER_NOM_KW, color='#ff8844', linestyle='--',
                         linewidth=0.8, label=f'Nominal {POWER_NOM_KW:.1f} kW')
        ax_power.axhline(POWER_PERI_KW, color='#ff4444', linestyle=':',
                         linewidth=0.7, label=f'Perihelion {POWER_PERI_KW:.1f} kW')
        ax_power.axhline(POWER_APO_KW, color='#4488ff', linestyle=':',
                         linewidth=0.7, label=f'Aphelion {POWER_APO_KW:.1f} kW')
        ax_power.set_ylim(0, pw_ymax)
        ax_power.legend(fontsize=5.5, facecolor=PANEL_BG,
                        labelcolor='white', loc='upper right',
                        framealpha=0.7, ncol=1)
        _style(ax_power, 'Total Power Output', 'kW', xl)

        # — Battery Level —
        batt = [x['avg_battery_pct'] for x in m]
        ax_battery.clear()
        ax_battery.fill_between(days, batt, alpha=0.18, color='#44aaff')
        ax_battery.plot(days, batt, color='#44aaff', linewidth=1.4)
        ax_battery.axhline(20, color='#ff4444', linestyle='--',
                           linewidth=0.8, label='20% critical')
        ax_battery.set_ylim(0, 105)
        ax_battery.legend(fontsize=5.5, facecolor=PANEL_BG,
                          labelcolor='#ff4444', loc='lower left',
                          framealpha=0.7)
        _style(ax_battery, 'Avg Battery Level (500 kWh/unit)', '%', xl)

        # — Orbital Drift from SRP —
        drift = [x['max_drift_km'] for x in m]
        ax_drift.clear()
        ax_drift.fill_between(days, drift, alpha=0.18, color='#ff6688')
        ax_drift.plot(days, drift, color='#ff6688', linewidth=1.4)
        ax_drift.set_xlabel('Simulation Day', color='#7777aa', fontsize=6.5)
        _style(ax_drift, 'Max Orbital Drift — Solar Radiation Pressure', 'km', xl)

        fig.canvas.draw_idle()


ani = animation.FuncAnimation(
    fig, update,
    frames=MAX_STEPS,
    interval=40,
    repeat=False,
)

plt.show()
