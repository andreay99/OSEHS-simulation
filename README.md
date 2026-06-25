# OSEHS — Orbital Solar Energy Harvesting Swarm

Digital twin + real-time simulation for the Star Maker team's NASA ORBIT Challenge Phase 3.

## Quick Start

```bash
pip install -r requirements.txt
python3 main.py
```

Controls:
- **Space** — pause / resume
- **S** — save screenshot (while paused)
- **=** / **-** — speed up / slow down

## Orbital Parameters (KesUraNu mission data)

All parameters updated with real orbital dynamics data from KesUraNu (Ranu Baylor):

| Parameter | Value |
|-----------|-------|
| Semimajor axis | 0.6722 AU |
| Eccentricity | 0.0760 |
| Inclination | 5.0° |
| Orbital period | 201.3 days |
| Perihelion | 0.621 AU |
| Aphelion | 0.723 AU |

## Physics Modelled

- **Keplerian propagation** — Newton-Raphson Kepler solver (mean → eccentric → true anomaly)
- **Solar radiation pressure (SRP)** — Gauss variational equations perturb semimajor axis, eccentricity, and RAAN over time. Realistic drift of ~1–2 km/day at 0.67 AU
- **Real panel power output** — irradiance × panel area × efficiency. Based on Jennifer's 2 m flat-to-flat hexagon CAD: 3.46 m², 29% triple-junction GaAs cells
- **Battery model** — 500 kWh storage per unit. Charges near perihelion (higher irradiance), drains near aphelion. Smooth cycle follows the 201-day orbit
- **Comms-range-limited coordination** — units only share state packets with neighbors within 0.52 AU (covers worst-case 9-unit spacing after 3 dropouts on elliptical orbit)
- **Shadow avoidance** — autonomous RAAN nudge if angular separation < 10°. No ground command needed
- **Collision avoidance** — true anomaly nudge if unit spacing < 750,000 km
- **Auto-rebalance on dropout** — RAAN redistributed evenly among surviving units automatically

## Power Output

| Condition | Total swarm power (12 units) |
|-----------|------------------------------|
| Perihelion (0.621 AU) | 42.5 kW |
| Nominal (0.6722 AU) | 36.3 kW |
| Aphelion (0.723 AU) | 31.4 kW |

## Dashboard

The real-time window shows:

- **3D orbit view** (left) — pearl necklace formation around the Sun. Cyan = active, orange = shadowed, purple = comms-isolated, red X = failed
- **Swarm Health %** (top right) — functional units vs 75% KPP threshold
- **Total Power Output (kW)** — live power with perihelion/nominal/aphelion reference lines
- **Avg Battery Level %** — charge state cycling with orbital distance
- **Max Orbital Drift (km)** — cumulative SRP-induced drift from nominal orbit

## Project Structure

```
OSEHS-simulation/
├── sim/
│   ├── orbital.py   — COE propagator, Kepler solver, ECI conversion, SRP perturbation, power model
│   ├── unit.py      — SwarmUnit: orbital state, battery, power output, comms range, drift tracking
│   ├── swarm.py     — Swarm: pearl necklace init, comms-limited broadcast, avoidance, fault tolerance
│   └── __init__.py
├── config.py        — All parameters (orbital, swarm, simulation, dropout events)
├── main.py          — Real-time 3D visualization + 4-panel metrics dashboard
├── requirements.txt — Python dependencies
└── README.md
```

## Configuration

Edit `config.py` to change parameters:

```python
ORBITAL_RADIUS = 0.6722 * AU   # semimajor axis — KesUraNu mission data
ECCENTRICITY   = 0.0760        # eccentricity   — KesUraNu mission data
INCLINATION    = np.radians(5) # inclination
N_UNITS        = 12            # number of collector panels
DT             = 3600 * 24     # time step (1 day)
MAX_STEPS      = 365 * 3       # 3-year simulation
DROPOUT_EVENTS = {100: 2, 200: 7, 300: 5}  # fault tolerance test cases
```

## Key Performance Parameters (KPPs)

| KPP | Target | Result |
|-----|--------|--------|
| Swarm health after 25% unit loss | ≥ 75% functional | ✅ 75% maintained after 3 dropouts |
| Autonomous correction | 95% without ground command | ✅ Shadow + collision avoidance fully autonomous |
| Safe spacing | No collisions | ✅ Zero collision violations over 1,095 days |
| Comms coverage | All units connected | ✅ 0 isolation events over full 3-year run |
| Power output | Above aphelion minimum | ✅ 23.5–42.5 kW depending on orbital position |

## Contact

- **Andrea Yanez Soto (Software)**: ay387@njit.edu
- **Team Lead**: Tatiana
- **Orbital data**: KesUraNu (Ranu Baylor)
