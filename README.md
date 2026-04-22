# OSEHS — Orbital Solar Energy Harvesting Swarm

Digital twin + real-time simulation for the Star Maker team's NASA ORBIT Challenge Phase 2 proposal.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
python3 main.py
```

A window opens showing:
- **3D orbit view** (left): Units in cyan, shadow-risk units in orange, failed units in red
- **Swarm health** (top right): % functional vs 94% KPP threshold
- **Energy collection** (bottom right): Cumulative energy harvested

Press Ctrl-C or close the window to stop.

## Configuration

Edit `config.py` to change simulation parameters:

```python
# Orbital parameters
ORBITAL_RADIUS = 1.0 * AU      # semimajor axis (m)
ECCENTRICITY = 0.0             # orbital eccentricity
INCLINATION = np.radians(5)    # orbital inclination (rad)

# Swarm parameters
N_UNITS = 12                    # number of collector panels
MIN_SAFE_DIST = 0.005 * AU      # collision avoidance threshold
MIN_ANGULAR_SEP = np.radians(10)  # shadow avoidance threshold (degrees)

# Simulation parameters
DT = 3600 * 24                  # time step (seconds; default: 1 day)
MAX_STEPS = 365 * 3             # simulation length (steps; default: 3 years)

# Dropout events (optional)
DROPOUT_EVENTS = {100: 2, 200: 7, 300: 5}  # {step: unit_id}
```

Once KesUraNu sends orbital dynamics data, update `ORBITAL_RADIUS`, `ECCENTRICITY`, and `INCLINATION` in `config.py` — no other changes needed.

## Project Structure

```
NASA COMP/
├── sim/
│   ├── orbital.py      — COE propagator, Kepler solver, ECI conversion
│   ├── unit.py         — SwarmUnit class (single panel state + energy)
│   ├── swarm.py        — Swarm class (pearl necklace formation, collision/shadow avoidance, fault tolerance)
│   └── __init__.py
├── config.py           — Simulation parameters (edit here!)
├── main.py             — Entry point, real-time 3D visualization + metrics
├── requirements.txt    — Python dependencies
└── README.md           — This file
```

## How It Works

### Orbital Mechanics
- **State representation**: Classical Orbital Elements (COE): semimajor axis, eccentricity, inclination, argument of perigee, RAAN, true anomaly
- **Propagation**: Kepler's equations via Newton-Raphson solver for mean → true anomaly
- **Conversion**: COE ↔ ECI (Earth-Centered Inertial) using rotation matrices

### Swarm Behavior
- **Formation**: Pearl necklace — all units share same a, e, i, ω; distributed by RAAN (right ascension of ascending node)
- **Collision avoidance**: If distance < MIN_SAFE_DIST, nudge true anomaly outward
- **Shadow avoidance**: If angular separation < MIN_ANGULAR_SEP, nudge RAAN outward (95% autonomous, no ground intervention)
- **Dropout recovery**: On unit failure, auto-rebalance remaining units' RAAN to close coverage gaps

### Key Performance Parameters (KPPs) Tracked

| KPP | Target | Status |
|-----|--------|--------|
| Swarm functional if 10% fail | ≥94% | ✅ Logged in metrics |
| Safe relative distance (99%) | 99% units > MIN_SAFE_DIST | ✅ Collision violations = 0 |
| Autonomous error correction | 95% without ground intervention | ✅ Auto-rebalance on dropout |
| Shadow violations | Minimized | ✅ Tracked, shadow avoidance active |

## Energy Collection

Each unit harvests solar energy proportional to inverse-square law:
```
irradiance = 1361 W/m² × (1 AU / distance)²
```

Total swarm energy shown in bottom-right plot. Drops when units fail (lost collection area).

## Running Tests

Check that orbital mechanics are correct:
```bash
python3 -c "
from sim.orbital import coe_to_eci, propagate_nu, AU
import numpy as np

# Test: 1 year orbit should return to start
nu0 = 0.0
nu1 = propagate_nu(AU, 0.0, nu0, dt=3600*24*365)
print(f'True anomaly after 1 year: {np.degrees(nu1):.2f}° (expect ~360°)')

# Test: position magnitude at 1 AU
r, _ = coe_to_eci(AU, 0.0, 0.0, 0.0, 0.0, nu0)
print(f'Position: {np.linalg.norm(r)/AU:.4f} AU (expect 1.0000)')
"
```

Expected output:
```
True anomaly after 1 year: 359.75° (expect ~360°)
Position: 1.0000 AU (expect 1.0000)
```

## Next Steps

1. **Await orbital data from KesUraNu** — update `config.py` with real mission parameters
2. **Integration with other teams**:
   - Kevin (power): Validate energy collection vs electrical generation specs
   - Tatiana/Jennifer/Natalie (mechanical): Validate structural mass assumptions, thermal models
3. **Advanced features** (optional):
   - Adaptive swarm algorithms (machine learning-based formation adjustment)
   - Power beaming simulation (DC → microwave conversion losses, targeting accuracy)
   - Orbital perturbations (solar radiation pressure, gravitational harmonics)

## Contact

- **Andrea (Software)**: ay387@njit.edu — simulation, swarm coordination, GUI
- **KesUraNu (Orbital Physics)**: orbital dynamics data input
- **Team Lead (Tatiana)**: Tay_8910

## Notes

- All orbital mechanics use the Sun's gravitational parameter: μ = 1.327×10²⁰ m³/s²
- Solar irradiance at 1 AU: 1361 W/m² (real physical constant)
- Simulation assumes frictionless heliocentric orbits (no atmospheric drag, minimal perturbations)
- GPU acceleration not needed; runs comfortably on CPU
