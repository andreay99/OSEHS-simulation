# OSEHS Simulation — Quick Start

## TL;DR

```bash
pip install -r requirements.txt
python3 main.py
```

Watch the swarm orbit in 3D. Check KPP metrics on the right.

---

## What This Does

✅ Simulates 12 autonomous solar collector panels orbiting as a "pearl necklace" formation  
✅ Each panel tracks position using Classical Orbital Elements (COE)  
✅ Collision avoidance: units spread apart if too close  
✅ Shadow avoidance: units adjust RAAN if they'd block each other's solar exposure  
✅ Fault tolerance: if units fail, remaining ones auto-rebalance  
✅ Energy tracking: logs cumulative solar energy collected  

---

## Customizing Parameters

Edit **`config.py`**:

```python
ORBITAL_RADIUS = 1.0 * AU   # ← Change orbital radius here
ECCENTRICITY = 0.0          # ← Change orbit shape here
INCLINATION = np.radians(5) # ← Change tilt here
N_UNITS = 12                # ← Change swarm size here
MAX_STEPS = 365 * 3         # ← Change simulation length here
```

**Once KesUraNu sends orbital dynamics data**, swap those three lines above — nothing else changes.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Runs the 3D visualization + metrics |
| `config.py` | **← Edit this to change everything** |
| `sim/orbital.py` | Kepler solver, COE math |
| `sim/unit.py` | Single panel class |
| `sim/swarm.py` | Swarm behavior, coordination, fault tolerance |
| `example_configs.py` | Sample orbital parameters |
| `README.md` | Full documentation |

---

## KPP Metrics (Right Side of Screen)

- **% Functional**: Alive units / Total units (goal: ≥94% if 10% fail) ✅
- **Shadow violations**: Units in angular shadow risk (goal: zero) ✅  
- **Energy**: Cumulative solar energy collected ✅

---

## Next: Waiting on KesUraNu

Once orbital dynamics team sends data:

1. Paste into `config.py`:
   ```python
   ORBITAL_RADIUS = <their semimajor axis>
   ECCENTRICITY = <their eccentricity>
   INCLINATION = <their inclination>
   ```
2. Run `python3 main.py`
3. Verify KPPs match proposal targets

Done ✅

---

## Questions?

- **How do I add new swarm behaviors?** → Edit `sim/swarm.py`, add method to `Swarm` class
- **How do I change collision distance?** → Edit `MIN_SAFE_DIST` in `config.py`
- **How do I log more metrics?** → Add to `self.metrics.append()` in `swarm.py:_log_metrics()`
- **Can I run longer simulations?** → Change `MAX_STEPS` in `config.py`
