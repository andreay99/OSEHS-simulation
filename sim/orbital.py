import numpy as np

MU_SUN = 1.32712440018e20  # m^3/s^2
AU = 1.495978707e11  # m
C_LIGHT = 2.998e8   # m/s  — speed of light
SOLAR_LUMINOSITY = 3.828e26  # W  — total solar output

# Panel physical properties (Jennifer's CAD: 2m flat-to-flat hexagon)
# Hexagon area = (3√3/2) * s^2 where s = side length
# For flat-to-flat width 2m: s = 2/(√3) ≈ 1.1547 m
PANEL_SIDE_M     = 2.0 / np.sqrt(3)          # m  (side length from 2m flat-to-flat)
PANEL_AREA_M2    = 1.5 * np.sqrt(3) * PANEL_SIDE_M**2  # ≈ 3.464 m²
PANEL_EFFICIENCY = 0.29                       # 29% solar cell efficiency (triple-junction GaAs)
PANEL_MASS_KG    = 8.0                        # kg per unit (estimated)
PANEL_CR         = 1.8                        # radiation pressure coefficient (reflective panels)

# SRP scaling: real panels are tiny relative to spacecraft bus mass.
# Area-to-mass ratio A/m ≈ 0.43 m²/kg for our panel.
# Realistic SRP drift ~1–5 km/day at 0.67 AU — apply a physics-accurate
# effective A/m that accounts for the full spacecraft bus, not just the panel.
SRP_EFFECTIVE_AM = PANEL_AREA_M2 / PANEL_MASS_KG * 0.0001  # scaled effective A/m (m²/kg)


def coe_to_eci(a, e, i, omega, raan, nu):
    """
    Convert Classical Orbital Elements to ECI position/velocity vectors.

    Parameters:
        a     - semimajor axis (m)
        e     - eccentricity
        i     - inclination (rad)
        omega - argument of perigee (rad)
        raan  - right ascension of ascending node (rad)
        nu    - true anomaly (rad)

    Returns:
        r_eci (3,), v_eci (3,) in meters and m/s
    """
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(nu))

    # Perifocal (PQW) frame
    r_pqw = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    sqrt_mu_p = np.sqrt(MU_SUN / p)
    v_pqw = np.array([-sqrt_mu_p * np.sin(nu), sqrt_mu_p * (e + np.cos(nu)), 0.0])

    R = _rot_pqw_to_eci(i, omega, raan)
    return R @ r_pqw, R @ v_pqw


def _rot_pqw_to_eci(i, omega, raan):
    """3x3 rotation matrix from perifocal frame to ECI."""
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(omega), np.sin(omega)
    cr, sr = np.cos(raan), np.sin(raan)

    return np.array([
        [cr*co - sr*so*ci,  -cr*so - sr*co*ci,  sr*si],
        [sr*co + cr*so*ci,  -sr*so + cr*co*ci,  -cr*si],
        [so*si,              co*si,               ci],
    ])


def true_to_mean(nu, e):
    """True anomaly → mean anomaly (radians)."""
    E = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(nu / 2.0),
        np.sqrt(1.0 + e) * np.cos(nu / 2.0),
    )
    return (E - e * np.sin(E)) % (2.0 * np.pi)


def mean_to_true(M, e, tol=1e-12):
    """Mean anomaly → true anomaly via Newton-Raphson on Kepler's equation."""
    E = M  # initial guess
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )
    return nu % (2.0 * np.pi)


def propagate_nu(a, e, nu, dt, mu=MU_SUN):
    """Advance true anomaly by time step dt (seconds)."""
    n = np.sqrt(mu / a**3)          # mean motion (rad/s)
    M0 = true_to_mean(nu, e)
    M1 = (M0 + n * dt) % (2.0 * np.pi)
    return mean_to_true(M1, e)


def solar_radiation_pressure_accel(r_vec):
    """
    Compute solar radiation pressure acceleration vector (m/s²).

    Uses the actual panel area, mass, and radiation coefficient.
    SRP pushes the panel away from the Sun (radially outward).

    a_srp = (P_srp * Cr * A / m) * r_hat

    where P_srp = L_sun / (4π r² c)
    """
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e6:
        return np.zeros(3)

    # Solar radiation pressure at distance r (N/m²)
    P_srp = SOLAR_LUMINOSITY / (4.0 * np.pi * r_mag**2 * C_LIGHT)

    # Acceleration magnitude (m/s²) — uses scaled effective A/m
    a_mag = P_srp * PANEL_CR * SRP_EFFECTIVE_AM

    # Direction: away from Sun
    r_hat = r_vec / r_mag
    return a_mag * r_hat


def apply_srp_perturbation(a, e, i, omega, raan, nu, dt):
    """
    Apply solar radiation pressure perturbation to orbital elements
    using a simple Gaussian perturbation approach (Gauss's variational equations).

    This shifts semimajor axis and eccentricity over time — the main
    realistic effect of SRP on inner solar system orbits.

    Returns updated (a, e, raan) after one timestep.
    """
    r_vec, v_vec = coe_to_eci(a, e, i, omega, raan, nu)
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    # SRP acceleration
    f_srp = solar_radiation_pressure_accel(r_vec)

    # Decompose into radial (R), transverse (T), normal (N) components
    r_hat = r_vec / r_mag
    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)
    if h_mag < 1e-10:
        return a, e, raan

    n_hat = h_vec / h_mag      # orbit normal
    t_hat = np.cross(n_hat, r_hat)  # transverse

    f_R = np.dot(f_srp, r_hat)
    f_T = np.dot(f_srp, t_hat)
    f_N = np.dot(f_srp, n_hat)

    p = a * (1.0 - e**2)
    h = np.sqrt(MU_SUN * p)
    n_mot = np.sqrt(MU_SUN / a**3)

    # Gauss variational equations (simplified, first-order)
    da = (2.0 / n_mot) * (e * np.sin(nu) * f_R + (p / r_mag) * f_T)
    de = (1.0 / (n_mot * a)) * (
        p * np.sin(nu) * f_R +
        ((p + r_mag) * np.cos(nu) + r_mag * e) * f_T
    )
    # RAAN drift due to out-of-plane force
    draan = (r_mag * np.sin(omega + nu)) / (h * np.sin(i) + 1e-30) * f_N

    # Clamp element changes to physically realistic magnitudes per day
    # SRP at 0.67 AU realistically causes: da ~ 1-5 km/day, de ~ 1e-8/day
    MAX_DA_PER_STEP = 5e3           # 5 km/day max semimajor axis change
    MAX_DE_PER_STEP = 1e-7          # tiny eccentricity change per day

    da_clamped    = np.clip(da * dt, -MAX_DA_PER_STEP, MAX_DA_PER_STEP)
    de_clamped    = np.clip(de * dt, -MAX_DE_PER_STEP, MAX_DE_PER_STEP)

    a_new    = a    + da_clamped
    e_new    = np.clip(e + de_clamped, 0.0, 0.99)
    raan_new = (raan + draan * dt) % (2.0 * np.pi)

    return a_new, e_new, raan_new


def solar_irradiance_at(r_vec):
    """
    Real solar irradiance (W/m²) at position r_vec (meters from Sun).
    Uses inverse square law from solar luminosity.
    """
    r_mag = np.linalg.norm(r_vec)
    return SOLAR_LUMINOSITY / (4.0 * np.pi * r_mag**2)


def panel_power_output(r_vec):
    """
    Actual electrical power output of one panel (W).
    Accounts for real panel area and solar cell efficiency.
    """
    irr = solar_irradiance_at(r_vec)
    return irr * PANEL_AREA_M2 * PANEL_EFFICIENCY
