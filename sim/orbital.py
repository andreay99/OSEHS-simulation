import numpy as np

MU_SUN = 1.32712440018e20  # m^3/s^2

AU = 1.495978707e11  # m


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
