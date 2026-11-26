"""Earth gravity (WGS 84 with J2) and rotation."""
import numpy as np

from .frames import cross3

from .constants import WGS84_A, WGS84_GM, WGS84_J2, OMEGA_EARTH

OMEGA_E = np.array([0.0, 0.0, OMEGA_EARTH])  # Earth rotation vector in ECEF/ECI, rad/s


def gravitation_ecef(r):
    """Gravitational acceleration (no centrifugal term) in ECEF, point-mass + J2."""
    x, y, z = r
    rn2 = x * x + y * y + z * z
    rn = rn2 ** 0.5
    k = 1.5 * WGS84_J2 * (WGS84_A * WGS84_A) / rn2
    zr2 = z * z / rn2
    c = -WGS84_GM / (rn2 * rn)
    return np.array([
        c * x * (1 + k * (1 - 5 * zr2)),
        c * y * (1 + k * (1 - 5 * zr2)),
        c * z * (1 + k * (3 - 5 * zr2)),
    ])


def gravity_ecef(r):
    """Effective gravity felt by a body at rest on the rotating Earth (gravitation + centrifugal)."""
    return gravitation_ecef(r) - cross3(OMEGA_E, cross3(OMEGA_E, r))
