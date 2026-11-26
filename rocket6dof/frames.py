"""Rotations and reference frames.

Quaternions are scalar-first Hamilton quaternions q = [w, x, y, z].
quat_to_dcm(q) returns C such that v_ref = C @ v_body.

Frames used:
  ECEF  Earth-centred, Earth-fixed (WGS 84)
  NED   local North-East-Down at a geodetic point
  body  x forward along the rocket axis (toward the nose), y and z complete a right-handed set.
        The body origin is the nose tip; a point at station s (metres aft of the tip) has x = -s.
"""
import math
import numpy as np

from .constants import WGS84_A, WGS84_E2


def cross3(a, b):
    """Cross product of two 3-vectors (much faster than np.cross for single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def quat_multiply(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_to_dcm(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_quat(C):
    """Shepperd's method; returns a unit quaternion with non-negative scalar part."""
    tr = C[0, 0] + C[1, 1] + C[2, 2]
    cand = [tr, C[0, 0], C[1, 1], C[2, 2]]
    i = int(np.argmax(cand))
    if i == 0:
        w = 0.5 * math.sqrt(1 + tr)
        x = (C[2, 1] - C[1, 2]) / (4 * w)
        y = (C[0, 2] - C[2, 0]) / (4 * w)
        z = (C[1, 0] - C[0, 1]) / (4 * w)
    elif i == 1:
        x = 0.5 * math.sqrt(1 + C[0, 0] - C[1, 1] - C[2, 2])
        w = (C[2, 1] - C[1, 2]) / (4 * x)
        y = (C[0, 1] + C[1, 0]) / (4 * x)
        z = (C[0, 2] + C[2, 0]) / (4 * x)
    elif i == 2:
        y = 0.5 * math.sqrt(1 - C[0, 0] + C[1, 1] - C[2, 2])
        w = (C[0, 2] - C[2, 0]) / (4 * y)
        x = (C[0, 1] + C[1, 0]) / (4 * y)
        z = (C[1, 2] + C[2, 1]) / (4 * y)
    else:
        z = 0.5 * math.sqrt(1 - C[0, 0] - C[1, 1] + C[2, 2])
        w = (C[1, 0] - C[0, 1]) / (4 * z)
        x = (C[0, 2] + C[2, 0]) / (4 * z)
        y = (C[1, 2] + C[2, 1]) / (4 * z)
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_rate(q, omega_body):
    """q_dot = 0.5 * q (x) [0, omega], omega = body rate relative to the reference frame, in body axes."""
    return 0.5 * quat_multiply(q, (0.0, omega_body[0], omega_body[1], omega_body[2]))


def geodetic_to_ecef(lat, lon, h):
    """Geodetic latitude/longitude (rad) and ellipsoidal height (m) to ECEF (m)."""
    sl, cl = math.sin(lat), math.cos(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sl * sl)
    return np.array([
        (N + h) * cl * math.cos(lon),
        (N + h) * cl * math.sin(lon),
        (N * (1 - WGS84_E2) + h) * sl,
    ])


def ecef_to_geodetic(r):
    """ECEF (m) to geodetic latitude, longitude (rad) and ellipsoidal height (m). Iterative, sub-mm."""
    x, y, z = r
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        sl = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sl * sl)
        h = p * math.cos(lat) + z * sl - WGS84_A * math.sqrt(1 - WGS84_E2 * sl * sl)
        lat_new = math.atan2(z, p * (1 - WGS84_E2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-13:
            lat = lat_new
            break
        lat = lat_new
    sl = math.sin(lat)
    h = p * math.cos(lat) + z * sl - WGS84_A * math.sqrt(1 - WGS84_E2 * sl * sl)
    return lat, lon, h


def dcm_ecef_to_ned(lat, lon):
    """Rotation matrix taking ECEF vectors to NED at (lat, lon)."""
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([
        [-sl * co, -sl * so, cl],
        [-so, co, 0.0],
        [-cl * co, -cl * so, -sl],
    ])


NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
