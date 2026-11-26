"""Atmosphere, mean wind and Dryden turbulence.

Atmosphere: US Standard Atmosphere 1976 (0-86 km), with an optional uniform temperature offset
(pressure re-integrated hydrostatically for the offset profile) and humidity (moist-air density).
Viscosity: Sutherland's law.
"""
import math
from collections import namedtuple

import numpy as np

from .constants import (G0, USSA_R_STAR, USSA_M0, USSA_R_AIR, USSA_EARTH_RADIUS, GAMMA_AIR,
                        R_WATER_VAPOUR, SUTHERLAND_BETA, SUTHERLAND_S)

AtmosphereState = namedtuple("AtmosphereState", "T P rho a mu nu")

# Geopotential layer bases (m) and lapse rates (K/m), USSA 1976
_H_BASE = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0)
_LAPSE = (-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0)
_T0 = 288.15
_P0 = 101325.0
_GMR = G0 * USSA_M0 / USSA_R_STAR  # K/m


class Atmosphere:
    def __init__(self, temperature_offset=0.0, relative_humidity=0.0, pressure_scale=1.0):
        self.dT = temperature_offset
        self.rh = relative_humidity
        self.p_scale = pressure_scale
        self.T_base = [_T0]
        for i in range(1, len(_H_BASE)):
            self.T_base.append(self.T_base[-1] + _LAPSE[i - 1] * (_H_BASE[i] - _H_BASE[i - 1]))
        self.P_base = [_P0]
        for i in range(1, len(_H_BASE)):
            self.P_base.append(self._layer_pressure(i - 1, _H_BASE[i]))

    def _layer_pressure(self, i, h):
        Tb = self.T_base[i] + self.dT
        L = _LAPSE[i]
        dh = h - _H_BASE[i]
        if L == 0.0:
            return self.P_base[i] * math.exp(-_GMR * dh / Tb)
        return self.P_base[i] * (Tb / (Tb + L * dh)) ** (_GMR / L)

    def properties(self, z):
        """State at geometric altitude z (m above mean sea level)."""
        h = USSA_EARTH_RADIUS * z / (USSA_EARTH_RADIUS + z)  # geopotential altitude
        i = 0
        for k in range(len(_H_BASE) - 1, -1, -1):
            if h >= _H_BASE[k]:
                i = k
                break
        T = self.T_base[i] + _LAPSE[i] * (h - _H_BASE[i]) + self.dT
        P = self._layer_pressure(i, h) * self.p_scale
        R_mix = USSA_R_AIR
        if self.rh > 0.0:
            Tc = T - 273.15
            e_sat = 611.21 * math.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))  # Buck (1996)
            e = min(self.rh * e_sat, 0.99 * P)
            rho = (P - e) / (USSA_R_AIR * T) + e / (R_WATER_VAPOUR * T)
            R_mix = P / (rho * T)
        else:
            rho = P / (USSA_R_AIR * T)
        a = math.sqrt(GAMMA_AIR * R_mix * T)
        mu = SUTHERLAND_BETA * T ** 1.5 / (T + SUTHERLAND_S)
        return AtmosphereState(T, P, rho, a, mu, mu / rho)


def standard_temperature(z):
    """USSA 1976 temperature at geometric altitude z (troposphere is enough for surface use)."""
    return Atmosphere().properties(z).T


class WindProfile:
    """Mean horizontal wind.

    With a table of (height above ground, speed, direction-from): wind vectors are interpolated
    linearly in height, held constant above the top entry, and below the lowest entry follow the
    power law V(h) = V_low (h / h_low)^p toward the ground.
    Without a table: V(h) = V_ref (h / h_ref)^p from a single reference wind.
    """

    def __init__(self, speed, reference_height, from_deg, exponent, table=()):
        self.exponent = exponent
        if not table:
            table = ((reference_height, speed, from_deg),)
        rows = sorted(table)
        self.h = np.array([r[0] for r in rows])
        vec = []
        for _, spd, frm in rows:
            to = math.radians(frm + 180.0)
            vec.append((spd * math.cos(to), spd * math.sin(to)))
        self.vec = np.array(vec)
        self.heading_to = math.atan2(self.vec[0, 1], self.vec[0, 0])

    def ned(self, h_agl):
        h = max(h_agl, 0.1)
        if h <= self.h[0]:
            n, e = self.vec[0] * (h / self.h[0]) ** self.exponent
        else:
            n = np.interp(h, self.h, self.vec[:, 0])
            e = np.interp(h, self.h, self.vec[:, 1])
        return np.array([n, e, 0.0])

    def speed_at(self, h_agl):
        w = self.ned(h_agl)
        return math.hypot(w[0], w[1])


class DrydenTurbulence:
    """MIL-F-8785C Dryden turbulence, exact discretisation of the shaping filters.

    Longitudinal u: first-order (Ornstein-Uhlenbeck) process.
    Lateral v and vertical w: second-order filters H(s) ~ (1 + sqrt(3) L s / V) / (1 + L s / V)^2.
    Output is held constant over each integration step (zero-order hold).
    Components are generated in the mean-wind frame (u along the wind, v to its right, w down)
    and returned in NED.
    """

    FT = 0.3048

    def __init__(self, wind: WindProfile, seed=1):
        self.wind = wind
        self.rng = np.random.default_rng(seed)
        self.u = self.rng.standard_normal()        # unit-variance states
        self.xv = np.zeros(2)
        self.xw = np.zeros(2)
        self.value_ned = np.zeros(3)

    def _scales(self, h_agl):
        """Turbulence scale lengths (m) and intensities (m/s)."""
        W20 = self.wind.speed_at(20 * self.FT)
        sigma_w = 0.1 * W20

        def low(h_ft):
            Lu = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2
            su = sigma_w / (0.177 + 0.000823 * h_ft) ** 0.4
            return Lu * self.FT, Lu * self.FT, h_ft * self.FT, su, su, sigma_w

        h_ft = max(h_agl / self.FT, 10.0)
        if h_ft <= 1000.0:
            return low(h_ft)
        lo = low(1000.0)
        hi = (1750 * self.FT,) * 3 + (lo[3],) * 3
        f = min((h_ft - 1000.0) / 1000.0, 1.0)
        return tuple(a + f * (b - a) for a, b in zip(lo, hi))

    def _second_order(self, x, a, dt):
        e = math.exp(-a * dt)
        Phi = e * np.array([[1 + a * dt, dt], [-a * a * dt, 1 - a * dt]])
        P_inf = np.diag([1 / (4 * a ** 3), 1 / (4 * a)])
        Q = P_inf - Phi @ P_inf @ Phi.T
        w, V = np.linalg.eigh(0.5 * (Q + Q.T))
        S = V @ np.diag(np.sqrt(np.maximum(w, 0.0)))
        x = Phi @ x + S @ self.rng.standard_normal(2)
        y_unit = (a * a * x[0] + math.sqrt(3) * a * x[1]) / math.sqrt(a)  # unit variance output
        return x, y_unit

    def step(self, h_agl, airspeed, dt):
        V = max(airspeed, 1.0)
        Lu, Lv, Lw, su, sv, sw = self._scales(h_agl)
        e = math.exp(-V * dt / Lu)
        self.u = e * self.u + math.sqrt(1 - e * e) * self.rng.standard_normal()
        self.xv, yv = self._second_order(self.xv, V / Lv, dt)
        self.xw, yw = self._second_order(self.xw, V / Lw, dt)
        u, v, w = su * self.u, sv * yv, sw * yw
        psi = self.wind.heading_to
        self.value_ned = np.array([u * math.cos(psi) - v * math.sin(psi),
                                   u * math.sin(psi) + v * math.cos(psi),
                                   w])
        return self.value_ned
