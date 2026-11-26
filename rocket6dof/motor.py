"""Certified commercial motor flown from its certification thrust curve (RASP .eng file).

State: tau, the time since ignition.
  Thrust      F = F_cert(tau) + (p_test - p_amb) A_e     (nozzle pressure-thrust correction for altitude)
  Mass flow   mdot = m_prop F_cert(tau) / I_total          (propellant consumed in proportion to impulse)
  Propellant  hollow cylinder whose core grows as mass is consumed (internal layout estimated)
"""
import math
from collections import namedtuple
from pathlib import Path

import numpy as np

from .massprops import part_from_cg, point_inertia

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MotorState = namedtuple("MotorState", "thrust mdot exit_area tau_dot burning")


def read_eng(path):
    """Parse a RASP .eng file. Returns header dict and (t, F) arrays starting at (0, 0)."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(";")]
    name, diam, length, delays, m_prop, m_tot, maker = lines[0].split()[:7]
    pts = [tuple(float(v) for v in ln.split()[:2]) for ln in lines[1:]]
    t = np.array([0.0] + [p[0] for p in pts])
    F = np.array([0.0] + [p[1] for p in pts])
    header = {"name": name, "diameter": float(diam) / 1000, "length": float(length) / 1000,
              "delays": delays, "propellant_mass": float(m_prop), "total_mass": float(m_tot),
              "manufacturer": maker}
    return header, t, F


class ThrustCurveMotor:
    def __init__(self, cfg, geom):
        path = Path(cfg.file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.header, self.t, self.F = read_eng(path)
        h = self.header
        self.ignition_time = cfg.ignition_time
        self.m_prop = h["propellant_mass"]
        self.I_cum = np.concatenate([[0.0], np.cumsum(0.5 * (self.F[1:] + self.F[:-1]) * np.diff(self.t))])
        self.I_total = float(self.I_cum[-1])
        self.t_end = float(self.t[-1])
        self.v_exhaust = self.I_total / self.m_prop          # effective exhaust velocity
        self.exit_area = math.pi * cfg.nozzle_exit_diameter ** 2 / 4
        self.p_test = cfg.test_ambient_pressure

        # Layout: motor aft end at the body's aft end (minus recess), propellant centred in the motor
        self.case_aft = geom.L - cfg.recess
        self.case_fwd = self.case_aft - h["length"]
        self.exit_station = self.case_aft
        self.propellant_centroid = np.array([-0.5 * (self.case_fwd + self.case_aft), 0.0, 0.0])
        self.Rg = cfg.grain_outer_diameter / 2
        self.Lg = cfg.grain_length
        self.rho_p = cfg.propellant_density
        if self.m_prop / (self.rho_p * math.pi * self.Lg) > self.Rg ** 2:
            raise ValueError("Propellant does not fit in the assumed grain envelope; enlarge grain_outer_diameter "
                             "or grain_length, or check propellant_density")
        if cfg.grain_outer_diameter >= h["diameter"]:
            raise ValueError("Grain outer diameter must be smaller than the motor diameter")

    def burned_out(self, tau):
        return tau >= self.t_end

    def evaluate(self, t, tau, P_amb):
        burning = t >= self.ignition_time and tau < self.t_end
        F_cert = float(np.interp(tau, self.t, self.F)) if burning else 0.0
        thrust = F_cert + (self.p_test - P_amb) * self.exit_area if F_cert > 0.0 else 0.0
        mdot = self.m_prop * F_cert / self.I_total
        return MotorState(thrust, mdot, self.exit_area, 1.0 if burning else 0.0, burning)

    def propellant_mass(self, tau):
        tau = min(max(tau, 0.0), self.t_end)
        return self.m_prop * (1.0 - float(np.interp(tau, self.t, self.I_cum)) / self.I_total)

    def propellant_terms(self, tau):
        """Mass, first moment and inertia about the body origin of the remaining propellant."""
        m = self.propellant_mass(tau)
        if m <= 0.0:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        R, L = self.Rg, self.Lg
        r2 = max(R * R - m / (self.rho_p * math.pi * L), 0.0)
        I_ax = m * (R * R + r2) / 2
        I_tr = m * (3 * (R * R + r2) + L * L) / 12
        p = self.propellant_centroid
        return m, m * p, np.diag([I_ax, I_tr, I_tr]) + point_inertia(m, p)

    def propellant_rates(self, tau, tau_dot):
        if tau_dot == 0.0 or self.burned_out(tau):
            return 0.0, np.zeros(3), np.zeros((3, 3))
        h = 1e-5
        a, b = max(tau - h, 0.0), min(tau + h, self.t_end)
        ma, fa, Ja = self.propellant_terms(a)
        mb, fb, Jb = self.propellant_terms(b)
        k = tau_dot / (b - a)
        return (mb - ma) * k, (fb - fa) * k, (Jb - Ja) * k

    def hardware_parts(self):
        h = self.header
        m_hw = h["total_mass"] - h["propellant_mass"]
        Ro = h["diameter"] / 2
        Ri = Ro - 0.0015
        L = h["length"]
        I_ax = m_hw * (Ro ** 2 + Ri ** 2) / 2
        I_tr = m_hw * (3 * (Ro ** 2 + Ri ** 2) + L ** 2) / 12
        return [part_from_cg(f"{h['manufacturer']} {h['name']} hardware", m_hw,
                             (-(self.case_fwd + L / 2), 0, 0), np.diag([I_ax, I_tr, I_tr]))]

    def performance(self):
        burning = self.F > 0.05 * self.F.max()
        t_b = float(self.t[burning][-1] - self.t[burning][0])
        idx = max(0, math.ceil(math.log2(self.I_total / 2.5)))
        return {"motor": f"{self.header['manufacturer']} {self.header['name']}",
                "total_impulse": self.I_total, "burn_time": t_b, "average_thrust": self.I_total / t_b,
                "max_thrust": float(self.F.max()), "propellant_mass": self.m_prop,
                "specific_impulse": self.I_total / (self.m_prop * 9.80665),
                "motor_class": chr(ord("A") + idx), "curve": {"t": self.t, "thrust": self.F}}
