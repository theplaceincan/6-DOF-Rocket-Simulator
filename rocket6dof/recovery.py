"""Recovery system: ejection, nose-cone separation, shock cord, suspension lines, canopy inflation.

Bodies after ejection (all in ECEF, with the same rotating-frame terms as the booster):
  booster   the 6-DOF rocket without its nose cone and parachute; aerodynamics of an open-fronted tube
  nose      3-DOF point mass with tumbling drag
  canopy    3-DOF point mass with drag area CdS(t) and apparent (added) air mass m_a(t)

Ejection (impulsive; lasts milliseconds, so external forces are neglected while it acts):
  black-powder gas in the parachute bay, ideal gas p0 = eta m R T / V0 with R = 119.2 J/(kg K),
  T = 1837 K for FFFF powder, expanding adiabatically (gamma = 1.25) as the nose shoulder slides out.
  Work W = integral over the shoulder length of ((p - p_amb) A - F_friction) dx.
  Separation speed from the two-body energy balance: v_rel = sqrt(2 W / mu), mu = m_f m_b / (m_f + m_b).
  Momentum is conserved exactly. If W <= 0 the nose stays on (failed separation).

Connections (tension only, elastic with viscous damping, slack carries no load):
  shock cord      booster front opening <-> nose cone, T = k (l - L) + c dl/dt, k = EA / L
  suspension lines nose cone <-> canopy
  Either one breaks (and carries no load from then on) if its tension exceeds its breaking strength.

Canopy inflation (Knacke): line stretch when the lines first come taut; filling time
  t_fill = n D0 / V_ls. Drag area grows as CdS = CdS_packed + (Cd S0 - CdS_packed) f^2, f = (t - t_ls)/t_fill.
  Apparent mass m_a = k_a rho (pi/6) (f D_p)^3, and the canopy obeys
  (m_c + m_a) a = m_c g + F_drag + F_lines - (dm_a/dt) v_rel     (momentum of the entrained air).
"""
import math
import numpy as np

from .earth import OMEGA_E, gravitation_ecef
from .frames import cross3, ecef_to_geodetic, dcm_ecef_to_ned, quat_to_dcm
from .state import POS, VEL, QUAT, OMEGA, TAU, NOSE_R, NOSE_V, CAN_R, CAN_V

R_BP = 119.2       # J/(kg K), black-powder combustion gas (22.16 ft lbf / lbm R)
T_BP = 1837.0      # K (3307 R)
GAMMA_BP = 1.25



class RecoverySystem:
    def __init__(self, model):
        self.model = model
        c = model.cfg.recovery
        self.c = c
        g, mass = model.geom, model.mass
        self.m_nose = mass.nose_mass
        self.m_chute = mass.chute_mass
        self.anchor = np.array([-g.L_nose, 0.0, 0.0])          # cord leaves the booster's open front
        self.k_cord = c.shock_cord_EA / c.shock_cord_length
        self.c_cord = 2 * c.shock_cord_damping_ratio * math.sqrt(self.k_cord * self.m_nose)
        self.k_line = c.line_EA / c.suspension_line_length
        self.c_line = 2 * c.line_damping_ratio * math.sqrt(self.k_line * self.m_chute)
        self.S0 = math.pi * c.canopy_diameter ** 2 / 4
        self.cds_full = c.drag_coefficient * self.S0
        self.Dp = c.projected_diameter_ratio * c.canopy_diameter
        self.nose_cds = c.nose_drag_coefficient * g.A_ref

        body = model.cfg.body
        R_in = g.R - body.wall
        self.A_bay = math.pi * R_in ** 2
        bay_fwd = g.L_nose + model.cfg.nose.shoulder_length
        bay_aft = g.L - model.cfg.internals.motor_tube_length
        self.V_bay = self.A_bay * max(bay_aft - bay_fwd, 0.01) * c.compartment_free_fraction
        self.shoulder = model.cfg.nose.shoulder_length

        self.separated = False
        self.t_sep = None
        self.t_ls = None
        self.t_fill = None
        self.landed = {"nose": False, "canopy": False}
        self.broken = {"cord": False, "lines": False}

    # ---------- canopy state ----------
    def fill_fraction(self, t):
        if self.t_ls is None:
            return 0.0
        return min((t - self.t_ls) / self.t_fill, 1.0)

    def drag_area(self, t):
        f = self.fill_fraction(t)
        return self.c.packed_drag_area + (self.cds_full - self.c.packed_drag_area) * f * f

    def added_mass(self, t, rho):
        f = self.fill_fraction(t)
        m_a = self.c.added_mass_coefficient * rho * math.pi / 6 * (f * self.Dp) ** 3
        dm_a = 0.0
        if self.t_ls is not None and f < 1.0:
            dm_a = 3 * m_a / max(f, 1e-9) / self.t_fill
        return m_a, dm_a

    # ---------- ejection ----------
    def ejection_work(self, P_amb):
        p0 = P_amb + self.c.charge_efficiency * self.c.ejection_charge_g * 1e-3 * R_BP * T_BP / self.V_bay
        x = np.linspace(0.0, self.shoulder, 201)
        p = p0 * (self.V_bay / (self.V_bay + self.A_bay * x)) ** GAMMA_BP
        F = (p - P_amb) * self.A_bay - self.c.nose_friction_N
        return float(np.trapezoid(F, x)), p0 - P_amb

    def separate(self, t, y):
        """Apply the ejection impulse. Returns (new state, info) or (y, info) with info['failed']."""
        m = self.model
        tau = y[TAU]
        C = quat_to_dcm(y[QUAT] / np.linalg.norm(y[QUAT]))
        P_amb = m.atmosphere.properties(ecef_to_geodetic(y[POS])[2]).P
        W, p_gauge = self.ejection_work(P_amb)
        info = {"work": W, "bay_pressure": p_gauge}
        if W <= 0.0:
            info["failed"] = True
            return y, info

        m_all, cg_all, _ = m.mass.properties(tau, False)
        m_b, cg_b, _ = m.mass.properties(tau, True)
        m_f = self.m_nose + self.m_chute
        mu = m_f * m_b / (m_f + m_b)
        v_rel = math.sqrt(2 * W / mu)
        axis = C[:, 0]
        w_rel = y[OMEGA] - C.T @ OMEGA_E

        def point(p_body):
            rb = p_body - cg_all
            return y[POS] + C @ rb, y[VEL] + C @ cross3(w_rel, rb)

        r_b, v_b = point(cg_b)
        r_n, v_n = point(self.model.mass.nose_cg)
        y = y.copy()
        y[POS] = r_b
        y[VEL] = v_b - (m_f / (m_f + m_b)) * v_rel * axis
        v_front = v_n + (m_b / (m_f + m_b)) * v_rel * axis
        y[NOSE_R] = r_n + self.shoulder * axis
        y[NOSE_V] = v_front
        y[CAN_R] = r_n + (self.shoulder - 0.05) * axis
        y[CAN_V] = v_front
        self.separated = True
        self.t_sep = t
        info.update({"separation_speed": v_rel})
        return y, info

    # ---------- forces ----------
    @staticmethod
    def _tension(pa, va, pb, vb, L, k, c):
        d = pb - pa
        l = math.sqrt(d @ d)
        if l <= L or l == 0.0:
            return 0.0, np.zeros(3)
        u = d / l
        T = k * (l - L) + c * ((vb - va) @ u)
        return max(T, 0.0), u

    def _air_velocity(self, r):
        m = self.model
        lat, lon, h = ecef_to_geodetic(r)
        h_agl = h - m.cfg.site.ground_altitude_m
        wind = dcm_ecef_to_ned(lat, lon).T @ (m.wind.ned(h_agl) + m.turb_ned)
        return wind, m.atmosphere.properties(h).rho

    def loads(self, t, y, C, cg, w_rel, cg_dot):
        """Returns (cord force on the booster ECEF, its moment about the CG in body axes,
        state derivative of nose and canopy (12), cord tension, line tension)."""
        r_n, v_n, r_c, v_c = y[NOSE_R], y[NOSE_V], y[CAN_R], y[CAN_V]
        ra = self.anchor - cg
        p_a = y[POS] + C @ ra
        v_a = y[VEL] + C @ (cross3(w_rel, ra) - cg_dot)
        T_cord, u_cord = self._tension(p_a, v_a, r_n, v_n, self.c.shock_cord_length, self.k_cord, self.c_cord)
        T_line, u_line = self._tension(r_n, v_n, r_c, v_c, self.c.suspension_line_length, self.k_line, self.c_line)
        if self.broken["cord"]:
            T_cord = 0.0
        if self.broken["lines"]:
            T_line = 0.0
        F_on_booster = T_cord * u_cord
        M_b = cross3(ra, C.T @ F_on_booster)

        def frame(r, v):
            return gravitation_ecef(r) - 2.0 * cross3(OMEGA_E, v) - cross3(OMEGA_E, cross3(OMEGA_E, r))

        dy = np.zeros(12)
        if not self.landed["nose"]:
            wind, rho = self._air_velocity(r_n)
            vr = v_n - wind
            F = -0.5 * rho * self.nose_cds * math.sqrt(vr @ vr) * vr - T_cord * u_cord + T_line * u_line
            dy[0:3], dy[3:6] = v_n, F / self.m_nose + frame(r_n, v_n)
        if not self.landed["canopy"]:
            wind, rho = self._air_velocity(r_c)
            vr = v_c - wind
            m_a, dm_a = self.added_mass(t, rho)
            F = -0.5 * rho * self.drag_area(t) * math.sqrt(vr @ vr) * vr - T_line * u_line - dm_a * vr
            g = gravitation_ecef(r_c) - cross3(OMEGA_E, cross3(OMEGA_E, r_c))
            dy[6:9] = v_c
            dy[9:12] = (self.m_chute * g + F) / (self.m_chute + m_a) - 2.0 * cross3(OMEGA_E, v_c)
        return F_on_booster, M_b, dy, T_cord, T_line
