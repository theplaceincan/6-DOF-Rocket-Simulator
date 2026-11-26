"""Aerodynamic forces and moments.

Normal forces are evaluated component by component with the local velocity at each component
(v_point = v_cg + omega x r), so pitch, yaw and roll damping come out of the same equations as the
static forces instead of from separate damping derivatives.

  Nose (potential flow)   Barrowman: C_N = C_Na (sin 2a)/2, C_Na = 2 A_base/A_ref, at x = L - V/A_base
  Body (viscous crossflow) Allen-Perkins: dN = 1/2 rho eta C_dc(M_c) 2r |v_c| v_c dx, strip by strip
  Fins (each panel)       subsonic: Diederich lift slope with Prandtl-Glauert beta, plus crossflow term
                          supersonic: Busemann second-order theory; linear blend across 0.9 < M < 1.1
                          body interference K_T(B) = 1 + r/(s + r); CP from the MAC (Barrowman / Niskanen)
  Axial drag              build-up per Niskanen (OpenRocket technical documentation, 2009):
                          skin friction (laminar / turbulent / roughness-limited, compressibility),
                          nose pressure, base (reduced while the motor fires), fin leading and trailing edge,
                          rail-button protuberances
"""
import math
import numpy as np

from .frames import cross3

_GAMMA = 1.4

# Allen-Perkins crossflow drag proportionality factor eta vs body fineness ratio
_ETA_FINENESS = (np.array([2, 4, 8, 12, 20, 40, 1e6]), np.array([0.56, 0.60, 0.66, 0.70, 0.75, 0.82, 1.0]))
# Crossflow drag coefficient of a circular cylinder vs crossflow Mach (subcritical Reynolds number)
_CDC_MACH = (np.array([0.0, 0.4, 0.6, 0.8, 1.0, 1.5, 3.0]), np.array([1.2, 1.2, 1.4, 1.7, 1.9, 1.6, 1.4]))


def stagnation_pressure_ratio(M):
    """q_stagnation / q_dynamic (Niskanen eq. 3.86)."""
    if M <= 1.0:
        return 1 + M * M / 4 + M ** 4 / 40
    return 1.84 - 0.76 / M ** 2 + 0.166 / M ** 4 + 0.035 / M ** 6


def base_drag_coefficient(M):
    return 0.12 + 0.13 * M * M if M < 1.0 else 0.25 / M


def skin_friction_coefficient(Re, M, length, roughness):
    if Re < 1e4:
        return 1.48e-2
    Re_crit = 51.0 * (roughness / length) ** -1.039
    if Re > Re_crit:
        cf = 0.032 * (roughness / length) ** 0.2
        return cf * (1 - 0.1 * M * M) if M < 1 else cf / (1 + 0.18 * M * M)
    cf = 1.0 / (1.50 * math.log(Re) - 5.6) ** 2
    return cf * (1 - 0.1 * M * M) if M < 1 else cf / (1 + 0.15 * M * M) ** 0.58


def fin_leading_edge_drag(M):
    if M < 0.9:
        return (1 - M * M) ** -0.417 - 1
    if M < 1.0:
        return 1 - 1.785 * (M - 0.9)
    return 1.214 - 0.502 / M ** 2 + 0.1095 / M ** 4


class Aerodynamics:
    def __init__(self, cfg, geom):
        self.g = geom
        self.roughness = cfg.body.surface_roughness
        g = geom

        # Body strips for viscous crossflow
        edges = np.concatenate([np.linspace(0, g.L_nose, 31), np.linspace(g.L_nose, g.L, 31)[1:]])
        s_mid = 0.5 * (edges[1:] + edges[:-1])
        r_mid = g.outer_radius(s_mid)
        self.strip_pos = np.column_stack([-s_mid, np.zeros_like(s_mid), np.zeros_like(s_mid)])
        self.strip_area = 2 * r_mid * np.diff(edges)
        self.booster_strips = s_mid >= g.L_nose     # strips left after the nose cone separates
        self.eta = float(np.interp(g.L / g.d, *_ETA_FINENESS))

        # Nose potential lift
        self.nose_cna = 2.0
        self.nose_cp = np.array([-(g.L_nose - g.nose_volume / g.A_ref), 0.0, 0.0])

        # Fins
        self.K_TB = 1 + g.R / (g.span + g.R)
        self.fin_normals = [np.array([0.0, -math.sin(p), math.cos(p)]) for p in g.fin_roll]
        self.fin_radial = [np.array([0.0, math.cos(p), math.sin(p)]) for p in g.fin_roll]
        self.fin_cp_radius = g.R + g.fin_mac_span

        # Drag reference quantities
        fineness = g.L / g.d
        t_over_c = g.t_fin / g.fin_mac
        self.friction_area = ((1 + 1 / (2 * fineness)) * (g.nose_wetted_area + g.body_wetted_area)
                              + (1 + 2 * t_over_c) * 2 * g.fin_area * g.n_fins)
        fins_wet = (1 + 2 * t_over_c) * 2 * g.fin_area * g.n_fins
        self.friction_area_booster = (1 + 1 / (2 * g.L_body / g.d)) * g.body_wetted_area + fins_wet
        self.fin_le_area = g.n_fins * g.t_fin * g.span
        self.fin_te_area = g.n_fins * g.t_fin * g.span
        self.button_area = len(cfg.rail.button_stations) * cfg.rail.button_frontal_area

    # ---------- fins ----------
    def fin_cna_subsonic(self, M):
        g = self.g
        beta = math.sqrt(max(1 - M * M, 1e-6))
        x = beta * g.span ** 2 / (g.fin_area * math.cos(g.fin_midchord_sweep))
        return 2 * math.pi * (g.span ** 2 / g.A_ref) / (1 + math.sqrt(1 + x * x))

    def _fin_cn_sub(self, alpha, M):
        ratio = self.g.fin_area / self.g.A_ref
        return (self.K_TB * self.fin_cna_subsonic(M) * math.sin(alpha) * math.cos(alpha)
                + 1.2 * ratio * math.sin(alpha) * abs(math.sin(alpha)))

    def _fin_cn_sup(self, alpha, M):
        beta = math.sqrt(M * M - 1)
        g = _GAMMA
        K1 = 2 / beta
        K3 = ((g + 1) * M ** 8 + (2 * g * g - 7 * g - 5) * M ** 6 + 10 * (g + 1) * M ** 4 + 8) / (6 * beta ** 7)
        a = max(min(alpha, 0.6), -0.6)
        ratio = self.g.fin_area / self.g.A_ref
        return (self.K_TB * ratio * (2 * K1 * a + 2 * K3 * a ** 3)
                + 1.2 * ratio * math.sin(alpha) * abs(math.sin(alpha)))

    def fin_cn(self, alpha, M):
        """Normal-force coefficient of one fin panel (reference area A_ref)."""
        if M <= 0.9:
            return self._fin_cn_sub(alpha, M)
        if M >= 1.1:
            return self._fin_cn_sup(alpha, M)
        f = (M - 0.9) / 0.2
        return (1 - f) * self._fin_cn_sub(alpha, 0.9) + f * self._fin_cn_sup(alpha, 1.1)

    def fin_cna_total(self, M):
        """Lift-curve slope of the fin set at zero angle of attack, per radian."""
        if M <= 0.9:
            one = self.K_TB * self.fin_cna_subsonic(M)
        elif M >= 1.1:
            one = self.K_TB * (self.g.fin_area / self.g.A_ref) * 4 / math.sqrt(M * M - 1)
        else:
            f = (M - 0.9) / 0.2
            one = ((1 - f) * self.K_TB * self.fin_cna_subsonic(0.9)
                   + f * self.K_TB * (self.g.fin_area / self.g.A_ref) * 4 / math.sqrt(1.1 ** 2 - 1))
        return self.g.n_fins / 2 * one

    def fin_cp_station(self, M):
        g = self.g
        if M <= 0.5:
            frac = 0.25
        else:
            AR = g.fin_aspect_ratio
            b2 = math.sqrt(3.0)
            frac2 = (AR * b2 - 0.67) / (2 * AR * b2 - 1)
            frac = 0.25 + (frac2 - 0.25) * min((M - 0.5) / 1.5, 1.0) if M < 2 else \
                (AR * math.sqrt(M * M - 1) - 0.67) / (2 * AR * math.sqrt(M * M - 1) - 1)
        return g.fin_root_le + g.fin_mac_le + frac * g.fin_mac

    def static_cp(self, M=0.3):
        """Centre of pressure station (m aft of nose tip) and total C_Na at small angle of attack."""
        cn_f = self.fin_cna_total(M)
        s_f = self.fin_cp_station(M)
        s_n = -self.nose_cp[0]
        cna = self.nose_cna + cn_f
        return (self.nose_cna * s_n + cn_f * s_f) / cna, cna

    # ---------- drag ----------
    def cd0(self, M, Re, thrusting, exit_area, nose_on=True):
        g = self.g
        cf = skin_friction_coefficient(Re, M, g.L if nose_on else g.L_body, self.roughness)
        cd = cf * (self.friction_area if nose_on else self.friction_area_booster) / g.A_ref

        if not nose_on:   # open, flat-faced front of the booster after separation
            cd += 0.8 * stagnation_pressure_ratio(M)
        elif M > 0.8:   # tangent ogive with a smooth joint has no subsonic pressure drag
            phi = math.degrees(g.nose_equivalent_half_angle)
            sup = lambda m: (0.083 + 0.096 / m ** 2) * (phi / 10.0) ** 1.69
            cd += sup(M) if M >= 1.3 else sup(1.3) * (M - 0.8) / 0.5

        base_area = g.A_ref - (exit_area if thrusting else 0.0)
        cd += base_drag_coefficient(M) * max(base_area, 0.0) / g.A_ref

        cd += fin_leading_edge_drag(M) * math.cos(g.fin_le_sweep) ** 2 * self.fin_le_area / g.A_ref
        cd += base_drag_coefficient(M) * self.fin_te_area / g.A_ref
        cd += 1.2 * stagnation_pressure_ratio(M) * self.button_area / g.A_ref
        return cd

    # ---------- forces ----------
    def forces(self, v_b, w_b, cg, rho, a_sound, nu, thrusting, exit_area, fin_deflections=None, nose_on=True):
        """Aerodynamic force and moment about the CG, body axes.

        v_b  velocity of the body point at the CG relative to the air (body axes)
        w_b  body angular velocity relative to the air mass (Earth-fixed), body axes
        fin_deflections  per-fin deflection (rad) of all-moving fins, positive like cant
        """
        g = self.g
        V = math.sqrt(v_b @ v_b)
        F = np.zeros(3)
        Mo = np.zeros(3)
        fin_normal = [0.0] * g.n_fins
        if V < 1e-6:
            return F, Mo, {"alpha": 0.0, "mach": 0.0, "q": 0.0, "cd0": 0.0, "reynolds": 0.0,
                           "fin_normal": fin_normal, "fin_cp_station": self.fin_cp_station(0.0)}
        M = V / a_sound
        Re = V * g.L / nu
        cd = self.cd0(M, Re, thrusting, exit_area, nose_on)
        F[0] -= 0.5 * rho * g.A_ref * cd * V * v_b[0]

        # Body crossflow strips
        keep = slice(None) if nose_on else self.booster_strips
        rel = self.strip_pos[keep] - cg
        rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]
        vy = v_b[1] + w_b[2] * rx - w_b[0] * rz        # crossflow components of v_b + w x rel
        vz = v_b[2] + w_b[0] * ry - w_b[1] * rx
        vc_mag = np.sqrt(vy * vy + vz * vz)
        cdc = np.interp(vc_mag / a_sound, *_CDC_MACH)
        k = -0.5 * rho * self.eta * cdc * self.strip_area[keep] * vc_mag
        fy, fz = k * vy, k * vz
        F[1] += fy.sum()
        F[2] += fz.sum()
        Mo += np.array([(ry * fz - rz * fy).sum(), -(rx * fz).sum(), (rx * fy).sum()])

        # Nose potential normal force
        rel_n = self.nose_cp - cg
        v_n = v_b + cross3(w_b, rel_n)
        vcn = np.array([0.0, v_n[1], v_n[2]])
        vcn_mag = math.sqrt(vcn @ vcn)
        if nose_on and vcn_mag > 1e-9:
            alpha_n = math.atan2(vcn_mag, v_n[0])
            q_n = 0.5 * rho * (v_n @ v_n)
            fn = -q_n * g.A_ref * self.nose_cna * 0.5 * math.sin(2 * alpha_n) * vcn / vcn_mag
            F += fn
            Mo += cross3(rel_n, fn)

        # Fin panels
        s_cp = self.fin_cp_station(M)
        for i, (n_hat, e_hat) in enumerate(zip(self.fin_normals, self.fin_radial)):
            p = np.array([-s_cp, 0.0, 0.0]) + self.fin_cp_radius * e_hat
            rel_f = p - cg
            v_i = v_b + cross3(w_b, rel_f)
            vn = v_i @ n_hat
            vx = v_i[0]
            Vp2 = vx * vx + vn * vn
            if Vp2 < 1e-12:
                continue
            delta = fin_deflections[i] if fin_deflections is not None else 0.0
            alpha_f = math.atan2(vn, vx) - g.fin_cant - delta
            alpha_f = (alpha_f + math.pi) % (2 * math.pi) - math.pi
            cn = self.fin_cn(alpha_f, math.sqrt(Vp2) / a_sound)
            N_i = -0.5 * rho * Vp2 * g.A_ref * cn
            fin_normal[i] = N_i
            ff = N_i * n_hat
            F += ff
            Mo += cross3(rel_f, ff)

        alpha = math.atan2(math.hypot(v_b[1], v_b[2]), v_b[0])
        return F, Mo, {"alpha": alpha, "mach": M, "q": 0.5 * rho * V * V, "cd0": cd, "reynolds": Re,
                       "fin_normal": fin_normal, "fin_cp_station": s_cp}
