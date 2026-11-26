"""Derived geometry shared by the mass, aerodynamic, motor and viewer models."""
import math
import numpy as np


class Geometry:
    def __init__(self, cfg):
        nose, body, fins, rail = cfg.nose, cfg.body, cfg.fins, cfg.rail

        # Airframe
        self.R = body.outer_diameter / 2
        self.d = body.outer_diameter
        self.A_ref = math.pi * self.R ** 2
        self.L_nose = nose.length
        self.L_body = body.length
        self.L = nose.length + body.length
        self.ogive_rho = (self.R ** 2 + self.L_nose ** 2) / (2 * self.R)

        s = np.linspace(0.0, self.L_nose, 401)
        y = self.nose_radius(s)
        dy = np.gradient(y, s)
        self.nose_volume = float(np.trapezoid(math.pi * y ** 2, s))
        self.nose_wetted_area = float(np.trapezoid(2 * math.pi * y * np.sqrt(1 + dy ** 2), s))
        self.body_wetted_area = 2 * math.pi * self.R * self.L_body
        self.nose_equivalent_half_angle = math.atan(self.R / self.L_nose)

        # Fins (planform in station / spanwise coordinates)
        self.n_fins = fins.count
        self.cr, self.ct, self.span, self.sweep, self.t_fin = (
            fins.root_chord, fins.tip_chord, fins.span, fins.sweep, fins.thickness)
        self.fin_root_le = self.L - fins.aft_offset - fins.root_chord
        self.fin_planform = [
            (self.fin_root_le, 0.0),
            (self.fin_root_le + self.sweep, self.span),
            (self.fin_root_le + self.sweep + self.ct, self.span),
            (self.fin_root_le + self.cr, 0.0),
        ]
        cr, ct, sp = self.cr, self.ct, self.span
        self.fin_area = 0.5 * (cr + ct) * sp
        self.fin_mac = (2.0 / 3.0) * (cr + ct - cr * ct / (cr + ct))
        self.fin_mac_le = self.sweep * (cr + 2 * ct) / (3 * (cr + ct))   # axial offset from root LE
        self.fin_mac_span = sp / 3 * (cr + 2 * ct) / (cr + ct)            # spanwise position of the MAC
        self.fin_midchord_sweep = math.atan((self.sweep + ct / 2 - cr / 2) / sp)
        self.fin_le_sweep = math.atan(self.sweep / sp)
        self.fin_aspect_ratio = 2 * sp ** 2 / self.fin_area                # two-panel convention
        self.fin_roll = [math.radians(fins.roll_offset_deg) + 2 * math.pi * i / fins.count
                         for i in range(fins.count)]
        self.fin_cant = math.radians(fins.cant_deg)

        # Rail buttons: contact points on the rail-slot line, in body coordinates
        phi_b = math.radians(rail.button_roll_deg)
        rc = self.R + rail.button_standoff
        self.button_roll = phi_b
        self.button_points = [np.array([-s_b, rc * math.cos(phi_b), rc * math.sin(phi_b)])
                              for s_b in rail.button_stations]


    def nose_radius(self, s):
        """Tangent-ogive radius at station s (0 <= s <= L_nose)."""
        s = np.asarray(s, dtype=float)
        rho, R, Ln = self.ogive_rho, self.R, self.L_nose
        return np.sqrt(np.maximum(rho ** 2 - (Ln - s) ** 2, 0.0)) + R - rho

    def outer_radius(self, s):
        s = np.asarray(s, dtype=float)
        return np.where(s < self.L_nose, self.nose_radius(np.minimum(s, self.L_nose)), self.R)
