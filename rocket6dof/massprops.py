"""Mass, centre of gravity and full inertia tensor, varying with propellant burn.

Every part is reduced to (mass, first moment, inertia tensor about the body origin = nose tip).
For the CG-referenced tensor: I_cg = J_O - m (|c|^2 E - c c^T).
Time derivatives of the CG and inertia (needed for the variable-mass equations) come from the
analytic chain rule, with d/dx of the grain terms by central differences.
"""
import math
import numpy as np

E3 = np.eye(3)


def point_inertia(m, p):
    return m * (p @ p * E3 - np.outer(p, p))


class Part:
    def __init__(self, name, mass, first, J):
        self.name, self.mass, self.first, self.J = name, mass, np.asarray(first, float), np.asarray(J, float)

    @property
    def cg(self):
        return self.first / self.mass


def part_from_cg(name, m, cg, I_c):
    cg = np.asarray(cg, float)
    return Part(name, m, m * cg, np.asarray(I_c, float) + point_inertia(m, cg))


def tube_part(name, r_out, r_in, length, s_fwd, density):
    m = density * math.pi * (r_out ** 2 - r_in ** 2) * length
    I_ax = m * (r_out ** 2 + r_in ** 2) / 2
    I_tr = m * (3 * (r_out ** 2 + r_in ** 2) + length ** 2) / 12
    return part_from_cg(name, m, (-(s_fwd + length / 2), 0, 0), np.diag([I_ax, I_tr, I_tr]))


def cylinder_part(name, m, radius, length, s_cg):
    I_ax = m * radius ** 2 / 2
    I_tr = m * (3 * radius ** 2 + length ** 2) / 12
    return part_from_cg(name, m, (-s_cg, 0, 0), np.diag([I_ax, I_tr, I_tr]))


def ogive_shell_part(geom, wall, density, n=400):
    s = np.linspace(0, geom.L_nose, n + 1)
    y = geom.nose_radius(s)
    sm, ym = 0.5 * (s[1:] + s[:-1]), 0.5 * (y[1:] + y[:-1])
    dl = np.hypot(np.diff(s), np.diff(y))
    dm = density * wall * 2 * math.pi * ym * dl
    m = dm.sum()
    first = np.array([-(dm * sm).sum(), 0, 0])
    Jxx = (dm * ym ** 2).sum()
    Jyy = (dm * (ym ** 2 / 2 + sm ** 2)).sum()
    return Part("nose cone", m, first, np.diag([Jxx, Jyy, Jyy]))


def fin_parts(geom, cfg, n=40):
    parts = []
    rho_t = cfg.fins.density * cfg.fins.thickness
    for i, phi in enumerate(geom.fin_roll):
        m_tot, first, J = 0.0, np.zeros(3), np.zeros((3, 3))
        for j in range(n):
            yf = (j + 0.5) / n * geom.span
            chord = geom.cr + (geom.ct - geom.cr) * yf / geom.span
            s_le = geom.fin_root_le + geom.sweep * yf / geom.span
            dA = chord / n * geom.span / n
            for k in range(n):
                s = s_le + (k + 0.5) / n * chord
                p = np.array([-s, (geom.R + yf) * math.cos(phi), (geom.R + yf) * math.sin(phi)])
                dm = rho_t * dA
                m_tot += dm
                first += dm * p
                J += point_inertia(dm, p)
        parts.append(Part(f"fin {i + 1}", m_tot, first, J))
    return parts


class MassModel:
    def __init__(self, cfg, geom, motor):
        self.geom, self.motor = geom, motor
        R = geom.R
        nose, body, it = cfg.nose, cfg.body, cfg.internals
        parts = [
            ogive_shell_part(geom, nose.wall, nose.density),
            tube_part("nose shoulder", R - body.wall, R - body.wall - nose.wall, nose.shoulder_length,
                      geom.L_nose, nose.density),
            tube_part("body tube", R, R - body.wall, body.length, geom.L_nose, body.density),
            tube_part("motor tube", it.motor_tube_outer_diameter / 2,
                      it.motor_tube_outer_diameter / 2 - it.motor_tube_wall, it.motor_tube_length,
                      geom.L - it.motor_tube_length, it.motor_tube_density),
        ]
        parts += motor.hardware_parts()
        for k, s_ring in enumerate(it.centering_ring_stations):
            parts.append(tube_part(f"centering ring {k + 1}", R - body.wall, it.motor_tube_outer_diameter / 2,
                                   it.centering_ring_thickness, s_ring - it.centering_ring_thickness / 2,
                                   it.centering_ring_density))
        parts += fin_parts(geom, cfg)
        for item in it.items:
            if item.length > 0:
                parts.append(cylinder_part(item.name, item.mass, item.radius, item.length, item.station))
            else:
                parts.append(part_from_cg(item.name, item.mass, (-item.station, 0, 0), np.zeros((3, 3))))
        if nose.ballast_mass > 0:
            parts.append(part_from_cg("nose ballast", nose.ballast_mass, (-nose.ballast_station, 0, 0),
                                      np.zeros((3, 3))))
        for k, p in enumerate(geom.button_points):
            pb = np.array([p[0], (R + 0.004) * math.cos(geom.button_roll), (R + 0.004) * math.sin(geom.button_roll)])
            parts.append(part_from_cg(f"rail button {k + 1}", cfg.rail.button_mass, pb, np.zeros((3, 3))))

        self.parts = parts
        self.dry_mass = sum(p.mass for p in parts)
        self.dry_first = sum(p.first for p in parts)
        self.dry_J = sum(p.J for p in parts)

        # Parts that leave the booster at ejection: the nose cone group and the packed parachute
        nose_names = {"nose cone", "nose shoulder", "nose ballast"}
        nose_parts = [p for p in parts if p.name in nose_names]
        chute_parts = [p for p in parts if p.name == "parachute"]
        if not chute_parts:
            raise ValueError("Internals must include an item named 'parachute'")
        self.nose_mass = sum(p.mass for p in nose_parts)
        self.nose_cg = sum(p.first for p in nose_parts) / self.nose_mass
        self.chute_mass = sum(p.mass for p in chute_parts)
        self.chute_cg = sum(p.first for p in chute_parts) / self.chute_mass
        gone = nose_parts + chute_parts
        self.booster_dry = (self.dry_mass - sum(p.mass for p in gone),
                            self.dry_first - sum(p.first for p in gone),
                            self.dry_J - sum(p.J for p in gone))
        self._cache = {}

    def _totals(self, tau, separated=False):
        mp, fp, Jp = self.motor.propellant_terms(tau)
        if separated:
            m0, f0, J0 = self.booster_dry
            return m0 + mp, f0 + fp, J0 + Jp
        return self.dry_mass + mp, self.dry_first + fp, self.dry_J + Jp

    def properties(self, tau, separated=False):
        """Mass, CG (body coordinates) and inertia tensor about the CG at motor burn time tau.
        separated=True gives the booster alone after the nose cone and parachute have left."""
        key = separated if self.motor.burned_out(tau) else None    # constant after burnout
        if key is not None and key in self._cache:
            return self._cache[key]
        m, first, J = self._totals(tau, separated)
        c = first / m
        out = (m, c, J - point_inertia(m, c))
        if key is not None:
            self._cache[key] = out
        return out

    def rates(self, tau, tau_dot, separated=False):
        """d/dt of mass, CG position and CG inertia tensor."""
        mdot, fdot, Jdot = self.motor.propellant_rates(tau, tau_dot)
        if mdot == 0.0:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        m, first, J = self._totals(tau, separated)
        c = first / m
        cdot = (fdot - mdot * c) / m
        Idot = (Jdot - mdot * (c @ c * E3 - np.outer(c, c))
                - m * (2 * (c @ cdot) * E3 - np.outer(cdot, c) - np.outer(c, cdot)))
        return mdot, cdot, Idot
