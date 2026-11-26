"""Design checks: static stability, motor performance, fin flutter."""
import math
import numpy as np


def static_margin(model, mach=0.3):
    """Static margin in calibers at ignition and at burnout."""
    s_cp, cna = model.aero.static_cp(mach)
    out = {"cp_station": s_cp, "cna": cna}
    for label, tau in (("ignition", 0.0), ("burnout", model.motor.t_end)):
        _, cg, _ = model.mass.properties(tau)
        out[f"cg_station_{label}"] = -cg[0]
        out[f"margin_{label}"] = (s_cp + cg[0]) / model.geom.d
    return out


def motor_performance(model):
    return model.motor.performance()


def fin_flutter_speed(model, altitude_msl):
    """Flutter boundary from NACA TN 4197 (Martin, 1958), as simplified for trapezoidal fins."""
    g = model.geom
    atm = model.atmosphere.properties(altitude_msl)
    AR = g.span ** 2 / g.fin_area
    lam = g.ct / g.cr
    t_c = g.t_fin / g.cr
    G = model.cfg.fins.shear_modulus
    denom = 1.337 * AR ** 3 * atm.P * (lam + 1) / (2 * (AR + 2) * t_c ** 3)
    return atm.a * math.sqrt(G / denom)
