"""Exhaust and ejection smoke as Lagrangian puffs, computed from a simulated flight.

Each puff is the smoke emitted during 1 / puff_rate seconds, released at the nozzle exit with the
exhaust velocity relative to the rocket (the effective exhaust velocity I_total / m_prop along -axis).

Jet phase (turbulent puff): the puff entrains air and grows linearly with the distance s it travels
through the air, r = r0 + a s. Momentum relative to the air is conserved apart from buoyancy,
  d(M u_rel)/dt = B z_hat,   M = m0 + rho (4/3) pi (r^3 - r0^3),
so the exhaust stops within a few metres. Buoyancy flux B = m0 g (T_exhaust - T_air) / T_air is
conserved as the hot gas mixes (it is diluted, not lost).
Drift: dx/dt = wind(z) + u_rel, with the measured mean wind profile.
Spreading after the jet phase: Richardson relative dispersion sigma^2 = C eps t^3, with the
surface-layer dissipation rate eps = u*^3 / (kappa z) and friction velocity u* = kappa U10 / ln(10 / z0).
Opacity through the puff centre: optical depth = k_ext f_p m0 / (pi r^2), alpha = 1 - exp(-depth).

At liftoff the jet strikes the blast deflector and spreads radially along the ground.
The ejection charge adds a black-powder puff at the booster's open front.
"""
import math

import numpy as np

from .frames import quat_to_dcm, NED_TO_ENU
from .recovery import T_BP
from .state import POS, VEL, QUAT, TAU

KAPPA = 0.41
SAMPLE_AGES = [0.0, 0.03, 0.08, 0.15, 0.3, 0.5] + [0.5 * k for k in range(2, 361)]


def _wind_enu(model, z):
    return NED_TO_ENU @ model.wind.ned(z)


def _puff(model, cfg, x0, v0, m0, T_hot, r0, depth_scale, t0, color):
    """Integrate one puff. x0, v0 are ENU position and ground velocity at release (z above ground)."""
    sc = cfg.smoke
    atm = model.atmosphere.properties(cfg.site.ground_altitude_m + max(x0[2], 0.0))
    rho, Ta = atm.rho, atm.T
    u10 = model.wind.speed_at(10.0)
    u_star = KAPPA * max(u10, 0.3) / math.log(10.0 / sc.roughness_length)
    B = m0 * 9.80665 * max(T_hot - Ta, 0.0) / Ta
    x = np.array(x0, float)
    P = m0 * (np.asarray(v0, float) - _wind_enu(model, x[2]))      # momentum relative to the air
    r, M, age = r0, m0, 0.0
    samples, k = [], 0
    while k < len(SAMPLE_AGES) and SAMPLE_AGES[k] <= sc.max_age:
        if age >= SAMPLE_AGES[k] - 1e-12:      # record a sample
            eps = u_star ** 3 / (KAPPA * max(x[2], 1.0))
            r_eff = math.sqrt(r * r + sc.richardson_constant * eps * age ** 3)
            alpha = 1.0 - math.exp(-depth_scale / (math.pi * r_eff * r_eff))
            samples.append([round(age, 3), round(float(x[0]), 2), round(float(x[1]), 2),
                            round(float(x[2]), 2), round(r_eff, 3), round(alpha, 4)])
            k += 1
            if alpha < 0.005 and age > 1.0:
                break
            continue
        u_rel = P / M
        speed = math.sqrt(u_rel @ u_rel)
        # resolve the fast jet phase (5 % radius growth per step), never step past the next sample
        dt = min(0.05 * r / max(sc.spread_rate * speed, 1e-9), 0.25, SAMPLE_AGES[k] - age)
        r += sc.spread_rate * speed * dt
        M = m0 + rho * 4.0 / 3.0 * math.pi * (r ** 3 - r0 ** 3)
        P[2] += B * dt
        x += (_wind_enu(model, max(x[2], 0.1)) + P / M) * dt
        if x[2] < 0.5 * r:                      # the ground stops the puff from sinking into it
            x[2] = 0.5 * r
            P[2] = max(P[2], 0.0)
        age += dt
    return {"t0": round(t0, 3), "c": color, "s": samples}


def compute_smoke(model, rec, events, t_from=0.0):
    """Puffs emitted at or after t_from, from flight records (as produced by Simulator)."""
    cfg = model.cfg
    if not cfg.smoke.enabled:
        return []
    sc, mot = cfg.smoke, model.motor
    T = NED_TO_ENU @ model.C_ne0
    ts = np.asarray(rec["t"])
    states = np.asarray(rec["state"])
    r_exit = math.sqrt(mot.exit_area / math.pi)
    deflector_half = cfg.rail.deflector_size / 2
    puffs = []

    def state_at(t):
        i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        f = (t - ts[i]) / (ts[i + 1] - ts[i])
        y = (1 - f) * states[i] + f * states[i + 1]
        y[QUAT] /= np.linalg.norm(y[QUAT])
        return y

    t0 = max(t_from, mot.ignition_time)
    t1 = min(events.get("burnout", ts[-1]), ts[-1])
    sep_t = events.get("ejection", np.inf)
    dt_e = 1.0 / sc.puff_rate_hz
    for t in np.arange(t0, t1, dt_e):
        y = state_at(t)
        ms = mot.evaluate(t, y[TAU], model.atmosphere.properties(cfg.site.ground_altitude_m).P)
        if ms.mdot <= 0.0:
            continue
        C = quat_to_dcm(y[QUAT])
        _, cg, _ = model.mass.properties(y[TAU], t >= sep_t)
        nozzle = T @ (y[POS] + C @ (model.exit_point - cg) - model.site_ecef)
        axis = T @ C[:, 0]
        v_ground = T @ y[VEL] - mot.v_exhaust * axis
        m0 = ms.mdot * dt_e
        depth = sc.mass_extinction * sc.particulate_fraction * m0
        if nozzle[2] < 1.2 and axis[2] > 0.7:      # jet striking the blast deflector: radial ground spread
            centre = np.array([nozzle[0], nozzle[1], 0.05])
            for j in range(6):
                a = 2 * math.pi * j / 6 + 0.4 * t
                d = np.array([math.cos(a), math.sin(a), 0.0])
                puffs.append(_puff(model, cfg, centre + deflector_half * d, 0.5 * mot.v_exhaust * d, m0 / 6,
                                   cfg.motor.exhaust_temperature_K, deflector_half, depth / 6, float(t), "e"))
        else:
            puffs.append(_puff(model, cfg, nozzle, v_ground, m0, cfg.motor.exhaust_temperature_K,
                               r_exit, depth, float(t), "e"))

    if "ejection" in events and events["ejection"] >= t_from and "separation_failed" not in events:
        te = events["ejection"]
        y = state_at(te)
        C = quat_to_dcm(y[QUAT])
        _, cg, _ = model.mass.properties(y[TAU], False)
        front = T @ (y[POS] + C @ (model.recovery.anchor - cg) - model.site_ecef)
        m0 = cfg.recovery.ejection_charge_g * 1e-3 * sc.black_powder_smoke_fraction
        v = T @ y[VEL] + 15.0 * (T @ C[:, 0])
        puffs.append(_puff(model, cfg, front, v, m0, T_BP, model.geom.R, sc.mass_extinction * m0, float(te), "b"))
    return puffs
