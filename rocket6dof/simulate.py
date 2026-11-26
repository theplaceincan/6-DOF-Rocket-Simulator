"""Time integration with flight-event detection.

Fourth-order Runge-Kutta with a step size chosen by flight phase (stiff rail contact, motor burn,
coast, parachute opening, descent). Turbulence is sampled once per step and held over it. The flight
computer runs at its own rate; its commands are held between cycles (zero-order hold). Events are
located by linear interpolation between steps.

Steady descent uses multirate integration: the light nose cone and canopy, held by stiff elastic
lines, take small sub-steps while the heavy booster takes one larger step. Within a macro step the
small bodies see the booster interpolated between its state now and an Euler prediction of its next
state; the booster then steps with the small bodies interpolated between their old and new states.

The Simulator can be snapshotted (copy.deepcopy) and resumed, and the ejection charge can be fired
at any time with force_ejection(t), which is how the viewer's deploy button works.
"""
import copy
import math

import numpy as np

from .control import FlightComputer
from .earth import OMEGA_E
from .frames import quat_to_dcm, ecef_to_geodetic, dcm_ecef_to_ned
from .dynamics import FlightModel
from .state import POS, VEL, QUAT, OMEGA, TAU, BOOSTER, RECOVERY, NOSE_R, NOSE_V, CAN_R, CAN_V

STOP_EVENTS = ("rail_exit", "burnout", "apogee", "landing")
RECORD_KEYS = ("altitude", "speed", "mach", "alpha", "thrust", "mdot", "mass", "q",
               "v_ned", "cord_tension", "line_tension", "canopy_fill")


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    y_new = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    y_new[QUAT] /= np.linalg.norm(y_new[QUAT])
    return y_new


def _interp(t0, t1, f0, f1):
    return t0 + (t1 - t0) * (-f0) / (f1 - f0) if f1 != f0 else t1


class Simulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = FlightModel(cfg)
        self.y = self.model.initial_state()
        self.t = 0.0
        self.events = {}
        self.rec = {k: [] for k in ("t", "state", *RECORD_KEYS, "control_on", "separated")}
        self.next_record = 0.0
        self.s0 = min(self.model.button_axial_positions(self.y))   # aft button resting on the rail stop
        self.vz_prev = 0.0
        self.h_prev = 0.0
        self.max_cord = self.max_line = 0.0
        # The flight computer always runs (it fires the ejection charge); fins move only if enabled
        self.computer = FlightComputer(self.model)
        _, pad = self.model.derivatives(cfg.motor.ignition_time - 10.0, self.y, info=True)
        self.computer.calibrate_on_pad(self.y[OMEGA], pad["accel_body"], pad["pressure"])
        self.next_ctrl = 0.0
        self.t_eject = None
        self.forced = False

    # ---------- control of the run ----------
    def snapshot(self):
        """Deep copy of the full simulation state, without the recorded history (to keep it small)."""
        rec, self.rec = self.rec, {k: [] for k in self.rec}
        try:
            return copy.deepcopy(self)
        finally:
            self.rec = rec

    def force_ejection(self, t):
        """Fire the ejection charge at time t (overrides the altimeter / motor-delay trigger)."""
        self.t_eject = t
        self.forced = True

    @property
    def done(self):
        return "landing" in self.events or self.t >= self.cfg.sim.max_time

    def run(self, stop_at="landing", until=None, callback=None):
        while not self.done and stop_at not in self.events and (until is None or self.t < until):
            self.step()
            if callback is not None:
                callback(self)
        return self

    # ---------- one step ----------
    def _dt(self):
        sim, m, rs = self.cfg.sim, self.model, self.model.recovery
        if m.rail_active:
            return sim.dt_rail, False
        if "burnout" not in self.events:
            return sim.dt_powered, False
        if rs.separated and (rs.t_ls is None or self.t < rs.t_ls + rs.t_fill + 1.0):
            return sim.dt_deploy, False
        if rs.separated:
            return sim.dt_descent_booster, True
        return sim.dt_coast, False

    def _record(self):
        m, rs, rec, computer = self.model, self.model.recovery, self.rec, self.computer
        _, info = m.derivatives(self.t, self.y, info=True)
        rec["t"].append(self.t)
        rec["state"].append(self.y.copy())
        for k in RECORD_KEYS:
            rec[k].append(info[k])
        rec["control_on"].append(bool(computer.enabled))
        rec["separated"].append(rs.separated)
        self.next_record += self.cfg.sim.record_interval

    def _multirate_step(self, dt):
        m, rs = self.model, self.model.recovery
        t, y = self.t, self.y
        sub = max(1, math.ceil(dt / self.cfg.sim.dt_descent - 1e-9))
        h = dt / sub
        dy0 = m.derivatives(t, y)
        yb0 = y[BOOSTER].copy()
        yb1 = yb0 + dt * dy0[BOOSTER]
        yb1[QUAT] /= np.linalg.norm(yb1[QUAT])
        _, cg, _ = m.mass.properties(y[TAU], True)
        work = y.copy()
        peak = [0.0, 0.0]

        def f_small(tt, yp):
            f = (tt - t) / dt
            yb = (1 - f) * yb0 + f * yb1
            yb[QUAT] /= np.linalg.norm(yb[QUAT])
            work[BOOSTER] = yb
            work[RECOVERY] = yp
            C = quat_to_dcm(yb[QUAT])
            _, _, d, Tc, Tl = rs.loads(tt, work, C, cg, yb[OMEGA] - C.T @ OMEGA_E, np.zeros(3))
            peak[0], peak[1] = max(peak[0], Tc), max(peak[1], Tl)
            return d

        yp0 = y[RECOVERY].copy()

        def small_pass():
            yp = yp0.copy()
            for k in range(sub):
                tk = t + k * h
                k1 = f_small(tk, yp)
                k2 = f_small(tk + h / 2, yp + h / 2 * k1)
                k3 = f_small(tk + h / 2, yp + h / 2 * k2)
                k4 = f_small(tk + h, yp + h * k3)
                yp = yp + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return yp

        yp1 = small_pass()      # small bodies against the Euler-predicted booster

        def f_big(tt, z):
            f = (tt - t) / dt
            zz = z.copy()
            zz[RECOVERY] = (1 - f) * yp0 + f * yp1
            d = m.derivatives(tt, zz)
            d[RECOVERY] = 0.0
            return d

        y_new = rk4_step(f_big, t, y, dt)
        y_new[RECOVERY] = yp1
        return y_new, peak

    def step(self):
        cfg, m, rs, ev, computer = self.cfg, self.model, self.model.recovery, self.events, self.computer
        t, y = self.t, self.y
        dt, multirate = self._dt()

        if t >= self.next_ctrl - 1e-12:
            _, cinfo = m.derivatives(t, y, info=True)
            m.fin_cmd = computer.step(t, y[OMEGA], cinfo["accel_body"], cinfo["pressure"]).copy()
            self.next_ctrl += computer.dt
            if computer.launched and "launch_detected" not in ev:
                ev["launch_detected"] = t
            if computer.enabled and "control_on" not in ev:
                ev["control_on"] = t
            if computer.baro_apogee_time is not None and "apogee_detected" not in ev:
                ev["apogee_detected"] = computer.baro_apogee_time

        # Ejection trigger
        if cfg.recovery.enabled and self.t_eject is None:
            if cfg.recovery.trigger == "altimeter" and computer.baro_apogee_time is not None:
                self.t_eject = computer.baro_apogee_time + cfg.recovery.deploy_delay
            elif cfg.recovery.trigger == "motor_delay" and "burnout" in ev:
                self.t_eject = ev["burnout"] + cfg.recovery.motor_delay
        if (self.t_eject is not None and not rs.separated and "separation_failed" not in ev
                and t >= self.t_eject):
            self.y, sinfo = rs.separate(t, y)
            y = self.y
            ev["ejection"] = t
            ev["ejection_forced"] = 1.0 if self.forced else 0.0
            ev["bay_pressure_kPa"] = sinfo["bay_pressure"] / 1000
            if sinfo.get("failed"):
                ev["separation_failed"] = t
            else:
                ev["separation_speed"] = sinfo["separation_speed"]
            dt, multirate = self._dt()

        if t >= self.next_record - 1e-12:
            self._record()

        m.update_turbulence(y, dt)
        if multirate:
            y_new, peak = self._multirate_step(dt)
        else:
            y_new, peak = rk4_step(m.derivatives, t, y, dt), [0.0, 0.0]
        t_new = t + dt

        if m.rail_active:
            pos = m.button_axial_positions(y_new)
            if "liftoff" not in ev and min(pos) > self.s0 + 1e-3:
                ev["liftoff"] = t_new
            if min(pos) > m.rail_length:
                ev["rail_exit"] = t_new
                ev["rail_exit_speed"] = float(np.linalg.norm(y_new[VEL]))
                m.rail_active = False
        if "burnout" not in ev and m.motor.burned_out(y_new[TAU]):
            ev["burnout"] = t_new

        if rs.separated:
            C_new = quat_to_dcm(y_new[QUAT])
            _, cg_new, _ = m.mass.properties(y_new[TAU], True)
            _, _, _, Tc, Tl = rs.loads(t_new, y_new, C_new, cg_new, y_new[OMEGA] - C_new.T @ OMEGA_E, np.zeros(3))
            Tc, Tl = max(Tc, peak[0]), max(Tl, peak[1])
            if Tc > self.max_cord:
                self.max_cord = Tc
                ev["max_cord_tension_N"], ev["max_cord_tension_t"] = Tc, t_new
            if Tl > self.max_line:
                self.max_line = Tl
                ev["max_line_tension_N"], ev["max_line_tension_t"] = Tl, t_new
            if not rs.broken["cord"] and Tc > cfg.recovery.shock_cord_strength_N:
                rs.broken["cord"] = True
                ev["shock_cord_broke"] = t_new
            if not rs.broken["lines"] and Tl > cfg.recovery.line_strength_N:
                rs.broken["lines"] = True
                ev["lines_broke"] = t_new
            if rs.t_ls is None:
                d = y_new[CAN_R] - y_new[NOSE_R]
                if np.linalg.norm(d) >= cfg.recovery.suspension_line_length:
                    lat, lon, h = ecef_to_geodetic(y_new[CAN_R])
                    wind = dcm_ecef_to_ned(lat, lon).T @ m.wind.ned(h - cfg.site.ground_altitude_m)
                    v_ls = float(np.linalg.norm(y_new[CAN_V] - wind))
                    rs.t_ls = t_new
                    rs.t_fill = cfg.recovery.fill_constant * cfg.recovery.canopy_diameter / max(v_ls, 1.0)
                    ev["line_stretch"] = t_new
                    ev["line_stretch_speed"] = v_ls
                    ev["full_inflation"] = t_new + rs.t_fill
            for name, R_, V_ in (("nose", NOSE_R, NOSE_V), ("canopy", CAN_R, CAN_V)):
                if not rs.landed[name] and m.altitude_agl_point(y_new[R_]) <= 0.0:
                    rs.landed[name] = True
                    ev[f"{name}_landing"] = t_new
                    ev[f"{name}_landing_speed"] = float(np.linalg.norm(y_new[V_]))
                    y_new[V_] = 0.0

        if not m.rail_active:
            vz = m.vertical_velocity(y_new)
            h = m.altitude_agl(y_new)
            if "apogee" not in ev and self.vz_prev > 0.0 >= vz:
                ev["apogee"] = _interp(t, t_new, self.vz_prev, vz)
                ev["apogee_altitude"] = h
            if h <= 0.0 and ("apogee" in ev or rs.separated) and not m.booster_landed:
                ev["booster_landing"] = _interp(t, t_new, self.h_prev, h)
                ev["booster_landing_speed"] = float(np.linalg.norm(y_new[VEL]))
                m.booster_landed = True
                y_new[VEL] = 0.0
                y_new[OMEGA] = quat_to_dcm(y_new[QUAT]).T @ OMEGA_E
            if m.booster_landed and (not rs.separated or all(rs.landed.values()) or rs.broken["cord"]):
                ev["landing"] = t_new
            self.vz_prev, self.h_prev = vz, h

        self.y, self.t = y_new, t_new

    def records(self):
        return {k: np.array(v) for k, v in self.rec.items()}


def simulate(cfg, stop_at="landing"):
    """Run the flight. Returns (records, events, model).

    records: dict of numpy arrays sampled every cfg.sim.record_interval
    events:  dict of event name -> time (s), plus a few summary numbers
    """
    if stop_at not in STOP_EVENTS:
        raise ValueError(f"stop_at must be one of {STOP_EVENTS}")
    sim = Simulator(cfg).run(stop_at)
    sim.model.computer = sim.computer
    return sim.records(), sim.events, sim.model
