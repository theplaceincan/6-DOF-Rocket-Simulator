"""Six-degree-of-freedom equations of motion of a variable-mass rocket over a rotating WGS 84 Earth.

State layout: see state.py.

Translation (ECEF, rotating frame):
  dv/dt = C F_b / m + g(r) - 2 Omega x v - Omega x (Omega x r)
  F_b = aerodynamics + thrust + jet Coriolis force + rail contact + shock cord

Rotation (body axes, about the instantaneous CG; variable-mass form):
  I dw/dt = M - w x (I w) - (dI/dt) w - mdot r_e x (w x r_e)
  The last two terms together give the net jet damping mdot (l_e^2 - k^2) for propellant
  burning near the CG. Jet Coriolis force: F = -2 mdot w x (r_exit - r_propellant).
  M = aerodynamics + thrust moment + rail + shock cord

Kinematics:
  dq/dt = 1/2 q (x) (w - C^T Omega)

Fin servos: d(delta)/dt = clip((delta_cmd - delta) / tau, +-omega_eff), with the torque-speed line
  omega_eff = omega_noload (1 - |H| / H_stall), H = fin normal force x (CP - shaft) the hinge moment,
  and a hard stop at the mechanical deflection limit. delta_cmd comes from the flight computer.

Launch rail: each rail button is a penalty (spring-damper) contact on the rail-slot line while it is
on the rail, with Coulomb friction along the rail, a one-sided stop at the rail bottom, and the button
flanges resisting twist about the rail line. Tip-off (pivoting about the aft button after the forward
one leaves) follows from the same equations.
"""
import math
import numpy as np

from .aero import Aerodynamics
from .atmosphere import Atmosphere, WindProfile, DrydenTurbulence
from .earth import OMEGA_E, gravitation_ecef, gravity_ecef
from .frames import cross3, quat_to_dcm, dcm_to_quat, quat_rate, geodetic_to_ecef, ecef_to_geodetic, dcm_ecef_to_ned
from .geometry import Geometry
from .massprops import MassModel
from .motor import ThrustCurveMotor
from .recovery import RecoverySystem
from .state import POS, VEL, QUAT, OMEGA, TAU, FINS, BOOSTER, RECOVERY, N_STATE


class FlightModel:
    def __init__(self, cfg):
        self.cfg = cfg
        site = cfg.site
        self.geom = Geometry(cfg)
        dT = site.temperature_offset_K
        if site.surface_temperature_C is not None:
            dT = site.surface_temperature_C + 273.15 - Atmosphere().properties(site.ground_altitude_m).T
        p_scale = 1.0
        if site.surface_pressure_hPa > 0:
            p_scale = site.surface_pressure_hPa * 100.0 / Atmosphere(dT).properties(site.ground_altitude_m).P
        self.atmosphere = Atmosphere(dT, site.relative_humidity, p_scale)
        self.motor = ThrustCurveMotor(cfg.motor, self.geom)
        self.mass = MassModel(cfg, self.geom, self.motor)
        self.aero = Aerodynamics(cfg, self.geom)
        self.wind = WindProfile(site.wind_speed_mps, site.wind_reference_height_m,
                                site.wind_from_deg, site.wind_shear_exponent, site.wind_profile)
        self.turbulence = DrydenTurbulence(self.wind, site.turbulence_seed) if site.turbulence else None
        self.turb_ned = np.zeros(3)

        # Launch site and rail geometry
        self.lat0 = math.radians(site.latitude_deg)
        self.lon0 = math.radians(site.longitude_deg)
        self.site_ecef = geodetic_to_ecef(self.lat0, self.lon0, site.ground_altitude_m)
        self.C_ne0 = dcm_ecef_to_ned(self.lat0, self.lon0)
        rail = cfg.rail
        el, az = math.radians(rail.elevation_deg), math.radians(rail.azimuth_deg)
        h_ned = np.array([math.cos(az), math.sin(az), 0.0])
        down = np.array([0.0, 0.0, 1.0])
        u_ned = math.cos(el) * h_ned - math.sin(el) * down           # along the rail, upward
        n_ned = math.sin(el) * h_ned + math.cos(el) * down           # rocket axis -> rail, perpendicular
        self.rail_u = self.C_ne0.T @ u_ned
        self.rail_n = self.C_ne0.T @ n_ned
        self.rail_base = self.site_ecef + rail.base_height * (self.C_ne0.T @ -down)
        self.rail_length = rail.length
        self.rail_k = rail.stiffness
        m0, _, I0 = self.mass.properties(0.0)
        n_btn = len(rail.button_stations)
        self.rail_c = 2 * rail.damping_ratio * math.sqrt(rail.stiffness * m0 / n_btn)
        self.rail_mu = rail.friction_coefficient
        self.rail_kt = rail.twist_stiffness
        I_line = I0[0, 0] + m0 * (self.geom.R + rail.button_standoff) ** 2   # roll inertia about the rail line
        self.rail_ct = 2 * rail.damping_ratio * math.sqrt(rail.twist_stiffness * I_line / n_btn)
        self.button_dir_body = np.array([0.0, math.cos(self.geom.button_roll), math.sin(self.geom.button_roll)])
        self.rail_active = True

        # Thrust line
        mis_z, mis_y = (math.radians(a) for a in cfg.motor.thrust_misalignment_deg)
        t_hat = np.array([1.0, math.tan(mis_z), math.tan(mis_y)])
        self.thrust_dir = t_hat / np.linalg.norm(t_hat)
        oy, oz = cfg.motor.thrust_offset_m
        self.exit_point = np.array([-self.motor.exit_station, oy, oz])

        # Fin servos (the state vector holds three)
        if self.geom.n_fins != 3:
            raise ValueError("The state vector holds three fin servos; set fins.count = 3")
        ctrl = cfg.control
        self.fin_cmd = np.zeros(3)
        self.hinge_station = (ctrl.hinge_station if ctrl.hinge_station >= 0
                              else self.aero.fin_cp_station(0.3) - 0.005)
        self.servo_speed = math.radians(ctrl.servo_speed_deg_s)
        self.servo_tau = ctrl.servo_time_constant
        self.servo_stall = ctrl.servo_stall_torque
        self.delta_limit = math.radians(ctrl.max_deflection_deg)
        self.fin_misalignment = np.radians(cfg.fins.misalignment_deg)

        self.recovery = RecoverySystem(self)
        self.booster_landed = False

    # ---------- initial condition ----------
    def initial_attitude(self):
        """Body x along the rail; rail buttons facing the rail."""
        u, n = self.rail_u, self.rail_n
        t = cross3(u, n)
        phi = self.geom.button_roll
        y_img = math.cos(phi) * n - math.sin(phi) * t
        z_img = math.sin(phi) * n + math.cos(phi) * t
        return np.column_stack([u, y_img, z_img])

    def initial_state(self):
        """Rocket at rest on the rail, settled into static equilibrium on its buttons and the rail stop."""
        C = self.initial_attitude()
        m0, cg0, _ = self.mass.properties(0.0)
        aft = int(np.argmin([p[0] for p in self.geom.button_points]))
        s0 = m0 * (gravity_ecef(self.rail_base) @ self.rail_u) / self.rail_k    # compression of the stop
        origin = self.rail_base + s0 * self.rail_u - C @ self.geom.button_points[aft]
        y = np.zeros(N_STATE)
        y[POS] = origin + C @ cg0
        y[QUAT] = dcm_to_quat(C)
        y[OMEGA] = C.T @ OMEGA_E
        return self._settle(y)

    def _settle(self, y, max_duration=3.0):
        """Integrate the damped rail contacts (motor not ignited, mean wind) until the rocket is at rest."""
        from .simulate import rk4_step
        t_before = self.cfg.motor.ignition_time - 10.0
        dt = self.cfg.sim.dt_rail
        for k in range(int(max_duration / dt)):
            y = rk4_step(self.derivatives, t_before, y, dt)
            if k % 100 == 99:
                dy = self.derivatives(t_before, y)
                _, _, I = self.mass.properties(0.0)
                if np.linalg.norm(dy[VEL]) < 1e-3 and np.linalg.norm(I @ dy[OMEGA]) < 1e-4:
                    break
        y[VEL] = 0.0
        y[OMEGA] = quat_to_dcm(y[QUAT]).T @ OMEGA_E
        return y

    # ---------- helpers ----------
    def button_axial_positions(self, y):
        C = quat_to_dcm(y[QUAT] / np.linalg.norm(y[QUAT]))
        _, cg, _ = self.mass.properties(y[TAU], self.recovery.separated)
        return [((y[POS] + C @ (b - cg)) - self.rail_base) @ self.rail_u for b in self.geom.button_points]

    def altitude_agl_point(self, r):
        return ecef_to_geodetic(r)[2] - self.cfg.site.ground_altitude_m

    def altitude_agl(self, y):
        return self.altitude_agl_point(y[POS])

    def vertical_velocity(self, y):
        lat, lon, _ = ecef_to_geodetic(y[POS])
        return -(dcm_ecef_to_ned(lat, lon) @ y[VEL])[2]

    def update_turbulence(self, y, dt):
        if self.turbulence is None:
            return
        lat, lon, h = ecef_to_geodetic(y[POS])
        h_agl = h - self.cfg.site.ground_altitude_m
        v_air = dcm_ecef_to_ned(lat, lon) @ y[VEL] - self.wind.ned(h_agl)
        self.turb_ned = self.turbulence.step(h_agl, float(np.linalg.norm(v_air)), dt)

    def _rail_forces(self, r, v, C, w_rel, cg, cg_dot):
        """Contact forces (ECEF) and moment about the CG (body) from all rail buttons."""
        F_e, M_b = np.zeros(3), np.zeros(3)
        u, n = self.rail_u, self.rail_n
        b_e = C @ self.button_dir_body
        b_p = b_e - (b_e @ u) * u
        twist = math.atan2(cross3(n, b_p) @ u, n @ b_p)
        twist_rate = (C @ w_rel) @ u
        for b in self.geom.button_points:
            rb = b - cg
            p = r + C @ rb
            vp = v + C @ (cross3(w_rel, rb) - cg_dot)
            d = p - self.rail_base
            s = d @ u
            if s > self.rail_length or s < -0.1:
                continue
            v_ax = vp @ u
            lat = d - s * u
            lat_v = vp - v_ax * u
            F_lat = -self.rail_k * lat - self.rail_c * lat_v
            N = math.sqrt(F_lat @ F_lat)
            F = F_lat - self.rail_mu * N * math.tanh(v_ax / 0.01) * u
            if s < 0.0:
                F += max(-self.rail_k * s - self.rail_c * v_ax, 0.0) * u
            F_e += F
            M_b += cross3(rb, C.T @ F)
            M_b += C.T @ ((-self.rail_kt * twist - self.rail_ct * twist_rate) * u)
        return F_e, M_b

    # ---------- equations of motion ----------
    def derivatives(self, t, y, info=False):
        r, v = y[POS], y[VEL]
        q = y[QUAT] / np.linalg.norm(y[QUAT])
        w = y[OMEGA]
        tau = y[TAU]
        delta = y[FINS]

        C = quat_to_dcm(q)
        Ct = C.T
        lat, lon, h = ecef_to_geodetic(r)
        h_agl = h - self.cfg.site.ground_altitude_m
        C_ne = dcm_ecef_to_ned(lat, lon)
        atm = self.atmosphere.properties(h)
        wind_e = C_ne.T @ (self.wind.ned(h_agl) + self.turb_ned)

        sep = self.recovery.separated
        ms = self.motor.evaluate(t, tau, atm.P)
        m, cg, I = self.mass.properties(tau, sep)
        mdot, cg_dot, I_dot = self.mass.rates(tau, ms.tau_dot, sep)

        w_rel = w - Ct @ OMEGA_E
        v_air_b = Ct @ (v - wind_e) - cg_dot

        F_b, M_b, ainfo = self.aero.forces(v_air_b, w_rel, cg, atm.rho, atm.a, atm.nu,
                                           ms.thrust > 0.0, ms.exit_area, delta + self.fin_misalignment,
                                           nose_on=not sep)

        # Thrust, jet Coriolis force and jet damping
        F_T = ms.thrust * self.thrust_dir
        r_e = self.exit_point - cg
        F_b = F_b + F_T - 2.0 * ms.mdot * cross3(w, self.exit_point - self.motor.propellant_centroid)
        M_b = M_b + cross3(r_e, F_T) - ms.mdot * cross3(r_e, cross3(w, r_e))

        # Recovery: shock cord on the booster, and the nose cone and canopy bodies
        rec_dy, T_cord, T_line = None, 0.0, 0.0
        if sep:
            F_cord_e, M_cord, rec_dy, T_cord, T_line = self.recovery.loads(t, y, C, cg, w_rel, cg_dot)
            F_b = F_b + Ct @ F_cord_e
            M_b = M_b + M_cord

        F_rail_e = np.zeros(3)
        if self.rail_active:
            F_rail_e, M_rail = self._rail_forces(r, v, C, w_rel, cg, cg_dot)
            M_b = M_b + M_rail

        dy = np.zeros(N_STATE)
        dy[POS] = v
        dy[VEL] = (C @ F_b + F_rail_e) / m + gravitation_ecef(r) \
            - 2.0 * cross3(OMEGA_E, v) - cross3(OMEGA_E, cross3(OMEGA_E, r))
        dy[QUAT] = quat_rate(q, w_rel)
        dy[OMEGA] = np.linalg.solve(I, M_b - cross3(w, I @ w) - I_dot @ w)
        dy[TAU] = ms.tau_dot

        lever = ainfo["fin_cp_station"] - self.hinge_station
        fin_rates = np.zeros(3)
        for i in range(3):
            hinge = abs(ainfo["fin_normal"][i] * lever)
            rate_max = self.servo_speed * max(0.0, 1.0 - hinge / self.servo_stall)
            rate = max(-rate_max, min(rate_max, (self.fin_cmd[i] - delta[i]) / self.servo_tau))
            if (delta[i] >= self.delta_limit and rate > 0) or (delta[i] <= -self.delta_limit and rate < 0):
                rate = 0.0
            fin_rates[i] = rate
        dy[FINS] = fin_rates
        if rec_dy is not None:
            dy[RECOVERY] = rec_dy
        if self.booster_landed:          # resting on the ground; the cord still holds nose and canopy
            dy[BOOSTER] = 0.0
        if not info:
            return dy
        return dy, {
            "altitude": h_agl, "v_ned": C_ne @ v, "speed": float(np.linalg.norm(v)),
            "mass": m, "thrust": ms.thrust, "mdot": ms.mdot,
            "alpha": ainfo["alpha"], "mach": ainfo["mach"], "q": ainfo["q"],
            "accel_body": (F_b + Ct @ F_rail_e) / m, "pressure": atm.P,
            "cord_tension": T_cord, "line_tension": T_line, "canopy_fill": self.recovery.fill_fraction(t),
        }
