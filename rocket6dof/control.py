"""Flight computer: barometric apogee detection for recovery, and optional active fin control.

What the flight computer knows: only its own sensor readings and preloaded design data
(nominal inertia, fin geometry, lift-curve slope). It never reads the true simulation state.

Sensors (sampled at the controller rate):
  gyro    w_true + bias + white noise          (bias constant after power-on)
  accel   specific force + bias + white noise
  baro    static pressure + white noise
  Discrete noise sigma = noise_density * sqrt(f_s / 2).

On the pad:   averages gyro (removes bias and Earth rate), accelerometer (gravity direction) and
              barometer (pad pressure). Launch is detected by sustained axial acceleration.
In flight:    strapdown integration of the gyro gives attitude relative to the pad frame
              (the body frame at launch); integrating the rotated accelerometer plus pad gravity
              gives velocity. Airspeed ~ |velocity| (no wind knowledge), density from the barometer.
Control law:  pitch/yaw hold the launch attitude, roll rate driven to zero:
                alpha_cmd = Kp e - Kd w,  alpha_cmd_x = -K_r w_x,   M_cmd = I_nominal alpha_cmd
              with Kp = w_n^2, Kd = 2 zeta w_n, e = x_body x (launch axis in body axes).
Allocation:   dM/d(delta_i) = q A_ref K_TB C_Na1(M) (r_i x n_i) for each fin; solve the 3 x 3
              system for the fin deflections (gain scheduling with dynamic pressure is implicit).
Limits:       deflection limit, command quantisation; fins centred before the enable speed
              and after apogee is detected.
Apogee:       barometric altitude, low-pass filtered (time constant 0.2 s); apogee is declared when
              the filtered altitude falls 1.5 m below its maximum, at least 2 s after launch.
              This time fires the ejection charge (recovery.trigger = "altimeter").
"""
import math
import numpy as np

from .frames import cross3, quat_multiply, quat_to_dcm


class FlightComputer:
    def __init__(self, model):
        self.model = model
        c = model.cfg.control
        self.c = c
        self.dt = 1.0 / c.controller_rate_hz
        rng = np.random.default_rng(c.sensor_seed)
        self.rng = rng
        self.gyro_bias = rng.normal(0.0, c.gyro_bias_sigma, 3)
        self.accel_bias = rng.normal(0.0, c.accel_bias_sigma, 3)
        bw = math.sqrt(c.controller_rate_hz / 2)
        self.gyro_sigma = c.gyro_noise_density * bw
        self.accel_sigma = c.accel_noise_density * bw

        self.Kp = c.attitude_bandwidth ** 2
        self.Kd = 2 * c.attitude_damping * c.attitude_bandwidth
        self.delta_max = math.radians(c.max_deflection_deg)
        self.quantum = math.radians(c.servo_resolution_deg)

        # Preloaded design data: mean of the liftoff and burnout mass properties
        m0, cg0, I0 = model.mass.properties(0.0)
        m1, cg1, I1 = model.mass.properties(model.motor.t_end)
        self.I_nom = 0.5 * (I0 + I1)
        self.cg_nom = 0.5 * (cg0 + cg1)
        aero = model.aero
        s_cp = aero.fin_cp_station(0.3)
        self.fin_arm = [cross3(np.array([-s_cp, 0.0, 0.0]) + aero.fin_cp_radius * e - self.cg_nom, n)
                        for n, e in zip(aero.fin_normals, aero.fin_radial)]

        # Filter / navigation state
        self.n_pad = 0
        self.gyro_cal = np.zeros(3)
        self.f_pad = np.zeros(3)
        self.p_pad = 0.0
        self.detect_time = 0.0
        self.launched = False
        self.t_launch = None
        self.q_est = np.array([1.0, 0.0, 0.0, 0.0])
        self.v_nav = np.zeros(3)
        self.g_pad = np.zeros(3)
        self.apogee = False
        self.enabled = False
        self.cmd = np.zeros(3)
        self.speed_est = 0.0
        self.baro_alt = 0.0
        self.alt_filt = 0.0
        self.alt_max = 0.0
        self.baro_apogee_time = None

    def measure(self, w_true, f_true, p_true):
        gyro = w_true + self.gyro_bias + self.rng.normal(0.0, self.gyro_sigma, 3)
        accel = f_true + self.accel_bias + self.rng.normal(0.0, self.accel_sigma, 3)
        baro = p_true + self.rng.normal(0.0, self.c.baro_noise)
        return gyro, accel, baro

    def calibrate_on_pad(self, w_true, f_true, p_true, duration=5.0):
        """Sensor averaging while the rocket sits on the rail before ignition."""
        for _ in range(int(duration / self.dt)):
            self._pad_sample(*self.measure(w_true, f_true, p_true))

    def _pad_sample(self, gyro, f, P):
        self.n_pad += 1
        k = 1.0 / self.n_pad
        self.gyro_cal += k * (gyro - self.gyro_cal)
        self.f_pad += k * (f - self.f_pad)
        self.p_pad += k * (P - self.p_pad)

    def step(self, t, w_true, f_true, p_true):
        """One controller cycle. Returns the fin deflection commands (rad)."""
        gyro, f, P = self.measure(w_true, f_true, p_true)
        dt = self.dt

        if not self.launched:
            g0 = 9.80665
            if abs(math.sqrt(f @ f) - g0) < 0.2 * g0:   # only quiet samples refine the pad calibration
                self._pad_sample(gyro, f, P)
            if f[0] > self.c.launch_detect_accel * g0:
                self.detect_time += dt
                if self.detect_time >= self.c.launch_detect_time:
                    self.launched = True
                    self.t_launch = t - self.detect_time
                    self.g_pad = -self.f_pad
            else:
                self.detect_time = 0.0
            self.cmd = np.zeros(3)
            return self.cmd

        # Strapdown attitude (body relative to the pad frame) and velocity
        w_m = gyro - self.gyro_cal
        ang = math.sqrt(w_m @ w_m) * dt
        if ang > 0.0:
            axis = w_m * dt / ang
            dq = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * axis)])
            self.q_est = quat_multiply(self.q_est, dq)
            self.q_est /= np.linalg.norm(self.q_est)
        R = quat_to_dcm(self.q_est)
        self.v_nav += (R @ f + self.g_pad) * dt
        self.speed_est = math.sqrt(self.v_nav @ self.v_nav)
        self.baro_alt = 44330.0 * (1.0 - (max(P, 1.0) / self.p_pad) ** (1 / 5.255))
        self.alt_filt += (dt / 0.2) * (self.baro_alt - self.alt_filt)
        self.alt_max = max(self.alt_max, self.alt_filt)
        if (self.baro_apogee_time is None and t - self.t_launch > 2.0
                and self.alt_filt < self.alt_max - 1.5):
            self.baro_apogee_time = t

        up = -self.g_pad / np.linalg.norm(self.g_pad)
        if not self.apogee and t - self.t_launch > 1.0 and self.v_nav @ up < 0.0:
            self.apogee = True

        self.enabled = (self.c.enabled and not self.apogee and t - self.t_launch > 0.1
                        and self.speed_est > self.c.enable_speed)
        if not self.enabled:
            self.cmd = np.zeros(3)
            return self.cmd

        # Control law
        target_b = R.T @ np.array([1.0, 0.0, 0.0])
        e = cross3(np.array([1.0, 0.0, 0.0]), target_b)
        alpha_cmd = self.Kp * e - self.Kd * w_m
        alpha_cmd[0] = -self.c.roll_rate_gain * w_m[0]
        M_cmd = self.I_nom @ alpha_cmd

        # Allocation with dynamic pressure from the barometer and velocity estimate
        T = 288.15 - 0.0065 * (self.model.cfg.site.ground_altitude_m + self.baro_alt)
        rho = P / (287.05 * T)
        mach = self.speed_est / math.sqrt(1.4 * 287.05 * T)
        qbar = 0.5 * rho * self.speed_est ** 2
        aero = self.model.aero
        k = qbar * self.model.geom.A_ref * aero.K_TB * aero.fin_cna_subsonic(min(mach, 0.9))
        B = np.column_stack([k * arm for arm in self.fin_arm])
        delta = np.linalg.lstsq(B, M_cmd, rcond=None)[0]
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        self.cmd = np.round(delta / self.quantum) * self.quantum
        return self.cmd
