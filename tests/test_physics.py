"""Reference-value checks. Run:  python -m unittest discover tests"""
import math
import unittest

import numpy as np

from rocket6dof import Config, FlightModel, simulate
from rocket6dof.atmosphere import Atmosphere
from rocket6dof.checks import static_margin
from rocket6dof.motor import read_eng
from rocket6dof.state import TAU, FINS, VEL, NOSE_V, CAN_V
from rocket6dof.earth import gravitation_ecef, gravity_ecef
from rocket6dof.frames import (quat_to_dcm, dcm_to_quat, geodetic_to_ecef, ecef_to_geodetic, cross3)
from rocket6dof.massprops import tube_part


class Atmosphere1976(unittest.TestCase):
    def test_table_values(self):
        atm = Atmosphere()
        sl = atm.properties(0.0)
        self.assertAlmostEqual(sl.T, 288.15, places=6)
        self.assertAlmostEqual(sl.P, 101325.0, places=3)
        self.assertAlmostEqual(sl.rho, 1.2250, places=4)
        self.assertAlmostEqual(sl.a, 340.29, places=2)
        self.assertAlmostEqual(sl.mu, 1.7894e-5, delta=1e-8)
        # 11 km geopotential = 11019.1 m geometric
        z = 11000.0 * 6356766.0 / (6356766.0 - 11000.0)
        tp = atm.properties(z)
        self.assertAlmostEqual(tp.T, 216.65, places=4)
        self.assertAlmostEqual(tp.P, 22632.06, delta=0.5)
        self.assertAlmostEqual(atm.properties(20000.0).P, 5529.3, delta=2.0)   # 20 km geometric

    def test_humidity_lowers_density(self):
        self.assertLess(Atmosphere(relative_humidity=1.0).properties(0).rho, Atmosphere().properties(0).rho)


class EarthAndFrames(unittest.TestCase):
    def test_gravity(self):
        eq = geodetic_to_ecef(0.0, 0.0, 0.0)
        pole = geodetic_to_ecef(math.pi / 2, 0.0, 0.0)
        # Normal gravity (WGS 84): 9.7803 m/s^2 at the equator, 9.8322 at the pole
        self.assertAlmostEqual(np.linalg.norm(gravity_ecef(eq)), 9.7803, delta=2e-3)
        self.assertAlmostEqual(np.linalg.norm(gravity_ecef(pole)), 9.8322, delta=2e-3)
        self.assertGreater(np.linalg.norm(gravitation_ecef(eq)), np.linalg.norm(gravity_ecef(eq)))

    def test_geodetic_round_trip(self):
        for lat, lon, h in [(0.3, -1.2, 1500.0), (-1.1, 2.9, 0.0), (1.5, 0.1, 30000.0)]:
            la, lo, hh = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
            self.assertAlmostEqual(la, lat, places=10)
            self.assertAlmostEqual(lo, lon, places=10)
            self.assertAlmostEqual(hh, h, delta=1e-4)

    def test_quaternion_dcm(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            q *= np.sign(q[0])
            C = quat_to_dcm(q)
            self.assertTrue(np.allclose(C @ C.T, np.eye(3), atol=1e-12))
            self.assertAlmostEqual(np.linalg.det(C), 1.0, places=12)
            self.assertTrue(np.allclose(dcm_to_quat(C), q, atol=1e-10))

    def test_cross(self):
        a, b = np.array([1.0, 2, 3]), np.array([-4.0, 5, 0.5])
        self.assertTrue(np.allclose(cross3(a, b), np.cross(a, b)))


class ThrustCurveMotor(unittest.TestCase):
    def setUp(self):
        self.motor = FlightModel(Config()).motor

    def test_certified_data(self):
        h, t, F = read_eng("motors/AeroTech_H128W.eng")
        self.assertEqual((h["name"], h["diameter"], h["propellant_mass"]), ("H128W", 0.029, 0.0942))
        # ThrustCurve.org lists 172.9 N s; trapezoidal integration from (0, 0) gives a few percent more
        self.assertAlmostEqual(self.motor.I_total, 172.9, delta=0.04 * 172.9)
        self.assertEqual(self.motor.performance()["motor_class"], "H")

    def test_propellant_consumed(self):
        m = self.motor
        self.assertAlmostEqual(m.propellant_mass(0.0), 0.0942, places=6)
        self.assertAlmostEqual(m.propellant_mass(m.t_end), 0.0, places=9)
        t = np.linspace(0, m.t_end, 4001)
        mdot = [m.evaluate(ti, ti, 101325.0).mdot for ti in t]
        self.assertAlmostEqual(np.trapezoid(mdot, t), 0.0942, delta=2e-4)

    def test_pressure_correction(self):
        m = self.motor
        sea = m.evaluate(0.3, 0.3, 101325.0).thrust
        high = m.evaluate(0.3, 0.3, 80000.0).thrust
        self.assertAlmostEqual(high - sea, 21325.0 * m.exit_area, places=6)


class MassProperties(unittest.TestCase):
    def test_tube_parallel_axis(self):
        p = tube_part("t", 0.03, 0.028, 0.5, 0.1, 1000.0)
        m = p.mass
        I_tr_about_tip = m * (3 * (0.03 ** 2 + 0.028 ** 2) + 0.5 ** 2) / 12 + m * 0.35 ** 2
        self.assertAlmostEqual(p.J[1, 1], I_tr_about_tip, places=12)

    def test_rates_match_finite_difference(self):
        mm = FlightModel(Config()).mass
        tau, h = 0.4, 1e-5
        mdot, cdot, Idot = mm.rates(tau, 1.0)
        m1, c1, I1 = mm.properties(tau + h)
        m0, c0, I0 = mm.properties(tau - h)
        self.assertAlmostEqual(mdot, (m1 - m0) / (2 * h), places=6)
        self.assertTrue(np.allclose(cdot, (c1 - c0) / (2 * h), atol=1e-7))
        self.assertTrue(np.allclose(Idot, (I1 - I0) / (2 * h), atol=1e-8))


class Flight(unittest.TestCase):
    def test_stable_design(self):
        sm = static_margin(FlightModel(Config()))
        self.assertGreater(sm["margin_ignition"], 1.0)
        self.assertGreater(sm["margin_burnout"], sm["margin_ignition"])

    def test_rail_phase(self):
        cfg = Config()
        cfg.site.turbulence = False
        rec, ev, model = simulate(cfg, stop_at="rail_exit")
        self.assertIn("liftoff", ev)
        self.assertGreater(ev["rail_exit_speed"], 15.0)
        # On the rail the axis stays aligned with the rail
        y = rec["state"][-1]
        axis = quat_to_dcm(y[6:10])[:, 0]
        self.assertGreater(axis @ model.rail_u, math.cos(math.radians(0.5)))

    def test_at_rest_on_pad(self):
        """Before ignition the rocket must stay still on the rail stop."""
        cfg = Config()
        cfg.site.turbulence = False
        cfg.site.wind_speed_mps = 0.0
        cfg.motor.ignition_time = 1.0
        model = FlightModel(cfg)
        y = model.initial_state()
        dy = model.derivatives(0.0, y)
        self.assertLess(np.linalg.norm(dy[VEL]), 5e-3)   # < 0.05 % of g
        _, _, I = model.mass.properties(y[TAU])
        self.assertLess(np.linalg.norm(I @ dy[10:13]), 1e-3)   # residual torque, N m


class FinControl(unittest.TestCase):
    def test_allocation_reproduces_moment(self):
        from rocket6dof.control import FlightComputer
        model = FlightModel(Config())
        fc = FlightComputer(model)
        k = 5000.0 * model.geom.A_ref * model.aero.K_TB * model.aero.fin_cna_subsonic(0.3)
        B = np.column_stack([k * arm for arm in fc.fin_arm])
        M = np.array([0.002, -0.05, 0.03])
        delta = np.linalg.solve(B, M)
        self.assertTrue(np.allclose(B @ delta, M))
        self.assertGreater(abs(np.linalg.det(B)), 0.0)

    def test_servo_rate_and_stop(self):
        model = FlightModel(Config())
        y = model.initial_state()
        model.fin_cmd = np.radians([30.0, -30.0, 5.0])
        dy = model.derivatives(0.5, y)
        speed = math.radians(model.cfg.control.servo_speed_deg_s)
        self.assertTrue(np.all(np.abs(dy[FINS]) <= speed + 1e-12))
        y[FINS.start] = model.delta_limit
        self.assertEqual(model.derivatives(0.5, y)[FINS.start], 0.0)

    def test_control_holds_attitude_in_wind(self):
        dev = {}
        for on in (False, True):
            cfg = Config()
            cfg.site.wind_profile = ()          # use a steady 5 m/s power-law crosswind
            cfg.site.wind_speed_mps = 5.0
            cfg.control.enabled = on
            rec, ev, model = simulate(cfg, stop_at="burnout")
            axis = quat_to_dcm(rec["state"][-1][6:10])[:, 0]
            dev[on] = math.degrees(math.acos(min(1.0, axis @ model.rail_u)))
            fins = np.degrees(np.abs([st[FINS] for st in rec["state"]]))
            self.assertLessEqual(fins.max(), cfg.control.max_deflection_deg + 1e-9)
        self.assertLess(dev[True], 0.5 * dev[False])


class Recovery(unittest.TestCase):
    def test_separation_conserves_momentum(self):
        model = FlightModel(Config())
        y = model.initial_state()
        y[VEL] = model.rail_u * 30.0                       # moving, so the check is not trivial
        m_all, _, _ = model.mass.properties(y[TAU], False)
        p_before = m_all * y[VEL]
        y2, info = model.recovery.separate(5.0, y)
        self.assertNotIn("failed", info)
        rs = model.recovery
        m_b, _, _ = model.mass.properties(y2[TAU], True)
        p_after = m_b * y2[VEL] + rs.m_nose * y2[NOSE_V] + rs.m_chute * y2[CAN_V]
        self.assertTrue(np.allclose(p_before, p_after, atol=1e-9))
        self.assertGreater(info["separation_speed"], 3.0)

    def test_weak_charge_fails_to_separate(self):
        cfg = Config()
        cfg.recovery.ejection_charge_g = 0.02
        model = FlightModel(cfg)
        _, info = model.recovery.separate(5.0, model.initial_state())
        self.assertTrue(info.get("failed"))

    def test_tension_only_cord(self):
        from rocket6dof.recovery import RecoverySystem
        T, _ = RecoverySystem._tension(np.zeros(3), np.zeros(3), np.array([0, 0, 3.0]), np.zeros(3), 4.0, 4000.0, 10.0)
        self.assertEqual(T, 0.0)                            # slack
        T, u = RecoverySystem._tension(np.zeros(3), np.zeros(3), np.array([0, 0, 4.1]), np.zeros(3), 4.0, 4000.0, 10.0)
        self.assertAlmostEqual(T, 400.0, places=6)
        self.assertTrue(np.allclose(u, [0, 0, 1]))


class Weather(unittest.TestCase):
    def test_wind_profile_interpolation(self):
        from rocket6dof.atmosphere import WindProfile
        w = WindProfile(0, 10, 0, 1 / 7, table=((10.0, 2.0, 270.0), (110.0, 6.0, 270.0)))
        self.assertTrue(np.allclose(w.ned(60.0), [0.0, 4.0, 0.0], atol=1e-12))   # from west -> blows east
        self.assertAlmostEqual(w.speed_at(500.0), 6.0)
        self.assertAlmostEqual(w.speed_at(5.0), 2.0 * 0.5 ** (1 / 7))

    def test_surface_conditions_match_config(self):
        cfg = Config()
        atm = FlightModel(cfg).atmosphere.properties(cfg.site.ground_altitude_m)
        self.assertAlmostEqual(atm.P / 100.0, cfg.site.surface_pressure_hPa, places=6)
        self.assertAlmostEqual(atm.T - 273.15, cfg.site.surface_temperature_C, places=6)



class Smoke(unittest.TestCase):
    def test_puffs_stop_and_drift(self):
        from rocket6dof.simulate import Simulator
        from rocket6dof.smoke import compute_smoke
        cfg = Config()
        cfg.site.turbulence = False
        sim = Simulator(cfg).run(stop_at="burnout")
        puffs = compute_smoke(sim.model, sim.rec, sim.events)
        self.assertGreater(len(puffs), 100)
        air = [p for p in puffs if p["s"][0][3] > 5.0]           # released in flight, not at the pad
        p = air[len(air) // 2]
        start, s1 = np.array(p["s"][0][1:4]), next(x for x in p["s"] if x[0] >= 1.0)
        # The exhaust leaves at ~1800 m/s relative to the rocket but stops within a few metres
        self.assertLess(np.linalg.norm(np.array(s1[1:4]) - start), 5.0)
        # Opacity falls as the puff spreads
        alphas = [x[5] for x in p["s"]]
        self.assertGreater(alphas[0], alphas[-1])


if __name__ == "__main__":
    unittest.main()
