"""Every input to the model: rocket, motor, launch station, site and simulation settings.

SI units throughout. Stations are measured in metres aft of the nose tip.
Values marked "estimate" are engineering guesses, not measured data; replace them with your own.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SiteConfig:
    # Black Rock Desert playa, Nevada: a flat dry lakebed used for high-power launches.
    # Approximate launch-area coordinates (AeroPac); the exact site moves year to year with playa conditions.
    name: str = "Black Rock Desert playa, Nevada"
    latitude_deg: float = 40.8707
    longitude_deg: float = -119.1070
    ground_altitude_m: float = 1191.0     # playa elevation above sea level (geoid offset to WGS 84 ignored)

    # Weather. Defaults are the Open-Meteo observation/forecast for 2026-11-26 10:30 America/Los_Angeles.
    # viewer.py refreshes them live; rocket6dof.weather.apply_weather(cfg, fetch(lat, lon)) does it in code.
    weather_source: str = "Open-Meteo, 2026-11-26T10:30 America/Los_Angeles"
    surface_temperature_C: float = 5.3    # sets the offset from the standard atmosphere (None: use offset below)
    temperature_offset_K: float = 0.0     # used only when surface_temperature_C is None
    surface_pressure_hPa: float = 885.4   # scales the pressure profile to the measured value (0: standard)
    relative_humidity: float = 0.85
    # Wind profile table: (height above ground m, speed m/s, direction wind blows FROM, deg)
    wind_profile: Tuple[Tuple[float, float, float], ...] = (
        (10.0, 1.28, 295.0), (80.0, 1.4, 252.0), (120.0, 3.42, 207.0), (180.0, 1.14, 218.0),
        (383.0, 5.82, 211.0), (1975.0, 10.55, 261.0))
    # Used only when wind_profile is empty: power law V(h) = V_ref (h / h_ref)^p
    wind_speed_mps: float = 1.28
    wind_reference_height_m: float = 10.0
    wind_from_deg: float = 295.0
    wind_shear_exponent: float = 1.0 / 7.0
    turbulence: bool = True               # Dryden turbulence (MIL-F-8785C), intensity from the 20 ft wind
    turbulence_seed: int = 1


@dataclass
class NoseConeConfig:
    length: float = 0.25                  # tangent ogive
    wall: float = 0.002
    density: float = 1240.0               # PLA
    shoulder_length: float = 0.06
    ballast_mass: float = 0.0
    ballast_station: float = 0.06


@dataclass
class BodyTubeConfig:
    outer_diameter: float = 0.054
    wall: float = 0.0015
    length: float = 0.95
    density: float = 1850.0               # glass-fibre epoxy
    surface_roughness: float = 20e-6      # m, smooth paint


@dataclass
class FinConfig:
    count: int = 3
    root_chord: float = 0.10
    tip_chord: float = 0.05
    span: float = 0.045
    sweep: float = 0.05                   # leading-edge sweep length (root LE to tip LE, axial)
    thickness: float = 0.003
    density: float = 1850.0               # G10
    shear_modulus: float = 4.0e9          # estimate for G10, used for the flutter check
    aft_offset: float = 0.0               # root trailing edge distance forward of the body's aft end
    cant_deg: float = 0.0
    # Measured angle of each fin to the body axis, deg (same sign as cant). Zero = perfectly aligned.
    # Enter values measured on your own rocket to reproduce its roll; nothing is assumed here.
    misalignment_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    roll_offset_deg: float = 0.0          # roll angle of fin 0 from body +y


@dataclass
class ItemConfig:
    """Internal item modelled as a solid cylinder on the axis (or a point mass if length = 0)."""
    name: str
    mass: float
    station: float                        # CG station
    length: float = 0.0
    radius: float = 0.0


@dataclass
class InternalsConfig:
    motor_tube_outer_diameter: float = 0.031   # 29 mm motor mount
    motor_tube_wall: float = 0.001
    motor_tube_length: float = 0.23
    motor_tube_density: float = 750.0     # spiral-wound paper
    centering_ring_thickness: float = 0.006
    centering_ring_density: float = 650.0  # birch plywood
    centering_ring_stations: Tuple[float, ...] = (0.98, 1.19)
    items: List[ItemConfig] = field(default_factory=lambda: [
        ItemConfig("altimeter bay", 0.080, 0.42, length=0.08, radius=0.022),
        ItemConfig("parachute", 0.090, 0.34, length=0.09, radius=0.023),
        ItemConfig("shock cord and nomex", 0.060, 0.52, length=0.08, radius=0.020),
        ItemConfig("bulkhead", 0.015, 0.31, length=0.006, radius=0.0255),
        ItemConfig("motor retainer", 0.030, 1.195, length=0.012, radius=0.018),
        ItemConfig("flight computer and IMU", 0.035, 0.45, length=0.06, radius=0.018),
        ItemConfig("battery 2S 450 mAh", 0.030, 0.47, length=0.05, radius=0.012),
        ItemConfig("fin servos x3 and linkages", 0.045, 0.93, length=0.04, radius=0.020),
    ])


@dataclass
class MotorConfig:
    """Certified commercial motor flown from its certification thrust curve (RASP .eng file).
    Impulse, masses, diameter and length come from the file; the internal layout below is estimated."""
    file: str = "motors/AeroTech_H128W.eng"     # relative to the project root
    propellant_density: float = 1600.0          # estimate, ammonium-perchlorate composite
    grain_outer_diameter: float = 0.024         # estimate
    grain_length: float = 0.150                 # estimate
    nozzle_exit_diameter: float = 0.011         # estimate, used for the altitude pressure correction
    test_ambient_pressure: float = 101325.0     # pressure during the certification test
    recess: float = 0.0                         # motor aft end distance forward of the body's aft end
    ignition_time: float = 0.0
    thrust_misalignment_deg: Tuple[float, float] = (0.0, 0.0)  # about body z and y
    thrust_offset_m: Tuple[float, float] = (0.0, 0.0)          # lateral offset of the thrust line (y, z)
    exhaust_temperature_K: float = 1500.0       # estimate, nozzle exit gas temperature (smoke buoyancy)


@dataclass
class LaunchRailConfig:
    length: float = 1.8
    elevation_deg: float = 87.0
    azimuth_deg: float = 90.0             # direction the rail leans toward, clockwise from north
    base_height: float = 0.35             # rail bottom above ground
    profile: float = 0.0254               # 1010 rail, 1 inch square
    button_stations: Tuple[float, ...] = (0.62, 1.12)
    button_roll_deg: float = 60.0
    button_standoff: float = 0.010        # body surface to rail-slot contact line
    button_mass: float = 0.004
    button_frontal_area: float = 1.2e-4
    stiffness: float = 2.0e5              # N/m per button, penalty contact
    damping_ratio: float = 0.7
    friction_coefficient: float = 0.2
    twist_stiffness: float = 50.0         # N m/rad per button: flange resisting rotation about the rail line
    tripod_leg_spread: float = 0.6        # station geometry (viewer)
    deflector_size: float = 0.25


@dataclass
class RecoveryConfig:
    """Single-deploy recovery: ejection charge pushes off the nose cone; nose cone, parachute and
    booster then fly as separate bodies joined by an elastic shock cord and suspension lines."""
    enabled: bool = True
    trigger: str = "altimeter"            # "altimeter": barometric apogee from the flight computer
    deploy_delay: float = 1.0             # s after detected apogee ("altimeter")
    motor_delay: float = 10.0             # s after burnout (trigger "motor_delay", motor ejection charge)
    # Ejection
    ejection_charge_g: float = 0.7        # black powder (FFFF)
    charge_efficiency: float = 0.7        # fraction of ideal pressure reached (heat loss, leakage); estimate
    compartment_free_fraction: float = 0.6  # free air volume of the parachute bay; estimate
    nose_friction_N: float = 30.0         # friction fit of the nose shoulder, no shear pins
    # Shock cord: booster front opening to the nose cone
    shock_cord_length: float = 4.0
    shock_cord_EA: float = 16000.0        # N, 1/4 in tubular nylon (about 4.4 kN at 27 % strain); estimate
    shock_cord_strength_N: float = 4400.0  # breaking load; estimate for 1/4 in tubular nylon
    shock_cord_damping_ratio: float = 0.1
    # Parachute: lines attach to the nose cone
    canopy_diameter: float = 0.91         # nominal (flat) diameter, 36 in
    drag_coefficient: float = 0.80        # on nominal area
    projected_diameter_ratio: float = 0.70  # inflated projected diameter / nominal
    suspension_line_length: float = 0.90
    line_EA: float = 16000.0              # N, all lines together; estimate
    line_strength_N: float = 5300.0       # breaking load of all lines together (8 x 150 lb); estimate
    line_damping_ratio: float = 0.1
    fill_constant: float = 8.0            # Knacke filling distance: t_fill = n D0 / V_line_stretch
    added_mass_coefficient: float = 0.5   # m_a = k rho (pi/6) Dp^3; estimate
    packed_drag_area: float = 0.003       # m^2, CdS of the packed canopy before line stretch
    nose_drag_coefficient: float = 0.8    # tumbling nose cone, on body cross-section; estimate


@dataclass
class ControlConfig:
    """Active fin control: each fin is all-moving, turned about a spanwise shaft by its own servo.

    The flight computer sees only its sensors (gyro, accelerometer, barometer), runs a strapdown
    attitude estimate from the pad, holds the launch attitude (pitch/yaw) and cancels roll rate.
    Values for sensors and servos are typical hobby-grade parts; replace with your datasheets.
    """
    enabled: bool = True
    hinge_station: float = -1.0           # fin shaft station; < 0 puts it 5 mm ahead of the fin CP
    max_deflection_deg: float = 10.0
    # Servo values are at the fin shaft, through a 6:1 reduction linkage from a hobby servo
    # (600 deg/s no-load, 0.25 N m stall, about 0.12 deg command resolution at the servo horn).
    servo_speed_deg_s: float = 100.0      # fin no-load speed
    servo_time_constant: float = 0.015    # s, first-order response of the position loop
    servo_stall_torque: float = 1.5       # N m at the fin shaft
    servo_resolution_deg: float = 0.02    # fin command quantisation
    controller_rate_hz: float = 100.0
    attitude_bandwidth: float = 8.0       # rad/s, closed-loop pitch/yaw natural frequency
    attitude_damping: float = 0.7
    roll_rate_gain: float = 12.0          # 1/s
    enable_speed: float = 25.0            # m/s estimated airspeed before fins move
    launch_detect_accel: float = 3.0      # g, sustained for launch_detect_time
    launch_detect_time: float = 0.03
    gyro_noise_density: float = 7e-5      # rad/s/sqrt(Hz)
    gyro_bias_sigma: float = 0.0035       # rad/s, constant per-axis bias after power-on
    accel_noise_density: float = 1.5e-3   # m/s^2/sqrt(Hz)
    accel_bias_sigma: float = 0.03        # m/s^2
    baro_noise: float = 1.5               # Pa rms
    sensor_seed: int = 7


@dataclass
class SmokeConfig:
    """Exhaust and ejection smoke as Lagrangian puffs (see smoke.py)."""
    enabled: bool = True
    puff_rate_hz: float = 120.0           # puffs emitted per second of burn
    particulate_fraction: float = 0.3     # visible aerosol mass / exhaust mass; estimate for AP composite
    mass_extinction: float = 3000.0       # m^2/kg, light extinction of fine smoke particles; estimate
    spread_rate: float = 0.25             # puff radius growth per metre travelled through the air
    richardson_constant: float = 0.5      # relative-dispersion constant (sigma^2 = C eps t^3)
    roughness_length: float = 0.0005      # m, dry-lake surface
    black_powder_smoke_fraction: float = 0.55  # solid residue fraction of the ejection charge
    max_age: float = 180.0                # s


@dataclass
class SimulationConfig:
    dt_rail: float = 2e-4
    dt_powered: float = 5e-4
    dt_coast: float = 2e-3
    dt_deploy: float = 5e-4               # ejection, cord snatch and canopy opening
    dt_descent: float = 2e-3              # steady descent, nose cone and canopy (stiff lines need small steps)
    dt_descent_booster: float = 5e-3      # steady descent, booster step (multirate); ~1 % descent-rate error vs 2e-3
    max_time: float = 600.0
    record_interval: float = 0.01


@dataclass
class Config:
    site: SiteConfig = field(default_factory=SiteConfig)
    nose: NoseConeConfig = field(default_factory=NoseConeConfig)
    body: BodyTubeConfig = field(default_factory=BodyTubeConfig)
    fins: FinConfig = field(default_factory=FinConfig)
    internals: InternalsConfig = field(default_factory=InternalsConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    rail: LaunchRailConfig = field(default_factory=LaunchRailConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
