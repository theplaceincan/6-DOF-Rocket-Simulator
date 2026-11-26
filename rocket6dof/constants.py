"""Physical constants. SI units throughout."""

# Standard gravity and gas constants
G0 = 9.80665                      # m/s^2, standard gravity
R_UNIVERSAL = 8.31446261815324    # J/(mol K), CODATA 2018

# US Standard Atmosphere 1976 uses its own values of R* and M0; keep them for table fidelity.
USSA_R_STAR = 8.31432             # J/(mol K)
USSA_M0 = 0.0289644               # kg/mol, mean molar mass of dry air
USSA_R_AIR = USSA_R_STAR / USSA_M0  # 287.053 J/(kg K)
USSA_EARTH_RADIUS = 6356766.0     # m, radius used for geopotential altitude
GAMMA_AIR = 1.4
R_WATER_VAPOUR = 461.495          # J/(kg K)

# Sutherland's law for air viscosity (USSA 1976)
SUTHERLAND_BETA = 1.458e-6        # kg/(m s K^0.5)
SUTHERLAND_S = 110.4              # K

# WGS 84 ellipsoid and gravity model
WGS84_A = 6378137.0               # m, semi-major axis
WGS84_F = 1.0 / 298.257223563     # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared
WGS84_GM = 3.986004418e14         # m^3/s^2
WGS84_J2 = 1.08262982131e-3       # second zonal harmonic (unnormalised)
OMEGA_EARTH = 7.292115e-5         # rad/s, Earth rotation rate
