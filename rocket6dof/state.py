"""Layout of the state vector y. Every module indexes y through these names.

  POS     booster CG position, ECEF (m)
  VEL     booster CG velocity relative to the Earth, ECEF axes (m/s)
  QUAT    attitude quaternion, body -> ECEF
  OMEGA   body angular velocity relative to inertial space, body axes (rad/s)
  TAU     motor burn time since ignition (s); thrust and propellant mass follow from it
  FINS    deflection of the three all-moving fins (rad)
  BOOSTER everything above: the booster's own state
  NOSE_R, NOSE_V, CAN_R, CAN_V
          nose cone and canopy position and velocity, ECEF, used after ejection
"""
POS, VEL, QUAT, OMEGA = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13)
TAU = 13
FINS = slice(14, 17)
BOOSTER = slice(0, 17)
NOSE_R, NOSE_V, CAN_R, CAN_V = slice(17, 20), slice(20, 23), slice(23, 26), slice(26, 29)
RECOVERY = slice(17, 29)
N_STATE = 29
