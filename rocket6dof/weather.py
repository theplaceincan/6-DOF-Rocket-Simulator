"""Live weather for the launch site from Open-Meteo (https://open-meteo.com, no API key).

Fetches the current surface conditions and the winds aloft for the current hour:
  10 m (current), 80 m, 120 m, 180 m above ground, and the 850 hPa and 700 hPa pressure levels
  (converted to height above ground with their geopotential heights).
apply_weather() writes them into a Config: wind profile table, temperature offset from the
standard atmosphere, humidity and a pressure scale matching the measured surface pressure.
"""
import json
import urllib.parse
import urllib.request

_HOURLY = ["wind_speed_80m", "wind_direction_80m", "wind_speed_120m", "wind_direction_120m",
           "wind_speed_180m", "wind_direction_180m", "wind_speed_850hPa", "wind_direction_850hPa",
           "geopotential_height_850hPa", "wind_speed_700hPa", "wind_direction_700hPa",
           "geopotential_height_700hPa"]
_CURRENT = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
            "wind_direction_10m", "wind_gusts_10m"]


def fetch(lat, lon, timeout=20):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "current": ",".join(_CURRENT), "hourly": ",".join(_HOURLY),
        "wind_speed_unit": "ms", "timezone": "auto", "forecast_days": 1})
    req = urllib.request.Request(f"https://api.open-meteo.com/v1/forecast?{q}",
                                 headers={"User-Agent": "rocket6dof-flight-sim"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c, h = d["current"], d["hourly"]
    i = [t[:13] for t in h["time"]].index(c["time"][:13])
    ground = d.get("elevation", 0.0)
    profile = [(10.0, c["wind_speed_10m"], c["wind_direction_10m"])]
    for hgt in (80, 120, 180):
        profile.append((float(hgt), h[f"wind_speed_{hgt}m"][i], h[f"wind_direction_{hgt}m"][i]))
    for lvl in ("850hPa", "700hPa"):
        z = h[f"geopotential_height_{lvl}"][i] - ground
        if z > profile[-1][0]:
            profile.append((z, h[f"wind_speed_{lvl}"][i], h[f"wind_direction_{lvl}"][i]))
    return {"time": c["time"], "timezone": d.get("timezone", ""), "elevation": ground,
            "temperature_C": c["temperature_2m"], "relative_humidity": c["relative_humidity_2m"] / 100.0,
            "surface_pressure_hPa": c["surface_pressure"], "gust_mps": c["wind_gusts_10m"],
            "wind_profile": tuple(profile)}


def apply_weather(cfg, wx):
    """Copy fetched conditions into cfg.site."""
    s = cfg.site
    s.wind_profile = wx["wind_profile"]
    s.wind_speed_mps, s.wind_from_deg = wx["wind_profile"][0][1], wx["wind_profile"][0][2]
    s.wind_reference_height_m = 10.0
    s.relative_humidity = wx["relative_humidity"]
    s.surface_pressure_hPa = wx["surface_pressure_hPa"]
    s.surface_temperature_C = wx["temperature_C"]
    s.weather_source = f"Open-Meteo, {wx['time']} {wx['timezone']}"
    return cfg
