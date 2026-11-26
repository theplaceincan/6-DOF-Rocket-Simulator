"""Build viewer.html: the rocket on its launch station at the configured site, plus a simulated flight
that plays back when you press Launch. Geometry, weather, flight and recovery all come from rocket6dof.

Run:  python viewer.py                simulate, write viewer.html, start the simulator at http://127.0.0.1:8765
                                       and open it (it re-simulates when you press Deploy parachute)
      python viewer.py --build-only   only write viewer.html (Deploy parachute is then unavailable)
      python viewer.py --no-weather   use the weather stored in rocket6dof/config.py, not live data
"""
import copy
import itertools
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from rocket6dof import Config
from rocket6dof.simulate import Simulator
from rocket6dof.frames import quat_to_dcm, NED_TO_ENU
from rocket6dof.smoke import compute_smoke
from rocket6dof.state import TAU, FINS, NOSE_R, CAN_R


def lathe(profile, segments=48):
    """Surface of revolution about body x. profile: list of (station, radius). Returns body-frame triangles."""
    th = np.linspace(0, 2 * math.pi, segments + 1)
    tris = []
    for (s0, r0), (s1, r1) in zip(profile[:-1], profile[1:]):
        for a, b in zip(th[:-1], th[1:]):
            p00 = (-s0, r0 * math.cos(a), r0 * math.sin(a))
            p01 = (-s0, r0 * math.cos(b), r0 * math.sin(b))
            p10 = (-s1, r1 * math.cos(a), r1 * math.sin(a))
            p11 = (-s1, r1 * math.cos(b), r1 * math.sin(b))
            tris += [(p00, p10, p11), (p00, p11, p01)]
    return tris


def prism(poly_top, poly_bottom):
    """Closed prism from two matching convex polygons (lists of 3D points)."""
    n = len(poly_top)
    tris = []
    for i in range(1, n - 1):
        tris.append((poly_top[0], poly_top[i], poly_top[i + 1]))
        tris.append((poly_bottom[0], poly_bottom[i + 1], poly_bottom[i]))
    for i in range(n):
        j = (i + 1) % n
        tris += [(poly_top[i], poly_bottom[i], poly_bottom[j]), (poly_top[i], poly_bottom[j], poly_top[j])]
    return tris


def beam(p0, p1, w, h=None, up=(0, 0, 1)):
    """Rectangular bar from p0 to p1 (any frame)."""
    h = h or w
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    up = np.asarray(up, float)
    if abs(ax @ up) > 0.95:
        up = np.array([1.0, 0, 0])
    e1 = np.cross(ax, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, ax)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    top = [tuple(p1 + a * e1 + b * e2) for a, b in corners]
    bot = [tuple(p0 + a * e1 + b * e2) for a, b in corners]
    return prism(top, bot)


def flat(tris, shift=(0.0, 0.0, 0.0)):
    sh = np.asarray(shift)
    return [round(float(c), 5) for tri in tris for v in tri for c in (np.asarray(v) - sh)]


def make_frames(model, rec, start=0):
    """Convert simulation records to viewer frames.
    0 t | 1-3 booster body origin ENU | 4-12 body->ENU matrix | 13-15 fins | 16 thrust | 17 canopy fill
    18 altitude | 19 speed | 20-22 velocity ENU | 23 fin control | 24 separated | 25-27 nose ENU
    28-30 canopy ENU | 31 cord tension | 32 line tension"""
    T = NED_TO_ENU @ model.C_ne0
    frames = []
    for k in range(start, len(rec["t"])):
        y = rec["state"][k]
        sep = bool(rec["separated"][k])
        C = quat_to_dcm(y[6:10])
        _, cg, _ = model.mass.properties(y[TAU], sep)
        origin = T @ (y[0:3] - C @ cg - model.site_ecef)
        R = T @ C
        v = NED_TO_ENU @ np.asarray(rec["v_ned"][k])
        nose = T @ (y[NOSE_R] - model.site_ecef) if sep else np.zeros(3)
        can = T @ (y[CAN_R] - model.site_ecef) if sep else np.zeros(3)
        frames.append([round(float(rec["t"][k]), 3), *[round(float(c), 3) for c in origin],
                       *[round(float(c), 5) for c in R.flatten()],
                       *[round(float(c), 5) for c in y[FINS]],
                       1 if rec["thrust"][k] > 0 else 0, round(float(rec["canopy_fill"][k]), 3),
                       round(float(rec["altitude"][k]), 2), round(float(rec["speed"][k]), 2),
                       *[round(float(c), 2) for c in v], 1 if rec["control_on"][k] else 0, 1 if sep else 0,
                       *[round(float(c), 3) for c in nose], *[round(float(c), 3) for c in can],
                       round(float(rec["cord_tension"][k]), 1), round(float(rec["line_tension"][k]), 1)])
    return frames


def ground_camera_position(cfg, distance=60.0, eye=1.7):
    """Spectator spot on the playa, side-on to the rail's lean (ENU, metres from the pad)."""
    az = math.radians(cfg.rail.azimuth_deg)
    return [round(-distance * math.cos(az), 2), round(distance * math.sin(az), 2), eye]


def clean_events(events):
    return {k: round(float(v), 2) for k, v in events.items()}


def main():
    cfg = Config()
    if "--no-weather" not in sys.argv:
        try:
            from rocket6dof.weather import fetch, apply_weather
            apply_weather(cfg, fetch(cfg.site.latitude_deg, cfg.site.longitude_deg))
            print(f"Live weather: {cfg.site.weather_source}", flush=True)
        except Exception as exc:  # offline or API change: keep the stored values
            print(f"Live weather unavailable ({exc}); using config values", flush=True)
    cfg.sim.record_interval = 1.0 / 30.0
    print("Simulating flight (a minute or two) ...", flush=True)
    sim = Simulator(cfg)
    snapshots = [(0.0, sim.snapshot())]

    def take_snapshot(s_):
        if not s_.model.recovery.separated and s_.t >= snapshots[-1][0] + 0.25:
            snapshots.append((s_.t, s_.snapshot()))
    sim.run(callback=take_snapshot)
    rec, events, model = sim.rec, sim.events, sim.model
    g = model.geom
    T = NED_TO_ENU @ model.C_ne0                    # ECEF -> local ENU at the pad

    def ecef_to_enu(p):
        return T @ (np.asarray(p) - model.site_ecef)

    # ---- Rocket in body coordinates (origin at the nose tip, x forward) ----
    body = []
    s = np.linspace(0, g.L_nose, 40)
    nose_tris = lathe(list(zip(s, g.nose_radius(s)))) + lathe([(g.L_nose, g.R), (g.L_nose, 0.0)])
    body.append({"color": "#eceff3", "positions": flat(lathe([(g.L_nose, g.R), (g.L, g.R)]))})
    body.append({"color": "#2b2f36", "positions": flat(lathe([(g.L, g.R), (g.L, 0.0)]))})
    s_exit = model.motor.exit_station
    r_exit = math.sqrt(model.motor.exit_area / math.pi)
    r_mount = cfg.internals.motor_tube_outer_diameter / 2
    body.append({"color": "#8d939b", "positions": flat(lathe([(s_exit - 0.012, r_mount - 0.001),
                                                             (s_exit, r_mount - 0.001), (s_exit, r_exit + 0.003)]))})
    body.append({"color": "#3a3a3a", "positions": flat(lathe([(s_exit, r_exit + 0.003), (s_exit + 0.001, r_exit)]))})
    for b in g.button_points:
        radial = np.array([0, math.cos(g.button_roll), math.sin(g.button_roll)])
        body.append({"color": "#111111", "positions": flat(beam(b - 0.008 * radial, b, 0.009))})
    nose_attached = {"color": "#d9482b", "positions": flat(nose_tris)}
    nose_free = {"color": "#d9482b", "positions": flat(nose_tris, model.mass.nose_cg)}   # about its own CG

    fins = []
    for phi in g.fin_roll:
        e = np.array([0.0, math.cos(phi), math.sin(phi)])
        nrm = np.array([0.0, -math.sin(phi), math.cos(phi)])
        hinge = np.array([-model.hinge_station, g.R * math.cos(phi), g.R * math.sin(phi)])

        def fp(st, sp, side):
            p = np.array([-st, (g.R + 0.001 + sp) * math.cos(phi), (g.R + 0.001 + sp) * math.sin(phi)])
            return p + side * g.t_fin / 2 * nrm - hinge
        tris = prism([fp(a, b, 1) for a, b in g.fin_planform], [fp(a, b, -1) for a, b in g.fin_planform])
        fins.append({"hinge": hinge.round(5).tolist(), "axis": e.round(6).tolist(), "positions": flat(tris)})

    # ---- Launch station in ENU ----
    station = []

    def add_enu(color, tris):
        station.append({"color": color, "positions": flat(tris)})

    rail = cfg.rail
    off = model.rail_n * (rail.profile / 2 - 0.004)
    add_enu("#b8bec7", beam(ecef_to_enu(model.rail_base - 0.05 * model.rail_u + off),
                            ecef_to_enu(model.rail_base + rail.length * model.rail_u + off), rail.profile))
    hub = ecef_to_enu(model.rail_base - 0.05 * model.rail_u + model.rail_n * rail.profile)
    for k in range(3):
        a = math.radians(30 + 120 * k)
        foot = np.array([hub[0] + rail.tripod_leg_spread * math.sin(a), hub[1] + rail.tripod_leg_spread * math.cos(a), 0.0])
        add_enu("#5d6672", beam(hub, foot, 0.025))
    y0 = rec["state"][0]
    C0 = quat_to_dcm(y0[6:10])
    _, cg0, _ = model.mass.properties(y0[TAU])
    nozzle0 = ecef_to_enu(y0[0:3] + C0 @ (np.array([-s_exit, 0, 0]) - cg0))
    d = rail.deflector_size / 2
    sq = [(-d, -d), (d, -d), (d, d), (-d, d)]
    c0 = np.array([nozzle0[0], nozzle0[1], 0.0])
    add_enu("#7b828c", prism([tuple(c0 + [a, b, 0.008]) for a, b in sq], [tuple(c0 + [a, b, 0.0]) for a, b in sq]))
    add_enu("#2b2f36", beam(ecef_to_enu(model.rail_base - 0.02 * model.rail_u + off),
                            ecef_to_enu(model.rail_base + off), rail.profile * 1.6))

    frames = make_frames(model, rec)

    rc = cfg.recovery
    data = {"body": body, "nose_attached": nose_attached, "nose_free": nose_free, "fins": fins,
            "station": station, "frames": frames,
            "anchor": [-g.L_nose, 0, 0], "canopy_dp": rc.projected_diameter_ratio * rc.canopy_diameter,
            "exit": [-s_exit, 0, 0], "exit_radius": r_exit, "length": g.L,
            "events": clean_events(events),
            "smoke": compute_smoke(model, rec, events),
            "ground_cam": ground_camera_position(cfg),
            "site": cfg.site.name, "weather": cfg.site.weather_source}
    out = Path(__file__).with_name("viewer.html")
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"Wrote {out}")
    for k in ("apogee_altitude", "separation_speed", "max_line_tension_N", "max_cord_tension_N",
              "booster_landing_speed", "landing"):
        if k in events:
            print(f"  {k}: {events[k]:.2f}")
    if "--build-only" not in sys.argv:
        serve(out, snapshots)


# ---------- local server: re-simulates when the parachute is deployed by hand ----------
# Plain GET requests only (no request bodies, no browser preflight), answered with JSON.
#   /api/ping                 {"service": "rocket6dof", "deploy_until": t}
#   /api/deploy?t=T           starts a re-simulation with the ejection charge fired at T -> {"job": id}
#   /api/job?id=N&since=K     new frames, events, smoke (when done), progress
class Job:
    def __init__(self):
        self.frames, self.events, self.smoke = [], {}, None
        self.done, self.error, self.sim_time = False, None, 0.0
        self.lock = threading.Lock()


def run_branch(job, snapshots, t_deploy):
    try:
        _, snap = max((sn for sn in snapshots if sn[0] <= t_deploy), key=lambda sn: sn[0])
        sim = copy.deepcopy(snap)
        sim.run(until=t_deploy)
        sim.force_ejection(sim.t)
        idx = [0]

        def push(s_):
            if len(s_.rec["t"]) - idx[0] >= 15 or s_.done:
                new = make_frames(s_.model, s_.rec, idx[0])
                idx[0] = len(s_.rec["t"])
                with job.lock:
                    job.frames.extend(f for f in new if f[0] >= t_deploy - 1e-9)
                    job.events = clean_events(s_.events)
                    job.sim_time = s_.t
        sim.run(callback=push)
        push(sim)
        job.smoke = compute_smoke(sim.model, sim.rec, sim.events, t_from=t_deploy)
    except Exception as exc:  # report to the page instead of failing silently
        job.error = repr(exc)
    job.done = True


def serve(html_path, snapshots, port=8765):
    from urllib.parse import urlparse, parse_qs
    jobs, ids = {}, itertools.count(1)
    t_last = snapshots[-1][0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")    # page may be opened as a file
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/viewer.html"):
                    self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/ping":
                    self._send(200, {"service": "rocket6dof", "deploy_until": t_last})
                elif url.path == "/api/deploy":
                    t = float(q.get("t", 0.0))
                    if t > t_last + 0.25:
                        return self._send(400, {"error": "the parachute is already out at that time"})
                    jid = next(ids)
                    jobs[jid] = Job()
                    threading.Thread(target=run_branch, args=(jobs[jid], snapshots, max(t, 0.0)),
                                     daemon=True).start()
                    self._send(200, {"job": jid})
                elif url.path == "/api/job":
                    job = jobs.get(int(q.get("id", 0)))
                    if job is None:
                        return self._send(404, {"error": "unknown job"})
                    n = int(q.get("since", 0))
                    with job.lock:
                        self._send(200, {"frames": job.frames[n:], "events": job.events, "done": job.done,
                                         "smoke": job.smoke if job.done else None,
                                         "sim_time": job.sim_time, "error": job.error})
                else:
                    self._send(404, {"error": f"no such address {url.path}"})
            except Exception as exc:
                self._send(500, {"error": repr(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Simulator running. Opening {url} (keep this window open; Ctrl+C to stop)", flush=True)
    if "--no-browser" not in sys.argv:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rocket Flight Viewer</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #a9c6e0; font: 13px system-ui, sans-serif; }
  canvas#c { display: block; width: 100%; height: 100%; }
  #ui { position: fixed; left: 12px; top: 12px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; z-index: 2; }
  button { font: 600 14px system-ui, sans-serif; padding: 8px 16px; border-radius: 6px; border: 1px solid #555; cursor: pointer; background: #fff; }
  #launch { background: #d9482b; color: #fff; border-color: #a5321c; }
  #launch:disabled, #deploy:disabled { opacity: .5; cursor: default; }
  #deploy { background: #f2c14e; border-color: #b98c1e; }
  #status { background: rgba(255,255,255,.9); padding: 6px 8px; border-radius: 6px; display: none; }
  label { display: flex; align-items: center; gap: 4px; background: rgba(255,255,255,.85); padding: 6px 8px; border-radius: 6px; }
  #hud { position: fixed; right: 12px; top: 12px; background: rgba(255,255,255,.9); padding: 10px 12px; z-index: 2;
         border-radius: 6px; font: 12px ui-monospace, Consolas, monospace; line-height: 1.6; color: #1f2430; min-width: 240px; }
  #hint { position: fixed; left: 12px; bottom: 10px; color: #2b2f36; z-index: 2; }
  #charts { position: fixed; left: 12px; top: 64px; width: 380px; background: rgba(255,255,255,.96); border-radius: 8px;
            box-shadow: 0 4px 18px rgba(0,0,0,.25); z-index: 3; overflow: hidden; }
  #charts header { cursor: move; padding: 7px 10px; background: #2b2f36; color: #fff; font-weight: 600;
                   display: flex; justify-content: space-between; align-items: center; user-select: none; }
  #charts header button { padding: 2px 8px; font-size: 12px; }
  #charts .cap { font: 11px system-ui, sans-serif; color: #444; padding: 4px 10px 0; }
  #alt { display: block; width: 380px; height: 170px; }
  #path { display: block; width: 380px; height: 260px; cursor: grab; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="ui">
  <button id="launch">Launch</button>
  <button id="reset">Reset</button>
  <button id="deploy" disabled>Deploy parachute</button>
  <button id="chartsBtn">Charts</button>
  <label>Speed <select id="speed"><option value="0.25">0.25x</option><option value="1" selected>1x</option><option value="4">4x</option><option value="10">10x</option></select></label>
  <label>View <select id="view"><option value="orbit">Orbit</option><option value="ground">Ground camera</option></select></label>
  <label><input type="checkbox" id="follow" checked> Follow</label>
  <span id="status"></span>
</div>
<div id="charts" hidden>
  <header id="chartsHead"><span>Flight data</span><button id="chartsClose">Close</button></header>
  <div class="cap">Altitude above ground (m) vs time (s)</div>
  <canvas id="alt" width="760" height="340"></canvas>
  <div class="cap" id="pathCap">Flight path in 3D (drag to rotate, scroll to zoom)</div>
  <canvas id="path"></canvas>
</div>
<div id="hud"></div>
<div id="hint"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D = __DATA__;
let F = D.frames, E = D.events, SMOKE = D.smoke;
const NOM_F = F, NOM_E = E, NOM_SMOKE = SMOKE;
const enu3 = (e, n, u) => new THREE.Vector3(e, u, -n);                    // ENU -> three.js (y up)
const P = new THREE.Matrix4().set(1,0,0,0, 0,0,1,0, 0,-1,0,0, 0,0,0,1);
document.getElementById('hint').textContent =
  `${D.site} | weather: ${D.weather} | drag to rotate, scroll to zoom | flight simulated by rocket6dof (viewer.py)`;

// ---------- Main scene ----------
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa9c6e0);
scene.fog = new THREE.Fog(0xcbd6dc, 2500, 15000);
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 30000);
const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.15;
controls.maxPolarAngle = Math.PI * 0.495;

scene.add(new THREE.HemisphereLight(0xdfeaf5, 0x9a8466, 0.6));
const sun = new THREE.DirectionalLight(0xfff1dc, 1.0);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.02;
Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.1, far: 30 });
scene.add(sun, sun.target);

// Nevada playa: tan-grey alkali mud with shallow polygon cracks. A soft, low-contrast 12 m tile
// (mipmapped, anisotropic) plus large-scale colour variation baked into the vertices, so the
// surface does not shimmer when the camera moves.
function playaTexture() {
  const N = 1024, cv = document.createElement('canvas');
  cv.width = cv.height = N;
  const g = cv.getContext('2d');
  g.fillStyle = '#ad9b7f';
  g.fillRect(0, 0, N, N);
  for (let i = 0; i < 2500; i++) {
    const v = Math.random() * 18 - 9, r = 6 + Math.random() * 24;
    g.fillStyle = `rgba(${173 + v | 0},${155 + v | 0},${127 + v | 0},0.25)`;
    g.beginPath(); g.arc(Math.random() * N, Math.random() * N, r, 0, Math.PI * 2); g.fill();
  }
  const n = 6, c = N / n, pts = [];
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++)
    pts.push([i * c + (i % n ? (Math.random() - 0.5) * c * 0.5 : 0), j * c + (j % n ? (Math.random() - 0.5) * c * 0.5 : 0)]);
  const Pt = (i, j) => pts[i * (n + 1) + j];
  g.strokeStyle = 'rgba(120,102,80,0.35)';
  g.lineWidth = 5;
  g.lineCap = 'round';
  const crack = (a, b) => {
    g.beginPath(); g.moveTo(a[0], a[1]);
    for (let k = 1; k <= 5; k++) {
      const f = k / 5, j = k < 5 ? 14 : 0;
      g.lineTo(a[0] + (b[0] - a[0]) * f + (Math.random() - 0.5) * j, a[1] + (b[1] - a[1]) * f + (Math.random() - 0.5) * j);
    }
    g.stroke();
  };
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
    if (i < n) crack(Pt(i, j), Pt(i + 1, j));
    if (j < n) crack(Pt(i, j), Pt(i, j + 1));
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(30000 / 12, 30000 / 12);
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  tex.encoding = THREE.sRGBEncoding;
  return tex;
}
const groundGeo = new THREE.PlaneGeometry(30000, 30000, 150, 150);
const gc = [];
for (let i = 0; i < groundGeo.attributes.position.count; i++) {
  const x = groundGeo.attributes.position.getX(i), y = groundGeo.attributes.position.getY(i);
  const v = 0.93 + 0.05 * Math.sin(x / 900 + 1.3) * Math.cos(y / 1300) + 0.03 * Math.sin((x + y) / 380);
  gc.push(v, v, v * 0.99);
}
groundGeo.setAttribute('color', new THREE.Float32BufferAttribute(gc, 3));
const ground = new THREE.Mesh(groundGeo, new THREE.MeshStandardMaterial({ map: playaTexture(), vertexColors: true, roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

(function mountains() {   // distant ranges, illustrative only
  const segs = 360, pos = [];
  const h = a => 380 + 320 * Math.sin(3 * a + 1) + 210 * Math.sin(7 * a + 2.3) + 120 * Math.sin(17 * a) + 60 * Math.sin(41 * a + 0.7);
  for (let i = 0; i < segs; i++) {
    const a0 = i / segs * Math.PI * 2, a1 = (i + 1) / segs * Math.PI * 2, r = 14000;
    const h0 = Math.max(h(a0), 40), h1 = Math.max(h(a1), 40);
    const p = (a, y) => [r * Math.cos(a), y, r * Math.sin(a)];
    pos.push(...p(a0, 0), ...p(a1, 0), ...p(a1, h1), ...p(a0, 0), ...p(a1, h1), ...p(a0, h0));
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.computeVertexNormals();
  scene.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: 0x8a8378, side: THREE.DoubleSide })));
})();

function mesh(positions, color, enu) {
  const pos = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    if (enu) { pos[i] = positions[i]; pos[i + 1] = positions[i + 2]; pos[i + 2] = -positions[i + 1]; }
    else { pos[i] = positions[i]; pos[i + 1] = positions[i + 1]; pos[i + 2] = positions[i + 2]; }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.15, side: THREE.DoubleSide }));
  m.castShadow = true; m.receiveShadow = true;
  return m;
}
for (const p of D.station) scene.add(mesh(p.positions, p.color, true));

// Smoke: one soft sprite per puff, position, radius and opacity interpolated from the physics samples
const smokeTex = (() => {
  const cv = document.createElement('canvas'); cv.width = cv.height = 64;
  const g = cv.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.6)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
})();
const smokeGroup = new THREE.Group();
scene.add(smokeGroup);
let smokeSprites = [];
function rebuildSmoke() {
  smokeSprites.forEach(sp => { smokeGroup.remove(sp); sp.material.dispose(); });
  smokeSprites = SMOKE.map(p => {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, color: p.c === 'b' ? 0x8f8a83 : 0xf1f0ec,
                                                            transparent: true, depthWrite: false, fog: true }));
    sp.visible = false; sp.userData.p = p; sp.userData.i = 0;
    smokeGroup.add(sp);
    return sp;
  });
}
function updateSmoke(t) {
  for (const sp of smokeSprites) {
    const p = sp.userData.p, S = p.s, age = t - p.t0;
    if (age < 0 || age > S[S.length - 1][0]) { sp.visible = false; continue; }
    let i = Math.min(sp.userData.i, S.length - 2);
    while (i > 0 && S[i][0] > age) i--;
    while (i < S.length - 2 && S[i + 1][0] <= age) i++;
    sp.userData.i = i;
    const a = S[i], b = S[i + 1] || a, f = b[0] > a[0] ? (age - a[0]) / (b[0] - a[0]) : 0;
    const L = j => a[j] + (b[j] - a[j]) * f;
    sp.position.copy(enu3(L(1), L(2), L(3)));
    sp.scale.setScalar(2.4 * L(4));
    sp.material.opacity = L(5);
    sp.visible = true;
  }
}
rebuildSmoke();

// Booster (children in body coordinates; group matrix = body -> three.js)
const rocket = new THREE.Group();
rocket.matrixAutoUpdate = false;
scene.add(rocket);
for (const p of D.body) rocket.add(mesh(p.positions, p.color, false));
const noseOn = mesh(D.nose_attached.positions, D.nose_attached.color, false);
rocket.add(noseOn);
const finPivots = D.fins.map(f => {
  const pivot = new THREE.Object3D();
  pivot.position.set(...f.hinge);
  pivot.add(mesh(f.positions, '#1f2a44', false));
  pivot.userData.axis = new THREE.Vector3(...f.axis);
  rocket.add(pivot);
  return pivot;
});
const flameGeo = new THREE.ConeGeometry(D.exit_radius * 1.6, 0.35, 20, 1, true);
flameGeo.translate(0, 0.175, 0);
flameGeo.rotateZ(Math.PI / 2);
const flame = new THREE.Mesh(flameGeo, new THREE.MeshBasicMaterial({ color: 0xffa040, transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
flame.position.set(...D.exit);
rocket.add(flame);

// Separated bodies
const noseFree = new THREE.Group();
noseFree.matrixAutoUpdate = false;
noseFree.add(mesh(D.nose_free.positions, D.nose_free.color, false));
scene.add(noseFree);
const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.5, 28, 12, 0, Math.PI * 2, 0, Math.PI / 2.2),
  new THREE.MeshStandardMaterial({ color: 0xe8492e, side: THREE.DoubleSide, roughness: 0.8 }));
canopy.castShadow = true;
const packed = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.12, 12), new THREE.MeshStandardMaterial({ color: 0xe8492e }));
scene.add(canopy, packed);
const ropeGeo = new THREE.BufferGeometry();
ropeGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((1 + 8) * 6), 3));
const ropes = new THREE.LineSegments(ropeGeo, new THREE.LineBasicMaterial({ color: 0x222222 }));
ropes.frustumCulled = false;
scene.add(ropes);

const B = new THREE.Matrix4(), M = new THREE.Matrix4(), NM = new THREE.Matrix4();
const Y = new THREE.Vector3(0, 1, 0), X = new THREE.Vector3(1, 0, 0);
const vA = new THREE.Vector3(), vN = new THREE.Vector3(), vC = new THREE.Vector3(), dir = new THREE.Vector3(), tmp = new THREE.Vector3();
const lerp = (a, b, f) => a + (b - a) * f;
function frameAt(t) {
  let lo = 0, hi = F.length - 1;
  if (t >= F[hi][0]) return [F[hi], F[hi], 0];
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (F[mid][0] <= t) lo = mid; else hi = mid; }
  return [F[lo], F[hi], (t - F[lo][0]) / (F[hi][0] - F[lo][0])];
}
const L3 = (a, b, f, i) => enu3(lerp(a[i], b[i], f), lerp(a[i + 1], b[i + 1], f), lerp(a[i + 2], b[i + 2], f));

function phaseText(t, k) {
  if (t <= 0) return 'On the pad';
  if (k[16]) return 'Boost';
  if (!k[24]) return E.apogee && t >= E.apogee ? 'Apogee, waiting for ejection' : 'Coast';
  if (E.line_stretch === undefined || t < E.line_stretch) return 'Ejection, nose cone separating';
  if (k[17] < 1) return 'Parachute inflating';
  if (E.landing && t >= (E.booster_landing || 1e9)) return 'Landed';
  return 'Descent under parachute';
}

function pose(t) {
  const [a, b, f] = frameAt(t);
  const k = f < 0.5 ? a : b;
  const R = k.slice(4, 13);
  B.set(R[0], R[1], R[2], lerp(a[1], b[1], f), R[3], R[4], R[5], lerp(a[2], b[2], f),
        R[6], R[7], R[8], lerp(a[3], b[3], f), 0, 0, 0, 1);
  M.multiplyMatrices(P, B);
  rocket.matrix.copy(M);
  rocket.matrixWorldNeedsUpdate = true;
  finPivots.forEach((p, i) => p.quaternion.setFromAxisAngle(p.userData.axis, -lerp(a[13 + i], b[13 + i], f)));
  flame.visible = k[16] === 1;
  if (flame.visible) flame.scale.set(1, 0.8 + 0.4 * Math.random(), 1);

  const sep = k[24] === 1 && a[24] === 1;
  noseOn.visible = !sep;
  noseFree.visible = canopy.visible = ropes.visible = packed.visible = sep;
  if (sep) {
    vA.set(...D.anchor).applyMatrix4(M);
    vN.copy(L3(a, b, f, 25));
    vC.copy(L3(a, b, f, 28));
    // Nose cone: lines pull at its base, so the tip points away from the canopy
    dir.subVectors(vN, vC);
    if (dir.lengthSq() < 1e-6) dir.subVectors(vN, vA);
    dir.normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(X, dir);
    NM.compose(vN, q, new THREE.Vector3(1, 1, 1));
    noseFree.matrix.copy(NM);
    noseFree.matrixWorldNeedsUpdate = true;
    // Canopy: open towards the nose cone, size from the inflation fraction
    const fill = lerp(a[17], b[17], f);
    canopy.visible = fill > 0.02;
    packed.visible = !canopy.visible;
    packed.position.copy(vC);
    const up = tmp.subVectors(vC, vN);
    if (up.lengthSq() < 1e-6) up.copy(Y);
    up.normalize();
    canopy.position.copy(vC);
    canopy.quaternion.setFromUnitVectors(Y, up);
    const rad = Math.max(D.canopy_dp * fill, 0.05);
    canopy.scale.set(rad, rad * 0.9, rad);
    const arr = ropeGeo.attributes.position.array;
    arr.set([vA.x, vA.y, vA.z, vN.x, vN.y, vN.z], 0);
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(0.5 * Math.cos(i * Math.PI / 4) * 0.98, 0.05, 0.5 * Math.sin(i * Math.PI / 4) * 0.98)
        .multiply(canopy.scale).applyQuaternion(canopy.quaternion).add(vC);
      arr.set([vN.x, vN.y, vN.z, p.x, p.y, p.z], (i + 1) * 6);
    }
    ropeGeo.attributes.position.needsUpdate = true;
  }
  const deg = v => (v * 180 / Math.PI).toFixed(2).padStart(6);
  document.getElementById('hud').innerHTML =
    `T+ ${t.toFixed(2)} s<br>${phaseText(t, k)}<br>Altitude ${k[18].toFixed(1)} m<br>Speed ${k[19].toFixed(1)} m/s<br>` +
    `Fin control: ${k[23] ? 'active' : 'centred'}<br>Fins ${deg(k[13])} ${deg(k[14])} ${deg(k[15])} deg<br>` +
    (sep ? `Shock cord ${k[31].toFixed(0)} N, lines ${k[32].toFixed(0)} N<br>` : '') +
    (E.apogee_altitude ? `Apogee ${E.apogee_altitude.toFixed(0)} m at ${E.apogee.toFixed(1)} s<br>` : '') +
    (E.max_line_tension_N ? `Opening shock ${E.max_line_tension_N.toFixed(0)} N (lines), ${E.max_cord_tension_N.toFixed(0)} N (cord)` : '');
  return k;
}

const centre = new THREE.Vector3();
function rocketCentre() { return centre.set(-D.length / 2, 0, 0).applyMatrix4(rocket.matrix).clone(); }
function placeCamera() {
  controls.target.copy(rocketCentre());
  camera.position.copy(controls.target).add(new THREE.Vector3(2.2, 0.6, 2.2));
}

// ---------- Charts panel ----------
const panel = document.getElementById('charts');
document.getElementById('chartsBtn').onclick = () => { panel.hidden = !panel.hidden; if (!panel.hidden) { rebuildCharts(); drawAlt(simT); } };
document.getElementById('chartsClose').onclick = () => { panel.hidden = true; };
(function draggable() {
  const head = document.getElementById('chartsHead');
  let dx = 0, dy = 0, drag = false;
  head.addEventListener('pointerdown', e => {
    if (e.target.tagName === 'BUTTON') return;
    drag = true; dx = e.clientX - panel.offsetLeft; dy = e.clientY - panel.offsetTop; head.setPointerCapture(e.pointerId);
  });
  head.addEventListener('pointermove', e => {
    if (!drag) return;
    panel.style.left = Math.min(Math.max(e.clientX - dx, 0), innerWidth - 60) + 'px';
    panel.style.top = Math.min(Math.max(e.clientY - dy, 0), innerHeight - 40) + 'px';
  });
  head.addEventListener('pointerup', () => { drag = false; });
})();

const altCv = document.getElementById('alt'), ac = altCv.getContext('2d');
let tMax = 0, hMax = 1;
const niceStep = v => { const p = Math.pow(10, Math.floor(Math.log10(v))); return [1, 2, 5, 10].map(m => m * p).find(s => v / s <= 6); };
function drawAlt(t) {
  const W = altCv.width, H = altCv.height, L = 70, Rm = 16, T = 16, Bm = 44;
  const X = v => L + (W - L - Rm) * v / tMax, Yv = v => H - Bm - (H - Bm - T) * v / (hMax * 1.05);
  ac.clearRect(0, 0, W, H);
  ac.font = '20px system-ui'; ac.fillStyle = '#555'; ac.strokeStyle = '#e3e3e3'; ac.lineWidth = 2;
  const hs = niceStep(hMax * 1.05), ts = niceStep(tMax);
  for (let v = 0; v <= hMax * 1.05; v += hs) {
    ac.beginPath(); ac.moveTo(L, Yv(v)); ac.lineTo(W - Rm, Yv(v)); ac.stroke();
    ac.textAlign = 'right'; ac.fillText(v.toFixed(0), L - 8, Yv(v) + 7);
  }
  for (let v = 0; v <= tMax; v += ts) { ac.textAlign = 'center'; ac.fillText(v.toFixed(0), X(v), H - Bm + 26); }
  ac.strokeStyle = '#2f6fd6'; ac.lineWidth = 3; ac.beginPath();
  F.forEach((r, i) => i ? ac.lineTo(X(r[0]), Yv(r[18])) : ac.moveTo(X(r[0]), Yv(r[18])));
  ac.stroke();
  const mark = (te, label) => { if (te === undefined || te > F[F.length - 1][0]) return; ac.fillStyle = '#d9482b'; ac.beginPath();
    const r = frameAt(te)[0];
    ac.arc(X(te), Yv(r[18]), 6, 0, Math.PI * 2); ac.fill(); ac.fillStyle = '#333'; ac.textAlign = 'left'; ac.fillText(label, X(te) + 9, Yv(r[18]) - 8); };
  mark(E.burnout, 'burnout'); mark(E.apogee, 'apogee ' + (E.apogee_altitude || 0).toFixed(0) + ' m'); mark(E.ejection, 'ejection');
  ac.strokeStyle = '#d9482b'; ac.lineWidth = 2; ac.beginPath(); ac.moveTo(X(t), T); ac.lineTo(X(t), H - Bm); ac.stroke();
}

const pathCv = document.getElementById('path');
const pr = new THREE.WebGLRenderer({ canvas: pathCv, antialias: true });
pr.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
pr.setSize(380, 260, false);
const ps = new THREE.Scene();
ps.background = new THREE.Color(0xf4f5f7);
const pcam = new THREE.PerspectiveCamera(45, 380 / 260, 1, 100000);
const pctl = new THREE.OrbitControls(pcam, pathCv);
pctl.enableDamping = true;
const pathGroup = new THREE.Group();
ps.add(pathGroup);
const marker = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshBasicMaterial({ color: 0xd9482b }));
ps.add(marker);
let camPlaced = false;
function rebuildCharts() {
  tMax = F[F.length - 1][0];
  hMax = Math.max(...F.map(r => r[18]), 1);
  pathGroup.children.slice().forEach(c => { pathGroup.remove(c); c.geometry && c.geometry.dispose(); });
  const pts = F.map(r => enu3(r[1], r[2], r[3]));
  const span = Math.max(...pts.map(p => Math.max(Math.abs(p.x), Math.abs(p.z))), 10) * 2.2;
  const gridStep = niceStep(Math.max(span, hMax) / 4);
  const gridSize = Math.ceil(Math.max(span, hMax * 0.8) / gridStep) * gridStep;
  pathGroup.add(new THREE.GridHelper(gridSize, gridSize / gridStep, 0x9aa0a8, 0xd0d3d8));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const colors = [];
  F.forEach(r => { const c = r[16] ? [0.85, 0.28, 0.17] : r[24] ? [0.95, 0.6, 0.1] : [0.18, 0.43, 0.84]; colors.push(...c); });
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  pathGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true })));
  pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(p.x, 0, p.z))),
    new THREE.LineBasicMaterial({ color: 0xb0b4ba })));
  pathGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, 0), gridStep, 0x2b2f36));
  marker.scale.setScalar(gridSize / 80);
  if (!camPlaced) {
    pctl.target.set(0, hMax / 2, 0);
    pcam.position.set(gridSize * 0.8, hMax * 0.9, gridSize * 0.8);
    camPlaced = true;
  }
  const last = F[F.length - 1];
  document.getElementById('pathCap').textContent =
    `3D flight path. Grid ${gridStep} m, arrow points north. Red boost, blue coast, orange recovery. ` +
    `Lands ${Math.hypot(last[1], last[2]).toFixed(0)} m from the pad.`;
}
rebuildCharts();

// ---------- Playback ----------
let playing = false, simT = 0, lastT = 0;
let branch = null;   // { job, tClick, since, done }
const launchBtn = document.getElementById('launch'), deployBtn = document.getElementById('deploy');
const statusEl = document.getElementById('status');
// The simulator (python viewer.py) re-runs the flight when the parachute is deployed by hand.
// Always addressed directly, so it works whether this page is served by it or opened as a file.
const API = 'http://127.0.0.1:8765';
const NO_SERVER = 'Deploying by hand needs the simulator: run "python viewer.py" and keep its window open.';
let server = null;
async function getJSON(path) {
  const r = await fetch(API + path);
  const type = r.headers.get('Content-Type') || '';
  if (!type.includes('application/json'))
    throw new Error(`${API} answered with ${type || 'no content type'} (HTTP ${r.status}); another program may be using port 8765`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
function findServer() {
  getJSON('/api/ping').then(j => {
    if (j.service !== 'rocket6dof') throw new Error('not the rocket simulator');
    server = j; deployBtn.title = '';
  }).catch(() => { server = null; deployBtn.title = NO_SERVER; setTimeout(findServer, 3000); });
}
findServer();

function setStatus(text) { statusEl.textContent = text; statusEl.style.display = text ? 'inline-block' : 'none'; }
function separatedAt(t) { return frameAt(t)[0][24] === 1; }
function updateDeployButton() {
  deployBtn.disabled = branch !== null || separatedAt(simT) || (server && simT > server.deploy_until + 0.25)
    || (E.landing !== undefined && simT >= E.landing);
}

deployBtn.onclick = async () => {
  if (deployBtn.disabled) return;
  if (!server) { setStatus(NO_SERVER); return; }
  branch = { job: null, tClick: simT, since: 0, done: false };
  updateDeployButton();
  setStatus('Simulating the deployment ...');
  try {
    branch.job = (await getJSON(`/api/deploy?t=${branch.tClick.toFixed(4)}`)).job;
    poll();
  } catch (err) { setStatus('Deploy failed: ' + err.message); branch = null; updateDeployButton(); }
};

async function poll() {
  if (!branch || branch.done) return;
  let j;
  try { j = await getJSON(`/api/job?id=${branch.job}&since=${branch.since}`); }
  catch (err) { setStatus('Lost the simulator: ' + err.message); branch.done = true; return; }
  if (j.error) { setStatus('Simulation error: ' + j.error); branch.done = true; return; }
  if (j.frames.length) {
    F = branch.since === 0 ? F.filter(row => row[0] < branch.tClick).concat(j.frames) : F.concat(j.frames);
    branch.since += j.frames.length;
    E = Object.assign({}, j.events);
    if (!panel.hidden) rebuildCharts();
  }
  branch.done = j.done;
  if (j.done) {
    SMOKE = NOM_SMOKE.filter(p => p.t0 < branch.tClick).concat(j.smoke || []);
    rebuildSmoke();
    if (!panel.hidden) rebuildCharts();
    setStatus('');
  } else {
    setStatus(`Simulating the deployment ... ${j.sim_time.toFixed(1)} s computed`);
    setTimeout(poll, 400);
  }
}

// ---------- Views ----------
// Orbit: free mouse camera that follows the rocket. Ground camera: a spectator on the playa at eye
// height, panning to keep the rocket centred and zooming like a telephoto (scroll to change framing).
const viewSel = document.getElementById('view');
const GROUND = enu3(...D.ground_cam);
let frameSize = 12;   // metres of scene across the ground camera's view
canvas.addEventListener('wheel', e => {
  if (viewSel.value === 'ground') { frameSize = Math.min(Math.max(frameSize * (e.deltaY > 0 ? 1.12 : 1 / 1.12), 1.5), 400); }
}, { passive: true });
viewSel.onchange = () => {
  const ground = viewSel.value === 'ground';
  controls.enabled = !ground;
  if (!ground) { camera.fov = 40; camera.updateProjectionMatrix(); placeCamera(); }
};
function subjectCentre(k) {
  // Before ejection the rocket; afterwards the midpoint of booster and parachute, so both stay in view
  const c = rocketCentre();
  if (k[24] === 1) c.add(enu3(k[28], k[29], k[30])).multiplyScalar(0.5);
  return c;
}

pose(0);
placeCamera();
launchBtn.onclick = () => { if (!playing) { playing = true; lastT = performance.now(); launchBtn.disabled = true; } };
document.getElementById('reset').onclick = () => {
  playing = false; simT = 0; launchBtn.disabled = false;
  if (branch) { branch.done = true; branch = null; F = NOM_F; E = NOM_E; SMOKE = NOM_SMOKE; rebuildSmoke(); setStatus(''); rebuildCharts(); }
  pose(0); if (viewSel.value === 'orbit') placeCamera(); updateDeployButton();
};

function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener('resize', resize);
resize();

(function loop(now) {
  const end = F[F.length - 1][0];
  const waiting = branch && !branch.done;
  if (playing) {
    const limit = waiting ? Math.max(end - 0.1, branch.tClick) : end;
    simT = Math.min(simT + (now - lastT) / 1000 * parseFloat(document.getElementById('speed').value), limit);
    if (!waiting && simT >= end) playing = false;
  }
  lastT = now;
  const before = rocketCentre();
  const k = pose(simT);
  updateSmoke(simT);
  if (viewSel.value === 'ground') {
    const target = subjectCentre(k);
    camera.position.copy(GROUND);
    camera.lookAt(target);
    const d = GROUND.distanceTo(target);
    camera.fov = Math.min(Math.max(2 * Math.atan(frameSize / 2 / d) * 180 / Math.PI, 0.5), 60);
    camera.updateProjectionMatrix();
    controls.target.copy(target);
  } else {
    if (document.getElementById('follow').checked && (playing || simT > 0)) {
      const d = rocketCentre().sub(before);
      controls.target.add(d);
      camera.position.add(d);
    }
    controls.update();
  }
  sun.position.copy(controls.target).add(new THREE.Vector3(4, 8, 3));
  sun.target.position.copy(controls.target);
  renderer.render(scene, camera);
  if (!panel.hidden) {
    marker.position.copy(enu3(k[1], k[2], k[3]));
    drawAlt(simT);
    pctl.update();
    pr.render(ps, pcam);
  }
  updateDeployButton();
  requestAnimationFrame(loop);
})(performance.now());
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
