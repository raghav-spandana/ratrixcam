"""
ratrixcam_web_server.py
-----------------------
A Flask-based web UI for ratrixcam.

Run with:
    pip install flask
    python web_server.py -c /path/to/config.json

Then open http://localhost:5000 in your browser.

How it works
------------
- Reads config.json (same file ratrix_multicam.py uses).
- Spawns one ratrix_cam_server subprocess per camera, exactly like multicam does.
- The cam_server writes preview JPEG stills to stills_path/ every preview_interval seconds.
- The web server reads those files and streams them to the browser as MJPEG (/stream/<name>).
- REST endpoints let the frontend start/stop individual cameras or all at once.
"""

import argparse
import multiprocessing
import os
import shutil
import signal
import sys
import time
from multiprocessing import Process
from multiprocessing.synchronize import Event
from threading import Thread
from typing import Optional

from flask import Flask, Response, jsonify, send_from_directory

# ── make sure the ratrixcam folder is on the path ────────────────────────────
RATRIXCAM_DIR = os.path.dirname(os.path.abspath(__file__))
# If web_server.py lives inside ratrixcam/, this is already correct.
# If you placed it elsewhere, set RATRIXCAM_DIR to the repo root explicitly:
#   RATRIXCAM_DIR = "/Users/username/Desktop/ratrixcam"
sys.path.insert(0, RATRIXCAM_DIR)

import ratrix_cam_server                           # noqa: E402  (after sys.path patch)
from ratrix_utils import Config, ensure_dir_exists, load_settings, reset_stills, still_path  # noqa: E402

# ── globals (populated after config loads) ───────────────────────────────────
app: Flask = Flask(__name__, static_folder="static", template_folder=".")
config: Optional[Config] = None
stop_events: dict[str, Event] = {}          # cam_name → multiprocessing.Event
cam_processes: dict[str, Optional[Process]] = {}   # cam_name → Process | None
cam_index: dict[str, int] = {}              # cam_name → device index (0-based)

BOUNDARY = b"--frame"


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _still_file(cam_name: str) -> str:
    assert config is not None
    return still_path(config.stills_path, cam_name)


def _is_alive(cam_name: str) -> bool:
    p = cam_processes.get(cam_name)
    return p is not None and p.is_alive()


def _run_without_handlers(cfg: Config, idx: int, ev: Event):
    """Entry point for camera subprocess — ignores SIGINT so parent handles it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ratrix_cam_server.run(cfg, idx, ev)


def _start_camera(cam_name: str):
    """Start a single camera process (no-op if already running)."""
    if _is_alive(cam_name):
        return
    assert config is not None

    # fresh stop_event for this run
    ev: Event = multiprocessing.Event()
    stop_events[cam_name] = ev

    idx = cam_index[cam_name]
    p = Process(target=_run_without_handlers, args=(config, idx, ev), daemon=True)
    p.start()
    cam_processes[cam_name] = p
    print(f"[web] started camera {cam_name} (device {idx})")


def _stop_camera(cam_name: str, timeout: float = 15.0):
    """Signal a camera process to stop and wait for it."""
    ev = stop_events.get(cam_name)
    if ev:
        ev.set()
    p = cam_processes.get(cam_name)
    if p and p.is_alive():
        p.join(timeout=timeout)
        if p.is_alive():
            p.kill()
    cam_processes[cam_name] = None
    # put offline placeholder back
    assert config is not None
    blank = config.blank_image
    dest = _still_file(cam_name)
    if os.path.exists(blank):
        shutil.copyfile(blank, dest)
    print(f"[web] stopped camera {cam_name}")


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG streaming
# ─────────────────────────────────────────────────────────────────────────────

def _mjpeg_generator(cam_name: str):
    """
    Yields MJPEG frames by re-reading the still JPEG that the cam_server
    writes periodically.  This adds zero extra CPU cost — no second decode.
    """
    assert config is not None
    path = _still_file(cam_name)
    blank = config.blank_image
    interval = 1.0 / 4  # poll at ~4 Hz — smooth enough for a status display

    while True:
        src = path if os.path.exists(path) else blank
        try:
            with open(src, "rb") as f:
                data = f.read()
        except OSError:
            data = b""

        if data:
            yield (
                BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                + data + b"\r\n"
            )
        time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/stream/<cam_name>")
def stream(cam_name: str):
    if config is None or cam_name not in cam_index:
        return "Camera not found", 404
    return Response(
        _mjpeg_generator(cam_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    """Return current state of every camera + global config info."""
    assert config is not None
    cameras = []
    for cam_cfg in config.cameras:
        name = cam_cfg.name
        cameras.append({
            "name": name,
            "row": cam_cfg.row,
            "col": cam_cfg.col if hasattr(cam_cfg, "col") else 1,
            "running": _is_alive(name),
        })
    return jsonify({
        "study_label": config.study_label,
        "rack_name": config.rack_name,
        "save_path": config.save_path,
        "cameras": cameras,
    })


@app.route("/api/start/<cam_name>", methods=["POST"])
def api_start(cam_name: str):
    if cam_name not in cam_index:
        return jsonify({"error": "unknown camera"}), 404
    _start_camera(cam_name)
    return jsonify({"status": "started", "camera": cam_name})


@app.route("/api/stop/<cam_name>", methods=["POST"])
def api_stop(cam_name: str):
    if cam_name not in cam_index:
        return jsonify({"error": "unknown camera"}), 404
    Thread(target=_stop_camera, args=(cam_name,), daemon=True).start()
    return jsonify({"status": "stopping", "camera": cam_name})


@app.route("/api/start_all", methods=["POST"])
def api_start_all():
    assert config is not None
    for cam_cfg in config.cameras:
        _start_camera(cam_cfg.name)
    return jsonify({"status": "all started"})


@app.route("/api/stop_all", methods=["POST"])
def api_stop_all():
    assert config is not None
    for cam_cfg in config.cameras:
        Thread(target=_stop_camera, args=(cam_cfg.name,), daemon=True).start()
    return jsonify({"status": "all stopping"})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global config

    parser = argparse.ArgumentParser(description="RatrixCam Web UI")
    parser.add_argument("-c", "--config", required=True, help="Path to config.json")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    args = parser.parse_args()

    config = load_settings(args.config)
    if config is None:
        print("ERROR: Could not load config file")
        sys.exit(1)

    # ensure required directories exist
    for path_attr in ("stills_path", "temp_path", "save_path"):
        p = getattr(config, path_attr)
        if not ensure_dir_exists(p):
            print(f"ERROR: Could not create directory: {p}")
            sys.exit(1)

    reset_stills(config)

    # build index structures
    for idx, cam_cfg in enumerate(config.cameras):
        name = cam_cfg.name
        cam_index[name] = idx
        cam_processes[name] = None
        stop_events[name] = multiprocessing.Event()

    print(f"[web] RatrixCam Web UI starting — http://{args.host}:{args.port}")
    print(f"[web] Study: {config.study_label} | {len(config.cameras)} cameras configured")

    # graceful shutdown
    def _shutdown(sig, frame):
        print("\n[web] Shutting down — stopping all cameras …")
        for name in list(cam_processes.keys()):
            _stop_camera(name, timeout=30)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # use_reloader=False is required — Flask's reloader forks the process and
    # breaks multiprocessing.Event ownership.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
