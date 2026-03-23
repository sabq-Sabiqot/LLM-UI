"""
main.py
=========
UAV Mission Designer web server.
Uses ONLY Python standard library -- no Flask, no Django, no framework.

http.server + socketserver handle all HTTP.
JSON is parsed/serialised with the stdlib json module.
Static files (CSS, JS) are read from disk and served directly.
Multi-user sessions are managed in a plain Python dict with threading.Lock.

File structure
--------------
  main.py                <- this file  (HTTP server + routing) BACKEND
  templates/index.html   <- HTML (served as-is)
  static/style.css       <- CSS  (served as-is)
  static/script.js       <- JS   (served as-is)

Run
---
  python main.py
  Open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import uuid
import time
import threading
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CONNECTION import (
    LLMConnector, build_mission, process_message,
    BACKEND, MISSION_SCHEMA,
)

# =============================================================================
# CONFIG
# =============================================================================

HOST           = "127.0.0.1"
PORT           = 5000
XFOIL_PATH     = r"../ML_Calculation/Airfoil_Prediction/Xfoil6.99/xfoil.exe"
MODEL_PATH     = "../ML_Calculation/Airfoil_Prediction/uav_model.pkl"
OUTPUT_DIR     = "llm_mission_output"
SESSION_COOKIE = "uav_session_id"
SESSION_TTL    = 3600    # seconds before idle session is pruned

CURRENT_DIR       = os.path.dirname(os.path.abspath(__file__))
BASE_DIR          = os.path.dirname(CURRENT_DIR)
TEMPLATE_DIR      = os.path.join(BASE_DIR, "templates")
STATIC_DIR        = os.path.join(BASE_DIR, "statics")
# print(STATIC_DIR)

# =============================================================================
# MULTI-USER SESSION STORE
# =============================================================================
# sessions[session_id] = {
#     "connector"        : LLMConnector,
#     "params"           : dict,      # accumulated mission params
#     "ready"            : bool,
#     "last_active"      : float,     # Unix timestamp
#     "pipeline_running" : bool,
#     "output_dir"       : str,
# }

_sessions: dict = {}
_sessions_lock  = threading.Lock()


def _new_session(session_id: str) -> dict:
    out_dir = os.path.join(OUTPUT_DIR, session_id[:8])
    os.makedirs(out_dir, exist_ok=True)
    return {
        "connector"        : LLMConnector(BACKEND),
        "params"           : {},
        "ready"            : False,
        "last_active"      : time.time(),
        "pipeline_running" : False,
        "output_dir"       : out_dir,
    }


def _get_session(session_id: str) -> dict:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = _new_session(session_id)
        sess = _sessions[session_id]
        sess["last_active"] = time.time()
        return sess


def _reset_session(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id]["connector"].reset()
            _sessions[session_id]["params"] = {}
            _sessions[session_id]["ready"]  = False


def _prune_sessions():
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, s in _sessions.items()
                   if now - s["last_active"] > SESSION_TTL]
        for sid in expired:
            del _sessions[sid]
    if expired:
        print(f"[session] Pruned {len(expired)} idle session(s). "
              f"Active: {len(_sessions)}")


def _pruner_loop():
    while True:
        time.sleep(600)
        _prune_sessions()


# =============================================================================
# COOKIE HELPERS
# =============================================================================

def _read_session_id(handler) -> str:
    """Parse session_id from Cookie header, or generate a new UUID."""
    cookie_header = handler.headers.get("Cookie", "")
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith(SESSION_COOKIE + "="):
            return part[len(SESSION_COOKIE) + 1:]
    return str(uuid.uuid4())


def _cookie_header(session_id: str) -> str:
    """Return a Set-Cookie header value."""
    return (
        f"{SESSION_COOKIE}={session_id}; "
        f"Max-Age={SESSION_TTL}; "
        f"HttpOnly; SameSite=Lax; Path=/"
    )


# =============================================================================
# FILE SERVING
# =============================================================================

def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


# =============================================================================
# RESPONSE HELPERS
# =============================================================================

def _send_json(handler, data: dict, status: int = 200, session_id: str = None):
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    if session_id:
        handler.send_header("Set-Cookie", _cookie_header(session_id))
    handler.end_headers()
    handler.wfile.write(body)


def _send_file(handler, path: str, session_id: str = None):
    try:
        body = _read_file(path)
    except FileNotFoundError:
        _send_json(handler, {"error": "not found"}, 404)
        return
    handler.send_response(200)
    handler.send_header("Content-Type", _mime(path))
    handler.send_header("Content-Length", str(len(body)))
    if session_id:
        handler.send_header("Set-Cookie", _cookie_header(session_id))
    handler.end_headers()
    handler.wfile.write(body)


def _send_html(handler, path: str, session_id: str = None):
    _send_file(handler, path, session_id)


def _read_json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


# =============================================================================
# ROUTE HANDLERS
# =============================================================================

def handle_index(handler, session_id: str):
    path = os.path.join(TEMPLATE_DIR, "index.html")
    _send_html(handler, path, session_id)


def handle_static(handler, url_path: str):
    # Strip /static/ prefix and resolve to static dir
    rel   = url_path[len("/statics/"):]
    # Prevent directory traversal
    rel   = os.path.normpath(rel).lstrip("/\\")
    fpath = os.path.join(STATIC_DIR, rel)
    if not fpath.startswith(STATIC_DIR):
        _send_json(handler, {"error": "forbidden"}, 403)
        return
    _send_file(handler, fpath)


def handle_status(handler, session_id: str):
    sess      = _get_session(session_id)
    conn      = sess["connector"]
    available = conn.is_available()
    models    = conn.available_models() if available else []
    _send_json(handler, {
        "available"      : available,
        "backend"        : BACKEND,
        "url"            : conn.url,
        "model"          : conn.model,
        "models"         : models,
        "session_id"     : session_id[:8],
        "active_sessions": len(_sessions),
    }, session_id=session_id)


def handle_chat(handler, session_id: str):
    data    = _read_json_body(handler)
    message = data.get("message", "").strip()

    if not message:
        _send_json(handler, {"error": "empty message"}, 400, session_id)
        return

    sess = _get_session(session_id)

    if sess["pipeline_running"]:
        _send_json(handler, {
            "reply"        : "Pipeline is already running. Please wait.",
            "params"       : sess["params"],
            "ready_to_run" : sess["ready"],
            "error"        : "pipeline_busy",
        }, session_id=session_id)
        return

    result = process_message(
        message    = message,
        connector  = sess["connector"],
        xfoil_path = XFOIL_PATH,
        model_path = MODEL_PATH,
        output_dir = sess["output_dir"],
        auto_run   = False,
    )

    if result.get("params"):
        sess["params"].update(result["params"])
    if result.get("ready_to_run"):
        sess["ready"] = True

    pipeline_result = None
    run_words = ["run", "yes", "go", "start", "execute",
                 "confirm", "proceed", "launch", "do it"]
    if sess["ready"] and any(w in message.lower() for w in run_words):
        sess["pipeline_running"] = True
        try:
            mission = build_mission(
                sess["params"],
                xfoil_path = XFOIL_PATH,
                model_path = MODEL_PATH,
                output_dir = sess["output_dir"],
            )
            from ...ML_Calculation.Airfoil_Prediction import POST_TRAIN
            pipeline_result = POST_TRAIN.run(mission)
        except Exception as e:
            pipeline_result = {"error": str(e)}
        finally:
            sess["pipeline_running"] = False

    _send_json(handler, {
        "reply"          : result["reply"],
        "params"         : sess["params"],
        "ready_to_run"   : sess["ready"],
        "pipeline_result": pipeline_result,
        "error"          : result.get("error"),
    }, session_id=session_id)


def handle_reset(handler, session_id: str):
    _reset_session(session_id)
    _send_json(handler, {"status": "ok"}, session_id=session_id)


def handle_run(handler, session_id: str):
    sess = _get_session(session_id)

    if sess["pipeline_running"]:
        _send_json(handler, {
            "status": "busy",
            "error" : "Pipeline already running for this session",
        }, 409, session_id)
        return

    data   = _read_json_body(handler)
    params = data.get("params", sess["params"])

    if not params.get("payload_kg") or not params.get("cruise_speed"):
        _send_json(handler, {
            "error" : "Missing required params: payload_kg and cruise_speed",
            "result": None,
        }, 400, session_id)
        return

    sess["pipeline_running"] = True
    try:
        mission = build_mission(
            params,
            xfoil_path = XFOIL_PATH,
            model_path = MODEL_PATH,
            output_dir = sess["output_dir"],
        )
        from ...ML_Calculation.Airfoil_Prediction import POST_TRAIN
        result = POST_TRAIN.run(mission)
        geo    = result.get("geometry", {})
        aero   = result.get("aero",     {})

        _send_json(handler, {
            "status"    : "ok",
            "session_id": session_id[:8],
            "result"    : {
                "MTOW_kg"   : geo.get("MTOW_kg"),
                "wingspan_m": geo.get("wingspan_m"),
                "S_m2"      : geo.get("S_m2"),
                "AR"        : geo.get("AR"),
                "c_mean_m"  : geo.get("c_mean_m"),
                "l_fus_m"   : geo.get("l_fus_m"),
                "P_motor_W" : geo.get("P_motor_W"),
                "CL"        : aero.get("CL_pred"),
                "CD"        : aero.get("CD_pred"),
                "LD"        : aero.get("LD_pred"),
                "breguet"   : aero.get("breguet_pred"),
                "report_path": os.path.join(
                    sess["output_dir"], "post_train_report.txt"),
            },
        }, session_id=session_id)

    except Exception as e:
        _send_json(handler, {"status": "error", "error": str(e)},
                   500, session_id)
    finally:
        sess["pipeline_running"] = False


def handle_admin_sessions(handler):
    now = time.time()
    with _sessions_lock:
        summary = [
            {
                "id"        : sid[:8],
                "idle_s"    : round(now - s["last_active"]),
                "params_set": len(s["params"]),
                "ready"     : s["ready"],
                "running"   : s["pipeline_running"],
            }
            for sid, s in _sessions.items()
        ]
    _send_json(handler, {"active": len(_sessions), "sessions": summary})


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class UAVHandler(BaseHTTPRequestHandler):

    # Silence default request log (replace with custom)
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()}  {fmt % args}")

    def log_error(self, fmt, *args):
        print(f"  [ERR] {self.address_string()}  {fmt % args}")

    # ── GET ──────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed     = urlparse(self.path)
        path       = parsed.path
        session_id = _read_session_id(self)
        _get_session(session_id)   # ensure session exists

        if path == "/" or path == "/index.html":
            handle_index(self, session_id)

        elif path.startswith("/statics/"):
            handle_static(self, path)

        elif path == "/status":
            handle_status(self, session_id)

        elif path == "/admin/sessions":
            handle_admin_sessions(self)

        else:
            _send_json(self, {"error": "not found"}, 404)

    # ── POST ─────────────────────────────────────────────────────────────

    def do_POST(self):
        parsed     = urlparse(self.path)
        path       = parsed.path
        session_id = _read_session_id(self)

        if path == "/chat":
            handle_chat(self, session_id)

        elif path == "/reset":
            handle_reset(self, session_id)

        elif path == "/run":
            handle_run(self, session_id)

        else:
            _send_json(self, {"error": "not found"}, 404)

    # ── OPTIONS (CORS preflight, not needed for same-origin but harmless) ─

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


# =============================================================================
# THREADED HTTP SERVER
# =============================================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in its own thread."""
    daemon_threads = True   # threads die when the server does


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Verify required files exist
    required = [
        os.path.join(TEMPLATE_DIR, "index.html"),
        os.path.join(STATIC_DIR, "index.css"),
        os.path.join(STATIC_DIR, "index.js"),
        os.path.join(STATIC_DIR, "rand.js")
    ]
    missing = [f for f in required if not os.path.isfile(f)]
    if missing:
        print("ERROR: Missing files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Start session pruner background thread
    threading.Thread(target=_pruner_loop, daemon=True).start()

    # Check LLM availability
    _probe = LLMConnector(BACKEND)

    print("=" * 60)
    print("  UAV Mission Designer  --  Web UI  (no framework)")
    print("=" * 60)
    print(f"  Backend  : {BACKEND}")
    print(f"  LLM URL  : {_probe.url}")
    print(f"  Model    : {_probe.model}")
    print(f"  XFoil    : {XFOIL_PATH}")
    print(f"  ML Model : {MODEL_PATH}")
    print(f"  Sessions : TTL={SESSION_TTL}s")
    print(f"  Server   : http://{HOST}:{PORT}")
    print("=" * 60)

    if not _probe.is_available():
        print("\nWARNING: LLM backend not reachable.")
        print("  Ollama   : ollama serve  &&  ollama pull llama3.2")
        print("  LM Studio: start the local server in the app\n")
    else:
        models = _probe.available_models()
        if models:
            print(f"\nAvailable models: {models}")
        print()

    server = ThreadedHTTPServer((HOST, PORT), UAVHandler)
    print(f"Serving on http://{HOST}:{PORT}  (Ctrl+C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()