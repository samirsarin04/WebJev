"""Local web UI:  python -m webjev.server   ->  http://127.0.0.1:5000"""
from __future__ import annotations

import queue
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .agent import MODES, describe_backends, run
from .config import Config

STATIC = Path(__file__).parent / "static"
CLARIFY_TIMEOUT_S = 900

app = Flask(__name__, static_folder=None)
SESSIONS: dict[str, "Session"] = {}


class Session:
    def __init__(self):
        self.events: list[dict] = []
        self.done = False
        self.replies: queue.Queue[str] = queue.Queue()

    def ask_user(self, question: str) -> str:
        try:
            return self.replies.get(timeout=CLARIFY_TIMEOUT_S)
        except queue.Empty:
            return "(no response)"


def _worker(session: Session, question: str, mode: str):
    try:
        run(question, mode, Config.from_env(), ask_user=session.ask_user, on_event=session.events.append)
    finally:
        session.done = True


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/info")
def info():
    return jsonify(describe_backends(Config.from_env()))


@app.post("/api/runs")
def start_run():
    body = request.get_json(force=True, silent=True) or {}
    question, mode = (body.get("question") or "").strip(), body.get("mode")
    if not question or mode not in MODES:
        return jsonify(error="Need a question and mode 'jev' or 'baseline'."), 400
    sid = uuid.uuid4().hex
    SESSIONS[sid] = session = Session()
    threading.Thread(target=_worker, args=(session, question, mode), daemon=True).start()
    return jsonify(id=sid)


@app.get("/api/runs/<sid>")
def poll(sid):
    session = SESSIONS.get(sid)
    if not session:
        return jsonify(error="Unknown run."), 404
    since = request.args.get("since", 0, type=int)
    done = session.done            # read before slicing so no event is missed on the final poll
    return jsonify(events=session.events[since:], done=done)


@app.post("/api/runs/<sid>/clarify")
def clarify(sid):
    session = SESSIONS.get(sid)
    if not session:
        return jsonify(error="Unknown run."), 404
    body = request.get_json(force=True, silent=True) or {}
    session.replies.put((body.get("answer") or "").strip())
    return jsonify(ok=True)


def main():
    print("WebJev UI on http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, threaded=True)


if __name__ == "__main__":
    main()
