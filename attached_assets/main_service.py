import os
import subprocess
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, request
from flask_cors import CORS
import queue

APP_DIR = r"C:\quantonium_os\apps"
UI_DIR = r"C:\quantonium_os\quantonium_ui\electron_test"

app = Flask(__name__, static_folder=UI_DIR, static_url_path="")
CORS(app, resources={r"/apps/*": {"origins": "http://localhost:5000"}})

handler = RotatingFileHandler(os.path.join(APP_DIR, 'app.log'), maxBytes=10000, backupCount=3)
handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
app.logger.addHandler(handler)
app.logger.addHandler(console_handler)
logging.basicConfig(level=logging.DEBUG)

running_processes = {}
output_queues = {}

def read_process_output(proc, app_name, queue):
    for line in iter(proc.stdout.readline, ''):
        queue.put((app_name, "stdout", line.strip()))
    for line in iter(proc.stderr.readline, ''):
        queue.put((app_name, "stderr", line.strip()))

@app.route("/")
def serve_home():
    app.logger.info(f"Attempting to serve index.html from {os.path.join(UI_DIR, 'index.html')}")
    if not os.path.exists(os.path.join(UI_DIR, "index.html")):
        app.logger.error("index.html not found in static folder")
        return jsonify({"error": "index.html not found"}), 404
    return app.send_static_file("index.html")

@app.route("/apps/start", methods=["POST"])
def start_app():
    app.logger.debug(f"Raw request data: {request.get_data(as_text=True)}")
    app.logger.debug(f"Request headers: {request.headers}")
    data = request.get_json(silent=True)
    app.logger.debug(f"Parsed JSON data: {data}")
    app_name = data.get("app") if data else None
    app.logger.debug(f"Extracted app_name: {app_name}")

    if not app_name:
        app.logger.error("No app provided in request")
        return jsonify({"status": "error", "message": "No app provided"}), 400

    if app_name in running_processes:
        app.logger.warning(f"{app_name} is already running")
        return jsonify({"status": "error", "message": f"{app_name} is already running"}), 400

    script_path = os.path.join(APP_DIR, app_name)
    if not os.path.exists(script_path):
        app.logger.error(f"Script not found: {script_path}")
        return jsonify({"status": "error", "message": f"Script not found: {script_path}"}), 404

    try:
        env = os.environ.copy()
        env["PATH"] = f"{os.path.dirname(sys.executable)};{env.get('PATH', '')}"

        proc = subprocess.Popen(
            ["python", script_path],
            cwd=APP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            shell=True
        )
        running_processes[app_name] = proc
        output_queues[app_name] = queue.Queue()
        app.logger.debug(f"Started {app_name} with PID {proc.pid}")

        output_thread = threading.Thread(target=read_process_output, args=(proc, app_name, output_queues[app_name]))
        output_thread.daemon = True
        output_thread.start()

        try:
            stdout_lines = []
            stderr_lines = []
            for _ in range(10):
                if not output_queues[app_name].empty():
                    app_name_q, stream, line = output_queues[app_name].get_nowait()
                    if stream == "stdout":
                        stdout_lines.append(line)
                        app.logger.info(f"{app_name} stdout: {line}")
                    elif stream == "stderr":
                        stderr_lines.append(line)
                        app.logger.error(f"{app_name} stderr: {line}")
            if stderr_lines:
                return jsonify({"status": "error", "message": f"Failed to start {app_name}: {''.join(stderr_lines)}"}), 500

            if proc.poll() is not None:
                app.logger.warning(f"{app_name} exited with code {proc.returncode}")
                return jsonify({"status": "error", "message": f"{app_name} exited with code {proc.returncode}"}), 500

            app.logger.info(f"{app_name} still running after initial check")
            return jsonify({"status": "success", "message": f"{app_name} started", "pid": proc.pid}), 200

        except queue.Empty:
            app.logger.info(f"No initial output from {app_name}, assuming success")
            return jsonify({"status": "success", "message": f"{app_name} started", "pid": proc.pid}), 200

    except Exception as e:
        app.logger.error(f"Failed to start {app_name}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/apps/stop/<app_name>", methods=["POST"])
def stop_app(app_name):
    app.logger.debug(f"Received request to stop: {app_name}")
    if app_name not in running_processes:
        return jsonify({"status": "error", "message": f"{app_name} is not running"}), 400

    proc = running_processes[app_name]
    proc.terminate()
    try:
        proc.wait(timeout=5)
        del running_processes[app_name]
        if app_name in output_queues:
            del output_queues[app_name]
        app.logger.debug(f"Stopped {app_name}")
    except subprocess.TimeoutExpired:
        proc.kill()
        del running_processes[app_name]
        if app_name in output_queues:
            del output_queues[app_name]
        app.logger.warning(f"Forced kill of {app_name}")
    return jsonify({"status": "success", "message": f"{app_name} stopped"}), 200

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "running_apps": list(running_processes.keys())}), 200

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(405)
def handle_405(e):
    return jsonify({"error": "Method Not Allowed"}), 405

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)