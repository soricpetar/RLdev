from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "navigation/artifacts/native_checkpoints"
DEFAULT_LOG_DIR = REPO_ROOT / "navigation/artifacts/native_logs"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "navigation/artifacts"
DEFAULT_POLICY_OUT_DIR = REPO_ROOT / "navigation/artifacts/policy_viewer"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.scripts import policy_viewer


ASSET_DIR = Path(__file__).resolve().parent / 'dashboard_assets'
HTML = (ASSET_DIR / 'training_dashboard.html').read_text()



def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def metric_history_from_final(data: dict) -> list[dict]:
    metrics = data.get("metrics", {})
    if not metrics:
        return []
    length = max((len(v) for v in metrics.values() if isinstance(v, list)), default=0)
    out = []
    for idx in range(length):
        point = {}
        for key, values in metrics.items():
            if isinstance(values, list) and idx < len(values):
                point[key] = values[idx]
        out.append(point)
    return out


def checkpoints_for_run(run_dir: Path) -> list[dict]:
    checkpoints = []
    for path in sorted(run_dir.glob("*.bin")):
        step_match = re.search(r"(\d+)\.bin$", path.name)
        step = int(step_match.group(1)) if step_match else 0
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0
        checkpoints.append({"path": str(path), "step": step, "mtime": mtime})
    return checkpoints


def training_processes(env: str) -> list[dict]:
    try:
        out = subprocess.check_output(["ps", "-axo", "pid,stat,etime,pcpu,pmem,command"], text=True)
    except Exception:
        return []
    processes = []
    for line in out.splitlines()[1:]:
        if "puffer train" not in line or env not in line:
            continue
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        processes.append(
            {
                "pid": parts[0],
                "stat": parts[1],
                "etime": parts[2],
                "cpu": parts[3],
                "mem": parts[4],
                "command": parts[5],
            }
        )
    return processes


def tailscale_ip() -> str | None:
    try:
        out = subprocess.check_output(["tailscale", "ip", "-4"], text=True).strip()
    except Exception:
        return None
    return out.splitlines()[0].strip() if out else None


def collect_state(env: str, checkpoint_dir: Path, log_dir: Path) -> dict:
    now = time.time()
    cp_env_dir = checkpoint_dir / env
    log_env_dir = log_dir / env
    run_ids = set()
    if cp_env_dir.exists():
        run_ids.update(path.name for path in cp_env_dir.iterdir() if path.is_dir())
    if log_env_dir.exists():
        for path in log_env_dir.glob("*.json"):
            if path.name == "latest.live.json":
                continue
            run_ids.add(path.name.split(".")[0])

    processes = training_processes(env)
    runs = []
    for run_id in sorted(run_ids):
        run_dir = cp_env_dir / run_id
        checkpoints = checkpoints_for_run(run_dir) if run_dir.exists() else []
        latest_cp = checkpoints[-1] if checkpoints else {}
        final_path = log_env_dir / f"{run_id}.json"
        live_path = log_env_dir / f"{run_id}.live.json"
        final_data = load_json(final_path) if final_path.exists() else None
        live_data = load_json(live_path) if live_path.exists() else None
        final_history = metric_history_from_final(final_data or {})
        latest_final = final_history[-1] if final_history else {}
        updated = None
        if live_data and live_data.get("updated_at"):
            updated = float(live_data["updated_at"])
        elif latest_cp:
            updated = latest_cp.get("mtime")
        elif final_path.exists():
            updated = final_path.stat().st_mtime

        checkpoint_history = [
            {"agent_steps": cp["step"], "checkpoint_step": cp["step"], "checkpoint_mtime": cp["mtime"]}
            for cp in checkpoints
        ]
        live_recent = live_data and now - float(live_data.get("updated_at", 0)) < 20
        paused_process = any("T" in proc.get("stat", "") for proc in processes)
        status = "live" if live_recent else ("finished" if final_data else ("paused" if paused_process else ("checkpointing" if checkpoints else "unknown")))
        status_class = "live" if status == "live" else ("dead" if status == "finished" else "stale")
        estimated_sps = 0.0
        if len(checkpoints) >= 2:
            a, b = checkpoints[-2], checkpoints[-1]
            dt = b["mtime"] - a["mtime"]
            if dt > 0:
                estimated_sps = (b["step"] - a["step"]) / dt

        runs.append(
            {
                "run_id": run_id,
                "status": (live_data and live_data.get("stage")) or status,
                "status_class": status_class,
                "latest_step": latest_cp.get("step", latest_final.get("agent_steps", 0)),
                "estimated_sps": estimated_sps,
                "checkpoint_count": len(checkpoints),
                "latest_checkpoint": latest_cp.get("path"),
                "updated_at": updated,
                "updated_age_s": None if updated is None else max(0, now - updated),
                "log_path": str(final_path) if final_path.exists() else None,
                "live": live_data,
                "latest_final": latest_final,
                "final_history": final_history,
                "checkpoint_history": checkpoint_history,
                "config": (final_data or {}).get("train") or (live_data or {}).get("config") or {},
            }
        )

    runs.sort(key=lambda r: (r["updated_at"] or 0, r["latest_step"] or 0), reverse=True)
    selected = runs[0] if runs else None
    total_timesteps = 0
    if selected:
        live_cfg = (selected.get("live") or {}).get("config") or {}
        total_timesteps = live_cfg.get("total_timesteps") or (selected.get("config") or {}).get("total_timesteps") or 0
    return {
        "env": env,
        "checkpoint_dir": str(checkpoint_dir),
        "log_dir": str(log_dir),
        "processes": processes,
        "runs": runs,
        "selected": selected,
        "total_timesteps": total_timesteps,
    }


class Handler(BaseHTTPRequestHandler):
    checkpoint_dir: Path
    log_dir: Path
    env: str
    artifacts_dir: Path
    policy_out_dir: Path

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            env = qs.get("env", [self.env])[0]
            payload = collect_state(env, self.checkpoint_dir, self.log_dir)
            self.send_json(payload)
            return

        if parsed.path == "/api/policies":
            policies = policy_viewer.load_policies(self.artifacts_dir)
            self.send_json({"policies": policies, "out_dir": str(self.policy_out_dir)})
            return

        if parsed.path.startswith("/artifact/"):
            rel = unquote(parsed.path[len("/artifact/") :])
            self.send_file((REPO_ROOT / rel).resolve())
            return

        if parsed.path == "/policy":
            body = policy_viewer.HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/render":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = policy_viewer.render_policy_payload(payload, self.policy_out_dir, self.artifacts_dir)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        root = REPO_ROOT.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return

        content_type = "application/octet-stream"
        if path.suffix == ".png":
            content_type = "image/png"
        elif path.suffix == ".gif":
            content_type = "image/gif"
        elif path.suffix == ".json":
            content_type = "application/json"

        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Serve a local FieldNav training dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--tailscale",
        action="store_true",
        help="Listen on all interfaces and print the Tailscale URL for remote monitoring.",
    )
    parser.add_argument("--env", default="field_nav")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--policy-out-dir", type=Path, default=DEFAULT_POLICY_OUT_DIR)
    args = parser.parse_args()

    ts_ip = tailscale_ip()
    if args.tailscale and args.host == "127.0.0.1":
        args.host = "0.0.0.0"

    Handler.checkpoint_dir = args.checkpoint_dir
    Handler.log_dir = args.log_dir
    Handler.env = args.env
    Handler.artifacts_dir = args.artifacts_dir
    Handler.policy_out_dir = args.policy_out_dir
    args.policy_out_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard_url=http://{args.host}:{args.port}")
    print(f"local_url=http://127.0.0.1:{args.port}")
    if ts_ip:
        print(f"tailscale_url=http://{ts_ip}:{args.port}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"log_dir={args.log_dir}")
    print(f"policy_viewer_url=http://127.0.0.1:{args.port}/policy")
    if ts_ip:
        print(f"policy_viewer_tailscale_url=http://{ts_ip}:{args.port}/policy")
    server.serve_forever()


if __name__ == "__main__":
    main()
