from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "navigation/artifacts/policy_viewer"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "navigation/artifacts"


ASSET_DIR = Path(__file__).resolve().parent / 'dashboard_assets'
HTML = (ASSET_DIR / 'policy_viewer.html').read_text()



def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:120] or "policy"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(\d+)\.bin$", path.name)
    return int(match.group(1)) if match else -1


def known_policies(artifacts_dir: Path) -> list[dict]:
    candidates = [
        (
            "camera_native_hard_100fov_best",
            "Camera native hard 100 FOV best",
            artifacts_dir
            / "front_camera_native_stage11_hard_narrow_fov_rr_schedule_30m_checkpoints/field_nav_camera_native_hard/1778156638753/0000000029982720.bin",
        ),
        (
            "reset_clearance_25m_best_score",
            "Reset clearance 25M best score",
            artifacts_dir
            / "polar_reset_clearance_25m_checkpoints/field_nav/1777285806862/0000000015986688.bin",
        ),
        (
            "reset_clearance_25m_final",
            "Reset clearance 25M final",
            artifacts_dir
            / "polar_reset_clearance_25m_checkpoints/field_nav/1777285806862/0000000024993792.bin",
        ),
        (
            "ent015_peak",
            "Entropy 0.015 peak before reset cleanup",
            artifacts_dir
            / "polar_50m_ent015_from_peak_checkpoints/field_nav/1777283181702/0000000019673088.bin",
        ),
        (
            "ent030_peak",
            "Entropy 0.030 continuation peak",
            artifacts_dir
            / "polar_50m_ent030_cont_checkpoints/field_nav/1777281708491/0000000025817088.bin",
        ),
    ]
    out = []
    for policy_id, name, path in candidates:
        if path.exists():
            out.append({"id": policy_id, "name": name, "path": str(path), "step": checkpoint_step(path)})
    return out


def discover_recent_policies(artifacts_dir: Path, existing_paths: set[str], limit: int = 12) -> list[dict]:
    checkpoints = sorted(
        list(artifacts_dir.glob("*checkpoints/field_nav/*/*.bin"))
        + list(artifacts_dir.glob("*checkpoints/field_nav_camera_native_hard/*/*.bin"))
        + list(artifacts_dir.glob("*checkpoints/field_nav_camera_native/*/*.bin")),
        key=lambda path: (path.stat().st_mtime, checkpoint_step(path)),
        reverse=True,
    )
    out = []
    for path in checkpoints:
        path_str = str(path)
        if path_str in existing_paths:
            continue
        run = path.parent.name
        exp = path.parents[2].name
        out.append(
            {
                "id": slug(f"{exp}_{run}_{path.stem}"),
                "name": f"{exp} {run} {path.stem}",
                "path": path_str,
                "step": checkpoint_step(path),
            }
        )
        existing_paths.add(path_str)
        if len(out) >= limit:
            break
    return out


def load_policies(artifacts_dir: Path) -> list[dict]:
    policies = known_policies(artifacts_dir)
    existing_paths = {p["path"] for p in policies}
    policies.extend(discover_recent_policies(artifacts_dir, existing_paths))
    return policies


def relative_url(root: Path, path: Path) -> str:
    return "/artifact/" + quote(str(path.resolve().relative_to(root.resolve())))


def render_policy_payload(payload: dict, out_dir_root: Path, artifacts_dir: Path) -> dict:
    policies = load_policies(artifacts_dir)
    policy_map = {policy["id"]: policy for policy in policies}
    policy = policy_map.get(str(payload.get("policy_id", "")))
    if policy is None:
        raise ValueError("Unknown policy")

    episodes = max(1, min(1000, int(payload.get("episodes", 80))))
    seed = int(payload.get("seed", 300000))
    top_k = max(1, min(12, int(payload.get("top_k", 6))))
    max_steps = max(10, min(1000, int(payload.get("max_steps", 400))))
    reset_clearance = bool(payload.get("reset_clearance", True))

    run_slug = slug(f"{policy['id']}_e{episodes}_s{seed}_k{top_k}_c{int(reset_clearance)}")
    out_dir = out_dir_root / run_slug
    cmd = [
        sys.executable,
        str(REPO_ROOT / "navigation/scripts/render_policy.py"),
        "--checkpoint",
        policy["path"],
        "--out-dir",
        str(out_dir),
        "--episodes",
        str(episodes),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
        "--gif-fps",
        "12",
        "--gif-max-frames",
        "160",
    ]
    if reset_clearance:
        cmd.extend(
            [
                "--reset-start-clearance-m",
                "1.0",
                "--reset-goal-clearance-m",
                "1.0",
                "--reset-forward-clearance-m",
                "1.5",
                "--reset-forward-margin-m",
                "0.25",
            ]
        )
    if "field_nav_camera_native" in policy["path"] or "camera_native" in policy["id"]:
        cmd.extend(
            [
                "--front-camera",
                "--front-camera-fov-deg",
                "100",
                "--show-camera-rays",
                "--hard-camera-env",
            ]
        )

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/codex_pycache")
    start = time.time()
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "render failed").strip())

    summary_path = out_dir / "render_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    assets = {
        "top_grid": out_dir / "top_policy_rollouts.png",
        "failure_grid": out_dir / "failure_policy_rollouts.png",
        "best_gif": out_dir / "best_policy_rollout.gif",
        "failure_gif": out_dir / "failure_policy_rollout.gif",
    }
    return {
        "policy": policy,
        "out_dir": str(out_dir),
        "elapsed_s": time.time() - start,
        "stdout": completed.stdout,
        "summary": summary,
        "assets": {key: relative_url(REPO_ROOT, path) for key, path in assets.items() if path.exists()},
    }


class ViewerHandler(BaseHTTPRequestHandler):
    out_dir: Path
    artifacts_dir: Path
    policies: list[dict]

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/policies":
            self.policies = load_policies(self.artifacts_dir)
            self.send_json({"policies": self.policies, "out_dir": str(self.out_dir)})
            return

        if parsed.path.startswith("/artifact/"):
            rel = unquote(parsed.path[len("/artifact/") :])
            self.send_file((REPO_ROOT / rel).resolve())
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
            result = self.render_policy(payload)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def render_policy(self, payload: dict) -> dict:
        return render_policy_payload(payload, self.out_dir, self.artifacts_dir)

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


def tailscale_ip() -> str | None:
    try:
        out = subprocess.check_output(["tailscale", "ip", "-4"], text=True).strip()
    except Exception:
        return None
    return out.splitlines()[0].strip() if out else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a FieldNav policy rollout viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--tailscale", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    if args.tailscale and args.host == "127.0.0.1":
        args.host = "0.0.0.0"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ViewerHandler.out_dir = args.out_dir
    ViewerHandler.artifacts_dir = args.artifacts_dir
    ViewerHandler.policies = load_policies(args.artifacts_dir)

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    ts_ip = tailscale_ip()
    print(f"policy_viewer_url=http://{args.host}:{args.port}", flush=True)
    print(f"local_url=http://127.0.0.1:{args.port}", flush=True)
    if ts_ip:
        print(f"tailscale_url=http://{ts_ip}:{args.port}", flush=True)
    print(f"out_dir={args.out_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
