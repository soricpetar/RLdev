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


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FieldNav Policy Viewer</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101214;
  --panel: #181c1f;
  --panel2: #20262a;
  --line: #303940;
  --text: #e8ecef;
  --muted: #9aa8b2;
  --green: #42d392;
  --red: #ff6b6b;
  --blue: #6ea8fe;
  --yellow: #f4bf50;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  padding: 18px 22px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
h1 { margin: 0; font-size: 18px; }
main { padding: 18px 22px 28px; }
.sub { color: var(--muted); font-size: 12px; }
.layout { display: grid; grid-template-columns: 380px 1fr; gap: 16px; align-items: start; }
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  min-width: 0;
}
label { display: block; color: var(--muted); font-size: 12px; margin: 10px 0 5px; }
select, input {
  width: 100%;
  background: var(--panel2);
  color: var(--text);
  border: 1px solid #3a454d;
  border-radius: 6px;
  padding: 8px 9px;
}
input[type="checkbox"] { width: auto; margin-right: 8px; }
button {
  width: 100%;
  margin-top: 14px;
  border: 1px solid #3a454d;
  border-radius: 6px;
  padding: 9px 10px;
  color: var(--text);
  background: var(--blue);
  cursor: pointer;
  font-weight: 700;
}
button:disabled { opacity: 0.5; cursor: wait; }
.grid { display: grid; grid-template-columns: repeat(5, minmax(110px, 1fr)); gap: 10px; margin-bottom: 14px; }
.metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.metric .k { color: var(--muted); font-size: 12px; }
.metric .v { font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; }
.images { display: grid; grid-template-columns: 1fr; gap: 14px; }
.image-row { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 14px; }
img {
  width: 100%;
  height: auto;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: #0b0d0f;
}
.path {
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
  margin-top: 8px;
}
.status { color: var(--yellow); min-height: 20px; }
pre {
  margin: 0;
  max-height: 380px;
  overflow: auto;
  color: #cbd5df;
  background: #0c0f11;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 12px;
  font-size: 12px;
}
@media (max-width: 1000px) {
  .layout { grid-template-columns: 1fr; }
  .grid { grid-template-columns: repeat(2, minmax(110px, 1fr)); }
  .image-row { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>FieldNav Policy Viewer</h1>
    <div class="sub">Render checkpoints on fixed seeds and inspect successes/failures.</div>
  </div>
  <div class="sub" id="serverInfo">Loading...</div>
</header>
<main class="layout">
  <section class="card">
    <label for="policy">Policy</label>
    <select id="policy"></select>
    <div class="path" id="checkpointPath"></div>

    <label for="episodes">Episodes</label>
    <input id="episodes" type="number" min="1" max="1000" value="80">

    <label for="seed">Seed</label>
    <input id="seed" type="number" value="300000">

    <label for="topK">Rollouts to show</label>
    <input id="topK" type="number" min="1" max="12" value="6">

    <label for="maxSteps">Max steps</label>
    <input id="maxSteps" type="number" min="10" max="1000" value="400">

    <label><input id="clearance" type="checkbox" checked>Use reset clearance</label>

    <button id="run">Render Policy</button>
    <div class="path status" id="status"></div>
  </section>

  <section>
    <div class="grid">
      <div class="metric"><div class="k">Success</div><div class="v" id="success">-</div></div>
      <div class="metric"><div class="k">Collision</div><div class="v" id="collision">-</div></div>
      <div class="metric"><div class="k">Timeout</div><div class="v" id="timeout">-</div></div>
      <div class="metric"><div class="k">Mean Return</div><div class="v" id="return">-</div></div>
      <div class="metric"><div class="k">Episodes</div><div class="v" id="episodeCount">-</div></div>
    </div>
    <div class="images">
      <div class="image-row">
        <div class="card">
          <div class="sub">Top Rollouts</div>
          <img id="topImage" alt="Top rollouts">
        </div>
        <div class="card">
          <div class="sub">Failure Rollouts</div>
          <img id="failureImage" alt="Failure rollouts">
        </div>
      </div>
      <div class="image-row">
        <div class="card">
          <div class="sub">Best Rollout GIF</div>
          <img id="bestGif" alt="Best rollout">
        </div>
        <div class="card">
          <div class="sub">Failure GIF</div>
          <img id="failureGif" alt="Failure rollout">
        </div>
      </div>
      <div class="card">
        <div class="sub">Summary JSON</div>
        <pre id="summary">{}</pre>
      </div>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const pct = v => Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '-';
const num = v => Number.isFinite(v) ? v.toFixed(2) : '-';
let policies = [];

function selectedPolicy() {
  return policies.find(p => p.id === $('policy').value);
}

function setStatus(text) {
  $('status').textContent = text;
}

function setImage(id, url) {
  const img = $(id);
  if (!url) {
    img.removeAttribute('src');
    return;
  }
  img.src = `${url}?t=${Date.now()}`;
}

async function loadPolicies() {
  const res = await fetch('/api/policies');
  const data = await res.json();
  policies = data.policies || [];
  $('serverInfo').textContent = data.out_dir || '';
  $('policy').innerHTML = policies.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
  updatePolicyPath();
}

function updatePolicyPath() {
  const p = selectedPolicy();
  $('checkpointPath').textContent = p ? p.path : '';
}

async function runRender() {
  const p = selectedPolicy();
  if (!p) return;
  $('run').disabled = true;
  setStatus('Rendering...');
  try {
    const payload = {
      policy_id: p.id,
      episodes: Number($('episodes').value),
      seed: Number($('seed').value),
      top_k: Number($('topK').value),
      max_steps: Number($('maxSteps').value),
      reset_clearance: $('clearance').checked,
    };
    const res = await fetch('/api/render', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    const s = data.summary || {};
    $('success').textContent = pct(s.success_rate);
    $('collision').textContent = pct(s.collision_rate);
    $('timeout').textContent = pct(s.timeout_rate);
    $('return').textContent = num(s.mean_return);
    $('episodeCount').textContent = s.episodes ?? '-';
    $('summary').textContent = JSON.stringify(s, null, 2);
    setImage('topImage', data.assets?.top_grid);
    setImage('failureImage', data.assets?.failure_grid);
    setImage('bestGif', data.assets?.best_gif);
    setImage('failureGif', data.assets?.failure_gif);
    setStatus(`Rendered in ${data.elapsed_s.toFixed(1)}s: ${data.out_dir}`);
  } catch (err) {
    setStatus(String(err.message || err));
  } finally {
    $('run').disabled = false;
  }
}

$('policy').addEventListener('change', updatePolicyPath);
$('run').addEventListener('click', runRender);
loadPolicies().catch(err => setStatus(String(err)));
</script>
</body>
</html>
"""


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:120] or "policy"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(\d+)\.bin$", path.name)
    return int(match.group(1)) if match else -1


def known_policies(artifacts_dir: Path) -> list[dict]:
    candidates = [
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
        artifacts_dir.glob("*checkpoints/field_nav/*/*.bin"),
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
