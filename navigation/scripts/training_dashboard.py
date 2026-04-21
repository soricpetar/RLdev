from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "navigation/artifacts/native_checkpoints"
DEFAULT_LOG_DIR = REPO_ROOT / "navigation/artifacts/native_logs"


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FieldNav Training Monitor</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101214;
  --panel: #181c1f;
  --panel2: #20262a;
  --text: #e8ecef;
  --muted: #9aa8b2;
  --cyan: #24d6d6;
  --green: #42d392;
  --red: #ff6b6b;
  --yellow: #f4bf50;
  --blue: #6ea8fe;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 22px;
  border-bottom: 1px solid #2b3338;
}
h1 { margin: 0; font-size: 18px; font-weight: 700; }
.sub { color: var(--muted); font-size: 12px; }
main { padding: 18px 22px 28px; }
.grid { display: grid; gap: 14px; }
.stats { grid-template-columns: repeat(6, minmax(120px, 1fr)); }
.charts { grid-template-columns: repeat(2, minmax(280px, 1fr)); margin-top: 14px; }
.wide { grid-column: 1 / -1; }
.card {
  background: var(--panel);
  border: 1px solid #2a3136;
  border-radius: 8px;
  padding: 14px;
  min-width: 0;
}
.metric-label { color: var(--muted); font-size: 12px; margin-bottom: 4px; white-space: nowrap; }
.metric-value { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.metric-small { color: var(--muted); font-size: 12px; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.controls {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.controls label { color: var(--muted); font-size: 12px; }
.controls button {
  background: var(--panel2);
  color: var(--text);
  border: 1px solid #344047;
  border-radius: 6px;
  padding: 6px 10px;
  cursor: pointer;
}
.controls button:hover { border-color: var(--cyan); }
.controls input[type="range"] { width: 160px; }
canvas { display: block; width: 100%; height: 240px; }
.runs { width: 100%; border-collapse: collapse; margin-top: 6px; }
.runs th, .runs td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #2a3136; font-variant-numeric: tabular-nums; }
.runs th { color: var(--muted); font-size: 12px; font-weight: 600; }
.pill { display: inline-block; border-radius: 999px; padding: 2px 8px; background: var(--panel2); color: var(--muted); font-size: 12px; }
.live { color: var(--green); }
.stale { color: var(--yellow); }
.dead { color: var(--muted); }
@media (max-width: 1000px) {
  .stats { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .charts { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>FieldNav Training Monitor</h1>
    <div class="sub" id="subtitle">Loading...</div>
  </div>
  <div class="sub" id="updated">-</div>
</header>
<main>
  <section class="grid stats">
    <div class="card"><div class="metric-label">Run</div><div class="metric-value" id="run">-</div><div class="metric-small" id="stage">-</div></div>
    <div class="card"><div class="metric-label">Steps</div><div class="metric-value" id="steps">-</div><div class="metric-small" id="progress">-</div></div>
    <div class="card"><div class="metric-label">SPS</div><div class="metric-value" id="sps">-</div><div class="metric-small" id="eta">-</div></div>
    <div class="card"><div class="metric-label">Success</div><div class="metric-value" id="success">-</div><div class="metric-small">goal reached episodes</div></div>
    <div class="card"><div class="metric-label">Collision</div><div class="metric-value" id="collision">-</div><div class="metric-small">collision episodes</div></div>
    <div class="card"><div class="metric-label">Return</div><div class="metric-value" id="score">-</div><div class="metric-small" id="epLen">-</div></div>
  </section>
  <section class="card controls">
    <button id="zoomIn" type="button">Zoom in</button>
    <button id="zoomOut" type="button">Zoom out</button>
    <button id="zoomAll" type="button">All</button>
    <label>Window <span id="windowLabel">100%</span></label>
    <input id="windowRange" type="range" min="5" max="100" step="5" value="100">
    <label>Moving avg <span id="smoothLabel">1</span></label>
    <input id="smoothRange" type="range" min="1" max="100" step="1" value="1">
  </section>
  <section class="grid charts">
    <div class="card"><canvas id="quality"></canvas></div>
    <div class="card"><canvas id="return"></canvas></div>
    <div class="card"><canvas id="loss"></canvas></div>
    <div class="card"><canvas id="speed"></canvas></div>
    <div class="card wide">
      <div class="metric-label">Runs and Checkpoints</div>
      <table class="runs">
        <thead><tr><th>Run</th><th>Status</th><th>Latest Step</th><th>Checkpoints</th><th>Updated</th><th>Path</th></tr></thead>
        <tbody id="runs"></tbody>
      </table>
    </div>
  </section>
</main>
<script>
const fmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 1});
const pct = v => Number.isFinite(v) ? `${(100*v).toFixed(1)}%` : '-';
const num = v => Number.isFinite(v) ? fmt.format(v) : '-';
const short = s => s ? String(s).slice(-6) : '-';
let viewPct = 100;
let smoothWindow = 1;

function get(obj, key) {
  if (!obj) return undefined;
  return obj[key];
}

function series(history, key) {
  return (history || []).map(p => [Number(p.agent_steps ?? p.steps ?? 0), Number(p[key])]).filter(p => Number.isFinite(p[1]));
}

function visibleSeries(data) {
  if (!data.length || viewPct >= 100) return data;
  const keep = Math.max(2, Math.ceil(data.length * viewPct / 100));
  return data.slice(-keep);
}

function smoothSeries(data) {
  if (smoothWindow <= 1 || data.length < 3) return data;
  const out = [];
  let sum = 0;
  const q = [];
  for (const point of data) {
    q.push(point[1]);
    sum += point[1];
    if (q.length > smoothWindow) sum -= q.shift();
    out.push([point[0], sum / q.length]);
  }
  return out;
}

function transformSeries(data) {
  return smoothSeries(visibleSeries(data));
}

function drawChart(canvas, title, lines) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, rect.width * dpr);
  canvas.height = Math.max(220, rect.height * dpr);
  ctx.scale(dpr, dpr);
  const w = canvas.width / dpr, h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#20262a';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#e8ecef';
  ctx.font = '600 13px system-ui';
  ctx.fillText(title, 12, 20);
  const left = 48, right = 12, top = 34, bottom = 28;
  const prepared = lines.map(l => ({...l, data: transformSeries(l.data)}));
  const pts = prepared.flatMap(l => l.data);
  if (!pts.length) {
    ctx.fillStyle = '#9aa8b2';
    ctx.font = '12px system-ui';
    ctx.fillText('No live history yet. Restart training after this patch for live metrics.', 12, 58);
    return;
  }
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax === ymin) { ymax += 1; ymin -= 1; }
  const pad = (ymax - ymin) * 0.08;
  ymin -= pad; ymax += pad;
  const xscale = x => left + (x - xmin) / (xmax - xmin) * (w - left - right);
  const yscale = y => top + (1 - (y - ymin) / (ymax - ymin)) * (h - top - bottom);
  ctx.strokeStyle = '#344047';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < 4; i++) {
    const y = top + i * (h - top - bottom) / 3;
    ctx.moveTo(left, y); ctx.lineTo(w - right, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#9aa8b2';
  ctx.font = '11px system-ui';
  ctx.fillText(num(ymax), 6, top + 4);
  ctx.fillText(num(ymin), 6, h - bottom);
  for (const line of prepared) {
    if (!line.data.length) continue;
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    line.data.forEach((p, i) => {
      const x = xscale(p[0]), y = yscale(p[1]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const last = line.data[line.data.length - 1];
    ctx.fillStyle = line.color;
    ctx.fillText(`${line.name}: ${num(last[1])}`, 12 + prepared.indexOf(line) * 150, h - 8);
  }
}

function updateControls() {
  document.getElementById('windowLabel').textContent = `${viewPct}%`;
  document.getElementById('smoothLabel').textContent = String(smoothWindow);
  document.getElementById('windowRange').value = viewPct;
  document.getElementById('smoothRange').value = smoothWindow;
}

document.getElementById('zoomIn').addEventListener('click', () => {
  viewPct = Math.max(5, Math.round(viewPct / 2));
  updateControls();
  refresh();
});
document.getElementById('zoomOut').addEventListener('click', () => {
  viewPct = Math.min(100, Math.round(viewPct * 2));
  updateControls();
  refresh();
});
document.getElementById('zoomAll').addEventListener('click', () => {
  viewPct = 100;
  updateControls();
  refresh();
});
document.getElementById('windowRange').addEventListener('input', event => {
  viewPct = Number(event.target.value);
  updateControls();
  refresh();
});
document.getElementById('smoothRange').addEventListener('input', event => {
  smoothWindow = Number(event.target.value);
  updateControls();
  refresh();
});

async function refresh() {
  const res = await fetch('/api/state');
  const data = await res.json();
  const run = data.selected || {};
  const live = run.live || {};
  const latest = live.latest || run.latest_final || {};
  const history = live.history || run.final_history || run.checkpoint_history || [];
  const cfg = live.config || run.config || {};
  const total = Number(cfg.total_timesteps || data.total_timesteps || 0);
  const steps = Number(latest.agent_steps || run.latest_step || 0);
  const sps = Number(latest.SPS || run.estimated_sps || 0);
  document.getElementById('subtitle').textContent = `${data.env} | ${data.checkpoint_dir}`;
  document.getElementById('updated').textContent = `Updated ${new Date().toLocaleTimeString()}`;
  document.getElementById('run').textContent = short(run.run_id);
  document.getElementById('stage').innerHTML = `<span class="${run.status_class || 'dead'}">${run.status || 'unknown'}</span>`;
  document.getElementById('steps').textContent = num(steps);
  const phase = live.stage || run.status || '';
  const over = total && steps > total;
  document.getElementById('progress').textContent = total
    ? `${pct(Math.min(steps / total, 1))} of ${num(total)}${over ? `, ${num(steps - total)} eval steps` : ''}`
    : 'total unknown';
  document.getElementById('sps').textContent = num(sps);
  document.getElementById('eta').textContent = total && sps ? `${num((total - steps) / sps / 60)} min remaining` : 'ETA unavailable';
  document.getElementById('success').textContent = pct(Number(latest['env/success_rate']));
  document.getElementById('collision').textContent = pct(Number(latest['env/collision_rate']));
  document.getElementById('score').textContent = num(Number(latest['env/score']));
  document.getElementById('epLen').textContent = `episode length ${num(Number(latest['env/episode_length']))}`;
  drawChart(document.getElementById('quality'), 'Success / Collision', [
    {name:'success', color:'#42d392', data:series(history, 'env/success_rate')},
    {name:'collision', color:'#ff6b6b', data:series(history, 'env/collision_rate')},
  ]);
  drawChart(document.getElementById('return'), 'Return and Episode Length', [
    {name:'score', color:'#6ea8fe', data:series(history, 'env/score')},
    {name:'ep len', color:'#f4bf50', data:series(history, 'env/episode_length')},
  ]);
  drawChart(document.getElementById('loss'), 'Losses', [
    {name:'value', color:'#f4bf50', data:series(history, 'loss/value_loss')},
    {name:'entropy', color:'#24d6d6', data:series(history, 'loss/entropy')},
  ]);
  drawChart(document.getElementById('speed'), 'Throughput', [
    {name:'SPS', color:'#42d392', data:series(history, 'SPS')},
  ]);
  document.getElementById('runs').innerHTML = data.runs.map(r => `
    <tr>
      <td>${short(r.run_id)}</td>
      <td><span class="pill ${r.status_class}">${r.status}</span></td>
      <td>${num(r.latest_step)}</td>
      <td>${r.checkpoint_count}</td>
      <td>${r.updated_age_s == null ? '-' : `${num(r.updated_age_s)}s ago`}</td>
      <td class="sub">${r.latest_checkpoint || r.log_path || ''}</td>
    </tr>`).join('');
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


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

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            env = qs.get("env", [self.env])[0]
            payload = collect_state(env, self.checkpoint_dir, self.log_dir)
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
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

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Serve a local FieldNav training dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--env", default="field_nav")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    Handler.checkpoint_dir = args.checkpoint_dir
    Handler.log_dir = args.log_dir
    Handler.env = args.env
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard_url=http://{args.host}:{args.port}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"log_dir={args.log_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
