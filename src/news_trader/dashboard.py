"""
"AI News Trader" tab for the existing Flask dashboard (src/dashboard.py).

Registered as a Blueprint so the BTC dashboard code is untouched apart from
one register call and a tab link.  Reads the news engine's journal and
status file; settings are written to the ``news_trader`` section of
config.yaml after validation with the same pydantic model the engine uses.

Setting ``mode`` to ``live`` here does NOT authorise real-money trading; the
engine still requires every gate in execution.LiveAuthorization.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, render_template_string, request
from pydantic import ValidationError

from src.news_trader.analytics import performance_report
from src.news_trader.config import AI_MODEL_PRICES, CATEGORIES, NewsTraderSettings, load_settings
from src.news_trader.journal import EventRow, Journal, OrderRow, PositionRow, SignalRow
from src.news_trader.sources.registry import ALL_PROFILES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

news_bp = Blueprint("news_trader", __name__)

_journals: dict[str, Journal] = {}


def _config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _settings() -> NewsTraderSettings:
    s = load_settings(_config())
    for attr in ("db_path", "status_path", "halt_file"):
        p = Path(getattr(s, attr))
        if not p.is_absolute():
            setattr(s, attr, str((PROJECT_ROOT / p).resolve()))
    return s


def _journal() -> Journal:
    path = _settings().db_path
    if path not in _journals:
        _journals[path] = Journal(path)
    return _journals[path]


def _row(r) -> dict:
    d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
    for k in ("rejection_codes", "evidence", "flags", "corroborating"):
        if isinstance(d.get(k), str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


def _limit(default: int = 100) -> int:
    try:
        return max(1, min(500, int(request.args.get("limit", default))))
    except ValueError:
        return default


@news_bp.route("/api/news/status")
def news_status():
    s = _settings()
    path = Path(s.status_path)
    status: dict = {"engine": "not_running"}
    if path.exists():
        try:
            status = json.loads(path.read_text())
            age = time.time() - float(status.get("heartbeat", 0))
            status["heartbeat_age_seconds"] = round(age, 1)
            status["engine"] = "running" if age < 30 else "stale"
        except (OSError, ValueError, json.JSONDecodeError) as e:
            status = {"engine": "unknown", "error": str(e)}
    status["config"] = {"mode": s.mode, "enabled": s.enabled, "emergency_stop": s.emergency_stop,
                        "halt_file_present": Path(s.halt_file).exists(),
                        "global_emergency_stop": (PROJECT_ROOT / "EMERGENCY_STOP").exists()}
    return jsonify(status)


@news_bp.route("/api/news/events")
def news_events():
    rows = _journal().rows(EventRow, EventRow.first_seen_at, _limit(150),
                           where=EventRow.is_duplicate.is_(False))
    return jsonify([_row(r) for r in rows])


@news_bp.route("/api/news/signals")
def news_signals():
    rows = _journal().rows(SignalRow, SignalRow.decision_at, _limit(100))
    return jsonify([_row(r) for r in rows])


@news_bp.route("/api/news/positions")
def news_positions():
    return jsonify([_row(r) for r in _journal().rows(PositionRow, PositionRow.opened_at, _limit(200))])


@news_bp.route("/api/news/orders")
def news_orders():
    return jsonify([_row(r) for r in _journal().rows(OrderRow, OrderRow.submitted_at, _limit(100))])


@news_bp.route("/api/news/performance")
def news_performance():
    mode = request.args.get("mode", "paper")
    return jsonify(performance_report(_journal(), "live" if mode == "live" else "paper"))


EDITABLE_RISK = ("max_trade_usd", "daily_loss_limit_usd", "min_edge", "min_confidence",
                 "max_news_age_seconds", "max_spread")


@news_bp.route("/api/news/settings", methods=["GET", "POST"])
def news_settings():
    config = _config()
    section = copy.deepcopy(config.get("news_trader") or {})
    current = load_settings(config)
    if request.method == "GET":
        return jsonify({
            "settings": current.model_dump(),
            "options": {"categories": list(CATEGORIES), "sources": {k: p.name for k, p in ALL_PROFILES.items()},
                        "ai_models": {k: {"input_per_mtok": v[0], "output_per_mtok": v[1]}
                                      for k, v in AI_MODEL_PRICES.items()}},
        })

    payload = request.get_json(silent=True) or {}
    if "mode" in payload:
        section["mode"] = payload["mode"]
    if "enabled" in payload:
        section["enabled"] = bool(payload["enabled"])
    if "emergency_stop" in payload:
        section["emergency_stop"] = bool(payload["emergency_stop"])
    if "categories" in payload:
        section["categories"] = payload["categories"]
    if "sources_enabled" in payload:
        unknown = [k for k in payload["sources_enabled"] if k not in ALL_PROFILES]
        if unknown:
            return jsonify({"error": f"unknown sources: {unknown}"}), 400
        section.setdefault("sources", {})["enabled"] = payload["sources_enabled"]
    risk = section.setdefault("risk", {})
    for key in EDITABLE_RISK:
        if key in payload:
            try:
                risk[key] = float(payload[key])
            except (TypeError, ValueError):
                return jsonify({"error": f"{key} must be a number"}), 400
    ai = section.setdefault("ai", {})
    if "ai_model" in payload:
        if payload["ai_model"] not in AI_MODEL_PRICES:
            return jsonify({"error": f"unsupported model {payload['ai_model']}"}), 400
        ai["model"] = payload["ai_model"]
    if "ai_daily_budget_usd" in payload:
        try:
            ai["daily_budget_usd"] = float(payload["ai_daily_budget_usd"])
        except (TypeError, ValueError):
            return jsonify({"error": "ai_daily_budget_usd must be a number"}), 400

    try:
        validated = NewsTraderSettings.model_validate(section)
    except ValidationError as e:
        return jsonify({"error": "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())}), 400

    config["news_trader"] = section
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False))
    note = "Saved. Risk, AI and matching settings apply within seconds; mode and sources apply on restart."
    if validated.mode == "live":
        note += (" Live mode is only a request: real orders also require the code release flag, "
                 "--enable-live-news-trading, NEWS_TRADER_LIVE_ACK and a passing Polymarket geoblock check.")
    return jsonify({"status": "ok", "note": note, "settings": validated.model_dump()})


@news_bp.route("/api/news/halt", methods=["POST"])
def news_halt():
    s = _settings()
    Path(s.halt_file).write_text(f"Halted from dashboard at {datetime.now(timezone.utc).isoformat()}\n")
    return jsonify({"status": "halted", "message": "News engine halted (other strategies unaffected)"})


@news_bp.route("/api/news/resume", methods=["POST"])
def news_resume():
    s = _settings()
    if Path(s.halt_file).exists():
        Path(s.halt_file).unlink()
    return jsonify({"status": "resumed", "message": "News engine halt cleared (a loss-limit latch needs a restart)"})


@news_bp.route("/news")
def news_page():
    return render_template_string(NEWS_HTML)


NEWS_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polybot — AI News Trader</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0e17; color: #e0e6f0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #111827, #1a1f35); border-bottom: 1px solid #1e293b; padding: 16px 24px;
            display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 16px; }
  .header h1 { font-size: 22px; font-weight: 700; background: linear-gradient(90deg, #60a5fa, #a78bfa);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .tabs { display: flex; justify-content: center; gap: 10px; }
  .tab { padding: 8px 16px; border: 1px solid #2a3652; border-radius: 999px; background: #0f172a; color: #94a3b8;
         font-size: 13px; font-weight: 600; cursor: pointer; text-decoration: none; }
  .tab.active { background: linear-gradient(135deg, #2563eb, #7c3aed); color: #fff; border-color: transparent; }
  .status { justify-self: end; display: flex; align-items: center; gap: 10px; font-size: 14px; color: #94a3b8; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #64748b; }
  .dot.ok { background: #22c55e; box-shadow: 0 0 8px #22c55e66; } .dot.bad { background: #ef4444; } .dot.warn { background: #f59e0b; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .banner { border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; font-size: 13px; border: 1px solid; }
  .banner.paper { background: #0b1f3a; border-color: #1d4ed8; color: #bfdbfe; }
  .banner.halt { background: #2e0505; border-color: #ef4444; color: #fecaca; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; align-items: center; }
  .btn { padding: 10px 20px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; color: #fff; }
  .btn-stop { background: linear-gradient(135deg, #dc2626, #b91c1c); } .btn-resume { background: linear-gradient(135deg, #16a34a, #15803d); }
  .btn-save { background: linear-gradient(135deg, #2563eb, #7c3aed); } .btn-ghost { background: #1e293b; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 24px; }
  .card { background: linear-gradient(135deg, #111827, #151c2e); border: 1px solid #1e293b; border-radius: 12px; padding: 14px; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 6px; }
  .card .value { font-size: 22px; font-weight: 700; color: #60a5fa; } .pos { color: #22c55e !important; } .neg { color: #ef4444 !important; }
  .card .sub { font-size: 11px; color: #64748b; margin-top: 4px; }
  .section-title { font-size: 15px; font-weight: 600; margin: 8px 0 10px; color: #94a3b8; }
  .table-wrap { background: #111827; border: 1px solid #1e293b; border-radius: 12px; overflow-x: auto; margin-bottom: 24px; max-height: 460px; overflow-y: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th { text-align: left; padding: 10px 12px; background: #1a1f35; color: #64748b; font-weight: 600; text-transform: uppercase; font-size: 10.5px; position: sticky; top: 0; }
  td { padding: 8px 12px; border-top: 1px solid #1e293b; vertical-align: top; }
  .clip { max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .wrap { max-width: 420px; white-space: normal; color: #94a3b8; }
  .badge { padding: 2px 8px; border-radius: 20px; font-size: 10.5px; font-weight: 600; border: 1px solid #334155; color: #cbd5e1; white-space: nowrap; }
  .b-ok { color: #22c55e; border-color: #22c55e55; } .b-bad { color: #ef4444; border-color: #ef444455; } .b-warn { color: #f59e0b; border-color: #f59e0b55; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; } @media (max-width: 1000px) { .two { grid-template-columns: 1fr; } }
  .panel { background: linear-gradient(135deg, #111827, #151c2e); border: 1px solid #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 24px; }
  .panel h3 { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 12px; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 10px; font-size: 13px; }
  .row label { color: #94a3b8; min-width: 190px; }
  input, select { padding: 6px 8px; border-radius: 6px; border: 1px solid #2a3652; background: #0b1220; color: #e2e8f0; font-size: 13px; }
  input[type=number] { width: 110px; } .chk { display: inline-flex; gap: 4px; align-items: center; margin-right: 10px; color: #cbd5e1; }
  .muted { color: #64748b; font-size: 12px; } .kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 13px; }
</style>
</head>
<body>
<div class="header">
  <h1>⚡ Polybot Dashboard</h1>
  <div class="tabs">
    <a class="tab" href="/#paper">Paper</a><a class="tab" href="/#live">Live</a>
    <a class="tab active" href="/news">🤖 AI News Trader</a>
  </div>
  <div class="status"><div id="dot" class="dot"></div><span id="engineState">—</span></div>
</div>
<div class="container">
  <div id="modeBanner" class="banner paper">PAPER MODE — no real orders are sent. Live execution is disabled in code.</div>
  <div id="haltBanner" class="banner halt" style="display:none"></div>

  <div class="controls">
    <button class="btn btn-stop" onclick="post('/api/news/halt')">⏸ Halt news engine</button>
    <button class="btn btn-resume" onclick="post('/api/news/resume')">▶ Resume news engine</button>
    <button class="btn btn-stop" onclick="globalStop()">🛑 GLOBAL EMERGENCY STOP (all engines)</button>
    <span id="ctlMsg" class="muted"></span>
  </div>

  <div class="grid" id="kpis"></div>

  <div class="two">
    <div class="panel"><h3>Engine</h3><div class="kv" id="engineKv"></div></div>
    <div class="panel"><h3>News sources</h3>
      <div class="table-wrap" style="margin:0;max-height:280px"><table><thead><tr><th>Source</th><th>State</th><th>Latency</th><th>Last ok</th><th>Items</th><th>Last error</th></tr></thead><tbody id="sources"></tbody></table></div>
    </div>
  </div>

  <div class="section-title">AI signals &amp; decisions</div>
  <div class="table-wrap"><table><thead><tr>
    <th>Time</th><th>Market</th><th>Action</th><th>Decision</th><th>P(yes) AI</th><th>Conf</th><th>YES bid/ask</th>
    <th>Edge</th><th>Detect→decide</th><th>Explanation / rejection</th></tr></thead><tbody id="signals"></tbody></table></div>

  <div class="section-title">Paper positions</div>
  <div class="table-wrap"><table><thead><tr>
    <th>Opened</th><th>Market</th><th>Side</th><th>Shares</th><th>Entry</th><th>Cost</th><th>Mark (bid)</th><th>Unreal.</th><th>Status</th><th>Realized</th><th>Source</th></tr></thead><tbody id="positions"></tbody></table></div>

  <div class="section-title">Orders</div>
  <div class="table-wrap"><table><thead><tr>
    <th>Submitted</th><th>Mode</th><th>Outcome</th><th>Limit</th><th>Requested $</th><th>Status</th><th>Filled</th><th>Avg</th><th>Fees</th><th>Ask at submit</th><th>Error</th></tr></thead><tbody id="orders"></tbody></table></div>

  <div class="section-title">Incoming headlines</div>
  <div class="table-wrap"><table><thead><tr>
    <th>First seen</th><th>Age</th><th>Source</th><th>Category</th><th>Verification</th><th>Headline</th><th>Outcome</th><th>Reason</th></tr></thead><tbody id="events"></tbody></table></div>

  <div class="two">
    <div class="panel"><h3>Why opportunities were skipped</h3><div id="rejections" class="kv"></div></div>
    <div class="panel"><h3>Performance breakdown</h3><div id="breakdown"></div></div>
  </div>

  <div class="panel"><h3>Settings</h3>
    <div class="row"><label>Mode</label><select id="s_mode"><option value="paper">paper</option><option value="live">live (request only)</option></select>
      <span class="muted">A dashboard toggle never authorises real money; see live-authorization status above.</span></div>
    <div class="row"><label>News categories</label><span id="s_categories"></span></div>
    <div class="row"><label>Sources</label><span id="s_sources"></span></div>
    <div class="row"><label>Max trade size ($)</label><input type="number" step="1" min="1" id="s_max_trade_usd"></div>
    <div class="row"><label>Daily loss limit ($)</label><input type="number" step="1" min="1" id="s_daily_loss_limit_usd"></div>
    <div class="row"><label>Min edge after costs</label><input type="number" step="0.005" min="0" max="1" id="s_min_edge"></div>
    <div class="row"><label>Min AI confidence</label><input type="number" step="0.05" min="0" max="1" id="s_min_confidence"></div>
    <div class="row"><label>Max headline age (s)</label><input type="number" step="30" min="30" id="s_max_news_age_seconds"></div>
    <div class="row"><label>Max spread</label><input type="number" step="0.01" min="0.01" max="1" id="s_max_spread"></div>
    <div class="row"><label>AI model</label><select id="s_ai_model"></select>
      <label style="min-width:auto">daily AI budget ($)</label><input type="number" step="0.5" min="0" id="s_ai_daily_budget_usd"></div>
    <div class="row"><button class="btn btn-save" onclick="saveSettings()">Save settings</button><span id="saveMsg" class="muted"></span></div>
  </div>
  <div class="muted" style="text-align:center">Auto-refreshes every 5 seconds · journal: data/news_trader.db</div>
</div>

<script>
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d=3) => (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toFixed(d);
const money = v => (v === null || v === undefined) ? '—' : (v < 0 ? '-$' : '$') + Math.abs(v).toFixed(2);
const pct = v => (v === null || v === undefined) ? '—' : (v * 100).toFixed(1) + '%';
const t = ts => ts ? new Date(ts * 1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}) : '—';
const dur = s => (s === null || s === undefined) ? '—' : s < 120 ? s.toFixed(0) + 's' : s < 7200 ? (s/60).toFixed(1) + 'm' : (s/3600).toFixed(1) + 'h';
const badge = (txt, cls='') => `<span class="badge ${cls}">${esc(txt)}</span>`;
async function j(u) { const r = await fetch(u); return r.json(); }
async function post(u, body) {
  const r = await fetch(u, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
  const d = await r.json(); document.getElementById('ctlMsg').textContent = d.message || d.error || ''; refresh(); return d;
}
function globalStop() { if (confirm('Stop ALL engines (BTC and news)?')) post('/api/emergency-stop'); }
function card(label, value, cls='', sub='') { return `<div class="card"><div class="label">${esc(label)}</div><div class="value ${cls}">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ''}</div>`; }

async function refreshStatus() {
  const s = await j('/api/news/status');
  const dot = document.getElementById('dot'), st = document.getElementById('engineState');
  const halted = s.halted || s.config.emergency_stop || s.config.halt_file_present || s.config.global_emergency_stop;
  dot.className = 'dot ' + (s.engine === 'running' ? (halted ? 'warn' : 'ok') : 'bad');
  st.textContent = s.engine === 'running' ? (halted ? 'HALTED' : 'Running') : (s.engine === 'stale' ? 'Stale heartbeat' : 'Not running');
  const hb = document.getElementById('haltBanner');
  if (halted) { hb.style.display = 'block'; hb.textContent = 'HALTED: ' + (s.halted || (s.config.global_emergency_stop ? 'global EMERGENCY_STOP file' : 'news halt')); } else hb.style.display = 'none';
  const auth = s.live_authorization;
  document.getElementById('modeBanner').innerHTML = (s.mode === 'live')
    ? '<b>LIVE EXECUTION ACTIVE</b>'
    : `PAPER MODE — no real orders are sent.${auth && !auth.granted ? ' Live authorization: <b>NOT GRANTED</b> — ' + esc(auth.reasons.join('; ')) : ''}`;
  const ai = s.ai || {}, ex = s.exposure || {}, cat = s.catalog || {}, lat = s.latency || {};
  document.getElementById('engineKv').innerHTML = [
    ['Engine', esc(s.engine) + (s.heartbeat_age_seconds !== undefined ? ` (heartbeat ${dur(s.heartbeat_age_seconds)} ago)` : '')],
    ['Mode', esc(s.mode || s.config.mode)], ['Requested mode', esc(s.requested_mode || s.config.mode)],
    ['AI model', esc(ai.model) + (ai.unavailable_reason ? ` <span class="b-bad badge">${esc(ai.unavailable_reason)} → NO TRADE</span>` : '')],
    ['AI calls today', `${esc(ai.calls_today ?? '—')} / ${esc(ai.max_calls)}`],
    ['AI spend today', `${money(ai.spent_today_usd)} / ${money(ai.budget_usd)}`],
    ['Market catalog', cat.markets ? `${cat.markets} markets, refreshed ${dur(cat.age_seconds)} ago${cat.error ? ' — ' + esc(cat.error) : ''}` : '—'],
    ['Queue', esc(s.queue_size ?? '—')], ['Open positions', esc(ex.open_positions ?? '—')],
    ['Bankroll (paper)', money(ex.bankroll)], ['Median detect→decide', dur(lat.median_detection_to_decision)],
  ].map(([k, v]) => `<div class="muted">${k}</div><div>${v}</div>`).join('');
  document.getElementById('sources').innerHTML = (s.providers || []).map(p => `<tr><td>${esc(p.name)}</td>
    <td>${badge(p.state, p.state === 'ok' ? 'b-ok' : p.state === 'pending' ? '' : 'b-bad')}</td><td>${esc(p.latency_class)} (~${dur(p.expected_latency_seconds)})</td>
    <td>${t(p.last_success_at)}</td><td>${esc(p.items_seen)}</td><td class="clip" title="${esc(p.last_error)}">${esc(p.last_error)}</td></tr>`).join('')
    || '<tr><td colspan="6" class="muted">Engine not running</td></tr>';
}

async function refreshPerformance() {
  const r = await j('/api/news/performance');
  const p = r.performance || {}, f = r.fills || {}, c = r.calibration || {}, v = r.vs_immediate_entry || {};
  document.getElementById('kpis').innerHTML = [
    card('Net realized P&L', money(p.net_pnl), p.net_pnl >= 0 ? 'pos' : 'neg', `max drawdown ${money(p.max_drawdown)}`),
    card('Unrealized', money(p.unrealized_pnl), p.unrealized_pnl >= 0 ? 'pos' : 'neg', `${p.open_positions || 0} open`),
    card('Win rate', pct(p.win_rate), '', `${p.trades || 0} closed trades`),
    card('Avg P&L / trade', money(p.avg_pnl), '', `win ${money(p.avg_win)} · loss ${money(p.avg_loss)}`),
    card('Trade signals', esc(r.signals.trade_signals), '', `${r.signals.assessed} assessed · ${r.events.qualifying} qualifying events`),
    card('Fill rate', pct(f.fill_rate_by_shares), '', `${f.filled} full · ${f.partial} partial · ${f.unfilled} unfilled`),
    card('Events', esc(r.events.unique), '', `${r.events.total} incl. duplicates`),
    card('AI cost', money(r.ai.cost_usd), '', `${r.ai.calls} calls · ${r.ai.cache_hits} cached · ${r.ai.failures} failed`),
    card('Brier: AI vs market', `${num(c.brier_model)} / ${num(c.brier_market_mid)}`, c.model_beats_market ? 'pos' : '', `${c.resolved_signals} resolved signals`),
    card('vs immediate entry', money(v.delay_cost), '', `delay cost over ${v.positions_compared || 0} trades`),
  ].join('');
  const rc = r.signals.rejection_counts || {};
  document.getElementById('rejections').innerHTML = Object.keys(rc).length
    ? Object.entries(rc).map(([k, n]) => `<div>${badge(k)}</div><div>${n}</div>`).join('') : '<div class="muted">No rejected signals yet</div>';
  const sect = (title, obj) => `<div class="muted" style="margin:8px 0 4px">${title}</div>` + (Object.keys(obj || {}).length
    ? Object.entries(obj).map(([k, s]) => `<div class="kv"><div>${esc(k)}</div><div>${s.trades} trades · ${money(s.net_pnl)} · win ${pct(s.win_rate)}</div></div>`).join('')
    : '<div class="muted">—</div>');
  document.getElementById('breakdown').innerHTML = sect('By source', r.by_source) + sect('By market category', r.by_category)
    + sect('By detection→decision latency', r.by_latency)
    + `<div class="muted" style="margin-top:8px">Missed opportunities (rejected, moved ≥5c in predicted direction within 1h): ${r.missed_opportunities.count}</div>`;
}

async function refreshTables() {
  const [sig, pos, ord, ev] = await Promise.all([j('/api/news/signals?limit=60'), j('/api/news/positions?limit=100'),
                                                 j('/api/news/orders?limit=50'), j('/api/news/events?limit=120')]);
  document.getElementById('signals').innerHTML = sig.map(s => `<tr><td>${t(s.decision_at)}</td>
    <td class="clip" title="${esc(s.question)}">${esc(s.question)}</td><td>${badge(s.action, s.action === 'NO_TRADE' ? '' : 'b-warn')}</td>
    <td>${badge(s.decision, s.decision === 'TRADE' ? 'b-ok' : 'b-bad')}</td><td>${num(s.p_ai_yes, 2)} <span class="muted">[${num(s.p_low, 2)}–${num(s.p_high, 2)}]</span></td>
    <td>${num(s.confidence, 2)}</td><td>${num(s.yes_bid_at_match, 3)} / ${num(s.yes_ask_at_match, 3)}</td><td>${num(s.edge_after_costs, 3)}</td>
    <td>${dur(s.detection_to_decision)}</td>
    <td class="wrap">${s.decision === 'REJECT' ? '<b>' + esc((s.rejection_codes || []).join(', ')) + '</b>: ' + esc(s.rejection_detail) + '<br>' : ''}${esc(s.rule_analysis || s.ai_error || '')}</td></tr>`).join('')
    || '<tr><td colspan="10" class="muted">No signals yet</td></tr>';
  document.getElementById('positions').innerHTML = pos.map(p => `<tr><td>${t(p.opened_at)}</td>
    <td class="clip" title="${esc(p.question)}">${esc(p.question)}</td><td>${esc(p.outcome)} (${esc(p.outcome_label)})</td><td>${num(p.shares, 2)}</td>
    <td>${num(p.entry_price, 3)}</td><td>${money(p.entry_cost)}</td><td>${num(p.last_mark_bid, 3)}</td>
    <td class="${p.unrealized_pnl >= 0 ? 'pos' : 'neg'}">${money(p.unrealized_pnl)}</td><td>${badge(p.status, p.status === 'OPEN' ? 'b-warn' : 'b-ok')}</td>
    <td class="${p.realized_pnl >= 0 ? 'pos' : 'neg'}">${money(p.realized_pnl)}</td><td>${esc(p.source_name)}</td></tr>`).join('')
    || '<tr><td colspan="11" class="muted">No paper positions yet</td></tr>';
  document.getElementById('orders').innerHTML = ord.map(o => `<tr><td>${t(o.submitted_at)}</td><td>${esc(o.mode)}</td><td>${esc(o.outcome)}</td>
    <td>${num(o.limit_price, 3)}</td><td>${money(o.requested_cost)}</td><td>${badge(o.status, o.status === 'FILLED' ? 'b-ok' : o.status === 'PARTIAL' ? 'b-warn' : 'b-bad')}</td>
    <td>${num(o.filled_shares, 2)}</td><td>${num(o.avg_price, 3)}</td><td>${money(o.fees)}</td><td>${num(o.side_ask_at_submit, 3)}</td><td class="clip">${esc(o.error)}</td></tr>`).join('')
    || '<tr><td colspan="11" class="muted">No orders yet</td></tr>';
  const now = Date.now() / 1000;
  document.getElementById('events').innerHTML = ev.map(e => `<tr><td>${t(e.first_seen_at)}</td><td>${dur(now - (e.published_at || e.first_seen_at))}</td>
    <td>${esc(e.source_name)}</td><td>${esc(e.category)}</td>
    <td>${badge(e.verification, e.verification === 'PRIMARY_SOURCE' || e.verification === 'CORROBORATED' ? 'b-ok' : e.verification === 'CONFLICTING' ? 'b-bad' : '')}</td>
    <td class="clip" title="${esc(e.title)}"><a href="${esc(e.url)}" target="_blank" rel="noopener noreferrer" style="color:#93c5fd">${esc(e.title)}</a></td>
    <td>${badge(e.status, e.status === 'traded' ? 'b-ok' : '')}</td><td class="clip" title="${esc(e.status_detail)}">${esc(e.status_detail)}</td></tr>`).join('')
    || '<tr><td colspan="8" class="muted">No headlines yet</td></tr>';
}

async function loadSettings() {
  const d = await j('/api/news/settings'), s = d.settings, o = d.options;
  document.getElementById('s_mode').value = s.mode;
  document.getElementById('s_categories').innerHTML = o.categories.map(c => `<label class="chk"><input type="checkbox" name="cat" value="${esc(c)}" ${s.categories.includes(c) ? 'checked' : ''}>${esc(c)}</label>`).join('');
  document.getElementById('s_sources').innerHTML = Object.entries(o.sources).map(([k, n]) => `<label class="chk" title="${esc(n)}"><input type="checkbox" name="src" value="${esc(k)}" ${s.sources.enabled.includes(k) ? 'checked' : ''}>${esc(k)}</label>`).join('');
  for (const k of ['max_trade_usd', 'daily_loss_limit_usd', 'min_edge', 'min_confidence', 'max_news_age_seconds', 'max_spread'])
    document.getElementById('s_' + k).value = s.risk[k];
  document.getElementById('s_ai_model').innerHTML = Object.entries(o.ai_models).map(([m, p]) => `<option value="${esc(m)}" ${m === s.ai.model ? 'selected' : ''}>${esc(m)} ($${p.input_per_mtok}/$${p.output_per_mtok} per MTok)</option>`).join('');
  document.getElementById('s_ai_daily_budget_usd').value = s.ai.daily_budget_usd;
}

async function saveSettings() {
  const body = { mode: document.getElementById('s_mode').value,
    categories: [...document.querySelectorAll('input[name=cat]:checked')].map(x => x.value),
    sources_enabled: [...document.querySelectorAll('input[name=src]:checked')].map(x => x.value),
    ai_model: document.getElementById('s_ai_model').value,
    ai_daily_budget_usd: parseFloat(document.getElementById('s_ai_daily_budget_usd').value) };
  for (const k of ['max_trade_usd', 'daily_loss_limit_usd', 'min_edge', 'min_confidence', 'max_news_age_seconds', 'max_spread'])
    body[k] = parseFloat(document.getElementById('s_' + k).value);
  const r = await fetch('/api/news/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const d = await r.json(), m = document.getElementById('saveMsg');
  m.textContent = r.ok ? d.note : ('Error: ' + d.error); m.style.color = r.ok ? '#22c55e' : '#ef4444';
  if (r.ok) loadSettings();
}

function refresh() { refreshStatus().catch(console.error); refreshPerformance().catch(console.error); refreshTables().catch(console.error); }
loadSettings(); refresh(); setInterval(refresh, 5000);
</script>
</body>
</html>
"""
