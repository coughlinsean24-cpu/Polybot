"""
Live web dashboard for Polybot.
Runs as a separate process alongside the bot.
Reads from the same SQLite database and controls the EMERGENCY_STOP file.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template_string, request
from sqlalchemy import create_engine, text

app = Flask(__name__)

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
EMERGENCY_STOP_FILE = PROJECT_ROOT / "EMERGENCY_STOP"


def get_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_db_engine():
    config = get_config()
    db_path = PROJECT_ROOT / config["logging"]["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def _normalize_mode(raw: str | None) -> str:
    mode = (raw or "paper").lower()
    if mode not in {"paper", "live", "all"}:
        return "paper"
    return mode


def _trade_filter(mode: str) -> tuple[str, dict]:
    if mode in {"paper", "live"}:
        return " AND paper_trade = :paper_trade", {"paper_trade": mode == "paper"}
    return "", {}


# ── API Routes ──────────────────────────────────────────────────────────────────


@app.route("/api/stats")
def api_stats():
    """Return bot statistics as JSON."""
    engine = get_db_engine()
    mode = _normalize_mode(request.args.get("mode"))
    trade_filter, trade_params = _trade_filter(mode)
    with engine.connect() as conn:
        trades = conn.execute(
            text(
                "SELECT * FROM trades WHERE outcome != 'PENDING'"
                f"{trade_filter} ORDER BY id DESC"
            ),
            trade_params,
        ).mappings().all()

        pending = conn.execute(
            text(
                "SELECT COUNT(*) as c FROM trades WHERE outcome = 'PENDING'"
                f"{trade_filter}"
            ),
            trade_params,
        ).scalar()

        try:
            if mode in {"paper", "live"}:
                signals = conn.execute(
                    text(
                        "SELECT COUNT(*) as c FROM signals s "
                        "JOIN bot_sessions b ON s.session_id = b.session_id "
                        "WHERE b.mode = :mode"
                    ),
                    {"mode": mode},
                ).scalar()
                signals_traded = conn.execute(
                    text(
                        "SELECT COUNT(*) as c FROM signals s "
                        "JOIN bot_sessions b ON s.session_id = b.session_id "
                        "WHERE s.traded = 1 AND b.mode = :mode"
                    ),
                    {"mode": mode},
                ).scalar()
            else:
                signals = conn.execute(
                    text("SELECT COUNT(*) as c FROM signals")
                ).scalar()
                signals_traded = conn.execute(
                    text("SELECT COUNT(*) as c FROM signals WHERE traded = 1")
                ).scalar()
        except Exception:
            signals = conn.execute(
                text("SELECT COUNT(*) as c FROM signals")
            ).scalar()
            signals_traded = conn.execute(
                text("SELECT COUNT(*) as c FROM signals WHERE traded = 1")
            ).scalar()

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    total_pnl = sum(t["profit_loss"] for t in trades)
    total_wagered = sum(t["bet_size"] for t in trades)
    config = get_config()
    max_bet = config.get("strategy", {}).get("max_bet", 10.0)
    initial_bet = config.get("strategy", {}).get("initial_bet", 10.0)

    # Pending trades cost
    with engine.connect() as conn:
        pending_cost = conn.execute(
            text(
                "SELECT COALESCE(SUM(bet_size), 0) FROM trades "
                "WHERE outcome = 'PENDING'" + trade_filter
            ),
            trade_params,
        ).scalar()

    return jsonify({
        "total_trades": len(trades),
        "pending_trades": pending,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": f"{len(wins) / len(trades) * 100:.1f}%" if trades else "0%",
        "total_pnl": round(total_pnl, 2),
        "total_wagered": round(total_wagered, 2),
        "pending_at_risk": round(pending_cost, 2),
        "avg_win": round(sum(t["profit_loss"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t["profit_loss"] for t in losses) / len(losses), 2) if losses else 0,
        "largest_win": round(max((t["profit_loss"] for t in wins), default=0), 2),
        "largest_loss": round(min((t["profit_loss"] for t in losses), default=0), 2),
        "bet_size": initial_bet,
        "max_bet": max_bet,
        "signals_total": signals,
        "signals_traded": signals_traded,
        "emergency_stop": EMERGENCY_STOP_FILE.exists(),
    })


@app.route("/api/trades")
def api_trades():
    """Return recent trades as JSON."""
    engine = get_db_engine()
    mode = _normalize_mode(request.args.get("mode"))
    trade_filter, trade_params = _trade_filter(mode)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM trades"
                f" WHERE 1=1{trade_filter} ORDER BY id DESC LIMIT 50"
            ),
            trade_params,
        ).mappings().all()
    return jsonify([dict(r) for r in rows])


@app.route("/api/signals")
def api_signals():
    """Return recent signals as JSON."""
    engine = get_db_engine()
    mode = _normalize_mode(request.args.get("mode"))
    with engine.connect() as conn:
        try:
            if mode in {"paper", "live"}:
                rows = conn.execute(
                    text(
                        "SELECT s.* FROM signals s "
                        "JOIN bot_sessions b ON s.session_id = b.session_id "
                        "WHERE b.mode = :mode "
                        "ORDER BY s.id DESC LIMIT 50"
                    ),
                    {"mode": mode},
                ).mappings().all()
            else:
                rows = conn.execute(
                    text("SELECT * FROM signals ORDER BY id DESC LIMIT 50")
                ).mappings().all()
        except Exception:
            rows = conn.execute(
                text("SELECT * FROM signals ORDER BY id DESC LIMIT 50")
            ).mappings().all()
    return jsonify([dict(r) for r in rows])


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    config = get_config()
    if request.method == "GET":
        return jsonify(config)

    payload = request.get_json(silent=True) or {}
    strategy = config.setdefault("strategy", {})

    # Bet size (initial_bet + max_bet kept in sync for flat-style cap)
    if "initial_bet" in payload or "bet_size" in payload:
        raw = payload.get("initial_bet", payload.get("bet_size"))
        try:
            bet_size = float(raw)
        except (TypeError, ValueError):
            return jsonify({"error": "initial_bet must be a number"}), 400
        if bet_size <= 0:
            return jsonify({"error": "initial_bet must be > 0"}), 400
        strategy["initial_bet"] = bet_size
        strategy["max_bet"] = bet_size  # cap Kelly to this amount

    # Max bet override (independent of initial_bet)
    if "max_bet" in payload:
        try:
            max_bet = float(payload["max_bet"])
        except (TypeError, ValueError):
            return jsonify({"error": "max_bet must be a number"}), 400
        if max_bet <= 0:
            return jsonify({"error": "max_bet must be > 0"}), 400
        strategy["max_bet"] = max_bet

    # Max loss limit
    if "max_loss" in payload:
        try:
            max_loss = float(payload["max_loss"])
        except (TypeError, ValueError):
            return jsonify({"error": "max_loss must be a number"}), 400
        if max_loss <= 0:
            return jsonify({"error": "max_loss must be > 0"}), 400
        strategy["max_loss"] = max_loss

    # Straddle settings
    straddle = config.setdefault("straddle", {})
    if "straddle_enabled" in payload:
        straddle["enabled"] = bool(payload["straddle_enabled"])
    if "straddle_price" in payload:
        try:
            straddle["limit_price"] = float(payload["straddle_price"])
        except (TypeError, ValueError):
            pass
    if "straddle_shares" in payload:
        try:
            straddle["shares"] = int(payload["straddle_shares"])
        except (TypeError, ValueError):
            pass
    if "straddle_max_cost" in payload:
        try:
            straddle["max_cost"] = float(payload["straddle_max_cost"])
        except (TypeError, ValueError):
            pass
    if "straddle_trigger_seconds" in payload:
        try:
            straddle["trigger_seconds"] = int(payload["straddle_trigger_seconds"])
        except (TypeError, ValueError):
            pass

    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False))
    return jsonify({"status": "ok", "config": config})


@app.route("/api/positions")
def api_positions():
    """Return open positions (PENDING trades) as JSON."""
    engine = get_db_engine()
    mode = _normalize_mode(request.args.get("mode"))
    trade_filter, trade_params = _trade_filter(mode)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM trades WHERE outcome = 'PENDING'"
                f"{trade_filter} ORDER BY id DESC LIMIT 50"
            ),
            trade_params,
        ).mappings().all()
    return jsonify([dict(r) for r in rows])


# ── Kill Switch ─────────────────────────────────────────────────────────────────


@app.route("/api/emergency-stop", methods=["POST"])
def emergency_stop():
    """Activate the kill switch — bot will halt on next scan cycle."""
    EMERGENCY_STOP_FILE.write_text(
        f"Activated from dashboard at {datetime.now(timezone.utc).isoformat()}\n"
    )
    return jsonify({"status": "stopped", "message": "EMERGENCY STOP activated"})


@app.route("/api/resume", methods=["POST"])
def resume():
    """Remove the kill switch — bot will resume trading."""
    if EMERGENCY_STOP_FILE.exists():
        EMERGENCY_STOP_FILE.unlink()
    return jsonify({"status": "running", "message": "Emergency stop cleared"})


@app.route("/api/learner")
def learner_stats():
    """Return adaptive-learner summary loaded from its JSON state file."""
    state_path = PROJECT_ROOT / "data" / "adaptive_state.json"
    if not state_path.exists():
        return jsonify({"status": "no_data", "message": "No adaptive state yet"})
    try:
        raw = json.loads(state_path.read_text())
        # The JSON stores buckets as top-level keys (not under 'buckets')
        buckets = raw if isinstance(raw, dict) else {}
        # In case a future format nests under 'buckets'
        if "buckets" in buckets and isinstance(buckets["buckets"], dict):
            buckets = buckets["buckets"]

        MIN_SAMPLES = 2  # matches adaptive_learner.py
        summary = {}
        for key, b in buckets.items():
            wins = b.get("wins", 0)
            losses = b.get("losses", 0)
            total = wins + losses
            if total == 0:
                continue
            wr = wins / total
            adjusted = b.get("adjusted_min_edge", 0.01)
            pnl = b.get("total_pnl", 0.0)
            avg_edge = b.get("avg_edge", 0.0)

            # Parse 3-part bucket key: volatility|momentum|edge_strength
            parts = key.split("|")
            vol = parts[0] if len(parts) > 0 else "?"
            mom = parts[1] if len(parts) > 1 else "?"
            edg = parts[2] if len(parts) > 2 else "?"

            summary[key] = {
                "wins": wins,
                "losses": losses,
                "total": total,
                "win_rate": round(wr, 3),
                "avg_edge": round(avg_edge, 4),
                "adjusted_min_edge": round(adjusted, 4),
                "total_pnl": round(pnl, 2),
                "samples_needed": max(0, MIN_SAMPLES - total),
                "is_active": total >= MIN_SAMPLES,
                "volatility": vol,
                "momentum": mom,
                "edge_strength": edg,
            }

        total_recorded = sum(v["total"] for v in summary.values())
        active_buckets = sum(1 for v in summary.values() if v["is_active"])
        total_pnl = sum(v["total_pnl"] for v in summary.values())
        overall_wins = sum(v["wins"] for v in summary.values())
        overall_losses = sum(v["losses"] for v in summary.values())

        return jsonify({
            "status": "ok",
            "total_trades_recorded": total_recorded,
            "total_buckets": len(summary),
            "active_buckets": active_buckets,
            "min_samples": MIN_SAMPLES,
            "overall_wins": overall_wins,
            "overall_losses": overall_losses,
            "overall_pnl": round(total_pnl, 2),
            "buckets": summary,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ── Dashboard UI ────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polybot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0a0e17;
            color: #e0e6f0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #111827, #1a1f35);
            border-bottom: 1px solid #1e293b;
            padding: 16px 24px;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 16px;
        }
        .header h1 {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .status {
            display: flex;
            align-items: center;
            gap: 12px;
            justify-self: end;
        }
        .status-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.running { background: #22c55e; box-shadow: 0 0 8px #22c55e66; }
        .status-dot.stopped { background: #ef4444; box-shadow: 0 0 8px #ef444466; animation: none; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 24px; }

        /* Tabs */
        .tabs {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .tab {
            padding: 8px 16px;
            border: 1px solid #2a3652;
            border-radius: 999px;
            background: #0f172a;
            color: #94a3b8;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab.active {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: #fff;
            border-color: transparent;
            box-shadow: 0 6px 18px #2563eb44;
        }

        /* Controls */
        .controls-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            margin-bottom: 24px;
        }

        /* Kill Switch */
        .kill-switch-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-stop {
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            color: white;
            box-shadow: 0 4px 14px #dc262644;
        }
        .btn-stop:hover { transform: translateY(-1px); box-shadow: 0 6px 20px #dc262666; }
        .btn-resume {
            background: linear-gradient(135deg, #16a34a, #15803d);
            color: white;
            box-shadow: 0 4px 14px #16a34a44;
        }
        .btn-resume:hover { transform: translateY(-1px); box-shadow: 0 6px 20px #16a34a66; }
        .btn-save {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            box-shadow: 0 4px 14px #7c3aed44;
        }
        .btn-save:hover { transform: translateY(-1px); box-shadow: 0 6px 20px #7c3aed66; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

        .bet-control {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 12px 14px;
        }
        .bet-control label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #64748b;
        }
        .bet-control input {
            width: 120px;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #2a3652;
            background: #0b1220;
            color: #e2e8f0;
            font-size: 14px;
        }
        .bet-status {
            font-size: 12px;
            color: #94a3b8;
            min-width: 120px;
        }

        /* Straddle + Learner panels */
        .panel-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 28px;
        }
        @media (max-width: 900px) {
            .panel-row { grid-template-columns: 1fr; }
        }
        .panel {
            background: linear-gradient(135deg, #111827, #151c2e);
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 18px;
        }
        .panel h3 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
            margin-bottom: 14px;
        }
        .panel .row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        .panel .row label {
            font-size: 12px;
            color: #94a3b8;
            min-width: 100px;
        }
        .panel .row input {
            width: 100px;
            padding: 6px 8px;
            border-radius: 6px;
            border: 1px solid #2a3652;
            background: #0b1220;
            color: #e2e8f0;
            font-size: 13px;
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 42px;
            height: 22px;
        }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background: #374151; border-radius: 22px;
            transition: .3s;
        }
        .toggle-slider:before {
            content: ""; position: absolute;
            height: 16px; width: 16px;
            left: 3px; bottom: 3px;
            background: #e2e8f0; border-radius: 50%;
            transition: .3s;
        }
        .toggle-switch input:checked + .toggle-slider { background: #22c55e; }
        .toggle-switch input:checked + .toggle-slider:before { transform: translateX(20px); }
        .learner-bucket {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #1e293b;
            font-size: 13px;
        }
        .learner-bucket:last-child { border-bottom: none; }
        .learner-bucket .key { color: #94a3b8; font-family: monospace; }
        .learner-bucket .val { color: #60a5fa; font-weight: 600; }
        .learner-empty { color: #475569; font-style: italic; font-size: 13px; }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }
        .stat-card {
            background: linear-gradient(135deg, #111827, #151c2e);
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 18px;
        }
        .stat-card .label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
            margin-bottom: 6px;
        }
        .stat-card .value {
            font-size: 26px;
            font-weight: 700;
        }
        .stat-card .value.positive { color: #22c55e; }
        .stat-card .value.negative { color: #ef4444; }
        .stat-card .value.neutral  { color: #60a5fa; }

        /* Tables */
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #94a3b8;
        }
        .table-wrap {
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 12px;
            overflow-x: auto;
            margin-bottom: 28px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th {
            text-align: left;
            padding: 12px 14px;
            background: #1a1f35;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }
        td {
            padding: 10px 14px;
            border-top: 1px solid #1e293b;
        }
        tr:hover td { background: #151c2e; }
        .badge {
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-win { background: #052e1633; color: #22c55e; border: 1px solid #22c55e44; }
        .badge-loss { background: #2e050533; color: #ef4444; border: 1px solid #ef444444; }
        .badge-pending { background: #1e293b; color: #f59e0b; border: 1px solid #f59e0b44; }

        .refresh-note {
            text-align: center;
            font-size: 12px;
            color: #475569;
            margin-top: 16px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ Polybot Dashboard</h1>
        <div class="tabs">
            <button id="tabPaper" class="tab active" onclick="setMode('paper')">Paper</button>
            <button id="tabLive" class="tab" onclick="setMode('live')">Live</button>
        </div>
        <div class="status">
            <div id="statusDot" class="status-dot running"></div>
            <span id="statusText" style="font-size: 14px; color: #94a3b8;">Running</span>
        </div>
    </div>

    <div class="container">
        <div class="controls-bar">
            <div class="kill-switch-bar">
                <button id="btnStop" class="btn btn-stop" onclick="emergencyStop()">
                    🛑 EMERGENCY STOP
                </button>
                <button id="btnResume" class="btn btn-resume" onclick="resumeBot()">
                    ▶️ Resume Trading
                </button>
            </div>
            <div class="bet-control">
                <label for="betSizeInput">Bet Size $</label>
                <input id="betSizeInput" type="number" min="0.01" step="0.01" placeholder="1.00" />
            </div>
            <div class="bet-control">
                <label for="maxBetInput">Max Bet $</label>
                <input id="maxBetInput" type="number" min="0.01" step="0.01" placeholder="1.00" />
            </div>
            <div class="bet-control">
                <label for="maxLossInput">Max Loss $</label>
                <input id="maxLossInput" type="number" min="1" step="5" placeholder="50" />
            </div>
            <div style="display:flex;align-items:center;gap:10px">
                <button class="btn btn-save" onclick="saveConfig()">Save Config</button>
                <span id="betSaveStatus" class="bet-status">—</span>
            </div>
        </div>

        <!-- Stats -->
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card"><div class="label">Bot P&L</div><div class="value" id="totalPnl">$—</div></div>
            <div class="stat-card"><div class="label">At Risk (Pending)</div><div class="value neutral" id="pendingRisk">$—</div></div>
            <div class="stat-card"><div class="label">Win Rate</div><div class="value neutral" id="winRate">—</div></div>
            <div class="stat-card"><div class="label">Total Trades</div><div class="value neutral" id="totalTrades">—</div></div>
            <div class="stat-card"><div class="label">Wins / Losses</div><div class="value neutral" id="winsLosses">—</div></div>
            <div class="stat-card"><div class="label">Bet Size</div><div class="value neutral" id="currentBet">$—</div></div>
            <div class="stat-card"><div class="label">Avg Win</div><div class="value positive" id="avgWin">$—</div></div>
            <div class="stat-card"><div class="label">Avg Loss</div><div class="value negative" id="avgLoss">$—</div></div>
            <div class="stat-card"><div class="label">Signals (traded)</div><div class="value neutral" id="signals">—</div></div>
        </div>

        <!-- Straddle + Learner -->
        <div class="panel-row">
            <div class="panel">
                <h3>🎯 Straddle Strategy</h3>
                <div class="row">
                    <label>Enabled</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="straddleEnabled" />
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <div class="row">
                    <label>Trigger (sec)</label>
                    <input id="straddleTriggerSec" type="number" min="10" step="10" placeholder="120" />
                </div>
                <div class="row">
                    <label>Limit Price $</label>
                    <input id="straddlePrice" type="number" min="0.01" step="0.01" placeholder="0.05" />
                </div>
                <div class="row">
                    <label>Shares</label>
                    <input id="straddleShares" type="number" min="1" step="10" placeholder="500" />
                </div>
                <div class="row">
                    <label>Max Cost $</label>
                    <input id="straddleMaxCost" type="number" min="1" step="5" placeholder="50" />
                </div>
            </div>
            <div class="panel">
                <h3>🧠 Adaptive Learner</h3>
                <div id="learnerSummary" style="margin-bottom:10px;">
                    <p class="learner-empty">Loading…</p>
                </div>
                <div id="learnerContent"></div>
            </div>
        </div>

        <!-- Adaptive Learner Detail Table -->
        <div class="section-title">🧠 Adaptive Learner — Bucket Detail</div>
        <div id="learnerDetailWrap" class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Volatility</th>
                        <th>Momentum</th>
                        <th>Edge Str</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>P&L</th>
                        <th>Avg Edge</th>
                        <th>Adj Min Edge</th>
                        <th>Progress</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="learnerDetailBody"></tbody>
            </table>
        </div>

        <!-- Recent Trades -->
        <div class="section-title" id="tradesTitle">Recent Trades — Paper</div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Direction</th>
                        <th>Bet Size</th>
                        <th>Entry Price</th>
                        <th>Close Price</th>
                        <th>Edge</th>
                        <th>BTC Open</th>
                        <th>BTC Close</th>
                        <th>BTC Δ%</th>
                        <th>Outcome</th>
                        <th>P&L</th>
                        <th>Bankroll</th>
                    </tr>
                </thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>

        <!-- Open Positions -->
        <div class="section-title" id="positionsTitle">Positions — Paper</div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Market</th>
                        <th>Direction</th>
                        <th>Bet Size</th>
                        <th>Entry Price</th>
                        <th>Status</th>
                        <th>Order ID</th>
                    </tr>
                </thead>
                <tbody id="positionsBody"></tbody>
            </table>
        </div>

        <!-- Recent Signals -->
        <div class="section-title" id="signalsTitle">Recent Signals — Paper</div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Direction</th>
                        <th>Δ%</th>
                        <th>Confidence</th>
                        <th>BTC Start</th>
                        <th>BTC End</th>
                        <th>Traded?</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody id="signalsBody"></tbody>
            </table>
        </div>

        <div class="refresh-note">Auto-refreshes every 5 seconds</div>
    </div>

    <script>
        // Restore last-used tab from localStorage / URL hash
        let currentMode = (function() {
            const hash = location.hash.replace('#', '');
            if (hash === 'live' || hash === 'paper') return hash;
            return localStorage.getItem('polybot_tab') || 'paper';
        })();

        function setMode(mode) {
            currentMode = mode;
            localStorage.setItem('polybot_tab', mode);
            location.hash = mode;
            document.getElementById('tabPaper').classList.toggle('active', mode === 'paper');
            document.getElementById('tabLive').classList.toggle('active', mode === 'live');
            document.getElementById('tradesTitle').textContent = `Recent Trades — ${mode === 'paper' ? 'Paper' : 'Live'}`;
            document.getElementById('positionsTitle').textContent = mode === 'paper'
                ? 'Positions — Paper'
                : 'Live Polymarket Positions';
            document.getElementById('signalsTitle').textContent = `Recent Signals — ${mode === 'paper' ? 'Paper' : 'Live'}`;
            refreshAll();
        }

        function apiUrl(path) {
            const glue = path.includes('?') ? '&' : '?';
            return `${path}${glue}mode=${currentMode}`;
        }

        async function fetchJSON(url) {
            const resp = await fetch(url);
            return resp.json();
        }

        function fmtMoney(v) {
            const n = Number(v);
            const s = '$' + Math.abs(n).toFixed(2);
            return n < 0 ? '-' + s : s;
        }

        function fmtTime(ts) {
            if (!ts) return '—';
            const d = new Date(ts);
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        async function refreshStats() {
            try {
                const stats = await fetchJSON(apiUrl('/api/stats'));

                // Status
                const dot = document.getElementById('statusDot');
                const txt = document.getElementById('statusText');
                const btnStop = document.getElementById('btnStop');
                const btnResume = document.getElementById('btnResume');
                if (stats.emergency_stop) {
                    dot.className = 'status-dot stopped';
                    txt.textContent = 'STOPPED';
                    txt.style.color = '#ef4444';
                    btnStop.disabled = true;
                    btnResume.disabled = false;
                } else {
                    dot.className = 'status-dot running';
                    txt.textContent = 'Running';
                    txt.style.color = '#22c55e';
                    btnStop.disabled = false;
                    btnResume.disabled = true;
                }

                // Stats cards
                const pnlEl = document.getElementById('totalPnl');
                pnlEl.textContent = fmtMoney(stats.total_pnl);
                pnlEl.className = 'value ' + (stats.total_pnl >= 0 ? 'positive' : 'negative');
                const riskEl = document.getElementById('pendingRisk');
                riskEl.textContent = fmtMoney(stats.pending_at_risk);
                document.getElementById('winRate').textContent = stats.win_rate;
                document.getElementById('totalTrades').textContent = stats.total_trades + (stats.pending_trades ? ' (+' + stats.pending_trades + ' pending)' : '');
                document.getElementById('winsLosses').textContent = stats.wins + ' / ' + stats.losses;
                document.getElementById('currentBet').textContent = '$' + (stats.bet_size || 10).toFixed(2) + ' / max $' + (stats.max_bet || 10).toFixed(2);
                document.getElementById('avgWin').textContent = fmtMoney(stats.avg_win);
                document.getElementById('avgLoss').textContent = fmtMoney(stats.avg_loss);
                document.getElementById('signals').textContent = stats.signals_total + ' (' + stats.signals_traded + ')';
            } catch (e) {
                console.error('Stats refresh failed:', e);
            }
        }

        async function refreshTrades() {
            try {
                const trades = await fetchJSON(apiUrl('/api/trades'));
                const tbody = document.getElementById('tradesBody');
                tbody.innerHTML = trades.map(t => `
                    <tr>
                        <td>${fmtTime(t.timestamp)}</td>
                        <td>${t.direction || '—'}</td>
                        <td>${fmtMoney(t.bet_size)}</td>
                        <td>${t.fill_price ? t.fill_price.toFixed(3) : '—'}</td>
                        <td>${t.outcome === 'WIN' ? '$1.00' : (t.outcome === 'LOSS' ? '$0.00' : '—')}</td>
                        <td>${t.edge_estimate ? t.edge_estimate.toFixed(3) : '—'}</td>
                        <td>${t.btc_price_start ? '$' + t.btc_price_start.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '—'}</td>
                        <td>${t.btc_price_end ? '$' + t.btc_price_end.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '—'}</td>
                        <td>${t.delta_pct ? t.delta_pct.toFixed(3) + '%' : '—'}</td>
                        <td><span class="badge badge-${(t.outcome||'pending').toLowerCase()}">${t.outcome || 'PENDING'}</span></td>
                        <td style="color: ${t.profit_loss >= 0 ? '#22c55e' : '#ef4444'}">${fmtMoney(t.profit_loss)}</td>
                        <td>${fmtMoney(t.bankroll_after)}</td>
                    </tr>
                `).join('');
                if (!trades.length) {
                    tbody.innerHTML = '<tr><td colspan="12" style="text-align:center; color:#475569; padding:24px;">No trades yet</td></tr>';
                }
            } catch (e) {
                console.error('Trades refresh failed:', e);
            }
        }

        async function refreshPositions() {
            try {
                const positions = await fetchJSON(apiUrl('/api/positions'));
                const tbody = document.getElementById('positionsBody');
                tbody.innerHTML = positions.map(p => `
                    <tr>
                        <td>${fmtTime(p.timestamp)}</td>
                        <td title="${p.market_question || ''}" style="max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${p.market_question || '—'}</td>
                        <td>${p.direction || '—'}</td>
                        <td>${fmtMoney(p.bet_size)}</td>
                        <td>${p.fill_price ? p.fill_price.toFixed(3) : '—'}</td>
                        <td><span class="badge badge-pending">PENDING</span></td>
                        <td>${p.order_id || '—'}</td>
                    </tr>
                `).join('');
                if (!positions.length) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:#475569; padding:24px;">No open positions</td></tr>';
                }
            } catch (e) {
                console.error('Positions refresh failed:', e);
            }
        }

        async function refreshSignals() {
            try {
                const signals = await fetchJSON(apiUrl('/api/signals'));
                const tbody = document.getElementById('signalsBody');
                tbody.innerHTML = signals.map(s => `
                    <tr>
                        <td>${fmtTime(s.timestamp)}</td>
                        <td>${s.direction}</td>
                        <td>${s.delta_pct ? s.delta_pct.toFixed(3) + '%' : '—'}</td>
                        <td>${s.confidence ? s.confidence.toFixed(2) : '—'}</td>
                        <td>$${s.btc_price_start ? s.btc_price_start.toLocaleString() : '—'}</td>
                        <td>$${s.btc_price_end ? s.btc_price_end.toLocaleString() : '—'}</td>
                        <td>${s.traded ? '✅' : '—'}</td>
                        <td style="max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${s.reason || ''}">${s.reason || '—'}</td>
                    </tr>
                `).join('');
                if (!signals.length) {
                    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:#475569; padding:24px;">No signals yet</td></tr>';
                }
            } catch (e) {
                console.error('Signals refresh failed:', e);
            }
        }

        async function emergencyStop() {
            if (!confirm('Are you sure you want to EMERGENCY STOP the bot?')) return;
            await fetch('/api/emergency-stop', { method: 'POST' });
            refreshStats();
        }

        async function resumeBot() {
            await fetch('/api/resume', { method: 'POST' });
            refreshStats();
        }

        async function loadConfig() {
            try {
                const cfg = await fetchJSON('/api/config');
                const s = cfg.strategy || {};
                if (s.initial_bet != null)
                    document.getElementById('betSizeInput').value = Number(s.initial_bet).toFixed(2);
                if (s.max_bet != null)
                    document.getElementById('maxBetInput').value = Number(s.max_bet).toFixed(2);
                if (s.max_loss != null)
                    document.getElementById('maxLossInput').value = Number(s.max_loss).toFixed(2);
                // Straddle
                const st = cfg.straddle || {};
                document.getElementById('straddleEnabled').checked = !!st.enabled;
                if (st.trigger_seconds != null)
                    document.getElementById('straddleTriggerSec').value = st.trigger_seconds;
                if (st.limit_price != null)
                    document.getElementById('straddlePrice').value = st.limit_price;
                if (st.shares != null)
                    document.getElementById('straddleShares').value = st.shares;
                if (st.max_cost != null)
                    document.getElementById('straddleMaxCost').value = st.max_cost;
            } catch (e) {
                console.error('Config load failed:', e);
            }
        }

        async function saveConfig() {
            const status = document.getElementById('betSaveStatus');
            const betSize = parseFloat(document.getElementById('betSizeInput').value);
            const maxBet = parseFloat(document.getElementById('maxBetInput').value);
            const maxLoss = parseFloat(document.getElementById('maxLossInput').value);

            const body = {};
            if (betSize > 0) body.initial_bet = betSize;
            if (maxBet > 0) body.max_bet = maxBet;
            if (maxLoss > 0 && !isNaN(maxLoss)) body.max_loss = maxLoss;

            // Straddle fields
            body.straddle_enabled = document.getElementById('straddleEnabled').checked;
            const trigSec = parseInt(document.getElementById('straddleTriggerSec').value);
            if (trigSec > 0) body.straddle_trigger_seconds = trigSec;
            const sPrice = parseFloat(document.getElementById('straddlePrice').value);
            if (sPrice > 0) body.straddle_price = sPrice;
            const sShares = parseInt(document.getElementById('straddleShares').value);
            if (sShares > 0) body.straddle_shares = sShares;
            const sMaxCost = parseFloat(document.getElementById('straddleMaxCost').value);
            if (sMaxCost > 0) body.straddle_max_cost = sMaxCost;

            status.textContent = 'Saving...';
            status.style.color = '#94a3b8';
            try {
                const resp = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await resp.json();
                if (resp.ok) {
                    status.textContent = '✓ Saved — restart bot to apply';
                    status.style.color = '#22c55e';
                    loadConfig();
                } else {
                    status.textContent = data.error || 'Save failed';
                    status.style.color = '#ef4444';
                }
            } catch (e) {
                status.textContent = 'Save failed';
                status.style.color = '#ef4444';
            }
        }

        async function refreshLearner() {
            try {
                const data = await fetchJSON('/api/learner');
                const sumEl = document.getElementById('learnerSummary');
                const listEl = document.getElementById('learnerContent');
                const detailBody = document.getElementById('learnerDetailBody');

                if (data.status === 'no_data') {
                    sumEl.innerHTML = '<p class="learner-empty">No trades recorded yet — learner needs data.</p>';
                    listEl.innerHTML = '';
                    detailBody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:#475569;padding:24px;">No learner data yet</td></tr>';
                    return;
                }
                if (data.status !== 'ok') {
                    sumEl.innerHTML = '<p class="learner-empty">Error: ' + (data.message || 'unknown') + '</p>';
                    listEl.innerHTML = '';
                    detailBody.innerHTML = '';
                    return;
                }

                // Summary panel
                const pnlColor = data.overall_pnl >= 0 ? '#22c55e' : '#ef4444';
                const overallWR = data.overall_wins + data.overall_losses > 0
                    ? ((data.overall_wins / (data.overall_wins + data.overall_losses)) * 100).toFixed(1)
                    : '0.0';
                sumEl.innerHTML = `
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:13px;">
                        <div><span style="color:#64748b;">Observations:</span> <strong>${data.total_trades_recorded}</strong></div>
                        <div><span style="color:#64748b;">Buckets:</span> <strong>${data.total_buckets}</strong> (<span style="color:#60a5fa;">${data.active_buckets} active</span>)</div>
                        <div><span style="color:#64748b;">Win Rate:</span> <strong>${overallWR}%</strong> (${data.overall_wins}W / ${data.overall_losses}L)</div>
                        <div><span style="color:#64748b;">Net P&L:</span> <strong style="color:${pnlColor};">${data.overall_pnl >= 0 ? '+' : ''}$${data.overall_pnl.toFixed(2)}</strong></div>
                        <div style="grid-column:1/-1;color:#475569;font-size:12px;margin-top:4px;">
                            ${data.active_buckets > 0
                                ? '✅ Learner is <strong style="color:#22c55e;">actively adjusting</strong> strategy in ' + data.active_buckets + ' bucket(s)'
                                : '⏳ Collecting data — needs <strong>' + data.min_samples + ' trades per bucket</strong> before adjusting'}
                        </div>
                    </div>
                `;

                // Compact list in side panel (top 6)
                const buckets = data.buckets || {};
                const keys = Object.keys(buckets).sort((a, b) => buckets[b].total - buckets[a].total);
                let listHtml = '';
                for (const k of keys.slice(0, 6)) {
                    const b = buckets[k];
                    const wr = (b.win_rate * 100).toFixed(0);
                    const color = b.win_rate >= 0.5 ? '#22c55e' : '#ef4444';
                    const badge = b.is_active
                        ? '<span style="background:#052e1633;color:#22c55e;border:1px solid #22c55e44;padding:1px 6px;border-radius:10px;font-size:10px;">ACTIVE</span>'
                        : '<span style="background:#1e293b;color:#f59e0b;border:1px solid #f59e0b44;padding:1px 6px;border-radius:10px;font-size:10px;">' + b.samples_needed + ' more</span>';
                    const edgLabel = b.edge_strength === 'big' ? '💪' : b.edge_strength === 'decent' ? '👌' : '🤏';
                    listHtml += `<div class="learner-bucket">
                        <span class="key">${b.volatility} ${b.momentum} ${edgLabel}</span>
                        <span style="display:flex;align-items:center;gap:8px;">
                            <span class="val" style="color:${color};">${wr}%</span>
                            ${badge}
                        </span>
                    </div>`;
                }
                if (keys.length === 0) {
                    listHtml = '<p class="learner-empty">Waiting for trade outcomes…</p>';
                }
                listEl.innerHTML = listHtml;

                // Detail table
                let tableHtml = '';
                for (const k of keys) {
                    const b = buckets[k];
                    const wr = (b.win_rate * 100).toFixed(1);
                    const wrColor = b.win_rate >= 0.6 ? '#22c55e' : b.win_rate >= 0.45 ? '#f59e0b' : '#ef4444';
                    const pnlC = b.total_pnl >= 0 ? '#22c55e' : '#ef4444';
                    const progress = Math.min(b.total / data.min_samples, 1.0);
                    const progressPct = (progress * 100).toFixed(0);
                    const progressColor = b.is_active ? '#22c55e' : '#3b82f6';

                    // Status badge
                    let statusBadge;
                    if (b.is_active && b.win_rate < 0.35 && b.total >= 6) {
                        statusBadge = '<span class="badge" style="background:#2e050533;color:#ef4444;border:1px solid #ef444444;">⛔ SKIP</span>';
                    } else if (b.is_active) {
                        statusBadge = '<span class="badge" style="background:#052e1633;color:#22c55e;border:1px solid #22c55e44;">✅ ACTIVE</span>';
                    } else {
                        statusBadge = '<span class="badge badge-pending">⏳ ' + b.samples_needed + ' more</span>';
                    }

                    // Momentum emoji
                    const momEmoji = b.momentum === 'trending' ? '📈' : b.momentum === 'reverting' ? '📉' : '🔄';
                    // Volatility emoji
                    const volEmoji = b.volatility === 'strong' ? '🔥' : b.volatility === 'moderate' ? '〰️' : '🧊';
                    // Edge strength emoji
                    const edgEmoji = b.edge_strength === 'big' ? '💪' : b.edge_strength === 'decent' ? '👌' : '🤏';

                    tableHtml += `<tr>
                        <td>${volEmoji} ${b.volatility}</td>
                        <td>${momEmoji} ${b.momentum}</td>
                        <td>${edgEmoji} ${b.edge_strength}</td>
                        <td>${b.total} (${b.wins}W/${b.losses}L)</td>
                        <td style="color:${wrColor};font-weight:600;">${wr}%</td>
                        <td style="color:${pnlC};font-weight:600;">${b.total_pnl >= 0 ? '+' : ''}$${b.total_pnl.toFixed(2)}</td>
                        <td>${b.avg_edge.toFixed(4)}</td>
                        <td>${b.adjusted_min_edge.toFixed(4)}</td>
                        <td>
                            <div style="background:#1e293b;border-radius:6px;height:8px;width:80px;overflow:hidden;" title="${progressPct}%">
                                <div style="background:${progressColor};height:100%;width:${progressPct}%;border-radius:6px;transition:width 0.3s;"></div>
                            </div>
                        </td>
                        <td>${statusBadge}</td>
                    </tr>`;
                }
                if (keys.length === 0) {
                    tableHtml = '<tr><td colspan="10" style="text-align:center;color:#475569;padding:24px;">No learner data yet — make some trades!</td></tr>';
                }
                detailBody.innerHTML = tableHtml;
            } catch (e) {
                console.error('Learner refresh failed:', e);
            }
        }

        // Initial load + auto-refresh
        function refreshAll() {
            refreshStats();
            refreshTrades();
            refreshPositions();
            refreshSignals();
            refreshLearner();
        }
        loadConfig();
        setMode(currentMode);  // apply restored tab on load
        setInterval(refreshAll, 5000);
    </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ── Entry Point ─────────────────────────────────────────────────────────────────

def main():
    port = int(os.environ.get("DASHBOARD_PORT", 8050))
    print(f"\n  ⚡ Polybot Dashboard running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
