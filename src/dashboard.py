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
PARLAY_STATE_FILE = PROJECT_ROOT / "data" / "parlay_state.json"


def get_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_db_engine():
    config = get_config()
    db_path = PROJECT_ROOT / config["logging"]["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


# -- API Routes ------------------------------------------------------------------


@app.route("/api/stats")
def api_stats():
    """Return bot statistics as JSON."""
    engine = get_db_engine()
    with engine.connect() as conn:
        trades = conn.execute(
            text(
                "SELECT * FROM trades WHERE outcome != 'PENDING'"
                " ORDER BY id DESC"
            ),
        ).mappings().all()

        pending = conn.execute(
            text(
                "SELECT COUNT(*) as c FROM trades WHERE outcome = 'PENDING'"
            ),
        ).scalar()

        try:
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
                "WHERE outcome = 'PENDING'"
            ),
        ).scalar()

    return jsonify({
        "total_trades": len(trades),
        "pending_trades": pending,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": f"{len(wins) / len(trades) * 100:.1f}%" if trades else "0%",
        "total_pnl": round(total_pnl, 2),
        "total_wagered": round(total_wagered, 2),
        "roi_pct": round((total_pnl / total_wagered) * 100, 2) if total_wagered else 0,
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
        "bankroll": round(trades[0]["bankroll_after"], 2) if trades and trades[0].get("bankroll_after") else round(config.get("strategy", {}).get("initial_bankroll", 200.0), 2),
        "last_trade_time": str(trades[0]["timestamp"]) if trades else None,
    })


@app.route("/api/trades")
def api_trades():
    """Return recent trades as JSON."""
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM trades"
                " ORDER BY id DESC LIMIT 50"
            ),
        ).mappings().all()
    return jsonify([dict(r) for r in rows])


@app.route("/api/signals")
def api_signals():
    """Return recent signals as JSON."""
    engine = get_db_engine()
    with engine.connect() as conn:
        try:
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
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM trades WHERE outcome = 'PENDING'"
                " ORDER BY id DESC LIMIT 50"
            ),
        ).mappings().all()
    return jsonify([dict(r) for r in rows])


# -- Kill Switch -----------------------------------------------------------------


@app.route("/api/emergency-stop", methods=["POST"])
def emergency_stop():
    """Activate the kill switch -- bot will halt on next scan cycle."""
    EMERGENCY_STOP_FILE.write_text(
        f"Activated from dashboard at {datetime.now(timezone.utc).isoformat()}\n"
    )
    return jsonify({"status": "stopped", "message": "EMERGENCY STOP activated"})


@app.route("/api/resume", methods=["POST"])
def resume():
    """Remove the kill switch -- bot will resume trading."""
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


# -- Double It & Pass (Parlay) ---------------------------------------------------


def _read_parlay_state():
    """Read parlay state from the shared JSON file."""
    if not PARLAY_STATE_FILE.exists():
        return {
            "active": False, "round": 0, "current_stake": 1.0,
            "streak_profit": 0.0, "initial_stake": 1.0,
            "max_rounds": 8, "take_profit": 100.0,
            "lifetime_profit": 0.0, "lifetime_lost": 0.0,
            "lifetime_net": 0.0, "history": [],
        }
    try:
        return json.loads(PARLAY_STATE_FILE.read_text())
    except Exception:
        return {"active": False, "error": "Could not read state file"}


@app.route("/api/parlay")
def parlay_status():
    """Return current parlay state."""
    cfg = get_config().get("parlay", {})
    state = _read_parlay_state()
    state["enabled"] = cfg.get("enabled", False)
    return jsonify(state)


@app.route("/api/parlay/start", methods=["POST"])
def parlay_start():
    """Signal the bot to start a parlay streak."""
    ctrl_path = PROJECT_ROOT / "data" / "parlay_control.json"
    ctrl_path.write_text(json.dumps({"action": "start"}))
    return jsonify({"status": "ok", "message": "Parlay start signal sent"})


@app.route("/api/parlay/pass", methods=["POST"])
def parlay_pass():
    """Signal the bot to cash out (pass) the current parlay streak."""
    ctrl_path = PROJECT_ROOT / "data" / "parlay_control.json"
    ctrl_path.write_text(json.dumps({"action": "pass"}))
    return jsonify({"status": "ok", "message": "Parlay pass signal sent"})


@app.route("/api/parlay/stop", methods=["POST"])
def parlay_stop():
    """Signal the bot to deactivate parlay mode entirely."""
    ctrl_path = PROJECT_ROOT / "data" / "parlay_control.json"
    ctrl_path.write_text(json.dumps({"action": "stop"}))
    return jsonify({"status": "ok", "message": "Parlay stop signal sent"})


# -- Dashboard UI ----------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moondog Terminal</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

        :root {
            --bg: #030a06;
            --bg2: #071310;
            --bg3: #0c1f18;
            --border: #0f3d2a;
            --border-glow: #00ff6644;
            --green: #00ff66;
            --green-dim: #00cc52;
            --green-dark: #00993d;
            --green-muted: #1a6b3a;
            --red: #ff3348;
            --red-dim: #cc2939;
            --amber: #ffb800;
            --amber-dim: #cc9300;
            --cyan: #00e5ff;
            --text: #b8f0cc;
            --text-dim: #5a8a6a;
            --text-muted: #2d5e40;
            --mono: 'Share Tech Mono', 'Consolas', monospace;
            --sans: 'Rajdhani', 'Segoe UI', system-ui, sans-serif;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--sans);
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }

        /* Matrix rain canvas */
        #matrixCanvas {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: 0;
            opacity: 0.06;
            pointer-events: none;
        }

        /* All content above the rain */
        .app-wrap { position: relative; z-index: 1; }

        /* ─── Header ─── */
        .header {
            background: linear-gradient(180deg, #081a10 0%, #040d08 100%);
            border-bottom: 1px solid var(--border);
            padding: 14px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 20px #00ff6612;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .logo-icon {
            font-size: 32px;
            filter: drop-shadow(0 0 6px #00ff6666);
        }
        .logo-text {
            font-family: var(--mono);
            font-size: 24px;
            font-weight: 700;
            color: var(--green);
            text-shadow: 0 0 16px #00ff6644, 0 0 40px #00ff6622;
            letter-spacing: 2px;
        }
        .logo-sub {
            font-family: var(--mono);
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 3px;
            text-transform: uppercase;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--bg2);
            font-family: var(--mono);
            font-size: 13px;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
        }
        .status-dot.running {
            background: var(--green);
            box-shadow: 0 0 8px var(--green);
            animation: blink 2s infinite;
        }
        .status-dot.stopped {
            background: var(--red);
            box-shadow: 0 0 8px var(--red);
            animation: none;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .clock {
            font-family: var(--mono);
            font-size: 13px;
            color: var(--text-dim);
        }

        /* ─── Container ─── */
        .container { max-width: 1280px; margin: 0 auto; padding: 24px; }

        /* ─── Controls Bar ─── */
        .controls-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 24px;
            padding: 14px 16px;
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .btn {
            padding: 10px 22px;
            border: none;
            border-radius: 6px;
            font-family: var(--mono);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.15s;
        }
        .btn-stop {
            background: var(--red);
            color: #fff;
            box-shadow: 0 0 12px #ff334844;
        }
        .btn-stop:hover { box-shadow: 0 0 20px #ff334888; }
        .btn-resume {
            background: var(--green-dark);
            color: #fff;
            box-shadow: 0 0 12px #00ff6622;
        }
        .btn-resume:hover { box-shadow: 0 0 20px #00ff6644; }
        .btn-save {
            background: var(--green-muted);
            color: var(--green);
            border: 1px solid var(--green-dark);
        }
        .btn-save:hover { background: var(--green-dark); }
        .btn:disabled { opacity: 0.3; cursor: not-allowed; }

        .ctrl-input {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .ctrl-input label {
            font-family: var(--mono);
            font-size: 11px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .ctrl-input input {
            width: 90px;
            padding: 7px 10px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: var(--bg);
            color: var(--green);
            font-family: var(--mono);
            font-size: 13px;
        }
        .ctrl-input input:focus {
            outline: none;
            border-color: var(--green-dark);
            box-shadow: 0 0 8px #00ff6622;
        }
        .save-status {
            font-family: var(--mono);
            font-size: 11px;
            color: var(--text-dim);
            min-width: 100px;
        }
        .controls-spacer { flex: 1; }

        /* ─── Stats Grid ─── */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 2px;
            background: var(--green-dark);
            opacity: 0.5;
        }
        .stat-card .lbl {
            font-family: var(--mono);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-dim);
            margin-bottom: 6px;
        }
        .stat-card .val {
            font-family: var(--mono);
            font-size: 22px;
            font-weight: 700;
        }
        .stat-card .val.pos { color: var(--green); text-shadow: 0 0 10px #00ff6633; }
        .stat-card .val.neg { color: var(--red); text-shadow: 0 0 10px #ff334833; }
        .stat-card .val.neu { color: var(--text); }
        .stat-card .val.warn { color: var(--amber); }
        .stat-card.hero {
            border-color: var(--green-dark);
            background: linear-gradient(135deg, #071a10 0%, #0a2618 100%);
        }
        .stat-card.hero::before { background: var(--green); opacity: 0.8; }

        /* ─── Panels (Straddle + Learner) ─── */
        .panel-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }
        @media (max-width: 900px) { .panel-row { grid-template-columns: 1fr; } }
        .panel {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px;
        }
        .panel-title {
            font-family: var(--mono);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--green-dim);
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }
        .panel .row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        .panel .row label {
            font-family: var(--mono);
            font-size: 11px;
            color: var(--text-dim);
            min-width: 110px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .panel .row input {
            width: 100px;
            padding: 6px 8px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: var(--bg);
            color: var(--green);
            font-family: var(--mono);
            font-size: 13px;
        }
        .panel .row input:focus {
            outline: none;
            border-color: var(--green-dark);
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 42px; height: 22px;
        }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background: #1a3328; border-radius: 22px;
            transition: .3s;
        }
        .toggle-slider:before {
            content: ""; position: absolute;
            height: 16px; width: 16px;
            left: 3px; bottom: 3px;
            background: var(--text-dim); border-radius: 50%;
            transition: .3s;
        }
        .toggle-switch input:checked + .toggle-slider { background: var(--green-dark); }
        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(20px);
            background: var(--green);
        }

        /* ─── Learner Display ─── */
        .learner-summary {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 13px;
            margin-bottom: 12px;
        }
        .learner-summary .kv {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
        }
        .learner-summary .k { color: var(--text-dim); font-size: 12px; }
        .learner-summary .v { font-family: var(--mono); font-weight: 600; }
        .learner-note {
            font-family: var(--mono);
            font-size: 11px;
            color: var(--text-dim);
            padding: 8px 10px;
            background: var(--bg);
            border-radius: 4px;
            border-left: 3px solid var(--green-dark);
            margin-bottom: 12px;
        }
        .learner-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid #0f2a1e;
            font-size: 13px;
        }
        .learner-bar:last-child { border-bottom: none; }
        .learner-tag {
            font-family: var(--mono);
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .learner-tag.active { background: #0a3d2044; color: var(--green); border: 1px solid var(--green-dark); }
        .learner-tag.learning { background: #3d2a0a44; color: var(--amber); border: 1px solid var(--amber-dim); }
        .learner-empty { color: var(--text-muted); font-family: var(--mono); font-size: 12px; }

        /* ─── Section Titles ─── */
        .section-title {
            font-family: var(--mono);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--green-dim);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-title::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        /* ─── Tables ─── */
        .table-wrap {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 24px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th {
            text-align: left;
            padding: 10px 14px;
            background: var(--bg3);
            color: var(--text-dim);
            font-family: var(--mono);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--border);
        }
        td {
            padding: 9px 14px;
            border-top: 1px solid #0a1f16;
            font-family: var(--mono);
            font-size: 12px;
        }
        tr:hover td { background: #081a1044; }

        .badge {
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            font-family: var(--mono);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-win { background: #00ff6615; color: var(--green); border: 1px solid var(--green-dark); }
        .badge-loss { background: #ff334815; color: var(--red); border: 1px solid var(--red-dim); }
        .badge-pending { background: #ffb80015; color: var(--amber); border: 1px solid var(--amber-dim); }
        .badge-cancelled { background: #ffffff10; color: var(--text-dim); border: 1px solid #2d5e40; }

        /* ─── Learner Detail Table ─── */
        .learner-detail-section { margin-bottom: 24px; }
        .learner-help {
            font-family: var(--mono);
            font-size: 11px;
            color: var(--text-dim);
            margin-bottom: 10px;
            padding: 10px 14px;
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 6px;
            line-height: 1.6;
        }
        .learner-help strong { color: var(--text); }

        /* Progress bar */
        .prog-bar {
            background: #0a1f16;
            border-radius: 4px;
            height: 6px;
            width: 70px;
            overflow: hidden;
        }
        .prog-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }

        /* ─── Footer ─── */
        .footer {
            text-align: center;
            font-family: var(--mono);
            font-size: 11px;
            color: var(--text-muted);
            padding: 16px;
        }

        /* ─── Responsive ─── */
        @media (max-width: 768px) {
            .header { flex-direction: column; gap: 10px; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .controls-bar { flex-direction: column; align-items: stretch; }
        }
    </style>
</head>
<body>
    <canvas id="matrixCanvas"></canvas>
    <div class="app-wrap">
        <div class="header">
            <div class="logo">
                <span class="logo-icon">&#x1F31D;</span>
                <div>
                    <div class="logo-text">MOONDOG</div>
                    <div class="logo-sub">Polymarket Trading Terminal</div>
                </div>
            </div>
            <div class="header-right">
                <div class="clock" id="clock">--:--:--</div>
                <div class="status-badge">
                    <div id="statusDot" class="status-dot running"></div>
                    <span id="statusText">ONLINE</span>
                </div>
            </div>
        </div>

        <div class="container">
            <!-- Controls -->
            <div class="controls-bar">
                <button id="btnStop" class="btn btn-stop" onclick="emergencyStop()">&#x26A0; KILL SWITCH</button>
                <button id="btnResume" class="btn btn-resume" onclick="resumeBot()">&#x25B6; RESUME</button>
                <div class="controls-spacer"></div>
                <div class="ctrl-input">
                    <label>Bet $</label>
                    <input id="betSizeInput" type="number" min="0.01" step="0.01" placeholder="1.00" />
                </div>
                <div class="ctrl-input">
                    <label>Max $</label>
                    <input id="maxBetInput" type="number" min="0.01" step="0.01" placeholder="1.00" />
                </div>
                <div class="ctrl-input">
                    <label>Loss Limit $</label>
                    <input id="maxLossInput" type="number" min="1" step="5" placeholder="50" />
                </div>
                <button class="btn btn-save" onclick="saveConfig()">SAVE</button>
                <span id="betSaveStatus" class="save-status">--</span>
            </div>

            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card hero"><div class="lbl">Net Profit / Loss</div><div class="val" id="totalPnl">$--</div></div>
                <div class="stat-card"><div class="lbl">Bankroll</div><div class="val neu" id="bankroll">$--</div></div>
                <div class="stat-card"><div class="lbl">ROI</div><div class="val" id="roi">--</div></div>
                <div class="stat-card"><div class="lbl">At Risk</div><div class="val" id="pendingRisk">$--</div></div>
                <div class="stat-card"><div class="lbl">Win Rate</div><div class="val" id="winRate">--</div></div>
                <div class="stat-card"><div class="lbl">Trades</div><div class="val neu" id="totalTrades">--</div></div>
                <div class="stat-card"><div class="lbl">W / L</div><div class="val neu" id="winsLosses">--</div></div>
                <div class="stat-card"><div class="lbl">Bet Size</div><div class="val neu" id="currentBet">$--</div></div>
                <div class="stat-card"><div class="lbl">Avg Win</div><div class="val pos" id="avgWin">$--</div></div>
                <div class="stat-card"><div class="lbl">Avg Loss</div><div class="val neg" id="avgLoss">$--</div></div>
                <div class="stat-card"><div class="lbl">Last Trade</div><div class="val neu" id="lastTrade" style="font-size:16px;">--</div></div>
                <div class="stat-card"><div class="lbl">Signals (Acted)</div><div class="val neu" id="signals">--</div></div>
            </div>

            <!-- Double It & Pass -->
            <div class="section-title">&#x1F3B0; Double It &amp; Pass</div>
            <div class="panel" style="margin-bottom:24px;">
                <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;">
                    <div style="flex:1;min-width:200px;">
                        <div class="learner-note" id="parlayExplain" style="display:block;margin-bottom:10px;">
                            Start with <strong style="color:var(--green);">$1</strong>, double your bet on every win.
                            Keep going or <strong>PASS</strong> to lock in profits.
                            If you lose, you only lost that initial dollar. Repeat.
                        </div>
                        <div style="display:flex;gap:10px;margin-bottom:12px;">
                            <button class="btn btn-resume" id="btnParlayStart" onclick="parlayStart()">&#x25B6; START</button>
                            <button class="btn btn-save" id="btnParlayPass" onclick="parlayPass()">&#x1F4B0; PASS</button>
                            <button class="btn btn-stop" id="btnParlayStop" onclick="parlayStop()" style="font-size:11px;padding:8px 14px;">STOP</button>
                        </div>
                        <div id="parlayStatus" style="font-family:var(--mono);font-size:13px;color:var(--text-dim);">Loading...</div>
                    </div>
                    <div style="flex:1;min-width:240px;">
                        <div class="stats-grid" style="grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:0;">
                            <div class="stat-card" style="padding:10px;">
                                <div class="lbl" style="font-size:9px;">Round</div>
                                <div class="val neu" id="parlayRound" style="font-size:20px;">0</div>
                            </div>
                            <div class="stat-card" style="padding:10px;">
                                <div class="lbl" style="font-size:9px;">Current Bet</div>
                                <div class="val" id="parlayStake" style="font-size:20px;">$1</div>
                            </div>
                            <div class="stat-card" style="padding:10px;">
                                <div class="lbl" style="font-size:9px;">Streak Profit</div>
                                <div class="val" id="parlayProfit" style="font-size:20px;">$0</div>
                            </div>
                            <div class="stat-card" style="padding:10px;">
                                <div class="lbl" style="font-size:9px;">Lifetime Won</div>
                                <div class="val pos" id="parlayLifeWon" style="font-size:16px;">$0</div>
                            </div>
                            <div class="stat-card" style="padding:10px;">
                                <div class="lbl" style="font-size:9px;">Lifetime Lost</div>
                                <div class="val neg" id="parlayLifeLost" style="font-size:16px;">$0</div>
                            </div>
                            <div class="stat-card" style="padding:10px;">
                                <div class="lbl" style="font-size:9px;">Net</div>
                                <div class="val" id="parlayLifeNet" style="font-size:16px;">$0</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div id="parlayHistory" style="margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--text-dim);"></div>
            </div>

            <!-- Straddle + Learner Panels -->
            <div class="panel-row">
                <div class="panel">
                    <div class="panel-title">&#x1F3AF; Straddle Strategy</div>
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
                    <div class="panel-title">&#x1F9E0; AI Pattern Learner</div>
                    <div id="learnerSummary" class="learner-summary"></div>
                    <div id="learnerNote" class="learner-note" style="display:none;"></div>
                    <div id="learnerContent"></div>
                </div>
            </div>

            <!-- Learner Detail -->
            <div class="learner-detail-section">
                <div class="section-title">&#x1F9E0; Pattern Learner &mdash; All Scenarios</div>
                <div id="learnerHelp" class="learner-help">
                    The bot groups every trade into a <strong>scenario</strong> based on three market conditions:
                    <strong>Volatility</strong> (how wild BTC is moving &mdash; calm / moderate / strong),
                    <strong>Momentum</strong> (price direction &mdash; trending / reverting / flat), and
                    <strong>Edge Strength</strong> (how big the statistical edge is &mdash; small / decent / big).
                    It tracks win rates in each scenario and automatically adjusts its minimum edge requirement.
                    A scenario needs at least <strong>2 trades</strong> before it starts influencing decisions.
                </div>
                <div id="learnerDetailWrap" class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Scenario</th>
                                <th>Trades</th>
                                <th>Win Rate</th>
                                <th>P&L</th>
                                <th>Avg Edge</th>
                                <th>Min Edge (auto)</th>
                                <th>Data</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="learnerDetailBody"></tbody>
                    </table>
                </div>
            </div>

            <!-- Recent Trades -->
            <div class="section-title">&#x1F4B0; Recent Trades</div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Direction</th>
                            <th>Bet</th>
                            <th>Entry</th>
                            <th>Payout</th>
                            <th>Edge</th>
                            <th>BTC Open</th>
                            <th>BTC Close</th>
                            <th>BTC &Delta;%</th>
                            <th>Result</th>
                            <th>P&L</th>
                            <th>Bankroll</th>
                        </tr>
                    </thead>
                    <tbody id="tradesBody"></tbody>
                </table>
            </div>

            <!-- Open Positions -->
            <div class="section-title">&#x23F3; Open Positions</div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Age</th>
                            <th>Market</th>
                            <th>Direction</th>
                            <th>Bet</th>
                            <th>Entry</th>
                            <th>Status</th>
                            <th>Order</th>
                        </tr>
                    </thead>
                    <tbody id="positionsBody"></tbody>
                </table>
            </div>

            <!-- Recent Signals -->
            <div class="section-title">&#x1F4E1; Signal Log</div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Direction</th>
                            <th>BTC &Delta;%</th>
                            <th>Confidence</th>
                            <th>BTC Start</th>
                            <th>BTC End</th>
                            <th>Traded</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody id="signalsBody"></tbody>
                </table>
            </div>

            <div class="footer" id="refreshNote">Refreshing... &#x23F1;</div>
        </div>
    </div>

    <script>
    // ─── Matrix Rain ───
    (function() {
        const c = document.getElementById('matrixCanvas');
        const ctx = c.getContext('2d');
        function resize() { c.width = window.innerWidth; c.height = window.innerHeight; }
        resize();
        window.addEventListener('resize', resize);
        const chars = 'MOONDOG01アイウエオカキクケコサシスセソ$%&=+<>';
        const fontSize = 14;
        let cols = Math.floor(c.width / fontSize);
        let drops = Array(cols).fill(1);
        function draw() {
            ctx.fillStyle = 'rgba(3,10,6,0.08)';
            ctx.fillRect(0, 0, c.width, c.height);
            ctx.fillStyle = '#00ff6620';
            ctx.font = fontSize + 'px monospace';
            for (let i = 0; i < drops.length; i++) {
                const ch = chars[Math.floor(Math.random() * chars.length)];
                ctx.fillText(ch, i * fontSize, drops[i] * fontSize);
                if (drops[i] * fontSize > c.height && Math.random() > 0.975) drops[i] = 0;
                drops[i]++;
            }
        }
        setInterval(draw, 50);
        window.addEventListener('resize', () => {
            cols = Math.floor(c.width / fontSize);
            drops = Array(cols).fill(1);
        });
    })();

    // ─── Clock ───
    function updateClock() {
        const d = new Date();
        document.getElementById('clock').textContent = d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
    }
    setInterval(updateClock, 1000);
    updateClock();

    // ─── Helpers ───
    function apiUrl(p) { return p; }

    async function fetchJSON(url) {
        const r = await fetch(url);
        return r.json();
    }

    function fmtMoney(v) {
        const n = Number(v);
        const s = '$' + Math.abs(n).toFixed(2);
        return n < 0 ? '-' + s : (n > 0 ? '+' + s : s);
    }

    function fmtMoneyPlain(v) {
        const n = Number(v);
        return '$' + Math.abs(n).toFixed(2);
    }

    function pnlClass(v) {
        const n = Number(v);
        return n > 0 ? 'pos' : n < 0 ? 'neg' : 'neu';
    }

    function pnlColor(v) {
        const n = Number(v);
        return n > 0 ? 'var(--green)' : n < 0 ? 'var(--red)' : 'var(--text)';
    }

    function wrColor(pct) {
        return pct >= 55 ? 'var(--green)' : pct >= 45 ? 'var(--amber)' : 'var(--red)';
    }

    function fmtTime(ts) {
        if (!ts) return '--';
        return new Date(ts).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
    }

    function fmtBtc(v) {
        if (!v) return '--';
        return '$' + Number(v).toLocaleString(undefined, {minimumFractionDigits:2,maximumFractionDigits:2});
    }

    // ─── Stats ───
    async function refreshStats() {
        try {
            const s = await fetchJSON(apiUrl('/api/stats'));
            const dot = document.getElementById('statusDot');
            const txt = document.getElementById('statusText');
            if (s.emergency_stop) {
                dot.className = 'status-dot stopped';
                txt.textContent = 'STOPPED';
                txt.style.color = 'var(--red)';
                document.getElementById('btnStop').disabled = true;
                document.getElementById('btnResume').disabled = false;
            } else {
                dot.className = 'status-dot running';
                txt.textContent = 'ONLINE';
                txt.style.color = 'var(--green)';
                document.getElementById('btnStop').disabled = false;
                document.getElementById('btnResume').disabled = true;
            }

            // P&L (hero card)
            const pnlEl = document.getElementById('totalPnl');
            pnlEl.textContent = fmtMoney(s.total_pnl);
            pnlEl.className = 'val ' + pnlClass(s.total_pnl);

            // Bankroll
            const bankEl = document.getElementById('bankroll');
            bankEl.textContent = fmtMoneyPlain(s.bankroll);

            // ROI
            const roiEl = document.getElementById('roi');
            const rv = s.roi_pct || 0;
            roiEl.textContent = (rv >= 0 ? '+' : '') + rv.toFixed(1) + '%';
            roiEl.className = 'val ' + pnlClass(rv);

            // At risk
            const riskEl = document.getElementById('pendingRisk');
            riskEl.textContent = fmtMoneyPlain(s.pending_at_risk);
            riskEl.className = 'val ' + (s.pending_at_risk > 0 ? 'warn' : 'neu');

            // Win rate
            const wrEl = document.getElementById('winRate');
            const wrText = s.win_rate || '0%';
            wrEl.textContent = wrText;
            const wrNum = parseFloat(wrText);
            wrEl.className = 'val ' + (wrNum >= 55 ? 'pos' : wrNum >= 45 ? 'warn' : wrNum > 0 ? 'neg' : 'neu');

            // Trades
            const trEl = document.getElementById('totalTrades');
            trEl.textContent = s.total_trades + (s.pending_trades ? ' (+' + s.pending_trades + ')' : '');

            // W/L
            document.getElementById('winsLosses').textContent = s.wins + ' / ' + s.losses;

            // Bet size
            document.getElementById('currentBet').textContent = '$' + (s.bet_size||10).toFixed(2) + ' / $' + (s.max_bet||10).toFixed(2);

            // Avg win/loss
            const awEl = document.getElementById('avgWin');
            awEl.textContent = s.avg_win > 0 ? fmtMoney(s.avg_win) : '$0.00';
            awEl.className = 'val ' + (s.avg_win > 0 ? 'pos' : 'neu');

            const alEl = document.getElementById('avgLoss');
            alEl.textContent = s.avg_loss < 0 ? fmtMoney(s.avg_loss) : '$0.00';
            alEl.className = 'val ' + (s.avg_loss < 0 ? 'neg' : 'neu');

            // Signals
            document.getElementById('signals').textContent = (s.signals_total||0) + ' (' + (s.signals_traded||0) + ')';

            // Last trade
            const ltEl = document.getElementById('lastTrade');
            if (s.last_trade_time) {
                const ago = Math.floor((Date.now() - new Date(s.last_trade_time).getTime()) / 1000);
                if (ago < 60) ltEl.textContent = ago + 's ago';
                else if (ago < 3600) ltEl.textContent = Math.floor(ago/60) + 'm ago';
                else ltEl.textContent = Math.floor(ago/3600) + 'h ago';
            } else ltEl.textContent = '--';
        } catch(e) { console.error('Stats:', e); }
    }

    // ─── Trades Table ───
    async function refreshTrades() {
        try {
            const trades = await fetchJSON(apiUrl('/api/trades'));
            const tbody = document.getElementById('tradesBody');
            if (!trades.length) {
                tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;color:var(--text-muted);padding:28px;font-family:var(--mono);">No trades yet &mdash; Moondog is watching...</td></tr>';
                return;
            }
            tbody.innerHTML = trades.map(t => {
                const oc = (t.outcome||'PENDING').toUpperCase();
                let badgeCls = 'badge-pending';
                if (oc === 'WIN') badgeCls = 'badge-win';
                else if (oc === 'LOSS') badgeCls = 'badge-loss';
                else if (oc === 'CANCELLED') badgeCls = 'badge-cancelled';

                const payout = oc === 'WIN' ? '$1.00' : oc === 'LOSS' ? '$0.00' : '--';
                const plVal = t.profit_loss != null ? t.profit_loss : 0;
                const plColor = pnlColor(plVal);
                const brAfter = t.bankroll_after != null ? fmtMoneyPlain(t.bankroll_after) : '--';

                return '<tr>' +
                    '<td>' + fmtTime(t.timestamp) + '</td>' +
                    '<td style="font-weight:600;">' + (t.direction||'--') + '</td>' +
                    '<td>' + fmtMoneyPlain(t.bet_size) + '</td>' +
                    '<td>' + (t.fill_price ? t.fill_price.toFixed(3) : '--') + '</td>' +
                    '<td>' + payout + '</td>' +
                    '<td>' + (t.edge_estimate ? t.edge_estimate.toFixed(3) : '--') + '</td>' +
                    '<td>' + fmtBtc(t.btc_price_start) + '</td>' +
                    '<td>' + fmtBtc(t.btc_price_end) + '</td>' +
                    '<td>' + (t.delta_pct ? t.delta_pct.toFixed(3)+'%' : '--') + '</td>' +
                    '<td><span class="badge '+badgeCls+'">'+oc+'</span></td>' +
                    '<td style="color:'+plColor+';font-weight:600;">' + (t.profit_loss != null ? fmtMoney(t.profit_loss) : '--') + '</td>' +
                    '<td>' + brAfter + '</td>' +
                '</tr>';
            }).join('');
        } catch(e) { console.error('Trades:', e); }
    }

    // ─── Positions Table ───
    async function refreshPositions() {
        try {
            const pos = await fetchJSON(apiUrl('/api/positions'));
            const tbody = document.getElementById('positionsBody');
            if (!pos.length) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:28px;font-family:var(--mono);">No open positions</td></tr>';
                return;
            }
            tbody.innerHTML = pos.map(p => {
                let age = '--';
                if (p.timestamp) {
                    const secs = Math.floor((Date.now() - new Date(p.timestamp).getTime()) / 1000);
                    if (secs < 60) age = secs + 's';
                    else if (secs < 3600) age = Math.floor(secs/60) + 'm ' + (secs%60) + 's';
                    else age = Math.floor(secs/3600) + 'h ' + Math.floor((secs%3600)/60) + 'm';
                }
                return '<tr>' +
                    '<td>' + fmtTime(p.timestamp) + '</td>' +
                    '<td style="color:var(--amber);font-weight:600;">' + age + '</td>' +
                    '<td title="'+(p.market_question||'')+'" style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + (p.market_question||'--') + '</td>' +
                    '<td style="font-weight:600;">' + (p.direction||'--') + '</td>' +
                    '<td>' + fmtMoneyPlain(p.bet_size) + '</td>' +
                    '<td>' + (p.fill_price ? p.fill_price.toFixed(3) : '--') + '</td>' +
                    '<td><span class="badge badge-pending">LIVE</span></td>' +
                    '<td style="font-size:10px;color:var(--text-dim);">' + (p.order_id ? p.order_id.substring(0,12)+'...' : '--') + '</td>' +
                '</tr>';
            }).join('');
        } catch(e) { console.error('Positions:', e); }
    }

    // ─── Signals Table ───
    async function refreshSignals() {
        try {
            const sigs = await fetchJSON(apiUrl('/api/signals'));
            const tbody = document.getElementById('signalsBody');
            if (!sigs.length) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:28px;font-family:var(--mono);">No signals yet</td></tr>';
                return;
            }
            tbody.innerHTML = sigs.map(s => '<tr>' +
                '<td>' + fmtTime(s.timestamp) + '</td>' +
                '<td style="font-weight:600;">' + (s.direction||'--') + '</td>' +
                '<td>' + (s.delta_pct ? s.delta_pct.toFixed(3)+'%' : '--') + '</td>' +
                '<td>' + (s.confidence ? s.confidence.toFixed(2) : '--') + '</td>' +
                '<td>' + fmtBtc(s.btc_price_start) + '</td>' +
                '<td>' + fmtBtc(s.btc_price_end) + '</td>' +
                '<td style="color:'+(s.traded?'var(--green)':'var(--text-dim)')+';font-weight:600;">' + (s.traded ? 'YES' : 'NO') + '</td>' +
                '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-dim);" title="'+(s.reason||'')+'">' + (s.reason||'--') + '</td>' +
            '</tr>').join('');
        } catch(e) { console.error('Signals:', e); }
    }

    // ─── Learner ───
    function friendlyScenario(vol, mom, edg) {
        const v = {calm:'&#x1F9CA; Calm',moderate:'&#x1F30A; Moderate',strong:'&#x1F525; Volatile'}[vol] || vol;
        const m = {trending:'&#x2197;&#xFE0F; Trending',reverting:'&#x2199;&#xFE0F; Reversing',flat:'&#x27A1;&#xFE0F; Flat'}[mom] || mom;
        const e = {big:'Strong Edge',decent:'OK Edge',small:'Weak Edge'}[edg] || edg;
        return v + ' &middot; ' + m + ' &middot; ' + e;
    }

    async function refreshLearner() {
        try {
            const data = await fetchJSON('/api/learner');
            const sumEl = document.getElementById('learnerSummary');
            const noteEl = document.getElementById('learnerNote');
            const listEl = document.getElementById('learnerContent');
            const detailBody = document.getElementById('learnerDetailBody');

            if (data.status === 'no_data' || data.status !== 'ok') {
                sumEl.innerHTML = '<div class="kv" style="grid-column:1/-1;"><span class="k">No trade data yet &mdash; the AI learner activates after your first trades.</span></div>';
                noteEl.style.display = 'none';
                listEl.innerHTML = '';
                detailBody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:28px;font-family:var(--mono);">Waiting for trades...</td></tr>';
                return;
            }

            // Summary
            const owTotal = data.overall_wins + data.overall_losses;
            const owPct = owTotal > 0 ? ((data.overall_wins / owTotal) * 100).toFixed(1) : '0.0';
            sumEl.innerHTML =
                '<div class="kv"><span class="k">Trades Learned</span><span class="v">' + data.total_trades_recorded + '</span></div>' +
                '<div class="kv"><span class="k">Scenarios</span><span class="v">' + data.total_buckets + ' <span style="color:var(--green-dim);">(' + data.active_buckets + ' active)</span></span></div>' +
                '<div class="kv"><span class="k">Win Rate</span><span class="v" style="color:' + wrColor(parseFloat(owPct)) + ';">' + owPct + '%</span></div>' +
                '<div class="kv"><span class="k">Net P&L</span><span class="v" style="color:' + pnlColor(data.overall_pnl) + ';">' + fmtMoney(data.overall_pnl) + '</span></div>';

            // Note
            if (data.active_buckets > 0) {
                noteEl.innerHTML = '&#x2705; AI is <strong style="color:var(--green);">actively adjusting</strong> strategy in ' + data.active_buckets + ' scenario(s)';
            } else {
                noteEl.innerHTML = '&#x23F3; Collecting data &mdash; needs <strong style="color:var(--amber);">' + data.min_samples + ' trades per scenario</strong> to start adjusting';
            }
            noteEl.style.display = 'block';

            // Compact list (top 6)
            const buckets = data.buckets || {};
            const keys = Object.keys(buckets).sort((a,b) => buckets[b].total - buckets[a].total);
            let listHtml = '';
            for (const k of keys.slice(0, 6)) {
                const b = buckets[k];
                const wr = (b.win_rate * 100).toFixed(0);
                const tagCls = b.is_active ? 'active' : 'learning';
                const tagLabel = b.is_active ? 'ACTIVE' : b.samples_needed + ' more';
                const vol = {calm:'&#x1F9CA;',moderate:'&#x1F30A;',strong:'&#x1F525;'}[b.volatility] || '?';
                const mom = {trending:'&#x2197;',reverting:'&#x2199;',flat:'&#x27A1;'}[b.momentum] || '?';
                listHtml += '<div class="learner-bar">' +
                    '<span style="font-family:var(--mono);font-size:12px;">' + vol + ' ' + mom + ' ' + b.edge_strength + '</span>' +
                    '<span style="display:flex;align-items:center;gap:8px;">' +
                        '<span style="font-family:var(--mono);font-weight:600;color:'+wrColor(parseFloat(wr))+';">'+wr+'%</span>' +
                        '<span class="learner-tag '+tagCls+'">' + tagLabel + '</span>' +
                    '</span>' +
                '</div>';
            }
            if (!keys.length) listHtml = '<p class="learner-empty">Waiting for trade outcomes...</p>';
            listEl.innerHTML = listHtml;

            // Detail table
            let tableHtml = '';
            for (const k of keys) {
                const b = buckets[k];
                const wr = (b.win_rate * 100).toFixed(1);
                const pnlC = pnlColor(b.total_pnl);
                const progress = Math.min(b.total / data.min_samples, 1.0);
                const pPct = (progress * 100).toFixed(0);
                const pColor = b.is_active ? 'var(--green)' : 'var(--cyan)';

                let statusHtml;
                if (b.is_active && b.win_rate < 0.35 && b.total >= 6) {
                    statusHtml = '<span class="badge badge-loss">AVOID</span>';
                } else if (b.is_active) {
                    statusHtml = '<span class="badge badge-win">ACTIVE</span>';
                } else {
                    statusHtml = '<span class="badge badge-pending">' + b.samples_needed + ' MORE</span>';
                }

                tableHtml += '<tr>' +
                    '<td style="font-size:12px;">' + friendlyScenario(b.volatility, b.momentum, b.edge_strength) + '</td>' +
                    '<td>' + b.total + ' <span style="color:var(--text-dim);">(' + b.wins + 'W/' + b.losses + 'L)</span></td>' +
                    '<td style="color:'+wrColor(parseFloat(wr))+';font-weight:600;">' + wr + '%</td>' +
                    '<td style="color:'+pnlC+';font-weight:600;">' + fmtMoney(b.total_pnl) + '</td>' +
                    '<td>' + b.avg_edge.toFixed(4) + '</td>' +
                    '<td>' + b.adjusted_min_edge.toFixed(4) + '</td>' +
                    '<td><div class="prog-bar"><div class="prog-fill" style="background:'+pColor+';width:'+pPct+'%;"></div></div></td>' +
                    '<td>' + statusHtml + '</td>' +
                '</tr>';
            }
            if (!keys.length) tableHtml = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:28px;font-family:var(--mono);">No learner data yet</td></tr>';
            detailBody.innerHTML = tableHtml;
        } catch(e) { console.error('Learner:', e); }
    }

    // ─── Parlay ───
    async function refreshParlay() {
        try {
            const p = await fetchJSON('/api/parlay');
            const round = p.round || 0;
            const stake = p.current_stake || p.initial_stake || 1;
            const profit = p.streak_profit || 0;
            const active = p.active;

            document.getElementById('parlayRound').textContent = round;
            const stakeEl = document.getElementById('parlayStake');
            stakeEl.textContent = fmtMoneyPlain(stake);
            stakeEl.className = 'val ' + (active ? 'warn' : 'neu');

            const profitEl = document.getElementById('parlayProfit');
            profitEl.textContent = fmtMoney(profit);
            profitEl.className = 'val ' + pnlClass(profit);

            const lwEl = document.getElementById('parlayLifeWon');
            lwEl.textContent = fmtMoney(p.lifetime_profit || 0);
            lwEl.className = 'val ' + pnlClass(p.lifetime_profit);

            const llEl = document.getElementById('parlayLifeLost');
            llEl.textContent = fmtMoney(-(p.lifetime_lost || 0));
            llEl.className = 'val neg';

            const netEl = document.getElementById('parlayLifeNet');
            const net = p.lifetime_net || 0;
            netEl.textContent = fmtMoney(net);
            netEl.className = 'val ' + pnlClass(net);

            // Status text
            const statusEl = document.getElementById('parlayStatus');
            if (active && round > 0) {
                statusEl.innerHTML = '<span style="color:var(--green);">&#x1F525; STREAK ACTIVE</span> &mdash; ' +
                    round + ' win(s), next bet <strong>$' + stake.toFixed(2) + '</strong>. ' +
                    'Hit PASS to lock in <strong style="color:var(--green);">$' + profit.toFixed(2) + '</strong>';
            } else if (active) {
                statusEl.innerHTML = '<span style="color:var(--cyan);">&#x23F3; WAITING</span> &mdash; parlay active, $' +
                    stake.toFixed(2) + ' on next trade';
            } else {
                statusEl.innerHTML = 'Parlay inactive &mdash; hit START to begin a $' +
                    (p.initial_stake || 1).toFixed(2) + ' streak';
            }

            // Button states
            document.getElementById('btnParlayStart').disabled = active;
            document.getElementById('btnParlayPass').disabled = !active || round === 0;
            document.getElementById('btnParlayStop').disabled = !active;

            // History
            const hist = p.history || [];
            if (hist.length) {
                let hHtml = '<span style="color:var(--text-muted);">Recent: </span>';
                hHtml += hist.slice(-6).map(h => {
                    const c = h.result === 'PASS' ? 'var(--green)' : 'var(--red)';
                    const sign = h.profit >= 0 ? '+' : '';
                    return '<span style="color:'+c+';">' + h.result + '(' + h.rounds + 'r) ' + sign + '$' + Math.abs(h.profit).toFixed(2) + '</span>';
                }).join(' &middot; ');
                document.getElementById('parlayHistory').innerHTML = hHtml;
            }
        } catch(e) { console.error('Parlay:', e); }
    }

    async function parlayStart() {
        await fetch('/api/parlay/start', {method:'POST'});
        setTimeout(refreshParlay, 500);
    }
    async function parlayPass() {
        if (!confirm('PASS: Lock in your streak profits?')) return;
        await fetch('/api/parlay/pass', {method:'POST'});
        setTimeout(refreshParlay, 500);
    }
    async function parlayStop() {
        await fetch('/api/parlay/stop', {method:'POST'});
        setTimeout(refreshParlay, 500);
    }

    // ─── Actions ───
    async function emergencyStop() {
        if (!confirm('KILL SWITCH: Are you sure you want to stop the bot immediately?')) return;
        await fetch('/api/emergency-stop', {method:'POST'});
        refreshStats();
    }

    async function resumeBot() {
        await fetch('/api/resume', {method:'POST'});
        refreshStats();
    }

    async function loadConfig() {
        try {
            const cfg = await fetchJSON('/api/config');
            const s = cfg.strategy || {};
            if (s.initial_bet != null) document.getElementById('betSizeInput').value = Number(s.initial_bet).toFixed(2);
            if (s.max_bet != null) document.getElementById('maxBetInput').value = Number(s.max_bet).toFixed(2);
            if (s.max_loss != null) document.getElementById('maxLossInput').value = Number(s.max_loss).toFixed(2);
            const st = cfg.straddle || {};
            document.getElementById('straddleEnabled').checked = !!st.enabled;
            if (st.trigger_seconds != null) document.getElementById('straddleTriggerSec').value = st.trigger_seconds;
            if (st.limit_price != null) document.getElementById('straddlePrice').value = st.limit_price;
            if (st.shares != null) document.getElementById('straddleShares').value = st.shares;
            if (st.max_cost != null) document.getElementById('straddleMaxCost').value = st.max_cost;
        } catch(e) { console.error('Config load:', e); }
    }

    async function saveConfig() {
        const status = document.getElementById('betSaveStatus');
        const body = {};
        const bet = parseFloat(document.getElementById('betSizeInput').value);
        const maxB = parseFloat(document.getElementById('maxBetInput').value);
        const maxL = parseFloat(document.getElementById('maxLossInput').value);
        if (bet > 0) body.initial_bet = bet;
        if (maxB > 0) body.max_bet = maxB;
        if (maxL > 0 && !isNaN(maxL)) body.max_loss = maxL;
        body.straddle_enabled = document.getElementById('straddleEnabled').checked;
        const tSec = parseInt(document.getElementById('straddleTriggerSec').value);
        if (tSec > 0) body.straddle_trigger_seconds = tSec;
        const sP = parseFloat(document.getElementById('straddlePrice').value);
        if (sP > 0) body.straddle_price = sP;
        const sS = parseInt(document.getElementById('straddleShares').value);
        if (sS > 0) body.straddle_shares = sS;
        const sC = parseFloat(document.getElementById('straddleMaxCost').value);
        if (sC > 0) body.straddle_max_cost = sC;

        status.textContent = 'SAVING...';
        status.style.color = 'var(--amber)';
        try {
            const resp = await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
            const data = await resp.json();
            if (resp.ok) {
                status.textContent = '\\u2705 SAVED';
                status.style.color = 'var(--green)';
                loadConfig();
            } else {
                status.textContent = data.error || 'FAILED';
                status.style.color = 'var(--red)';
            }
        } catch(e) {
            status.textContent = 'FAILED';
            status.style.color = 'var(--red)';
        }
    }

    // ─── Refresh Loop ───
    function refreshAll() {
        refreshStats();
        refreshTrades();
        refreshPositions();
        refreshSignals();
        refreshLearner();
        refreshParlay();
        document.getElementById('refreshNote').innerHTML = 'Live &middot; refreshing every 2s &middot; ' + new Date().toLocaleTimeString() + ' &#x23F1;';
    }
    loadConfig();
    refreshAll();
    setInterval(refreshAll, 2000);
    </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# -- Entry Point -----------------------------------------------------------------

def main():
    port = int(os.environ.get("DASHBOARD_PORT", 8050))
    print(f"\n  * Polybot Dashboard running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
