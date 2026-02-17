# Polybot Quick Reference

## 🚀 Start Bot
```bash
python -m src.main
```

## ⏸️ Stop Bot
- **Graceful**: Press `Ctrl+C`
- **Emergency**: `touch EMERGENCY_STOP`
- **Pause**: `export TRADING_ENABLED=false`

## 📊 Monitor Performance
```bash
# Recent trades
python check_trades.py

# Statistics
python check_stats.py

# Live logs
tail -f data/polybot.log
```

## 💰 Key Features

### Bankroll Sync
- Auto-syncs with Polymarket every 60 seconds
- Ensures bot bankroll = actual Polymarket cash
- Logs: `"[$] Bankroll synced: $X.XX -> $Y.YY"`

### Trading Strategy
- **Edge**: Exploits 30-second delay in Polymarket prices
- **Scan**: Every 1 second for minimum latency
- **Entry**: Immediate when BTC moves vs window start
- **Exit**: Limit sell placed immediately after entry

### Risk Limits
- **Per Trade**: Max $25
- **Per Window**: Max $100 total
- **Session**: Halts after -$50 loss
- **Fill Price**: Min $0.30 (0% WR below)

## ⚙️ Configuration
Edit `config.yaml`:
```yaml
strategy:
  initial_bankroll: 350.0    # Your starting capital
  initial_bet: 5.0           # Base bet size
  min_edge: 0.05             # 5% minimum edge to trade
  min_fill_price: 0.30       # Don't buy below this
  kelly_fraction: 0.35       # Quarter-Kelly (safer)
```

## 🔐 Environment (.env)
```bash
POLYMARKET_API_KEY=xxx
POLYMARKET_API_SECRET=xxx
POLYMARKET_PASSPHRASE=xxx
POLYMARKET_PRIVATE_KEY=xxx
```

## 📈 Compounding Settings
```yaml
compound:
  enabled: true
  floor: 300.0              # Protect base capital
  profit_reinvest_pct: 0.60 # Reinvest 60% of wins
  lock_in:
    pct: 0.25               # Lock 25% of each win
```

## 🛠️ Troubleshooting

### Bot won't start
```bash
# Check if already running
rm data/polybot.pid  # (only if you're sure no instance is running)

# Verify credentials
cat .env

# Check logs
tail -50 data/polybot.log
```

### No trades happening
- Check `min_edge` is not too high
- Verify BTC is moving (> ±0.05% from window start)
- Check session hasn't hit loss limit
- Ensure `TRADING_ENABLED=true`

### Losses accumulating
- Bot halts after -$50 session loss (restart required)
- Review `min_edge` - increase for more selective trades
- Check `min_fill_price` - may need to increase
- Verify strategy parameters in config

## 📝 Files

### Important Files
- `config.yaml` - All strategy parameters
- `.env` - API credentials (never commit!)
- `data/trades.db` - Trade history
- `data/polybot.log` - Detailed logs

### Control Files
- `EMERGENCY_STOP` - Create to halt bot
- `data/parlay_control.json` - Control parlay mode

## 🎯 Strategy Logic

1. **Price Feed**: Parallel WebSockets (Coinbase, Binance, Kraken, Bybit)
2. **Window Detection**: Identifies current 5-min BTC window
3. **Target Price**: Chainlink oracle (BTC at window start)
4. **Edge Calculation**: Current BTC vs target + market odds
5. **Decision**: Trade if edge > min_edge and fills safety checks
6. **Sizing**: Kelly fraction with compounding adjustments
7. **Entry**: Market order for shares
8. **Exit**: Immediate limit sell at target profit
9. **Resolution**: Window ends, record W/L
10. **Sync**: Update bankroll from Polymarket

## 📞 Support

See `CLOUD_SESSION_SUMMARY.md` for detailed documentation.

**Key Commands**:
```bash
# Start
python -m src.main

# Stop
Ctrl+C  # or
touch EMERGENCY_STOP

# Check
python check_trades.py
tail -f data/polybot.log

# Emergency
rm data/polybot.pid
```

---
**Status**: Production Ready ✅  
**Strategy**: 30-Second Delay Latency Arbitrage  
**Risk**: Multiple safety layers active
