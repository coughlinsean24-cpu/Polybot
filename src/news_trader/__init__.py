"""
AI news-trading engine for Polymarket (paper mode by default).

Pipeline: news ingestion -> normalisation/verification -> market matching ->
resolution-rule validation -> AI signal -> edge/EV -> deterministic risk gate
-> execution adapter -> journal/analytics.  Isolated from the BTC engine in
src/main.py; it shares only config.yaml, the dashboard process, the
EMERGENCY_STOP kill switch and (for live mode) PolymarketClient.
"""
