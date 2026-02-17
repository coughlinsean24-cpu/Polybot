import sqlite3

conn = sqlite3.connect('data/trades.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Recent trades
cur.execute('''
    SELECT direction, outcome, bet_size, profit_loss, fill_price, bankroll_after, timestamp
    FROM trades ORDER BY timestamp DESC LIMIT 25
''')
rows = cur.fetchall()

print("=== LAST 25 TRADES ===")
for r in rows:
    pnl = r["profit_loss"] or 0
    bank = r["bankroll_after"] or 0
    print(f'{r["timestamp"]} | {r["direction"]:>4} | {str(r["outcome"] or "PENDING"):>9} | '
          f'bet=${r["bet_size"]:.2f} | fill={r["fill_price"]:.3f} | pnl=${pnl:+.2f} | bank=${bank:.2f}')

# Summary stats
cur.execute("SELECT COUNT(*) FROM trades WHERE outcome='WIN'")
wins = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM trades WHERE outcome='LOSS'")
losses = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM trades WHERE outcome='CANCELLED'")
cancelled = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM trades WHERE outcome IS NULL OR outcome='PENDING'")
pending = cur.fetchone()[0]
cur.execute("SELECT SUM(profit_loss) FROM trades WHERE profit_loss IS NOT NULL")
total_pnl = cur.fetchone()[0] or 0
cur.execute("SELECT bankroll_after FROM trades WHERE bankroll_after IS NOT NULL ORDER BY timestamp DESC LIMIT 1")
row = cur.fetchone()
current_bank = row[0] if row else 0

total = wins + losses
wr = (wins / total * 100) if total > 0 else 0

print(f"\n=== SUMMARY ===")
print(f"Wins: {wins} | Losses: {losses} | Cancelled: {cancelled} | Pending: {pending}")
print(f"Win Rate: {wr:.1f}%")
print(f"Total P&L: ${total_pnl:+.2f}")
print(f"Current Bankroll: ${current_bank:.2f}")

conn.close()
