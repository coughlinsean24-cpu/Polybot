import sqlite3
conn = sqlite3.connect('data/trades.db')
c = conn.cursor()
c.execute("SELECT outcome, COUNT(*), SUM(profit_loss) FROM trades WHERE outcome IN ('WIN','LOSS') GROUP BY outcome")
rows = {r[0]: (r[1], r[2]) for r in c.fetchall()}
w, wp = rows.get('WIN', (0, 0))
l, lp = rows.get('LOSS', (0, 0))
t = w + l
print(f"Wins: {w}  Losses: {l}  Total: {t}")
print(f"Win rate: {w/t*100:.1f}%")
print(f"Win P&L: +${wp:.2f}")
print(f"Loss P&L: -${abs(lp):.2f}")
print(f"Net P&L: ${wp+lp:.2f}")
conn.close()
