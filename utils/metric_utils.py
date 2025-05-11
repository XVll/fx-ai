# Sharpe ratio, drawdown, winrate, and other metrics calculation functions
def calculate_metrics(trades, entry_price):
    total_trades = len(trades)
    total_profit = sum([trade[1] - entry_price for trade in trades if trade[2] == 'sell'])
    win_trades = [t for t in trades if t[2] == 'sell' and (t[1] - entry_price) > 0]
    win_rate = len(win_trades) / max(1, total_trades) * 100
    avg_pnl = total_profit / max(1, len(win_trades))
    return {
        "Total Trades": total_trades,
        "Total Profit": total_profit,
        "Win Rate (%)": win_rate,
        "Average PnL": avg_pnl
    }
