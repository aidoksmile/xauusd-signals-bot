def backtest_strategy(df, signals):
    results = []
    profits = []
    for sig in signals:
        mask = df.index >= sig['timestamp']
        trade = df.loc[mask].iloc[:5]  # 5 дней
        profit = (trade['Close'][-1] - sig['entry']) * (1 if sig['direction'] == 'BUY' else -1)
        profits.append(profit)
        results.append(profit > 0)

    win_rate = sum(results) / len(results)
    gross_profit = sum([p for p in profits if p > 0])
    gross_loss = abs(sum([p for p in profits if p < 0]))
    profit_factor = gross_profit / (gross_loss or 1)

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_profit': sum(profits),
        'trades': len(signals)
    }
