import pandas as pd
from risk_system import RiskCalculationSystem

# 1) 读取数据
portfolio = pd.read_csv("/Users/henrywang/Desktop/portfolio.csv")
prices = pd.read_csv("/Users/henrywang/Desktop/prices.csv", parse_dates=["date"]).set_index("date")

# 2) 可选：清洗数据
prices = prices.apply(pd.to_numeric, errors="coerce").dropna()

# 3) 初始化系统
system = RiskCalculationSystem(
    portfolio=portfolio,
    price_history=prices,
    default_confidence=0.99
)

# 4) Historical VaR / ES
hist_res = system.historical_var_es(confidence=0.99, lookback=252 * 5)
print(hist_res)
print("Historical VaR/ES")
print(hist_res)

# 5) Parametric VaR / ES
param_res = system.parametric_var_es(confidence=0.99, lookback=252 * 5)
print("\nParametric VaR/ES")
print(param_res)

# 6) Monte Carlo VaR / ES
mc_res = system.monte_carlo_var_es(confidence=0.99, lookback=252 * 5, n_sims=50000, seed=44)
print("\nMonte Carlo VaR/ES")
print(mc_res)

# 7) Backtest
#bt = system.backtest_var(method="historical", confidence=0.99, lookback=150)
#print("\nBacktest Summary")
#print(bt["summary"])
# 7) Backtest - Historical
bt_hist = system.backtest_var(
    method="historical",
    confidence=0.99,
    lookback=252 * 5
)
print("\nBacktest - Historical")
print(bt_hist["summary"])


# 8) Backtest - Parametric
bt_param = system.backtest_var(
    method="parametric",
    confidence=0.99,
    lookback=252 * 5
)
print("\nBacktest - Parametric")
print(bt_param["summary"])


# 9) Backtest - Monte Carlo
bt_mc = system.backtest_var(
    method="monte_carlo",
    confidence=0.99,
    lookback=252 * 5,
    n_sims=50000,   # 可以先小一点，跑得快
    seed=44
)
print("\nBacktest - Monte Carlo")
print(bt_mc["summary"])
# 8) 保存 backtest 序列
#bt["series"].to_csv("backtest_series.csv")