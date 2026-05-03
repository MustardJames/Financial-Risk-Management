from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility math functions
# ---------------------------------------------------------------------------

def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.asarray(x) ** 2) / sqrt(2.0 * pi)


def _norm_ppf(p: float) -> float:
    """
    Acklam's inverse-normal approximation.
    Accurate enough for risk management and avoids a SciPy dependency.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1.")

    # Coefficients in rational approximations.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = sqrt(-2.0 * log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = sqrt(-2.0 * log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Standardized portfolio row.

    Supported instruments:
    - stock
    - option  (European call/put on one underlying)

    quantity is signed: positive for long, negative for short.
    """

    instrument_type: Literal["stock", "option"]
    symbol: str
    quantity: float
    underlying: Optional[str] = None
    option_type: Optional[Literal["call", "put"]] = None
    strike: Optional[float] = None
    maturity: Optional[float] = None   # years to maturity
    volatility: Optional[float] = None  # annualized implied / input volatility
    rate: float = 0.0  # continuously compounded annual rate
    contract_multiplier: float = 1.0  # 1 for stocks, usually 100 for equity options
    @property
    def risk_symbol(self) -> str:
        return self.symbol if self.instrument_type == "stock" else str(self.underlying)


class BlackScholesPricer:
    """European Black-Scholes pricer with delta."""

    @staticmethod
    def price(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        option_type: Literal["call", "put"],
    ) -> float:
        if maturity <= 0:
            if option_type == "call":
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)
        if volatility <= 0:
            fwd_intrinsic = spot - strike * exp(-rate * maturity)
            if option_type == "call":
                return max(fwd_intrinsic, 0.0)
            return max(-fwd_intrinsic, 0.0)

        d1 = (
            log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity
        ) / (volatility * sqrt(maturity))
        d2 = d1 - volatility * sqrt(maturity)

        if option_type == "call":
            return spot * _norm_cdf(d1) - strike * exp(-rate * maturity) * _norm_cdf(d2)
        return strike * exp(-rate * maturity) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

    @staticmethod
    def delta(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        option_type: Literal["call", "put"],
    ) -> float:
        if maturity <= 0:
            if option_type == "call":
                return 1.0 if spot > strike else 0.0
            return -1.0 if spot < strike else 0.0
        if volatility <= 0:
            if option_type == "call":
                return 1.0 if spot > strike * exp(-rate * maturity) else 0.0
            return -1.0 if spot < strike * exp(-rate * maturity) else 0.0

        d1 = (
            log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity
        ) / (volatility * sqrt(maturity))
        if option_type == "call":
            return float(_norm_cdf(d1))
        return float(_norm_cdf(d1) - 1.0)


# ---------------------------------------------------------------------------
# Main risk system
# ---------------------------------------------------------------------------

class RiskCalculationSystem:
    """
    Risk calculation system for portfolios of stocks and European options.

    Core capabilities aligned with the project requirements:
    - take a portfolio of stock and option positions as input
    - calibrate to historical data and/or accept user-supplied parameters
    - compute historical, parametric, and Monte Carlo VaR
    - compute historical and Monte Carlo ES
    - backtest computed VaR against history

    Design notes:
    - Historical and Monte Carlo methods use full revaluation of the portfolio.
    - Parametric VaR/ES uses a delta-normal approximation for nonlinear positions.
    - Risk factors are underlying asset prices.
    - Historical calibration is done from price history of the underlying symbols.
    """

    REQUIRED_COLUMNS = {
        "instrument_type",
        "symbol",
        "quantity",
        "underlying",
    }

    def __init__(
        self,
        portfolio: pd.DataFrame,
        price_history: pd.DataFrame,
        current_spots: Optional[Dict[str, float]] = None,
        default_confidence: float = 0.99,
        trading_days: int = 252,
    ) -> None:
        self.trading_days = trading_days
        self.default_confidence = default_confidence
        self.pricer = BlackScholesPricer()
        self.portfolio = self._coerce_portfolio(portfolio)
        self.price_history = self._coerce_price_history(price_history)

        self.risk_symbols = sorted(set(pos.risk_symbol for pos in self.portfolio))
        missing = [sym for sym in self.risk_symbols if sym not in self.price_history.columns]
        if missing:
            raise ValueError(f"Missing price history for risk symbols: {missing}")

        if current_spots is None:
            current_series = self.price_history[self.risk_symbols].iloc[-1]
            self.current_spots = current_series.to_dict()
        else:
            self.current_spots = {k: float(v) for k, v in current_spots.items()}

        self.log_returns = np.log(self.price_history[self.risk_symbols] / self.price_history[self.risk_symbols].shift(1)).dropna()
        self.simple_returns = self.price_history[self.risk_symbols].pct_change().dropna()

    # ------------------------------------------------------------------
    # Constructors / input handling
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        portfolio_csv: str | Path,
        prices_csv: str | Path,
        date_col: str = "date",
        default_confidence: float = 0.99,
        trading_days: int = 252,
    ) -> "RiskCalculationSystem":
        portfolio = pd.read_csv(portfolio_csv)
        prices = pd.read_csv(prices_csv)
        prices[date_col] = pd.to_datetime(prices[date_col])
        prices = prices.set_index(date_col)
        return cls(
            portfolio=portfolio,
            price_history=prices,
            default_confidence=default_confidence,
            trading_days=trading_days,
        )

    def _coerce_portfolio(self, portfolio: pd.DataFrame) -> List[Position]:
        """
        Parse portfolio.csv into Position objects.

        Supports the new portfolio format:
        - instrument_id
        - instrument_type
        - symbol
        - underlying
        - quantity
        - option_type
        - strike
        - maturity
        - implied_vol
        - risk_free_rate
        - contract_multiplier
        """

        cols = {c.lower() for c in portfolio.columns}
        missing = self.REQUIRED_COLUMNS - cols
        if missing:
            raise ValueError(f"Portfolio is missing required columns: {sorted(missing)}")

        df = portfolio.copy()
        df.columns = [c.lower() for c in df.columns]

        out: List[Position] = []

        today = pd.Timestamp.today().normalize()

        for _, row in df.iterrows():
            instrument_type = str(row["instrument_type"]).strip().lower()

            if instrument_type not in {"stock", "option"}:
                raise ValueError(f"Unsupported instrument_type: {instrument_type}")

            symbol = str(row["symbol"]).strip()
            quantity = float(row["quantity"])

            if instrument_type == "stock":
                out.append(
                    Position(
                        instrument_type="stock",
                        symbol=symbol,
                        quantity=quantity,
                        underlying=symbol,
                        option_type=None,
                        strike=None,
                        maturity=None,
                        volatility=None,
                        rate=0.0,
                        contract_multiplier=1.0,
                    )
                )

            else:
                underlying = str(row["underlying"]).strip()

                option_type = str(row["option_type"]).strip().lower()
                if option_type not in {"call", "put"}:
                    raise ValueError(
                        f"Option {symbol} has invalid option_type: {option_type}"
                    )

                strike = float(row["strike"])

                # Convert maturity date into years to maturity
                maturity_raw = row["maturity"]
                maturity_date = pd.to_datetime(maturity_raw, errors="coerce")

                if pd.isna(maturity_date):
                    raise ValueError(f"Option {symbol} has invalid maturity: {maturity_raw}")

                maturity_years = max((maturity_date - today).days / self.trading_days, 0.0)

                # New portfolio uses implied_vol, but old code used volatility
                if "implied_vol" in df.columns and not pd.isna(row.get("implied_vol")):
                    volatility = float(row.get("implied_vol"))
                elif "volatility" in df.columns and not pd.isna(row.get("volatility")):
                    volatility = float(row.get("volatility"))
                else:
                    raise ValueError(f"Option {symbol} is missing implied_vol or volatility.")

                # New portfolio uses risk_free_rate, but old code used rate
                if "risk_free_rate" in df.columns and not pd.isna(row.get("risk_free_rate")):
                    rate = float(row.get("risk_free_rate"))
                elif "rate" in df.columns and not pd.isna(row.get("rate")):
                    rate = float(row.get("rate"))
                else:
                    rate = 0.0

                if "contract_multiplier" in df.columns and not pd.isna(row.get("contract_multiplier")):
                    contract_multiplier = float(row.get("contract_multiplier"))
                else:
                    contract_multiplier = 100.0

                out.append(
                    Position(
                        instrument_type="option",
                        symbol=symbol,
                        quantity=quantity,
                        underlying=underlying,
                        option_type=option_type,  # type: ignore[arg-type]
                        strike=strike,
                        maturity=maturity_years,
                        volatility=volatility,
                        rate=rate,
                        contract_multiplier=contract_multiplier,
                    )
                )

        return out

    def _coerce_price_history(self, price_history: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and align historical price data.
        Handles missing values using forward-fill.
        """

        if not isinstance(price_history.index, pd.DatetimeIndex):
            raise ValueError("price_history must have a DatetimeIndex.")

        prices = price_history.sort_index().copy()

        prices = prices.apply(pd.to_numeric, errors="coerce")

        missing_before = int(prices.isna().sum().sum())

        # Forward-fill missing prices
        prices = prices.ffill()

        # Drop rows that are still missing at the beginning
        prices = prices.dropna()

        missing_after = int(prices.isna().sum().sum())

        if missing_before > 0:
            print(f"Warning: detected {missing_before} missing price values.")
            print(f"After forward-fill, remaining missing values: {missing_after}")

        if prices.empty:
            raise ValueError("Price history is empty after cleaning.")

        if (prices <= 0).any().any():
            raise ValueError("All prices must be strictly positive for log-return calibration.")

        return prices

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate_from_history(
        self,
        lookback: Optional[int] = None,
        use_log_returns: bool = True,
    ) -> Dict[str, pd.Series | pd.DataFrame]:
        data = self.log_returns if use_log_returns else self.simple_returns
        if lookback is not None:
            data = data.tail(lookback)
        mu_daily = data.mean()
        cov_daily = data.cov()
        vol_annual = data.std() * sqrt(self.trading_days)
        corr = data.corr()
        return {
            "mu_daily": mu_daily,
            "cov_daily": cov_daily,
            "vol_annual": vol_annual,
            "corr": corr,
        }

    def covariance_psd_check(
            self,
            lookback: Optional[int] = None,
            use_log_returns: bool = True,
            tol: float = 1e-10,
    ) -> Dict[str, object]:
        """
        Test 3.1: Check whether the covariance matrix is symmetric
        and positive semi-definite.

        A valid covariance matrix should be symmetric and have
        non-negative eigenvalues.
        """

        data = self.log_returns if use_log_returns else self.simple_returns

        if lookback is not None:
            data = data.tail(lookback)

        cov = data.cov()
        cov_matrix = cov.to_numpy()

        is_symmetric = bool(np.allclose(cov_matrix, cov_matrix.T, atol=tol))

        eigenvalues = np.linalg.eigvalsh(cov_matrix)

        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))

        is_psd = bool(is_symmetric and min_eigenvalue >= -tol)

        if abs(min_eigenvalue) < tol:
            adjusted_min_eigenvalue = 0.0
        else:
            adjusted_min_eigenvalue = min_eigenvalue

        condition_number = (
            float(max_eigenvalue / adjusted_min_eigenvalue)
            if adjusted_min_eigenvalue > 0
            else np.inf
        )

        return {
            "num_assets": int(cov.shape[0]),
            "is_symmetric": is_symmetric,
            "is_psd": is_psd,
            "min_eigenvalue": min_eigenvalue,
            "max_eigenvalue": max_eigenvalue,
            "condition_number": condition_number,
            "covariance_matrix": cov,
            "eigenvalues": eigenvalues,
        }

    def distribution_test(
            self,
            lookback: Optional[int] = None,
            use_log_returns: bool = True,
    ) -> pd.DataFrame:
        """
        Test 3.2: Jarque-Bera normality test for each asset's returns.

        The Jarque-Bera test checks whether skewness and kurtosis
        are consistent with a normal distribution.

        For chi-square distribution with 2 degrees of freedom:
        p-value = exp(-JB / 2)
        """

        data = self.log_returns if use_log_returns else self.simple_returns

        if lookback is not None:
            data = data.tail(lookback)

        records = []

        for symbol in data.columns:
            x = data[symbol].dropna()
            n = len(x)

            if n < 8:
                records.append(
                    {
                        "symbol": symbol,
                        "n": n,
                        "skewness": np.nan,
                        "kurtosis": np.nan,
                        "jarque_bera": np.nan,
                        "p_value": np.nan,
                        "reject_normality_5pct": None,
                    }
                )
                continue

            mean = float(x.mean())
            std = float(x.std(ddof=0))

            if std == 0:
                records.append(
                    {
                        "symbol": symbol,
                        "n": n,
                        "skewness": np.nan,
                        "kurtosis": np.nan,
                        "jarque_bera": np.nan,
                        "p_value": np.nan,
                        "reject_normality_5pct": None,
                    }
                )
                continue

            z = (x - mean) / std

            skewness = float(np.mean(z ** 3))
            kurtosis = float(np.mean(z ** 4))

            jb_stat = float((n / 6.0) * (skewness ** 2 + ((kurtosis - 3.0) ** 2) / 4.0))

            # Chi-square survival function with 2 degrees of freedom
            p_value = float(np.exp(-jb_stat / 2.0))

            records.append(
                {
                    "symbol": symbol,
                    "n": int(n),
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "jarque_bera": jb_stat,
                    "p_value": p_value,
                    "reject_normality_5pct": bool(p_value < 0.05),
                }
            )

        return pd.DataFrame(records)

    def option_volatility_calibration_check(
            self,
            lookback: Optional[int] = 252,
    ) -> pd.DataFrame:
        """
        Test 3.3: Check whether option valuation uses implied volatility
        rather than historical volatility.

        For each option position, compare:
        - implied volatility from portfolio input
        - historical annualized volatility from underlying returns
        - option price using implied volatility
        - option price using historical volatility
        """

        hist_vol = self.calibrate_from_history(
            lookback=lookback,
            use_log_returns=True
        )["vol_annual"]

        records = []

        for pos in self.portfolio:
            if pos.instrument_type != "option":
                continue

            if pos.underlying is None:
                raise ValueError(f"Option {pos.symbol} is missing underlying.")

            if pos.option_type not in {"call", "put"}:
                raise ValueError(f"Option {pos.symbol} has invalid option_type.")

            if pos.strike is None or pos.maturity is None:
                raise ValueError(f"Option {pos.symbol} is missing strike or maturity.")

            if pos.volatility is None:
                raise ValueError(
                    f"Option {pos.symbol} is missing implied volatility. "
                    "Test 3.3 requires implied volatility input."
                )

            spot = self.current_spots[pos.underlying]
            implied_vol = float(pos.volatility)
            historical_vol = float(hist_vol[pos.underlying])

            price_using_implied_vol = self.pricer.price(
                spot=spot,
                strike=pos.strike,
                maturity=max(pos.maturity, 0.0),
                rate=pos.rate,
                volatility=implied_vol,
                option_type=pos.option_type,
            )

            price_using_historical_vol = self.pricer.price(
                spot=spot,
                strike=pos.strike,
                maturity=max(pos.maturity, 0.0),
                rate=pos.rate,
                volatility=historical_vol,
                option_type=pos.option_type,
            )

            position_value_using_implied_vol = (
                    pos.quantity
                    * pos.contract_multiplier
                    * price_using_implied_vol
            )

            position_value_using_historical_vol = (
                    pos.quantity
                    * pos.contract_multiplier
                    * price_using_historical_vol
            )

            records.append(
                {
                    "option_symbol": pos.symbol,
                    "underlying": pos.underlying,
                    "option_type": pos.option_type,
                    "strike": pos.strike,
                    "maturity_years": pos.maturity,
                    "spot": spot,
                    "implied_vol_used": implied_vol,
                    "historical_vol_reference": historical_vol,
                    "option_price_using_implied_vol": price_using_implied_vol,
                    "option_price_using_historical_vol": price_using_historical_vol,
                    "position_value_using_implied_vol": position_value_using_implied_vol,
                    "position_value_using_historical_vol": position_value_using_historical_vol,
                    "pricing_vol_source": "implied_vol",
                }
            )

        if not records:
            raise ValueError("No option positions found in portfolio.")

        return pd.DataFrame(records)

    def stress_test(
            self,
            equity_shock: float = -0.20,
            implied_vol_shock: float = 0.50,
    ) -> Dict[str, object]:
        """
        Test 6.2: Hypothetical stress scenario.

        Apply:
        - deterministic equity price shock, e.g. -20%
        - deterministic implied volatility shock, e.g. +50%

        Then fully revalue the portfolio and report stress loss.
        """

        current_value = self.portfolio_value()

        shocked_spots = {
            symbol: spot * (1.0 + equity_shock)
            for symbol, spot in self.current_spots.items()
        }

        records = []
        stressed_value = 0.0

        for pos in self.portfolio:
            base_value = self.position_value(pos, self.current_spots)

            if pos.instrument_type == "stock":
                stressed_position_value = pos.quantity * shocked_spots[pos.symbol]

                records.append(
                    {
                        "symbol": pos.symbol,
                        "instrument_type": pos.instrument_type,
                        "base_value": base_value,
                        "stressed_value": stressed_position_value,
                        "stress_pnl": stressed_position_value - base_value,
                        "stress_loss": base_value - stressed_position_value,
                        "equity_shock": equity_shock,
                        "implied_vol_shock": 0.0,
                    }
                )

            else:
                if pos.underlying is None:
                    raise ValueError(f"Option {pos.symbol} is missing underlying.")
                if pos.option_type not in {"call", "put"}:
                    raise ValueError(f"Option {pos.symbol} has invalid option_type.")
                if pos.strike is None or pos.maturity is None:
                    raise ValueError(f"Option {pos.symbol} is missing strike or maturity.")

                if pos.volatility is None:
                    hist_vol = self.calibrate_from_history()["vol_annual"]
                    base_vol = float(hist_vol[pos.underlying])
                else:
                    base_vol = float(pos.volatility)

                shocked_vol = base_vol * (1.0 + implied_vol_shock)

                shocked_option_price = self.pricer.price(
                    spot=shocked_spots[pos.underlying],
                    strike=pos.strike,
                    maturity=max(pos.maturity, 0.0),
                    rate=pos.rate,
                    volatility=shocked_vol,
                    option_type=pos.option_type,
                )

                contract_multiplier = getattr(pos, "contract_multiplier", 100.0)

                stressed_position_value = (
                        pos.quantity
                        * contract_multiplier
                        * shocked_option_price
                )

                records.append(
                    {
                        "symbol": pos.symbol,
                        "instrument_type": pos.instrument_type,
                        "underlying": pos.underlying,
                        "option_type": pos.option_type,
                        "strike": pos.strike,
                        "base_vol": base_vol,
                        "shocked_vol": shocked_vol,
                        "base_value": base_value,
                        "stressed_value": stressed_position_value,
                        "stress_pnl": stressed_position_value - base_value,
                        "stress_loss": base_value - stressed_position_value,
                        "equity_shock": equity_shock,
                        "implied_vol_shock": implied_vol_shock,
                    }
                )

            stressed_value += stressed_position_value

        stress_loss = current_value - stressed_value
        stress_pnl = stressed_value - current_value

        details = pd.DataFrame(records)

        return {
            "equity_shock": equity_shock,
            "implied_vol_shock": implied_vol_shock,
            "current_value": current_value,
            "stressed_value": float(stressed_value),
            "stress_pnl": float(stress_pnl),
            "stress_loss": float(stress_loss),
            "stress_loss_pct": float(stress_loss / current_value),
            "details": details,
        }

    def stress_scenario_suite(
            self,
            scenarios: Optional[List[Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Run multiple hypothetical stress scenarios and summarize portfolio-level impact.

        This extends Test 6.2 by identifying which market scenario creates
        the largest portfolio loss.
        """

        if scenarios is None:
            scenarios = [
                {
                    "scenario": "Equity selloff + volatility spike",
                    "equity_shock": -0.20,
                    "implied_vol_shock": 0.50,
                },
                {
                    "scenario": "Equity selloff only",
                    "equity_shock": -0.20,
                    "implied_vol_shock": 0.00,
                },
                {
                    "scenario": "Moderate selloff + volatility crush",
                    "equity_shock": -0.10,
                    "implied_vol_shock": -0.30,
                },
                {
                    "scenario": "Volatility crush only",
                    "equity_shock": 0.00,
                    "implied_vol_shock": -0.50,
                },
                {
                    "scenario": "Equity rally + volatility crush",
                    "equity_shock": 0.20,
                    "implied_vol_shock": -0.30,
                },
                {
                    "scenario": "Equity rally + volatility spike",
                    "equity_shock": 0.20,
                    "implied_vol_shock": 0.50,
                },
            ]

        records = []

        for sc in scenarios:
            res = self.stress_test(
                equity_shock=sc["equity_shock"],
                implied_vol_shock=sc["implied_vol_shock"],
            )

            records.append(
                {
                    "scenario": sc["scenario"],
                    "equity_shock": sc["equity_shock"],
                    "implied_vol_shock": sc["implied_vol_shock"],
                    "current_value": res["current_value"],
                    "stressed_value": res["stressed_value"],
                    "stress_pnl": res["stress_pnl"],
                    "stress_loss": res["stress_loss"],
                    "stress_loss_pct": res["stress_loss_pct"],
                }
            )

        result = pd.DataFrame(records)

        # Largest loss first
        result = result.sort_values(
            by="stress_loss",
            ascending=False
        ).reset_index(drop=True)

        return result

    # ------------------------------------------------------------------
    # Valuation helpers
    # ------------------------------------------------------------------
    def position_value(self, position: Position, spots: Dict[str, float]) -> float:
        if position.instrument_type == "stock":
            return position.quantity * spots[position.symbol]

        if position.underlying is None:
            raise ValueError(f"Option {position.symbol} is missing an underlying symbol.")
        if position.option_type not in {"call", "put"}:
            raise ValueError(f"Option {position.symbol} must specify option_type as 'call' or 'put'.")
        if position.strike is None or position.maturity is None:
            raise ValueError(f"Option {position.symbol} must specify strike and maturity.")

        sigma = position.volatility
        if sigma is None:
            hist_vol = self.calibrate_from_history()["vol_annual"]
            sigma = float(hist_vol[position.underlying])

        spot = spots[position.underlying]

        price = self.pricer.price(
            spot=spot,
            strike=position.strike,
            maturity=max(position.maturity, 0.0),
            rate=position.rate,
            volatility=sigma,
            option_type=position.option_type,  # type: ignore[arg-type]
        )

        return position.quantity * position.contract_multiplier * price

    def portfolio_value(self, spots: Optional[Dict[str, float]] = None) -> float:
        use_spots = self.current_spots if spots is None else spots
        return float(sum(self.position_value(pos, use_spots) for pos in self.portfolio))

    def _portfolio_path_values(self, scenario_spots: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        scenario_spots : array, shape (n_scenarios, n_assets)
            Asset prices for each scenario, ordered by self.risk_symbols.
        """
        n_scenarios = scenario_spots.shape[0]
        values = np.zeros(n_scenarios)
        symbol_to_idx = {sym: i for i, sym in enumerate(self.risk_symbols)}
        for pos in self.portfolio:
            if pos.instrument_type == "stock":
                idx = symbol_to_idx[pos.symbol]
                values += pos.quantity * scenario_spots[:, idx]
            else:
                if pos.underlying is None or pos.option_type is None or pos.strike is None or pos.maturity is None:
                    raise ValueError(f"Option {pos.symbol} is missing required fields.")
                sigma = pos.volatility
                if sigma is None:
                    hist_vol = self.calibrate_from_history()["vol_annual"]
                    sigma = float(hist_vol[pos.underlying])
                idx = symbol_to_idx[pos.underlying]
                option_values = np.array([
                    self.pricer.price(
                        spot=float(s),
                        strike=pos.strike,
                        maturity=max(pos.maturity, 0.0),
                        rate=pos.rate,
                        volatility=sigma,
                        option_type=pos.option_type,
                    )
                    for s in scenario_spots[:, idx]
                ])

                values += pos.quantity * pos.contract_multiplier * option_values
        return values

    def _losses_from_values(self, future_values: np.ndarray, current_value: Optional[float] = None) -> np.ndarray:
        v0 = self.portfolio_value() if current_value is None else current_value
        return v0 - future_values

    @staticmethod
    def _var_from_losses(losses: np.ndarray, confidence: float) -> float:
        return float(np.quantile(losses, confidence, method="linear"))

    @staticmethod
    def _es_from_losses(losses: np.ndarray, confidence: float) -> float:
        var = float(np.quantile(losses, confidence, method="linear"))
        tail = losses[losses > var]
        if tail.size == 0:
            tail = losses[losses >= var]
        return float(np.mean(tail))

    # ------------------------------------------------------------------
    # Historical VaR / ES
    # ------------------------------------------------------------------
    def historical_var_es(
        self,
        confidence: Optional[float] = None,
        lookback: Optional[int] = None,
        horizon_days: int = 1,
        shock_type: Literal["relative", "absolute"] = "relative",
    ) -> Dict[str, float]:
        p = self.default_confidence if confidence is None else confidence

        if shock_type == "relative":
            returns = np.log(self.price_history[self.risk_symbols] / self.price_history[self.risk_symbols].shift(horizon_days)).dropna()
            if lookback is not None:
                returns = returns.tail(lookback)
            scenario_spots = np.exp(returns.to_numpy()) * np.array([self.current_spots[s] for s in self.risk_symbols])
        elif shock_type == "absolute":
            deltas = self.price_history[self.risk_symbols].diff(horizon_days).dropna()
            if lookback is not None:
                deltas = deltas.tail(lookback)
            scenario_spots = deltas.to_numpy() + np.array([self.current_spots[s] for s in self.risk_symbols])
            if (scenario_spots <= 0).any():
                raise ValueError("Absolute shocks produced nonpositive scenario prices.")
        else:
            raise ValueError("shock_type must be 'relative' or 'absolute'.")

        future_values = self._portfolio_path_values(scenario_spots)
        losses = self._losses_from_values(future_values)
        return {
            "current_value": self.portfolio_value(),
            "var": self._var_from_losses(losses, p),
            "es": self._es_from_losses(losses, p),
            "num_scenarios": int(len(losses)),
        }

    # ------------------------------------------------------------------
    # Parametric VaR / ES (delta-normal approximation)
    # ------------------------------------------------------------------
    def _delta_exposures(self) -> pd.Series:
        exposures = pd.Series(0.0, index=self.risk_symbols, dtype=float)
        for pos in self.portfolio:
            if pos.instrument_type == "stock":
                exposures[pos.symbol] += pos.quantity
            else:
                if pos.underlying is None or pos.option_type is None or pos.strike is None or pos.maturity is None:
                    raise ValueError(f"Option {pos.symbol} is missing required fields.")
                sigma = pos.volatility
                if sigma is None:
                    hist_vol = self.calibrate_from_history()["vol_annual"]
                    sigma = float(hist_vol[pos.underlying])
                delta = self.pricer.delta(
                    spot=self.current_spots[pos.underlying],
                    strike=pos.strike,
                    maturity=max(pos.maturity, 0.0),
                    rate=pos.rate,
                    volatility=sigma,
                    option_type=pos.option_type,
                )

                exposures[pos.underlying] += pos.quantity * pos.contract_multiplier * delta
        return exposures

    def parametric_var_es(
        self,
        confidence: Optional[float] = None,
        lookback: Optional[int] = None,
        horizon_days: int = 1,
        mu_daily: Optional[pd.Series] = None,
        cov_daily: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        p = self.default_confidence if confidence is None else confidence
        z = _norm_ppf(p)

        if mu_daily is None or cov_daily is None:
            calib = self.calibrate_from_history(lookback=lookback, use_log_returns=False)
            if mu_daily is None:
                mu_daily = calib["mu_daily"]  # type: ignore[assignment]
            if cov_daily is None:
                cov_daily = calib["cov_daily"]  # type: ignore[assignment]

        mu_daily = mu_daily.loc[self.risk_symbols]
        cov_daily = cov_daily.loc[self.risk_symbols, self.risk_symbols]

        exposures = self._delta_exposures()
        spots = pd.Series(self.current_spots).loc[self.risk_symbols]

        # Approximate price changes using dS ≈ S * r for simple returns.
        pnl_mean_1d = float(np.dot(exposures * spots, mu_daily))
        pnl_var_1d = float((exposures * spots).T @ cov_daily @ (exposures * spots))

        pnl_mean_h = horizon_days * pnl_mean_1d
        pnl_std_h = sqrt(max(horizon_days * pnl_var_1d, 0.0))

        # Loss = -PnL, so VaR_p = -mu_PnL + sigma_PnL * z_p
        var = -pnl_mean_h + pnl_std_h * z
        es = -pnl_mean_h + pnl_std_h * (_norm_pdf(z) / (1.0 - p))

        return {
            "current_value": self.portfolio_value(),
            "var": float(max(var, 0.0)),
            "es": float(max(es, 0.0)),
            "pnl_mean": float(pnl_mean_h),
            "pnl_std": float(pnl_std_h),
        }

    # ------------------------------------------------------------------
    # Monte Carlo VaR / ES
    # ------------------------------------------------------------------
    def monte_carlo_var_es(
        self,
        confidence: Optional[float] = None,
        lookback: Optional[int] = None,
        horizon_days: int = 1,
        n_sims: int = 10000,
        mu_daily: Optional[pd.Series] = None,
        cov_daily: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        p = self.default_confidence if confidence is None else confidence
        rng = np.random.default_rng(seed)

        if mu_daily is None or cov_daily is None:
            calib = self.calibrate_from_history(lookback=lookback, use_log_returns=True)
            if mu_daily is None:
                mu_daily = calib["mu_daily"]  # type: ignore[assignment]
            if cov_daily is None:
                cov_daily = calib["cov_daily"]  # type: ignore[assignment]

        mu_daily = mu_daily.loc[self.risk_symbols]
        cov_daily = cov_daily.loc[self.risk_symbols, self.risk_symbols]

        mean_h = mu_daily.to_numpy() * horizon_days
        cov_h = cov_daily.to_numpy() * horizon_days
        shocks = rng.multivariate_normal(mean=mean_h, cov=cov_h, size=n_sims)

        s0 = np.array([self.current_spots[s] for s in self.risk_symbols])
        scenario_spots = s0 * np.exp(shocks)
        future_values = self._portfolio_path_values(scenario_spots)
        losses = self._losses_from_values(future_values)

        return {
            "current_value": self.portfolio_value(),
            "var": self._var_from_losses(losses, p),
            "es": self._es_from_losses(losses, p),
            "num_scenarios": int(n_sims),
        }

    def monte_carlo_convergence_test(
            self,
            confidence: Optional[float] = None,
            lookback: Optional[int] = 252,
            horizon_days: int = 1,
            n_sims_list: Optional[List[int]] = None,
            seed: int = 42,
    ) -> pd.DataFrame:
        """
        Test 4.3: Monte Carlo convergence test.

        Run Monte Carlo VaR / ES using increasing numbers of simulation paths.
        The VaR and ES estimates should become more stable as n_sims increases.
        """

        if n_sims_list is None:
            n_sims_list = [1000, 10000, 100000]

        records = []

        previous_var = None

        for n_sims in n_sims_list:
            res = self.monte_carlo_var_es(
                confidence=confidence,
                lookback=lookback,
                horizon_days=horizon_days,
                n_sims=n_sims,
                seed=seed,
            )

            var = float(res["var"])
            es = float(res["es"])

            if previous_var is None:
                var_change = np.nan
                var_change_pct = np.nan
            else:
                var_change = var - previous_var
                var_change_pct = var_change / previous_var

            records.append(
                {
                    "n_sims": n_sims,
                    "current_value": res["current_value"],
                    "var": var,
                    "es": es,
                    "var_change_from_previous": var_change,
                    "var_change_pct_from_previous": var_change_pct,
                }
            )

            previous_var = var

        return pd.DataFrame(records)

    def greeks_based_benchmark(
            self,
            confidence: Optional[float] = None,
            lookback: Optional[int] = 252,
            horizon_days: int = 1,
            n_sims: int = 10000,
            seed: int = 42,
            tolerance_pct: float = 0.20,
    ) -> Dict[str, object]:
        """
        Test 7.1: Greeks-based benchmarking.

        Compare:
        - First-order Delta-Normal Parametric VaR
        - Full-revaluation Monte Carlo VaR

        For portfolios with options, a large difference indicates nonlinear
        option risk that cannot be fully captured by delta-only approximation.
        """

        param_res = self.parametric_var_es(
            confidence=confidence,
            lookback=lookback,
            horizon_days=horizon_days,
        )

        mc_res = self.monte_carlo_var_es(
            confidence=confidence,
            lookback=lookback,
            horizon_days=horizon_days,
            n_sims=n_sims,
            seed=seed,
        )

        param_var = float(param_res["var"])
        mc_var = float(mc_res["var"])

        difference = mc_var - param_var
        difference_pct = difference / param_var if param_var != 0 else np.nan

        within_tolerance = bool(abs(difference_pct) <= tolerance_pct)

        if within_tolerance:
            interpretation = (
                "Monte Carlo VaR and Delta-Normal VaR are within the tolerance range. "
                "The portfolio behaves approximately linearly over the tested horizon."
            )
        else:
            interpretation = (
                "Monte Carlo VaR and Delta-Normal VaR differ materially. "
                "This indicates nonlinear option risk and shows the limitation of "
                "a first-order delta approximation."
            )

        return {
            "confidence": self.default_confidence if confidence is None else confidence,
            "lookback": lookback,
            "horizon_days": horizon_days,
            "n_sims": n_sims,
            "parametric_var": param_var,
            "monte_carlo_var": mc_var,
            "difference": difference,
            "difference_pct": difference_pct,
            "tolerance_pct": tolerance_pct,
            "within_tolerance": within_tolerance,
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    def _one_day_realized_pnl_series(self) -> pd.Series:
        dates = self.price_history.index[1:]
        scenario_spots = self.price_history[self.risk_symbols].iloc[1:].to_numpy()
        values = self._portfolio_path_values(scenario_spots)
        current_values = np.array([
            self._portfolio_path_values(self.price_history[self.risk_symbols].iloc[i:i+1].to_numpy())[0]
            for i in range(len(self.price_history) - 1)
        ])
        pnl = values - current_values
        return pd.Series(pnl, index=dates, name="realized_pnl")

    @staticmethod
    def _kupiec_test(exceptions: int, observations: int, alpha: float) -> Dict[str, float]:
        """
        Kupiec unconditional coverage statistic.
        alpha = tail probability = 1 - confidence.
        Returns LR_uc and an approximate p-value using chi-square(1).
        """
        if observations <= 0:
            raise ValueError("observations must be positive")
        x = exceptions
        n = observations
        pi_hat = x / n
        if pi_hat in {0.0, 1.0}:
            # Degenerate case: still report LR using a small epsilon.
            pi_hat = min(max(pi_hat, 1e-12), 1.0 - 1e-12)
        term_null = ((1 - alpha) ** (n - x)) * (alpha ** x)
        term_alt = ((1 - pi_hat) ** (n - x)) * (pi_hat ** x)
        lr_uc = -2.0 * log(term_null / term_alt)
        # For df=1, survival function = 2 * (1 - Phi(sqrt(x)))
        p_value = 2.0 * (1.0 - float(_norm_cdf(sqrt(max(lr_uc, 0.0)))))
        return {"lr_uc": float(lr_uc), "p_value": float(max(min(p_value, 1.0), 0.0))}

    def backtest_var(
        self,
        method: Literal["historical", "parametric", "monte_carlo"] = "historical",
        confidence: Optional[float] = None,
        lookback: int = 252,
        horizon_days: int = 1,
        n_sims: int = 5000,
        shock_type: Literal["relative", "absolute"] = "relative",
        seed: Optional[int] = 123,
    ) -> Dict[str, object]:
        if horizon_days != 1:
            raise NotImplementedError("This backtest implementation currently supports only 1-day VaR.")

        p = self.default_confidence if confidence is None else confidence
        alpha = 1.0 - p
        if len(self.price_history) <= lookback + 1:
            raise ValueError("Not enough history for the requested rolling backtest.")

        realized_pnl = self._one_day_realized_pnl_series()
        var_records: List[Tuple[pd.Timestamp, float, float, int]] = []

        original_history = self.price_history.copy()
        original_spots = self.current_spots.copy()

        try:
            for end_idx in range(lookback, len(original_history) - 1):
                window_prices = original_history.iloc[end_idx - lookback : end_idx + 1]
                self.price_history = window_prices
                self.current_spots = window_prices[self.risk_symbols].iloc[-1].to_dict()
                self.log_returns = np.log(window_prices[self.risk_symbols] / window_prices[self.risk_symbols].shift(1)).dropna()
                self.simple_returns = window_prices[self.risk_symbols].pct_change().dropna()

                if method == "historical":
                    res = self.historical_var_es(confidence=p, lookback=lookback, horizon_days=1, shock_type=shock_type)
                elif method == "parametric":
                    res = self.parametric_var_es(confidence=p, lookback=lookback, horizon_days=1)
                elif method == "monte_carlo":
                    res = self.monte_carlo_var_es(confidence=p, lookback=lookback, horizon_days=1, n_sims=n_sims, seed=seed)
                else:
                    raise ValueError("Unknown backtest method.")

                test_date = original_history.index[end_idx + 1]
                pnl = float(realized_pnl.loc[test_date])
                loss = -pnl
                var = float(res["var"])
                exception = int(loss > var)
                var_records.append((test_date, var, loss, exception))
        finally:
            self.price_history = original_history
            self.current_spots = original_spots
            self.log_returns = np.log(self.price_history[self.risk_symbols] / self.price_history[self.risk_symbols].shift(1)).dropna()
            self.simple_returns = self.price_history[self.risk_symbols].pct_change().dropna()

        backtest_df = pd.DataFrame(var_records, columns=["date", "var", "realized_loss", "exception"]).set_index("date")
        exceptions = int(backtest_df["exception"].sum())
        observations = int(len(backtest_df))
        kupiec = self._kupiec_test(exceptions, observations, alpha)

        return {
            "summary": {
                "method": method,
                "confidence": p,
                "tail_probability": alpha,
                "observations": observations,
                "exceptions": exceptions,
                "exception_rate": exceptions / observations,
                "expected_exception_rate": alpha,
                **kupiec,
            },
            "series": backtest_df,
        }


# ---------------------------------------------------------------------------
# Optional convenience helpers
# ---------------------------------------------------------------------------

def example_portfolio_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_type": "stock",
                "symbol": "AAPL",
                "quantity": 100.0,
                "underlying": np.nan,
                "option_type": np.nan,
                "strike": np.nan,
                "maturity": np.nan,
                "volatility": np.nan,
                "rate": 0.0,
            },
            {
                "instrument_type": "option",
                "symbol": "AAPL_C_210",
                "quantity": 10.0,
                "underlying": "AAPL",
                "option_type": "call",
                "strike": 210.0,
                "maturity": 0.5,
                "volatility": 0.25,
                "rate": 0.04,
            },
            {
                "instrument_type": "stock",
                "symbol": "MSFT",
                "quantity": -50.0,
                "underlying": np.nan,
                "option_type": np.nan,
                "strike": np.nan,
                "maturity": np.nan,
                "volatility": np.nan,
                "rate": 0.0,
            },
        ]
    )


def example_usage() -> None:
    """
    Example only. Requires a CSV with columns like:
        date,AAPL,MSFT
        2025-01-02,185.64,414.34
        ...
    """
    portfolio = example_portfolio_dataframe()
    # prices = pd.read_csv("prices.csv", parse_dates=["date"]).set_index("date")
    # system = RiskCalculationSystem(portfolio, prices)
    # print(system.historical_var_es(confidence=0.99, lookback=250))
    # print(system.parametric_var_es(confidence=0.99, lookback=250))
    # print(system.monte_carlo_var_es(confidence=0.99, lookback=250, n_sims=20000, seed=7))
    # bt = system.backtest_var(method="historical", confidence=0.99, lookback=250)
    # print(bt["summary"])
    pass


if __name__ == "__main__":
    print(
        "risk_system.py created. Import RiskCalculationSystem and load portfolio / price history data to run it."
    )
