import json
import os
import yfinance as yf
import httpx
import logging
import sys
import pandas as pd
from functools import lru_cache
import joblib
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from pandas.tseries.offsets import BDay
from typing import Any, List, Dict
from datetime import datetime

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("financial-datasets-mcp")

# ========== MCP Init ==========
mcp = FastMCP("financial-datasets")

# Load env variables
load_dotenv()

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Savings Quartile Support ----------
EXPENSE_COLS = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare",
    "Education", "Miscellaneous"
]

def _find_benchmark_csv() -> str | None:
    candidates = [
        "/mnt/data/data.csv",                                # chat upload path (if you used it)
        os.path.join(BASE_DIR, "data", "data.csv"),         # repo path (your screenshot)
        os.path.join(BASE_DIR, "data.csv"),                 # fallback
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _load_quartiles() -> tuple[dict, str | None]:
    src = _find_benchmark_csv()
    if not src:
        print("⚠️ data.csv not found for quartiles", file=sys.stderr)
        return {}, None
    try:
        dfb = pd.read_csv(src)
        keep = [c for c in EXPENSE_COLS if c in dfb.columns]
        if not keep:
            print("⚠️ data.csv has no matching expense columns", file=sys.stderr)
            return {}, src
        for c in keep:
            dfb[c] = pd.to_numeric(dfb[c], errors="coerce")
        q = {}
        for c in keep:
            q[c] = {
                "Q1": float(dfb[c].quantile(0.25)),
                "Q2": float(dfb[c].quantile(0.50)),
                "Q3": float(dfb[c].quantile(0.75)),
            }
        print(f"✅ Loaded benchmark data for quartiles: {src}", file=sys.stderr)
        return q, src
    except Exception as e:
        print(f"⚠️ Failed loading quartiles: {e}", file=sys.stderr)
        return {}, src

QUARTILES, QUARTILE_SRC = _load_quartiles()

# ========== API Helper ==========
async def make_request(url: str) -> dict[str, Any] | None:
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"Error": str(e)}

# ------------------- TOOLS -------------------

@mcp.tool()
async def get_current_stock_price(ticker: str) -> str:
    try:
        # Auto-correct common Indian stock tickers
        ticker_map = {
            "TCS": "TCS.NS",
            "INFY": "INFY.NS",
            "RELIANCE": "RELIANCE.NS",
            "HDFCBANK": "HDFCBANK.NS",
            "ICICIBANK": "ICICIBANK.NS",
            "SBIN": "SBIN.NS",
        }

        original_ticker = ticker
        ticker = ticker_map.get(ticker.upper(), ticker)

        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = (
            info.get('currentPrice') or
            info.get('regularMarketPrice') or
            info.get('previousClose')
        )

        if current_price is None:
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])

        if current_price is None:
            return json.dumps({
                "error": f"Price not found for {ticker}",
                "suggestion": "Try adding .NS for NSE or .BO for BSE (e.g., TCS.NS)",
                "original_ticker": original_ticker
            }, indent=2)

        snapshot = {
            "ticker": ticker,
            "original_ticker": original_ticker if original_ticker != ticker else None,
            "price": round(float(current_price), 2),
            "currency": info.get('currency', 'USD'),
            "market": info.get('exchange', 'Unknown'),
            "company_name": info.get('longName') or info.get('shortName', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "market_cap": info.get('marketCap'),
            "day_high": info.get('dayHigh'),
            "day_low": info.get('dayLow'),
            "previous_close": info.get('previousClose'),
            "volume": info.get('volume'),
            "fifty_two_week_high": info.get('fiftyTwoWeekHigh'),
            "fifty_two_week_low": info.get('fiftyTwoWeekLow'),
        }
        snapshot = {k: v for k, v in snapshot.items() if v is not None}
        return json.dumps(snapshot, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error fetching price for {ticker}",
            "details": str(e),
            "ticker": ticker
        }, indent=2)

@mcp.tool()
async def portfolio_summary(holdings: dict,
                            purchase_prices: dict | None = None,
                            base_currency: str = "INR") -> dict:
    """
    Enhanced portfolio summary with FX, sector allocation, weighted dividend, and perf.
    """
    def _fx_pair(src: str, dst: str) -> str:
        return f"{src}{dst}=X"

    def _get_price_and_meta(tk: str) -> tuple[float | None, dict]:
        t = yf.Ticker(tk)
        info = t.info or {}
        price = info.get("currentPrice")
        if price is None:
            h = t.history(period="1d")
            if not h.empty:
                price = float(h["Close"].iloc[-1])
        return price, info

    def _fx_rate(src: str, dst: str) -> float:
        s = src.upper(); d = dst.upper()
        if s == d:
            return 1.0
        pair = _fx_pair(s, d)
        try:
            fx = yf.Ticker(pair).history(period="5d")["Close"].dropna()
            if fx.empty:
                return 1.0
            return float(fx.iloc[-1])
        except Exception:
            return 1.0

    purchase_prices = purchase_prices or {}
    base_currency = base_currency.upper()

    rows = []
    fx_cache: dict[tuple[str, str], float] = {}
    total_value_base = 0.0

    for tk, qty in holdings.items():
        qty = float(qty or 0.0)
        if qty <= 0:
            continue

        price, info = _get_price_and_meta(tk)
        if price is None:
            rows.append({"ticker": tk, "quantity": qty, "error": "Price not found"})
            continue

        native_ccy = (info.get("currency") or "USD").upper()
        fx_key = (native_ccy, base_currency)
        if fx_key not in fx_cache:
            fx_cache[fx_key] = _fx_rate(native_ccy, base_currency)
        rate = fx_cache[fx_key]

        value_native = price * qty
        value_base = value_native * rate
        total_value_base += value_base

        ppx = purchase_prices.get(tk)
        ret_pct = None
        if ppx is not None:
            try:
                ret_pct = ((price - float(ppx)) / float(ppx)) * 100.0
            except Exception:
                ret_pct = None

        rows.append({
            "ticker": tk,
            "quantity": qty,
            "price": round(price, 2),
            "currency": native_ccy,
            "value_native": round(value_native, 2),
            "value_base": round(value_base, 2),
            "purchase_price": round(float(ppx), 2) if ppx is not None else None,
            "return_pct_since_purchase": round(ret_pct, 2) if ret_pct is not None else None,
            "company_name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "market_cap": info.get("marketCap"),
        })

    if not rows:
        return {"error": "No valid holdings after parsing"}

    for r in rows:
        r["weight"] = round((r["value_base"] / total_value_base) if total_value_base else 0.0, 6)

    sector_alloc: dict[str, float] = {}
    for r in rows:
        sec = r.get("sector") or "Unknown"
        sector_alloc[sec] = sector_alloc.get(sec, 0.0) + r["value_base"]
    sector_alloc = {k: round(v / total_value_base, 6) for k, v in sector_alloc.items()}

    perf_rows = [r for r in rows if r.get("return_pct_since_purchase") is not None]
    top = sorted(perf_rows, key=lambda x: x["return_pct_since_purchase"], reverse=True)[:3]
    worst = sorted(perf_rows, key=lambda x: x["return_pct_since_purchase"])[:3]

    w_div = 0.0
    for r in rows:
        dy = r.get("dividend_yield")
        if dy is not None:
            w_div += dy * r["weight"]
    weighted_dividend_yield = round(float(w_div), 4) if w_div else None

    return {
        "base_currency": base_currency,
        "portfolio_value_base": round(float(total_value_base), 2),
        "analytics": {
            "num_holdings": len(rows),
            "weighted_dividend_yield": weighted_dividend_yield,
            "top_performers": top,
            "worst_performers": worst,
            "sector_allocation": sector_alloc,
            "fx_used": {f"{k[0]}->{k[1]}": v for k, v in fx_cache.items()},
        },
        "holdings": rows
    }

@mcp.tool()
async def screen_stocks(criteria: dict) -> List[Dict[str, Any]]:
    def format_market_cap(value: int | None) -> str | None:
        if value is None:
            return None
        trillion = 1_000_000_000_000
        billion = 1_000_000_000
        million = 1_000_000
        if value >= trillion:
            return f"₹{round(value / trillion, 2)}T"
        if value >= billion:
            return f"₹{round(value / billion, 2)}B"
        if value >= million:
            return f"₹{round(value / million, 2)}M"
        return str(value)

    if "tickers" not in criteria:
        return [{"error": "Please provide a list of tickers"}]

    tickers = criteria["tickers"]
    max_pe = criteria.get("max_pe", None)
    min_market_cap = criteria.get("min_market_cap", None)

    results: List[Dict[str, Any]] = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            pe_ratio = info.get("trailingPE")
            market_cap = info.get("marketCap")

            if max_pe is not None and (pe_ratio is None or pe_ratio > max_pe):
                continue
            if min_market_cap is not None and (market_cap is None or market_cap < min_market_cap):
                continue

            result = {
                "ticker": ticker,
                "company_name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "current_price": info.get("currentPrice"),
                "currency": info.get("currency"),
                "pe_ratio": pe_ratio,
                "eps": info.get("trailingEps"),
                "market_cap": format_market_cap(market_cap),
                "market_cap_raw": market_cap,
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow")
            }

            result["summary"] = (
                f"{result['company_name']} trades at a PE of {pe_ratio}, "
                f"with a market cap of {result['market_cap']}. "
                f"52-week range: {result['fifty_two_week_low']} - {result['fifty_two_week_high']}. "
                f"Dividend yield: {result['dividend_yield']}."
            )

            results.append(result)

        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    return results

# --------- Stock predictions (tz fix, exog, caching) ---------
@mcp.tool()
async def predict_stock_profit(ticker: str, purchase_date: str, future_date: str) -> dict:
    """
    Predicts future stock price using ARIMA/SARIMAX + fundamentals.
    Fixes tz-aware/naive errors and ensures forecasting is beyond training window.
    """
    try:
        ticker_map = {
            "TCS": "TCS.NS",
            "INFY": "INFY.NS",
            "RELIANCE": "RELIANCE.NS",
            "HDFCBANK": "HDFCBANK.NS",
            "ICICIBANK": "ICICIBANK.NS",
            "SBIN": "SBIN.NS",
            "MRF": "MRF.NS",
        }
        raw_ticker = ticker.strip().upper()
        ticker_clean = ticker_map.get(raw_ticker, raw_ticker)

        ticker_obj = yf.Ticker(ticker_clean)
        # Fetch appropriate date range to keep fitting fast and accurate
        p_dt = pd.to_datetime(purchase_date).tz_localize(None)
        f_dt = pd.to_datetime(future_date).tz_localize(None)
        
        # Start at least 1 year before purchase date or 3 years ago
        start_bound = min(p_dt - pd.Timedelta(days=365), pd.Timestamp.now() - pd.Timedelta(days=730))
        hist = ticker_obj.history(start=start_bound.strftime("%Y-%m-%d"))

        # Fallback if raw ticker has no data and might be Indian stock without suffix
        if hist.empty and not any(ticker_clean.endswith(sfx) for sfx in [".NS", ".BO"]) and "." not in ticker_clean:
            alt_ticker = ticker_clean + ".NS"
            alt_obj = yf.Ticker(alt_ticker)
            alt_hist = alt_obj.history(start=start_bound.strftime("%Y-%m-%d"))
            if not alt_hist.empty:
                ticker_clean = alt_ticker
                ticker_obj = alt_obj
                hist = alt_hist

        if hist.empty:
            hist = ticker_obj.history(period="2y")

        if hist.empty:
            return {"error": f"No historical data found for {ticker_clean}"}

        series = hist["Close"].dropna().asfreq("B").ffill()
        series.index = series.index.tz_localize(None)
        if series.empty:
            return {"error": f"No valid closing prices found for {ticker_clean}"}

        today = series.index[-1]

        purchase_date_dt = p_dt
        future_date_dt = f_dt
        if future_date_dt <= today:
            return {"error": "Future date must be later than the last market date."}

        # Safe alignment for purchase date without infinite loops
        if purchase_date_dt <= series.index[0]:
            purchase_date_dt = series.index[0]
        elif purchase_date_dt >= series.index[-1]:
            purchase_date_dt = series.index[-1]
        else:
            valid_dates = series.index[series.index >= purchase_date_dt]
            if not valid_dates.empty:
                purchase_date_dt = valid_dates[0]
            else:
                purchase_date_dt = series.index[-1]

        purchase_price = float(series.loc[purchase_date_dt])
        latest_price = float(series.iloc[-1])

        # Exogenous fundamentals (if available)
        has_exog = False
        exog_aligned = pd.DataFrame()
        try:
            balance_sheet = getattr(ticker_obj, "balance_sheet", pd.DataFrame())
            income_statement = getattr(ticker_obj, "financials", pd.DataFrame())
            if not balance_sheet.empty or not income_statement.empty:
                exog_df = pd.concat([balance_sheet.T, income_statement.T], axis=1)
                exog_df = exog_df.apply(pd.to_numeric, errors="coerce").ffill().replace([np.inf, -np.inf], 0)
                exog_df.index = pd.to_datetime(exog_df.index).tz_localize(None)
                exog_aligned = exog_df.reindex(series.index).ffill().bfill().fillna(0)
                features = [
                    "Total Assets", "Total Current Liabilities", "Total Liab",
                    "Total Revenue", "Gross Profit", "Operating Income", "Net Income"
                ]
                exog_aligned = exog_aligned[[c for c in features if c in exog_aligned.columns]]
                if not exog_aligned.empty and len(exog_aligned.columns) > 0:
                    has_exog = True
        except Exception as e:
            logger.warning(f"Could not load exogenous features for {ticker_clean}: {e}")
            has_exog = False

        forecast_index = pd.bdate_range(today + BDay(1), future_date_dt)
        days = len(forecast_index)
        if days <= 0:
            days = 1

        # Use recent subset of series (max 500 business days) for fast and responsive fitting
        series_train = series.iloc[-500:] if len(series) > 500 else series

        def fit_arima():
            model = pm.auto_arima(series_train, seasonal=False, error_action="ignore", suppress_warnings=True, maxiter=20)
            return ARIMA(series_train, order=model.order).fit()

        def fit_sarimax():
            exog_train = exog_aligned.reindex(series_train.index).ffill().bfill().fillna(0)
            model = pm.auto_arima(series_train, exogenous=exog_train, seasonal=True, m=5, error_action="ignore", suppress_warnings=True, maxiter=20)
            return SARIMAX(
                series_train,
                order=model.order,
                seasonal_order=getattr(model, "seasonal_order", (0, 0, 0, 0)),
                exog=exog_train
            ).fit(disp=False)

        forecast = None
        if days > 60 and has_exog:
            try:
                fitted = fit_sarimax()
                future_exog = pd.DataFrame(
                    [exog_aligned.iloc[-1].values] * days,
                    columns=exog_aligned.columns,
                    index=forecast_index
                )
                forecast = fitted.predict(
                    start=len(series_train), end=len(series_train) + days - 1, exog=future_exog
                )
            except Exception as e:
                logger.warning(f"SARIMAX failed, falling back to ARIMA: {e}")
                forecast = None

        if forecast is None:
            fitted = fit_arima()
            forecast = fitted.predict(start=len(series_train), end=len(series_train) + days - 1)

        forecast_price = float(forecast.iloc[-1])
        pnl = forecast_price - purchase_price
        pct = (pnl / purchase_price) * 100.0 if purchase_price else 0.0

        return {
            "ticker": ticker_clean,
            "purchase_date": str(purchase_date_dt.date()),
            "purchase_price": round(purchase_price, 2),
            "latest_price": round(latest_price, 2),
            "predicted_future_price": round(forecast_price, 2),
            "profit_loss_purchase_to_future": round(pnl, 2),
            "pct_change": round(pct, 2),
            "days_forecasted": days
        }

    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        return {"error": str(e)}

# ================================
# 📌 Savings Forecast Tool (+ quartiles)
# ================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "savings_model.joblib")

try:
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        pipeline = saved["pipeline"]
        feature_cols = saved["feature_cols"]
        target_cols = saved["target_cols"]
        print(f"✅ Savings model loaded successfully from: {MODEL_PATH}", file=sys.stderr)
    else:
        pipeline, feature_cols, target_cols = None, [], []
        print(f"⚠ Model file not found at {MODEL_PATH}.", file=sys.stderr)
except Exception as e:
    pipeline, feature_cols, target_cols = None, [], []
    print(f"⚠ Model load failed: {e}", file=sys.stderr)

@mcp.tool()
async def forecast_savings(json_path: str = "", desired_savings_percentage: float = 20.0, budget_data: dict | None = None) -> dict:
    """Predict potential savings for a user's monthly budget + quartile feedback."""
    if not pipeline:
        return {"error": "Model not loaded. Please ensure models/savings_model.joblib exists."}

    input_data = None
    if budget_data and isinstance(budget_data, dict):
        input_data = budget_data
    else:
        # Resolve json_path
        if not json_path:
            json_path = os.path.join(BASE_DIR, "data", "user.json")
        candidates = [
            json_path,
            os.path.join(BASE_DIR, json_path),
            os.path.join(BASE_DIR, "data", json_path),
            os.path.join(BASE_DIR, "data", "user.json")
        ]
        resolved = None
        for c in candidates:
            if os.path.exists(c):
                resolved = c
                break

        if not resolved:
            return {"error": f"JSON budget file not found: {json_path}"}

        try:
            with open(resolved, "r", encoding="utf-8") as f:
                input_data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to read JSON: {e}"}

    X_in = pd.DataFrame([input_data])

    for col in feature_cols:
        if col not in X_in.columns:
            X_in[col] = 0.0

    X_in["Desired_Savings_Percentage"] = float(desired_savings_percentage)

    expense_cols = EXPENSE_COLS
    X_in["Total_Expenses"] = X_in[expense_cols].sum(axis=1)
    X_in["Disposable_Income"] = X_in["Income"] - X_in["Total_Expenses"]
    X_in["Desired_Savings"] = (X_in["Income"] * desired_savings_percentage / 100)

    X_pred = X_in[feature_cols]

    try:
        preds = pipeline.predict(X_pred)
        preds_df = pd.DataFrame(preds, columns=target_cols)
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    derived = {
        "Income": float(X_in["Income"].iloc[0]),
        "Desired_Savings_Percentage": float(desired_savings_percentage),
        "Desired_Savings": float(X_in["Desired_Savings"].iloc[0]),
        "Total_Expenses": float(X_in["Total_Expenses"].iloc[0]),
        "Disposable_Income": float(X_in["Disposable_Income"].iloc[0])
    }

    # ----- Quartile feedback & table -----
    feedback = {"Slightly_High_Q2_Q3": [], "Highly_Overspending_Above_Q3": [], "Below_Q1_Frugal": []}
    quart_out = {}

    if QUARTILES:
        for cat, qvals in QUARTILES.items():
            user_val = float(X_in[cat].iloc[0]) if cat in X_in.columns else None
            if user_val is not None:
                q1, q2, q3 = qvals["Q1"], qvals["Q2"], qvals["Q3"]
                if q2 <= user_val < q3:
                    feedback["Slightly_High_Q2_Q3"].append(cat)
                elif user_val >= q3:
                    feedback["Highly_Overspending_Above_Q3"].append(cat)
                elif user_val < q1:
                    feedback["Below_Q1_Frugal"].append(cat)
                quart_out[cat] = {"Q1": q1, "Q2": q2, "Q3": q3}

    return {
        "Derived_Values": derived,
        "Predicted_Savings": preds_df.iloc[0].to_dict(),
        "Quartiles": quart_out,
        "Quartile_Feedback": feedback,
        "Quartile_Source": QUARTILE_SRC
    }

# ================================
# 🚀 Run MCP
# ================================
if __name__ == "__main__":
    logger.info("Starting MCP Server with Financial + Personal Finance Tools...")
    mcp.run(transport="stdio")
