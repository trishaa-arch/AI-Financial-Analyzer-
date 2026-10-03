import streamlit as st
import asyncio
import json
import math
import os
import pandas as pd
from datetime import date, datetime
from typing import Any, Dict, List, Union
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import sys, platform
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ==== .env & Gemini ====
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_ENABLED:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        GEMINI_ENABLED = False

# ---------- Windows async policy ----------
if platform.system().lower().startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# -----------------------
# Event Loop Management
# -----------------------
@st.cache_resource
def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

loop = get_event_loop()

# -----------------------
# Session State
# -----------------------
if "connected" not in st.session_state:
    st.session_state.connected = False
if "tools" not in st.session_state:
    st.session_state.tools = []
if "filtered_tickers" not in st.session_state:
    st.session_state.filtered_tickers = None
if "filtered_details" not in st.session_state:
    st.session_state.filtered_details = None
for key in ("mcp_client", "mcp_session", "read_stream", "write_stream"):
    if key not in st.session_state:
        st.session_state[key] = None

# Artifacts to feed Gemini
ARTIFACT_DIR = os.path.join(BASE_DIR, "data", "session_exports")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
if "artifact_paths" not in st.session_state:
    st.session_state.artifact_paths = []   # newest last
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None

def _save_artifact(name: str, payload: Any):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ARTIFACT_DIR, f"{ts}_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        st.session_state.artifact_paths.append(path)
        # Keep the last ~10 only
        st.session_state.artifact_paths = st.session_state.artifact_paths[-10:]
        return path
    except Exception:
        return None

# -----------------------
# App Header
# -----------------------
st.title("Finance MCP")
st.markdown("Interact with your MCP financial server.")

# -----------------------
# Connect / Disconnect UI
# -----------------------
st.header("📡 Connection")
col1, col2 = st.columns(2)

async def _connect_async():
    server_script = os.path.join(BASE_DIR, "server.py")
    server = StdioServerParameters(command=sys.executable, args=[server_script], cwd=BASE_DIR)
    client = stdio_client(server)
    read_stream, write_stream = await client.__aenter__()
    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()
    init = await session.initialize()
    tools = await session.list_tools()

    st.session_state.mcp_client = client
    st.session_state.read_stream = read_stream
    st.session_state.write_stream = write_stream
    st.session_state.mcp_session = session

    st.session_state.tools = [
        {"name": t.name, "description": t.description, "schema": getattr(t, "inputSchema", None)}
        for t in tools.tools
    ]
    st.session_state.connected = True
    st.success(f"✅ Connected to {init.serverInfo.name} ({init.serverInfo.version})")

def connect_to_server():
    try:
        loop.run_until_complete(_connect_async())
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        st.exception(e)

async def _disconnect_async():
    if st.session_state.mcp_session is not None:
        try:
            await st.session_state.mcp_session.__aexit__(None, None, None)
        except Exception:
            pass
    if st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
        except Exception:
            pass
    st.session_state.mcp_client = None
    st.session_state.mcp_session = None
    st.session_state.read_stream = None
    st.session_state.write_stream = None

with col1:
    st.button("🔌 Connect", disabled=st.session_state.connected, on_click=connect_to_server)

with col2:
    def _disconnect_click():
        try:
            loop.run_until_complete(_disconnect_async())
        finally:
            st.session_state.connected = False
            st.session_state.tools = []
            st.session_state.filtered_tickers = None
            st.session_state.filtered_details = None
            st.rerun()
    st.button("🔴 Disconnect", disabled=not st.session_state.connected, on_click=_disconnect_click)

st.divider()

# -----------------------
# Helpers
# -----------------------
def human_currency(n: Union[int, float, None], currency: str = "USD") -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "-"
    sign = "-" if float(n) < 0 else ""
    n = abs(float(n))
    unit = ""
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000.0:
            break
        n /= 1000.0
    symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}
    sym = symbols.get(currency.upper(), "")
    return f"{sign}{sym}{n:,.2f}{unit}"

def infer_currency_from_ticker(ticker: str | None) -> str:
    if not ticker:
        return "USD"
    tick = str(ticker).upper()
    if tick.endswith(".NS") or tick.endswith(".BO"):
        return "INR"
    return "USD"

def tidy_float(x: Any) -> Any:
    try:
        f = float(x)
        if abs(f) >= 100:
            return f"{f:,.0f}"
        return f"{f:,.2f}"
    except Exception:
        return x

# -----------------------
# Universal Renderer
# -----------------------
def render_result_payload(payload: Any, context: Dict[str, Any] | None = None):
    # Persist last payload (for AI chat & saving)
    st.session_state.last_payload = payload

    if isinstance(payload, str):
        st.code(payload)
        return

    # --- SAVINGS TOOL ---
    if isinstance(payload, dict) and "Derived_Values" in payload and "Predicted_Savings" in payload:
        # Save artifact
        _save_artifact("forecast_savings", payload)

        derived = payload["Derived_Values"]
        savings = payload["Predicted_Savings"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Income", human_currency(derived.get("Income"), "INR"))
        c2.metric("Desired Savings (%)", f"{derived.get('Desired_Savings_Percentage', 0)}%")
        c3.metric("Desired Savings", human_currency(derived.get("Desired_Savings"), "INR"))
        c4.metric("Disposable Income", human_currency(derived.get("Disposable_Income"), "INR"))

        st.markdown("### 📊 Potential Monthly Savings (by category)")
        rows = []
        for k, v in savings.items():
            label = k.replace("Potential_Savings_", "").replace("_", " ")
            rows.append({"Category": label, "Potential_Saving": float(v or 0)})
        df = pd.DataFrame(rows).sort_values("Potential_Saving", ascending=False)
        df["Potential_Saving_Display"] = df["Potential_Saving"].apply(lambda x: human_currency(x, "INR"))
        st.dataframe(df[["Category", "Potential_Saving_Display"]], use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Category")["Potential_Saving"])

        # ---- Savings Split Pie ----
        st.markdown("#### Savings Split (Pie)")
        if not df.empty:
            fig, ax = plt.subplots()
            ax.pie(df["Potential_Saving"], labels=df["Category"], autopct="%1.1f%%", startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        # ---- Quartiles table ----
        quart = payload.get("Quartiles") or {}
        if quart:
            qt_rows = []
            for cat, qvals in quart.items():
                qt_rows.append({
                    "Category": cat,
                    "Q1": qvals.get("Q1"),
                    "Q2 (Median)": qvals.get("Q2"),
                    "Q3": qvals.get("Q3"),
                })
            st.markdown("### Benchmark Quartiles (from data.csv)")
            st.dataframe(pd.DataFrame(qt_rows), use_container_width=True, hide_index=True)
            src = payload.get("Quartile_Source")
            # if src:
            #     st.caption(f"Benchmarks loaded from: `{src}`")

        # ---- Feedback buckets ----
        fb = payload.get("Quartile_Feedback") or {}
        st.markdown("### 🧭 Expense Feedback")
        c1, c2 = st.columns(2)
        slightly = fb.get("Slightly_High_Q2_Q3", [])
        high = fb.get("Highly_Overspending_Above_Q3", [])
        c1.markdown("**Slightly High (Q2–Q3):**")
        c1.write(", ".join(slightly) if slightly else "All good 👍")
        c2.markdown("**Highly Overspending (Above Q3):**")
        c2.write(", ".join(high) if high else "All good 👍")
        return

    # --- ENHANCED PORTFOLIO ---
    if isinstance(payload, dict) and "analytics" in payload and "holdings" in payload:
        # Save artifact
        _save_artifact("portfolio_summary", payload)

        base = payload.get("base_currency", "INR")

        k1, k2 = st.columns(2)
        k1.metric("Total Value", human_currency(payload.get("portfolio_value_base"), base))
        k2.metric("# Holdings", payload.get("analytics", {}).get("num_holdings", 0))

        alloc = payload.get("analytics", {}).get("sector_allocation", {})
        if alloc:
            alloc_df = pd.DataFrame({"Sector": list(alloc.keys()),
                                     "Weight%": [v*100 for v in alloc.values()]}).sort_values("Weight%", ascending=False).set_index("Sector")
            st.markdown("### Sector Allocation (%)")
            st.bar_chart(alloc_df["Weight%"])

        analytics = payload.get("analytics", {})
        top = analytics.get("top_performers") or []
        worst = analytics.get("worst_performers") or []
        if top or worst:
            st.markdown("### Performance Highlights")
            ctop, cworst = st.columns(2)
            if top:
                ctop.markdown("**Top performers**")
                top_df = pd.DataFrame(top)
                if "return_pct_since_purchase" in top_df.columns:
                    top_df.rename(columns={"return_pct_since_purchase": "Return %"}, inplace=True)
                ctop.dataframe(top_df, use_container_width=True, hide_index=True)
            if worst:
                cworst.markdown("**Worst performers**")
                worst_df = pd.DataFrame(worst)
                if "return_pct_since_purchase" in worst_df.columns:
                    worst_df.rename(columns={"return_pct_since_purchase": "Return %"}, inplace=True)
                cworst.dataframe(worst_df, use_container_width=True, hide_index=True)

        table = pd.DataFrame(payload["holdings"])
        cols_order = ["ticker","quantity","price","currency","value_native","value_base",
                      "weight","purchase_price","return_pct_since_purchase","sector","pe_ratio","dividend_yield"]
        table = table[[c for c in cols_order if c in table.columns]]
        table.rename(columns={
            "value_native": f"Value (native)",
            "value_base":   f"Value ({base})",
            "weight": "Weight",
            "price": "Live Price",
            "purchase_price": "Purchase Price",
            "return_pct_since_purchase": "Return %"
        }, inplace=True)
        if "Weight" in table.columns:
            table["Weight"] = table["Weight"].apply(lambda x: f"{float(x)*100:.2f}%")
        st.markdown("### Holdings")
        st.dataframe(table, use_container_width=True, hide_index=True)
        return

    # --- PRICE TOOL ---
    if isinstance(payload, dict) and "price" in payload:
        cur = infer_currency_from_ticker(payload.get("ticker"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", payload.get("ticker", "—"))
        c2.metric("Company", payload.get("company_name", "—"))
        c3.metric("Price", human_currency(payload.get("price"), payload.get("currency", cur)))
        c4.metric("Market", payload.get("market", "—"))
        return

    # --- PREDICT STOCK PROFIT ---
    if isinstance(payload, dict) and {"purchase_price", "latest_price", "predicted_future_price"} <= payload.keys():
        tick = payload.get("ticker", "—")
        cur = infer_currency_from_ticker(tick)
        purchase = payload.get("purchase_price")
        latest = payload.get("latest_price")
        future = payload.get("predicted_future_price")
        pnl = payload.get("profit_loss_purchase_to_future")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", tick)
        c2.metric("Buy Price", human_currency(purchase, cur))
        c3.metric("Latest Price", human_currency(latest, cur))
        c4.metric("Forecast Price", human_currency(future, cur))

        try:
            pct = ((future - purchase) / purchase) * 100.0
        except Exception:
            pct = None

        colA, colB = st.columns(2)
        colA.metric("Expected P/L", human_currency(pnl, cur), f"{pct:.2f}%" if pct is not None else None)
        mini = pd.DataFrame({"Price": [purchase, latest, future]}, index=["Buy", "Latest", "Forecast"])
        colB.bar_chart(mini)
        return

    # --- SCREEN STOCKS ---
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        df = pd.DataFrame(payload)
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    if isinstance(payload, dict):
        df = pd.DataFrame([[k, v] for k, v in payload.items()], columns=["Key", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    if isinstance(payload, list):
        st.dataframe(pd.DataFrame(payload), use_container_width=True, hide_index=True)
        return

    st.code(json.dumps(payload, indent=2))

def render_result(raw: Any):
    if isinstance(raw, list):
        for chunk in raw:
            try:
                data = json.loads(chunk.text)
            except Exception:
                data = getattr(chunk, "text", chunk)
            render_result_payload(data)
    else:
        try:
            data = json.loads(raw)
        except Exception:
            data = raw
        render_result_payload(data)

# -----------------------
# Tool Call Helpers
# -----------------------
def _require_session() -> ClientSession | None:
    if not st.session_state.connected or st.session_state.mcp_session is None:
        st.warning("Please connect to the server first.")
        return None
    return st.session_state.mcp_session

def execute_tool(tool_name: str, args: dict):
    session = _require_session()
    if session is None:
        return

    async def run(sess: ClientSession):
        try:
            with st.spinner(f"⚡ Running {tool_name}..."):
                result = await sess.call_tool(tool_name, args)
            st.success("✅ Execution complete")
            st.subheader("📊 Result")
            render_result(result.content)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

    loop.run_until_complete(run(session))

def call_tool_return_json(tool_name: str, args: dict):
    session = _require_session()
    if session is None:
        return None

    async def run(sess: ClientSession):
        result = await sess.call_tool(tool_name, args)
        rows = []
        for item in result.content:
            try:
                rows.append(json.loads(item.text))
            except Exception:
                rows.append(item.text)
        if len(rows) == 1:
            return rows[0]
        return rows

    return loop.run_until_complete(run(session))

# -----------------------
# yfinance Universe
# -----------------------
@st.cache_data(ttl=60 * 60 * 24)
def get_yf_universe():
    try:
        if hasattr(yf, "tickers_sp500") and callable(yf.tickers_sp500):
            return sorted(set(yf.tickers_sp500() + yf.tickers_dow()))
    except Exception:
        pass
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BRK-B",
        "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "MRF.NS"
    ]

# -----------------------
# Tool UI
# -----------------------
if st.session_state.connected and st.session_state.tools:
    st.header("🛠️ Tools")

    tool_names = [t["name"] for t in st.session_state.tools]
    selected_tool = st.selectbox("Select Tool", tool_names)

    st.divider()

    # --- get_current_stock_price ---
    if selected_tool == "get_current_stock_price":
        ticker = st.text_input("Enter Ticker", "AAPL")
        st.button("▶️ Run", on_click=lambda: execute_tool(selected_tool, {"ticker": ticker}))

    # --- portfolio_summary ---
    elif selected_tool == "portfolio_summary":
        st.markdown("### Build Portfolio")
        base_ccy = st.selectbox("Base currency", ["INR", "USD", "EUR", "GBP"], index=0)

        st.caption("Optional: Upload CSV with columns: Ticker, Quantity, PurchasePrice")
        up = st.file_uploader("Upload holdings CSV", type=["csv"], accept_multiple_files=False)

        st.markdown("#### Add/Modify Holdings")
        universe = get_yf_universe()
        with st.form("portfolio_form"):
            init_df = pd.DataFrame(
                [{"Ticker": "AAPL", "Quantity": 10.0, "PurchasePrice": ""}]
            )
            edited = st.data_editor(
                init_df,
                num_rows="dynamic",
                column_config={
                    "Ticker": st.column_config.SelectboxColumn(options=universe + ["MRF.NS", "TCS.NS", "INFY.NS"]),
                    "Quantity": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f"),
                    "PurchasePrice": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.2f"),
                },
                use_container_width=True
            )
            st.form_submit_button("Apply")

        if up is not None:
            try:
                df_csv = pd.read_csv(up)
                df_csv.columns = [c.strip().lower() for c in df_csv.columns]
                map_rows = []
                for _, r in df_csv.iterrows():
                    map_rows.append({
                        "Ticker": str(r.get("ticker") or r.get("symbol") or "").strip(),
                        "Quantity": float(r.get("quantity") or r.get("qty") or 0.0),
                        "PurchasePrice": float(r.get("purchaseprice") or r.get("ppx") or 0.0) if (r.get("purchaseprice") or r.get("ppx")) else ""
                    })
                edited = pd.DataFrame(map_rows)
            except Exception as e:
                st.error(f"CSV parse failed: {e}")

        if not edited.empty:
            preview_rows = []
            for _, r in edited.iterrows():
                tk = str(r["Ticker"]).strip()
                if not tk:
                    continue
                qty = float(r.get("Quantity") or 0.0)
                ppx = r.get("PurchasePrice")
                try:
                    info = yf.Ticker(tk).info or {}
                    price = info.get("currentPrice")
                    if price is None:
                        hist = yf.Ticker(tk).history(period="1d")
                        price = float(hist["Close"].iloc[-1]) if not hist.empty else None
                except Exception:
                    price = None

                val = (price * qty) if (price is not None) else None
                ret = None
                if (ppx not in [None, ""]) and (price is not None):
                    try:
                        ret = ((float(price) - float(ppx)) / float(ppx)) * 100.0
                    except Exception:
                        ret = None

                preview_rows.append({
                    "Ticker": tk,
                    "Quantity": qty,
                    "PurchasePrice": ppx if ppx not in [None, ""] else None,
                    "LivePrice": round(float(price), 2) if price is not None else None,
                    "RowValue": round(float(val), 2) if val is not None else None,
                    "Return%": round(float(ret), 2) if ret is not None else None
                })

            st.markdown("#### Preview (live)")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        if st.button("▶️ Run"):
            rows = []
            for _, r in edited.iterrows():
                tk = str(r["Ticker"]).strip()
                if not tk:
                    continue
                qty = float(r.get("Quantity") or 0.0)
                ppx = r.get("PurchasePrice")
                rows.append((tk, qty, ppx))

            if not rows:
                st.warning("Please add at least one holding.")
            else:
                holdings = {tk: qty for tk, qty, _ in rows}
                ppx = {tk: float(pp) for tk, _, pp in rows if pp not in [None, ""]}
                execute_tool(
                    selected_tool,
                    {"holdings": holdings, "purchase_prices": ppx, "base_currency": base_ccy}
                )

    # --- screen_stocks ---
    elif selected_tool == "screen_stocks":
        st.markdown("### Stock Screener (Filter → Then Select)")

        max_pe = st.number_input("Max P/E", value=40.0, step=1.0, min_value=0.0)
        mcap_options = {
            "100M": 100_000_000, "500M": 500_000_000, "1B": 1_000_000_000,
            "5B": 5_000_000_000, "10B": 10_000_000_000, "50B": 50_000_000_000,
            "100B": 100_000_000_000,
        }
        mcap_label = st.selectbox("Min Market Cap (preset)", list(mcap_options.keys()), index=2)
        min_mcap = mcap_options[mcap_label]

        st.caption("Screener uses only S&P 500 + DOW 30. Manual tickers are not filtered.")

        if st.button("🔍 Find Eligible Stocks"):
            universe = get_yf_universe()
            with st.spinner(f"Filtering {len(universe)} tickers..."):
                result = call_tool_return_json(
                    "screen_stocks",
                    {"criteria": {"tickers": universe, "max_pe": max_pe, "min_market_cap": min_mcap}},
                )
                eligible = [row.get("ticker") for row in (result or []) if isinstance(row, dict) and not row.get("error")]
                st.session_state.filtered_details = result
                st.session_state.filtered_tickers = sorted(set([t for t in eligible if t]))

        if st.session_state.filtered_tickers is not None:
            count = len(st.session_state.filtered_tickers)
            st.success(f"✅ {count} matching stocks found.")
            selected_from_list = st.multiselect("Matching Stocks", st.session_state.filtered_tickers, default=[])
        else:
            selected_from_list = []

        manual = st.text_input("Or add custom tickers (comma-separated)", "")

        if st.button("▶️ Run"):
            final_tickers = list(selected_from_list)
            final_tickers += [t.strip() for t in manual.split(",") if t.strip()]
            final_tickers = sorted(set(final_tickers))

            if not final_tickers:
                st.warning("Please select at least one ticker.")
            else:
                if manual.strip():
                    execute_tool("screen_stocks", {"criteria": {"tickers": final_tickers}})
                else:
                    execute_tool(
                        "screen_stocks",
                        {"criteria": {"tickers": final_tickers, "max_pe": max_pe, "min_market_cap": min_mcap}},
                    )

    # --- predict_stock_profit ---
    elif selected_tool == "predict_stock_profit":
        ticker = st.text_input("Ticker", "MRF.NS")
        purchase_date = st.date_input("Purchase Date", date(2025, 11, 3))
        future_date = st.date_input("Future Date", date(2026, 1, 25))

        if st.button("▶️ Run"):
            execute_tool(
                selected_tool,
                {"ticker": ticker, "purchase_date": str(purchase_date), "future_date": str(future_date)},
            )

    # --- forecast_savings ---
    elif selected_tool == "forecast_savings":
        st.subheader("Monthly Budget Inputs")

        income = st.number_input("Income", min_value=0.0, value=90000.0, step=1000.0)

        st.markdown("#### Expense Breakdown")
        rent = st.number_input("Rent", min_value=0.0, value=15000.0, step=500.0)
        loan = st.number_input("Loan Repayment", min_value=0.0, value=8000.0, step=500.0)
        insurance = st.number_input("Insurance", min_value=0.0, value=3000.0, step=500.0)
        groceries = st.number_input("Groceries", min_value=0.0, value=9000.0, step=500.0)
        transport = st.number_input("Transport", min_value=0.0, value=3000.0, step=500.0)
        eating_out = st.number_input("Eating Out", min_value=0.0, value=2000.0, step=500.0)
        entertainment = st.number_input("Entertainment", min_value=0.0, value=2500.0, step=500.0)
        utilities = st.number_input("Utilities", min_value=0.0, value=4000.0, step=500.0)
        healthcare = st.number_input("Healthcare", min_value=0.0, value=2000.0, step=500.0)
        education = st.number_input("Education", min_value=0.0, value=0.0, step=500.0)
        misc = st.number_input("Miscellaneous", min_value=0.0, value=3000.0, step=500.0)

        desired_pct = st.slider("Desired Savings %", min_value=1, max_value=80, value=20, step=1)

        if st.button("▶️ Run"):
            budget_dict = {
                "Income": income, "Rent": rent, "Loan_Repayment": loan, "Insurance": insurance,
                "Groceries": groceries, "Transport": transport, "Eating_Out": eating_out,
                "Entertainment": entertainment, "Utilities": utilities, "Healthcare": healthcare,
                "Education": education, "Miscellaneous": misc,
            }

            execute_tool(
                selected_tool,
                {"budget_data": budget_dict, "desired_savings_percentage": desired_pct},
            )

else:
    st.info("👆 Connect to server to load tools")

st.divider()
# st.caption("Polished outputs with auto-charts • Theme matches your Streamlit settings")

# =========================
# 💬 Talk to your money
# =========================
st.markdown("---")
st.header("💬 Talk to your money")

col_chat1, col_chat2 = st.columns([3, 1])
with col_chat2:
    custom_gemini_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password", placeholder="AIzaSy...")

# Show the most recent artifacts we’ll send as context (latest 3)
recent_context = list(reversed(st.session_state.artifact_paths))[:3]
if recent_context:
    with st.expander("📁 Loaded Context Data (latest session outputs)"):
        for p in recent_context:
            st.write(f"• `{os.path.basename(p)}`")

prompt = st.text_area("Ask anything about your savings/portfolio:", height=100, placeholder="e.g., Where am I overspending? What’s my sector risk? How can I optimize savings?")

def _read_file(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()[:100_000]  # keep request compact
    except Exception:
        return ""

def _local_financial_advisor(prompt_text: str, context_text: str) -> str:
    """Built-in rule-based advisor when AI API key is not configured."""
    lines = ["🤖 **Financial Analysis Summary:**\n"]
    if not context_text or context_text == "No context files yet.":
        return "ℹ️ No recent analysis data found. Please run **Portfolio Summary** or **Forecast Savings** first to generate context."
    
    if "forecast_savings" in context_text:
        lines.append("### 💰 Budget & Savings Insights:")
        lines.append("- Based on your budget model, we analyzed your expenses against benchmark quartiles.")
        lines.append("- Categories above Q3 benchmark represent prime areas to reduce spending and redirect into high-yield investments or emergency reserves.")
    
    if "portfolio_summary" in context_text:
        lines.append("### 📈 Investment Portfolio Insights:")
        lines.append("- Your holdings have been evaluated with live market quotes and converted to your base currency.")
        lines.append("- Monitor your top and bottom performers to maintain optimal rebalancing and risk mitigation.")
        
    lines.append("\n💡 *Tip: Enter a valid Gemini API Key above to unlock conversational deep-dives and custom financial reasoning!*")
    return "\n".join(lines)

if st.button("▶️ Ask"):
    if not prompt.strip():
        st.warning("Please type a question.")
    else:
        # Build context text from artifacts (savings/portfolio JSONs)
        context_blobs = []
        for fp in recent_context:
            txt = _read_file(fp)
            if txt:
                context_blobs.append(f"FILE: {os.path.basename(fp)}\n{txt}")
        context_text = "\n\n".join(context_blobs) if context_blobs else "No context files yet."

        active_key = (custom_gemini_key.strip() or GEMINI_API_KEY).strip()
        answer = None

        if active_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=active_key)
                sysmsg = (
                    "You are a helpful financial assistant. Use the provided JSON context (savings forecasts and portfolio "
                    "analytics) to answer questions. If data is missing, say so. Be concise and actionable."
                )
                full_prompt = f"{sysmsg}\n\n=== CONTEXT START ===\n{context_text}\n=== CONTEXT END ===\n\nUSER QUESTION:\n{prompt.strip()}"
                
                # Try preferred model and fallbacks
                models_to_try = [GEMINI_MODEL, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
                for m_name in models_to_try:
                    try:
                        model = genai.GenerativeModel(m_name)
                        resp = model.generate_content(full_prompt)
                        if resp and resp.text:
                            answer = resp.text.strip()
                            break
                    except Exception:
                        continue
                if not answer:
                    st.warning("Could not reach Gemini with the provided key. Falling back to local advisor:")
                    answer = _local_financial_advisor(prompt, context_text)
            except Exception as e:
                st.warning(f"Gemini connection error ({e}). Showing local financial analysis:")
                answer = _local_financial_advisor(prompt, context_text)
        else:
            answer = _local_financial_advisor(prompt, context_text)

        st.markdown("**Answer:**")
        st.write(answer)
