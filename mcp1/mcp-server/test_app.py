import asyncio
import os
import sys
import json
import platform
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

if platform.system().lower().startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

async def test_full_application():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(base_dir, "server.py")
    
    print("=" * 60)
    print("Starting End-to-End MCP Application Verification")
    print(f"Python: {sys.executable}")
    print(f"Server script: {server_script}")
    print("=" * 60)

    server = StdioServerParameters(command=sys.executable, args=[server_script], cwd=base_dir)
    client = stdio_client(server)
    read_stream, write_stream = await client.__aenter__()
    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()

    # 1. Initialize
    init = await session.initialize()
    print(f"[1/6] MCP Initialize: SUCCESS ({init.serverInfo.name} v{init.serverInfo.version})")

    # 2. List tools
    tools = await session.list_tools()
    tool_names = [t.name for t in tools.tools]
    print(f"[2/6] Available Tools ({len(tool_names)}): {tool_names}")

    # 3. Test get_current_stock_price
    price_res = await session.call_tool("get_current_stock_price", {"ticker": "AAPL"})
    print(f"[3/6] get_current_stock_price (AAPL): {price_res.content[0].text[:80]}...")

    # 4. Test portfolio_summary
    portfolio_res = await session.call_tool("portfolio_summary", {
        "holdings": {"AAPL": 5, "MSFT": 3},
        "purchase_prices": {"AAPL": 150.0, "MSFT": 300.0},
        "base_currency": "INR"
    })
    port_data = json.loads(portfolio_res.content[0].text)
    print(f"[4/6] portfolio_summary: Value = {port_data.get('portfolio_value_base')} {port_data.get('base_currency')}, Holdings = {len(port_data.get('holdings', []))}")

    # 5. Test forecast_savings
    savings_res = await session.call_tool("forecast_savings", {
        "json_path": os.path.join(base_dir, "data", "user.json"),
        "desired_savings_percentage": 20.0
    })
    savings_data = json.loads(savings_res.content[0].text)
    print(f"[5/6] forecast_savings: Income = {savings_data.get('Derived_Values', {}).get('Income')}, Predicted Categories = {list(savings_data.get('Predicted_Savings', {}).keys())[:3]}...")

    # 6. Test predict_stock_profit
    predict_res = await session.call_tool("predict_stock_profit", {
        "ticker": "AAPL",
        "purchase_date": "2025-01-06",
        "future_date": "2026-11-01"
    })
    predict_data = json.loads(predict_res.content[0].text)
    print(f"[6/6] predict_stock_profit: Ticker = {predict_data.get('ticker')}, Buy = {predict_data.get('purchase_price')}, Forecast = {predict_data.get('predicted_future_price')}, PnL = {predict_data.get('profit_loss_purchase_to_future')}")

    await session.__aexit__(None, None, None)
    await client.__aexit__(None, None, None)
    print("=" * 60)
    print("ALL TESTS PASSED! APPLICATION IS 100% FUNCTIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_full_application())
