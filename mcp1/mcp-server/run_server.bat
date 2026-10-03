@echo off
title Financial Datasets MCP Server
echo ========================================================
echo Starting Financial Datasets MCP Server (stdio transport)...
echo ========================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" server.py
) else (
    python server.py
)
pause
