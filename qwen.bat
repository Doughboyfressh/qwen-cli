@echo off
set PYTHONUTF8=1
set PYDANTIC_DISABLE_ANNOTATIONLIB=1

REM ---------------------------------------------------------------------------
REM Qwen quick launcher — starts CLI (and web UI if not already running).
REM The LLM server (port 8080) must be running; use start-qwen.bat to launch
REM the full stack from scratch.
REM ---------------------------------------------------------------------------

REM --- Warn if LLM server is not reachable ---
netstat -ano | findstr ":8080 " >nul 2>&1
if errorlevel 1 (
    echo [qwen] WARNING: LLM server does not appear to be running on port 8080.
    echo [qwen]          Run start-qwen.bat to launch the full stack, or start
    echo [qwen]          llama-server manually before using the CLI.
    echo.
)

REM --- Web UI (skip if already running on 7860) ---
netstat -ano | findstr ":7860 " >nul 2>&1
if errorlevel 1 (
    start "Qwen Web" /min "%USERPROFILE%\.qwen-cli\.venv\Scripts\python.exe" "%USERPROFILE%\.qwen-cli\qwen-web.py"
)

REM --- CLI in foreground (pass through any args) ---
"%USERPROFILE%\.qwen-cli\.venv\Scripts\python.exe" "%USERPROFILE%\.qwen-cli\qwen-cli.py" %*
