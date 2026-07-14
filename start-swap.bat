@echo off
setlocal enabledelayedexpansion
REM ===========================================================================
REM start-swap.bat — run the CLI behind llama-swap, so /model actually switches
REM
REM Plain llama-server loads one model and ignores the OpenAI "model" field, so
REM /model can only relabel. llama-swap is an OpenAI-compatible proxy that reads
REM that field, starts the right llama-server and stops the previous one.
REM
REM   start-swap.bat            -> proxy on :8090, CLI on the "swap" provider
REM
REM Models are defined in llama-swap.yaml, NOT here.
REM start-qwen.bat is untouched and still works if you want the old two-server
REM setup back.
REM ===========================================================================

set "SWAP_EXE=%USERPROFILE%\.qwen-cli\bin\llama-swap.exe"
set "SWAP_CFG=%USERPROFILE%\.qwen-cli\llama-swap.yaml"

if not exist "%SWAP_EXE%" (
    echo [ERROR] llama-swap is not installed.
    echo.
    echo   It is a third-party open-source proxy ^(github.com/mostlygeek/llama-swap^).
    echo   Install it yourself so you can see exactly what you are running:
    echo.
    echo     curl -L -o "%%TEMP%%\llama-swap.zip" https://github.com/mostlygeek/llama-swap/releases/download/v239/llama-swap_239_windows_amd64.zip
    echo     tar -xf "%%TEMP%%\llama-swap.zip" -C "%USERPROFILE%\.qwen-cli\bin"
    echo.
    pause
    exit /b 1
)

if not exist "%SWAP_CFG%" (
    echo [ERROR] Missing %SWAP_CFG%
    pause
    exit /b 1
)

REM --- Verify the llama-server path in the YAML still exists ------------------
REM LM Studio bumps its backend version directory on update, which silently
REM breaks every model in the config. Catch it here with a clear message rather
REM than letting each model fail to start one at a time.
for /f "delims=" %%D in ('dir /b /ad "%USERPROFILE%\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-*" 2^>nul ^| sort') do (
    if exist "%USERPROFILE%\.lmstudio\extensions\backends\%%D\llama-server.exe" set "LLAMA_PATH=%USERPROFILE%\.lmstudio\extensions\backends\%%D\llama-server.exe"
)
if defined LLAMA_PATH (
    findstr /C:"%LLAMA_PATH:\=/%" "%SWAP_CFG%" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] llama-swap.yaml points at a different llama-server than the newest one found:
        echo        newest: %LLAMA_PATH%
        echo        Update the 'llama:' macro in llama-swap.yaml if models fail to start.
        echo.
    )
)

REM --- Free the ports (proxy + the two upstreams it manages) ------------------
for %%P in (8090 8091 8092) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr ":%%P .*LISTENING"') do (
        echo [start-swap] Killing stale process on port %%P ^(PID %%A^)
        taskkill /F /PID %%A >nul 2>&1
    )
)

echo [start-swap] Proxy on http://localhost:8090  (models load on first use)
start "llama-swap" "%SWAP_EXE%" --config "%SWAP_CFG%" --listen 127.0.0.1:8090

REM --- Wait for the proxy itself (not a model — those load on demand) ---------
set "_WAIT=0"
:wait_proxy
timeout /t 1 /nobreak >nul
curl -s -o nul http://localhost:8090/v1/models 2>nul && goto proxy_ready
set /a _WAIT+=1
if !_WAIT! GEQ 20 (
    echo [ERROR] llama-swap did not come up on :8090
    pause
    exit /b 1
)
goto wait_proxy

:proxy_ready
echo [start-swap] Ready. /model switches models; the first request to a model loads it.
echo.

REM QWEN_PROVIDER selects the [providers.swap] profile in config.toml — same
REM mechanism as `provider = "swap"` there, but scoped to this launcher so
REM start-qwen.bat keeps using the direct llama-server setup.
set "QWEN_PROVIDER=swap"

cd /d "%USERPROFILE%\.qwen-cli"
".venv\Scripts\python.exe" qwen-cli.py %*
