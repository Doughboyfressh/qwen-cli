@echo off
set PYTHONUTF8=1
set PYDANTIC_DISABLE_ANNOTATIONLIB=1

REM CLI input-token budget. Keep well under (server -c below) minus the preset
REM max_tokens output reservation (32768 for thinking/code) -- 65536 - 32768 =
REM 32768 ceiling, so 28000 leaves ~4.7k headroom for tokenizer-estimate drift
REM and tool schemas. This is the committed fallback for a fresh clone with no
REM config.toml (gitignored, personal); a config.toml token_limit still wins.
set QWEN_TOKEN_LIMIT=28000

REM CUDA 12 runtime DLLs
set PATH=%USERPROFILE%\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2;%PATH%

REM ---------------------------------------------------------------------------
REM Qwen full-stack launcher
REM ---------------------------------------------------------------------------

REM --- Stop old server ---
set "_LLM_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8080 " ^| findstr "LISTENING"') do set "_LLM_PID=%%P"
if defined _LLM_PID (
    echo [start-qwen] Stopping existing LLM server...
    taskkill /F /PID %_LLM_PID% >nul 2>&1
)

REM Wait for port free
set "_WAIT=0"
:wait_free
netstat -ano | findstr ":8080 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 goto port_free
set /a _WAIT+=1
if %_WAIT% geq 25 goto port_free
ping -n 2 127.0.0.1 >nul 2>&1
goto wait_free
:port_free

REM --- Auto-detect latest llama-server.exe ---
set "LLAMA_PATH="
for /f "delims=" %%D in ('dir /b /ad "%USERPROFILE%\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-*" 2^>nul ^| sort') do (
    if exist "%USERPROFILE%\.lmstudio\extensions\backends\%%D\llama-server.exe" set "LLAMA_PATH=%USERPROFILE%\.lmstudio\extensions\backends\%%D\llama-server.exe"
)

if not defined LLAMA_PATH (
    echo [ERROR] Cannot find llama-server.exe! Open LM Studio and update the CUDA backend.
    pause
    exit /b 1
)

REM --- Start LLM Server ---
start "Qwen LLM Server" "%LLAMA_PATH%" ^
  -m "%USERPROFILE%\.qwen-cli\models\Qwen3.6-27B-UD-Q6_K_XL.gguf" ^
  --mmproj "%USERPROFILE%\.qwen-cli\models\mmproj-F32.gguf" ^
  --alias "Qwen3.6-27B" ^
  --port 8080 ^
  --host 127.0.0.1 ^
  -ngl 64 ^
  -c 65536 ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  -np 1 ^
  -t 16 ^
  --flash-attn on ^
  --image-min-tokens 1024

echo [start-qwen] Loading model (30-90 seconds)...
set "_WAIT=0"
:wait_ready
curl.exe -sf http://127.0.0.1:8080/health -o NUL >nul 2>&1
if not errorlevel 1 goto server_ready
set /a _WAIT+=1
if %_WAIT% geq 150 goto ready_timeout
ping -n 3 127.0.0.1 >nul 2>&1
goto wait_ready
:ready_timeout
echo [WARNING] Server not ready after timeout.
goto after_ready
:server_ready
echo [start-qwen] LLM server is ready.
:after_ready

REM --- CLI ---
start "Qwen CLI" "%USERPROFILE%\.qwen-cli\.venv\Scripts\python.exe" "%USERPROFILE%\.qwen-cli\qwen-cli.py"
