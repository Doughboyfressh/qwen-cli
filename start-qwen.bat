@echo off
set PYTHONUTF8=1
set PYDANTIC_DISABLE_ANNOTATIONLIB=1

REM CLI input-token budget. Keep well under (server -c below) minus the preset
REM max_tokens output reservation (16384, see core/config.py) -- 49152 - 16384
REM = 32768 ceiling, so 28000 leaves ~4.7k headroom for tokenizer-estimate
REM drift and tool schemas. -c was traded from 65536 down to 49152 to afford
REM q8_0 KV cache (q4_0 K measurably hurts long-context quality).
REM
REM K and V cache types MUST MATCH with --flash-attn on: mixed q8_0/q4_0
REM falls off the FA CUDA fast path on this build (2.24.0, RTX 5090) --
REM prompt processing collapsed 2500 t/s -> 69 t/s, hanging every long
REM prompt (benchmarked live 2026-07-11). Matched q8_0/q8_0: 1900+ t/s.
REM Costs ~660 MiB more VRAM than q4_0/q4_0@65536 (~760 MiB headroom left
REM with aux up). To reclaim headroom at the cost of long-context quality:
REM q4_0/q4_0 and -c 65536, and raise preset max_tokens back to 32768.
REM
REM This is the committed fallback for a fresh clone with no config.toml
REM (gitignored, personal); a config.toml token_limit still wins.
set QWEN_TOKEN_LIMIT=28000

REM CUDA 12 runtime DLLs
set PATH=%USERPROFILE%\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2;%PATH%

REM ---------------------------------------------------------------------------
REM Qwen full-stack launcher
REM ---------------------------------------------------------------------------

REM --- Stop old servers (main :8080, aux :8081) ---
set "_LLM_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8080 " ^| findstr "LISTENING"') do set "_LLM_PID=%%P"
if defined _LLM_PID (
    echo [start-qwen] Stopping existing main LLM server...
    taskkill /F /PID %_LLM_PID% >nul 2>&1
)
set "_AUX_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8081 " ^| findstr "LISTENING"') do set "_AUX_PID=%%P"
if defined _AUX_PID (
    echo [start-qwen] Stopping existing aux LLM server...
    taskkill /F /PID %_AUX_PID% >nul 2>&1
)

REM Wait for both ports free (8081 too — the aux server can otherwise race
REM its own dying predecessor and fail to bind)
set "_WAIT=0"
:wait_free
netstat -ano | findstr /C:":8080 " /C:":8081 " | findstr "LISTENING" >nul 2>&1
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
  -c 49152 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  -np 1 ^
  -t 16 ^
  --flash-attn on ^
  --image-min-tokens 1024 ^
  --log-file "%USERPROFILE%\.qwen-cli\llama-main.log"

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

REM --- Aux LLM Server (fast MoE for background work + /model switch) ---
REM Started AFTER the main model has claimed its VRAM. All MoE expert layers
REM live in system RAM (--n-cpu-moe 999, mmap'd); only attention/shared layers
REM and the KV cache use the ~4 GB of VRAM the 27B leaves free. No mmproj:
REM vision stays on the main model. Thinking is disabled at the template level:
REM background calls use small max_tokens (15-250) and a thinking model burns
REM them all on reasoning, returning empty content (verified live).
REM If this server fails to start (e.g. OOM), the CLI runs fine without it -
REM aux is strictly optional.
set "AUX_MODEL_PATH=%USERPROFILE%\.lmstudio\models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
if not exist "%AUX_MODEL_PATH%" (
    echo [start-qwen] Aux model not found - skipping aux server.
    goto aux_done
)
echo [start-qwen] Starting aux server: Qwen3.6-35B-A3B on :8081...
start "Qwen Aux LLM Server" "%LLAMA_PATH%" ^
  -m "%AUX_MODEL_PATH%" ^
  --alias "Qwen3.6-35B-A3B" ^
  --port 8081 ^
  --host 127.0.0.1 ^
  -ngl 99 ^
  --n-cpu-moe 999 ^
  -c 40960 ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  -np 1 ^
  -t 16 ^
  --flash-attn on ^
  --chat-template-kwargs "{\"enable_thinking\":false}" ^
  --log-file "%USERPROFILE%\.qwen-cli\llama-aux.log"
set "_WAIT=0"
:wait_aux
curl.exe -sf http://127.0.0.1:8081/health -o NUL >nul 2>&1
if not errorlevel 1 goto aux_ready
set /a _WAIT+=1
if %_WAIT% geq 60 goto aux_timeout
ping -n 3 127.0.0.1 >nul 2>&1
goto wait_aux
:aux_timeout
echo [start-qwen] [warning] Aux server not ready yet - CLI will run without it.
goto aux_done
:aux_ready
echo [start-qwen] Aux server is ready.
:aux_done

REM --- CLI ---
start "Qwen CLI" "%USERPROFILE%\.qwen-cli\.venv\Scripts\python.exe" "%USERPROFILE%\.qwen-cli\qwen-cli.py"
