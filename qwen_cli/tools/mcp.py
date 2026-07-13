"""Minimal MCP (Model Context Protocol) client — stdio transport only.

Servers are configured in config.toml and exposed to the model as
mcp_<server>_<tool> in the on-demand "mcp" tool group (enable_tools('mcp')):

    [mcp.servers.github]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-github"]
    # optional:
    # env = { GITHUB_PERSONAL_ACCESS_TOKEN = "..." }   # merged over os.environ
    # cwd = "C:/somewhere"

Implements the client side of initialize / tools/list / tools/call over
newline-delimited JSON-RPC 2.0 (the MCP stdio framing) with no external
dependency. Server processes start lazily on enable_tools('mcp') and are
terminated atexit.
"""

import atexit
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading

_logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2025-03-26"
_INIT_TIMEOUT = 30.0  # seconds for initialize / tools/list (npx may download)
_CALL_TIMEOUT = 120.0  # seconds for a tools/call
_MAX_RESULT_CHARS = 24_000  # raw cap; _smart_cap shapes it further upstream
_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


class MCPServer:
    """One configured MCP server: child process + JSON-RPC over its stdio."""

    def __init__(self, name: str, command: str, args: list[str], env: dict | None = None, cwd: str | None = None):
        self.name = name
        self.command = command
        self.args = [str(a) for a in (args or [])]
        self.env = {str(k): str(v) for k, v in (env or {}).items()}
        self.cwd = cwd
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict] = []  # raw defs from tools/list
        self.error = ""
        self._id = 0
        self._lock = threading.Lock()  # one in-flight request per server
        self._lines: queue.Queue = queue.Queue()

    # -- transport ----------------------------------------------------------

    def start(self) -> bool:
        """Spawn the process, run the MCP handshake, list tools. False + .error on failure."""
        try:
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            self.proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.cwd or None,
                env={**os.environ, **self.env},
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **kwargs,
            )
        except Exception as e:
            self.error = f"failed to launch: {e}"
            return False
        # Blocking readline can't time out on Windows pipes — a reader thread
        # feeds a queue and requests pop from it with a deadline instead.
        threading.Thread(target=self._reader, daemon=True, name=f"mcp-{self.name}-reader").start()
        try:
            self._request(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "qwen-cli", "version": "1.0"},
                },
                timeout=_INIT_TIMEOUT,
            )
            self._notify("notifications/initialized")
            listed = self._request("tools/list", {}, timeout=_INIT_TIMEOUT)
            self.tools = [t for t in listed.get("tools", []) if t.get("name")]
            return True
        except Exception as e:
            self.error = str(e)
            self.stop()
            return False

    def _reader(self) -> None:
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self._lines.put(line)
        except Exception:
            # A broken pipe here is why a later tool call times out — leave a trace.
            _logger.debug("MCP stdout reader stopped", exc_info=True)
        self._lines.put(None)  # EOF sentinel

    def _send(self, msg: dict) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise ConnectionError("server process is not running")
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str, params: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        self._send(msg)

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        """Send a request and wait for its response, servicing anything else
        the server sends in between (notifications ignored, ping answered)."""
        with self._lock:
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            import time as _time

            deadline = _time.monotonic() + timeout
            while True:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{method}: no response within {timeout:.0f}s")
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError(f"{method}: no response within {timeout:.0f}s") from None
                if line is None:
                    raise ConnectionError(f"{method}: server exited")
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue  # startup banners / stray output
                if msg.get("id") == rid and ("result" in msg or "error" in msg):
                    if "error" in msg:
                        err = msg["error"]
                        raise RuntimeError(f"{method}: {err.get('message', err)}")
                    return msg.get("result") or {}
                # Server-initiated request: answer ping, decline the rest.
                if msg.get("method") and msg.get("id") is not None:
                    if msg["method"] == "ping":
                        self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
                    else:
                        self._send(
                            {
                                "jsonrpc": "2.0",
                                "id": msg["id"],
                                "error": {"code": -32601, "message": "not supported by qwen-cli"},
                            }
                        )
                # Anything else (notifications, stale responses): ignore.

    # -- tools ---------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = self._request(
            "tools/call", {"name": tool_name, "arguments": arguments or {}}, timeout=_CALL_TIMEOUT
        )
        parts: list[str] = []
        for item in result.get("content", []):
            kind = item.get("type")
            if kind == "text":
                parts.append(item.get("text", ""))
            elif kind == "image":
                parts.append(f"[image content ({item.get('mimeType', '?')}) — not displayable in this CLI]")
            elif kind == "resource":
                res = item.get("resource", {})
                parts.append(res.get("text") or f"[resource {res.get('uri', '?')}]")
        text = "\n".join(p for p in parts if p) or json.dumps(result)[:2000]
        if len(text) > _MAX_RESULT_CHARS:
            text = text[:_MAX_RESULT_CHARS] + f"\n... [mcp result truncated at {_MAX_RESULT_CHARS:,} chars]"
        if result.get("isError"):
            return f"[mcp tool error: {text}]"
        return text

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                _logger.debug("MCP server kill failed during shutdown", exc_info=True)
        self.proc = None


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_servers: dict[str, MCPServer] = {}
_registry: dict[str, tuple[str, str]] = {}  # exposed name -> (server, tool)
_started = False
_start_lock = threading.Lock()


def start_all(configs: dict) -> str:
    """Start every configured server (once per process). Returns a summary line."""
    global _started
    with _start_lock:
        if not _started:
            _started = True
            for name, cfg in (configs or {}).items():
                if not isinstance(cfg, dict) or not cfg.get("command"):
                    continue
                srv = MCPServer(
                    name,
                    str(cfg["command"]),
                    cfg.get("args") or [],
                    cfg.get("env") or {},
                    cfg.get("cwd"),
                )
                _servers[name] = srv
                if srv.start():
                    _register(srv)
                    _logger.info("mcp server '%s' up: %d tool(s)", name, len(srv.tools))
                else:
                    _logger.warning("mcp server '%s' failed: %s", name, srv.error)
    up = [f"{s.name} ({len(s.tools)} tools)" for s in _servers.values() if s.tools]
    down = [f"{s.name}: {s.error}" for s in _servers.values() if not s.tools]
    parts = []
    if up:
        parts.append("connected: " + ", ".join(up))
    if down:
        parts.append("failed: " + "; ".join(down))
    return " | ".join(parts) or "no MCP servers configured"


def _register(srv: MCPServer) -> None:
    for t in srv.tools:
        exposed = _NAME_RE.sub("-", f"mcp_{srv.name}_{t['name']}")[:64]
        n, base = 2, exposed
        while exposed in _registry:  # collision (e.g. sanitized twins)
            exposed = f"{base[:61]}-{n}"
            n += 1
        _registry[exposed] = (srv.name, t["name"])


def tool_schemas() -> list:
    """OpenAI-style schemas for every connected server's tools."""
    out = []
    by_pair = {(s, t): exposed for exposed, (s, t) in _registry.items()}
    for srv in _servers.values():
        for t in srv.tools:
            exposed = by_pair.get((srv.name, t["name"]))
            if not exposed:
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": f"[MCP:{srv.name}] {(t.get('description') or t['name'])[:350]}",
                        "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                    },
                }
            )
    return out


def dispatch(name: str, args: dict) -> str:
    """Route an exposed mcp_* tool call to its server."""
    pair = _registry.get(name)
    if pair is None:
        return f"[unknown mcp tool: {name} — enable_tools('mcp') lists what's available]"
    server, tool = pair
    srv = _servers.get(server)
    if srv is None or srv.proc is None:
        return f"[mcp server '{server}' is not running]"
    try:
        return srv.call_tool(tool, args)
    except Exception as e:
        return f"[mcp error calling {server}/{tool}: {e}]"


def stop_all() -> None:
    for srv in _servers.values():
        srv.stop()


atexit.register(stop_all)
