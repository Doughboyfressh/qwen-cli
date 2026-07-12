"""Tests for the minimal MCP stdio client (qwen_cli/tools/mcp.py) and its
integration with tool-group gating. Uses a real child process running a fake
MCP server that speaks newline-delimited JSON-RPC."""

import sys
import textwrap

import pytest

FAKE_SERVER = textwrap.dedent(
    """
    import json, sys

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    print("startup banner that is not JSON")  # clients must tolerate this
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        m = msg.get("method")
        if m == "initialize":
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "0"}}})
            # exercise the server-initiated request path
            send({"jsonrpc": "2.0", "id": 999, "method": "ping"})
        elif m == "notifications/initialized":
            continue
        elif m == "tools/list":
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
                {"name": "echo", "description": "Echo text back",
                 "inputSchema": {"type": "object",
                                 "properties": {"text": {"type": "string"}},
                                 "required": ["text"]}},
                {"name": "fail", "description": "Always errors",
                 "inputSchema": {"type": "object", "properties": {}}}]}})
        elif m == "tools/call":
            name = msg["params"]["name"]
            if name == "echo":
                send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                    "content": [{"type": "text",
                                 "text": "echo: " + msg["params"]["arguments"]["text"]}]}})
            else:
                send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                    "content": [{"type": "text", "text": "boom"}], "isError": True}})
        else:
            send({"jsonrpc": "2.0", "id": msg.get("id"),
                  "error": {"code": -32601, "message": "unknown method"}})
    """
)


@pytest.fixture
def fake_server_path(tmp_path):
    p = tmp_path / "fake_mcp_server.py"
    p.write_text(FAKE_SERVER, encoding="utf-8")
    return str(p)


@pytest.fixture
def mcp_mod():
    from qwen_cli.tools import mcp

    # isolate module state per test
    saved = (dict(mcp._servers), dict(mcp._registry), mcp._started)
    mcp._servers.clear()
    mcp._registry.clear()
    mcp._started = False
    yield mcp
    mcp.stop_all()
    mcp._servers.clear()
    mcp._servers.update(saved[0])
    mcp._registry.clear()
    mcp._registry.update(saved[1])
    mcp._started = saved[2]


class TestMCPServer:
    def test_handshake_list_and_call(self, mcp_mod, fake_server_path):
        srv = mcp_mod.MCPServer("fake", sys.executable, [fake_server_path])
        try:
            assert srv.start(), srv.error
            assert [t["name"] for t in srv.tools] == ["echo", "fail"]
            assert srv.call_tool("echo", {"text": "hi"}) == "echo: hi"
            assert srv.call_tool("fail", {}) == "[mcp tool error: boom]"
        finally:
            srv.stop()

    def test_launch_failure_is_reported(self, mcp_mod):
        srv = mcp_mod.MCPServer("ghost", "definitely-not-a-real-binary-xyz", [])
        assert srv.start() is False
        assert srv.error

    def test_start_all_registers_prefixed_tools(self, mcp_mod, fake_server_path):
        summary = mcp_mod.start_all({"fake": {"command": sys.executable, "args": [fake_server_path]}})
        assert "fake (2 tools)" in summary
        names = [t["function"]["name"] for t in mcp_mod.tool_schemas()]
        assert names == ["mcp_fake_echo", "mcp_fake_fail"]
        schema = mcp_mod.tool_schemas()[0]["function"]
        assert schema["description"].startswith("[MCP:fake]")
        assert schema["parameters"]["properties"]["text"]["type"] == "string"
        assert mcp_mod.dispatch("mcp_fake_echo", {"text": "yo"}) == "echo: yo"
        assert "unknown mcp tool" in mcp_mod.dispatch("mcp_fake_nope", {})


class TestMCPGatingIntegration:
    def _fresh(self, qwen_cli, monkeypatch, configs):
        monkeypatch.setattr(qwen_cli, "TOOL_GROUPS_MODE", "auto")
        monkeypatch.setattr(qwen_cli, "_enabled_tool_groups", set())
        monkeypatch.setattr(qwen_cli.console, "print", lambda *a, **k: None)
        monkeypatch.setitem(qwen_cli._CFG, "mcp", {"servers": configs})

    def test_enable_mcp_without_config_errors(self, qwen_cli, monkeypatch):
        self._fresh(qwen_cli, monkeypatch, {})
        assert "no MCP servers configured" in qwen_cli.do_enable_tools("mcp")
        assert "mcp" not in qwen_cli._enabled_tool_groups

    def test_enable_mcp_connects_and_exposes_tools(self, qwen_cli, monkeypatch, mcp_mod, fake_server_path):
        self._fresh(qwen_cli, monkeypatch, {"fake": {"command": sys.executable, "args": [fake_server_path]}})
        result = qwen_cli.do_enable_tools("mcp")
        assert "mcp_fake_echo" in result
        names = {t["function"]["name"] for t in qwen_cli.active_tools()}
        assert "mcp_fake_echo" in names and "browser_action" not in names
        # dynamic tools route through _dispatch_interactive's mcp_ fallback
        assert qwen_cli._dispatch_interactive("mcp_fake_echo", {"text": "hi"}) == "echo: hi"

    def test_enable_mcp_when_server_fails(self, qwen_cli, monkeypatch, mcp_mod):
        self._fresh(qwen_cli, monkeypatch, {"ghost": {"command": "definitely-not-a-real-binary-xyz"}})
        result = qwen_cli.do_enable_tools("mcp")
        assert "no MCP server came up" in result
        assert "mcp" not in qwen_cli._enabled_tool_groups
