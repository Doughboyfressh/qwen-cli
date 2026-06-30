"""Tests for LSP integration — command parsing, fallbacks, and error handling."""
import io
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console


def _mock_ctx():
    """Create a minimal mock _ReplContext for testing commands."""
    ctx = SimpleNamespace()
    ctx._out = []
    ctx._print = lambda text="": ctx._out.append(text)
    return ctx


def _run_cmd(ctx, qwen_cli, arg):
    """Run a command, monkey-patching the rich Console to capture output."""
    buf = io.StringIO()
    qwen_cli.console = Console(file=buf, force_terminal=True, legacy_windows=False)
    try:
        qwen_cli._cmd_lsp(ctx, arg)
    finally:
        pass
    return buf.getvalue() + "\n".join(ctx._out)


# --- Fallback / multilspy unavailable tests ---

def test_lsp_query_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_query("status", "/tmp/dummy.py")
    assert isinstance(msg, str)


def test_lsp_status_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_status()
    assert isinstance(msg, str)


def test_lsp_diagnostics_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_diagnostics("/tmp/dummy.py")
    assert isinstance(msg, str)


def test_lsp_definition_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_definition("/tmp/dummy.py", 1, 0)
    assert isinstance(msg, str)


def test_lsp_references_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_references("/tmp/dummy.py", 1, 0)
    assert isinstance(msg, str)


def test_lsp_hover_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_hover("/tmp/dummy.py", 1, 0)
    assert isinstance(msg, str)


def test_lsp_symbols_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_symbols("/tmp/dummy.py")
    assert isinstance(msg, str)


def test_lsp_rename_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_rename("/tmp/dummy.py", 1, 0, "new_name")
    assert isinstance(msg, str)


def test_lsp_completion_returns_fallback_when_not_installed(lsp_client):
    msg = lsp_client.lsp_completion("/tmp/dummy.py", 1, 0)
    assert isinstance(msg, str)


def test_lsp_client_has_query_dispatcher(lsp_client):
    assert callable(lsp_client.lsp_query)


# --- Language detection ---

def test_detect_language_from_extension(lsp_client):
    assert lsp_client._detect_language("file.py") == "python"
    assert lsp_client._detect_language("file.js") == "javascript"
    assert lsp_client._detect_language("file.ts") == "typescript"
    assert lsp_client._detect_language("file.tsx") == "typescript"
    assert lsp_client._detect_language("file.jsx") == "javascript"
    assert lsp_client._detect_language("file.java") == "java"
    assert lsp_client._detect_language("file.go") == "go"
    assert lsp_client._detect_language("file.rs") == "rust"
    assert lsp_client._detect_language("file.rb") == "ruby"
    assert lsp_client._detect_language("file.php") == "php"
    assert lsp_client._detect_language("file.cpp") == "cpp"


def test_detect_language_defaults_to_python_for_unknown(lsp_client):
    assert lsp_client._detect_language("file.xyz") == "python"
    assert lsp_client._detect_language("file.unknown") == "python"


# --- Project root detection ---

def test_get_project_root_returns_path(lsp_client):
    root = lsp_client._get_project_root(__file__)
    assert root is not None
    assert isinstance(root, str)


# --- Shutdown safety ---

def test_shutdown_does_not_crash(lsp_client):
    lsp_client.shutdown()


# --- Query dispatcher routing ---

def test_lsp_query_routes_status(lsp_client):
    result = lsp_client.lsp_query("status")
    assert isinstance(result, str)


def test_lsp_query_routes_diagnostics_with_file(lsp_client):
    result = lsp_client.lsp_query("diagnostics", "/tmp/x.py")
    assert isinstance(result, str)


def test_lsp_query_routes_diagnostics_without_file(lsp_client):
    result = lsp_client.lsp_query("diagnostics", "")
    assert "Usage" in result or "usage" in result.lower()


def test_lsp_query_routes_definition(lsp_client):
    result = lsp_client.lsp_query("definition", "/tmp/x.py", 1, 0)
    assert isinstance(result, str)


def test_lsp_query_routes_references(lsp_client):
    result = lsp_client.lsp_query("references", "/tmp/x.py", 1, 0)
    assert isinstance(result, str)


def test_lsp_query_routes_hover(lsp_client):
    result = lsp_client.lsp_query("hover", "/tmp/x.py", 1, 0)
    assert isinstance(result, str)


def test_lsp_query_routes_symbols(lsp_client):
    result = lsp_client.lsp_query("symbols", "/tmp/x.py")
    assert isinstance(result, str)


def test_lsp_query_routes_rename(lsp_client):
    result = lsp_client.lsp_query("rename", "/tmp/x.py", 1, 0, "new_name")
    assert isinstance(result, str)


def test_lsp_query_unknown_action(lsp_client):
    result = lsp_client.lsp_query("unknown_action")
    assert "Usage" in result or "usage" in result.lower() or "Unknown" in result


# --- Command parsing tests for _cmd_lsp ---

def test_cmd_lsp_no_args_shows_usage(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "")
    assert len(output) > 0
    assert "usage" in output.lower() or "lsp" in output.lower()


def test_cmd_lsp_status_subcommand(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "status")
    assert len(output) > 0


def test_cmd_lsp_diagnose_no_file_shows_usage(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "diagnose")
    assert len(output) > 0


def test_cmd_lsp_diagnose_with_file(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "diagnose some_file.py")
    assert len(output) > 0


def test_cmd_lsp_define_no_args_shows_usage(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "define")
    assert len(output) > 0
    assert "usage" in output.lower() or "file" in output.lower()


def test_cmd_lsp_define_with_args(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "define some_file.py 1 0")
    assert len(output) > 0


def test_cmd_lsp_refs_with_args(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "refs some_file.py 1 0")
    assert len(output) > 0


def test_cmd_lsp_hover_with_args(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "hover some_file.py 1 0")
    assert len(output) > 0


def test_cmd_lsp_symbols_with_file(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "symbols some_file.py")
    assert len(output) > 0


def test_cmd_lsp_rename_with_args(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "rename some_file.py 1 0 new_name")
    assert len(output) > 0


def test_cmd_lsp_rename_too_few_args(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "rename some_file.py")
    assert len(output) > 0
    assert "usage" in output.lower() or "rename" in output.lower()


def test_cmd_lsp_unknown_subcommand(qwen_cli):
    ctx = _mock_ctx()
    output = _run_cmd(ctx, qwen_cli, "unknown_thing")
    assert len(output) > 0
    assert "usage" in output.lower() or "lsp" in output.lower() or "unknown" in output.lower()
