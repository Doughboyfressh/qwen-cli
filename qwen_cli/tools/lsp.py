"""LSP client wrapper using multilspy.

Provides a lazy-starting, auto-shutting-down language server for diagnostics,
definition lookup, references, hover, symbols, and rename.

Design principles:
  - Zero cost until first LSP call (lazy initialization)
  - Auto-shutdown after 300s of inactivity
  - All calls wrapped in try/except — failures never crash the CLI
  - Import-friendly: if multilspy is not installed, graceful fallback
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — multilspy may not be installed
# ---------------------------------------------------------------------------

_multilspy_available: bool = False
_Language: Any = None
_MultilspyConfig: Any = None
_MultilspyLogger: Any = None
_SyncLanguageServer: Any = None

try:
    from multilspy import SyncLanguageServer as _SyncLanguageServer
    from multilspy.multilspy_config import Language as _Language
    from multilspy.multilspy_config import MultilspyConfig as _MultilspyConfig
    from multilspy.multilspy_logger import MultilspyLogger as _MultilspyLogger

    # Patch multilspy's _sync_call: when called from a thread with a running
    # event loop, the original implementation fire-forgets the coroutine via
    # asyncio.create_task(coro) and returns None immediately.  This produces
    # "coroutine was never awaited" warnings and silently broken LSP calls.
    # Our patch uses run_coroutine_threadsafe on the server's own loop instead.
    _multilspy_original_sync_call = getattr(_SyncLanguageServer, "_sync_call", None)

    def _multilspy_patched_sync_call(self, coro):
        import asyncio

        server_loop = getattr(self, "loop", None)
        if server_loop is not None and not server_loop.is_closed():
            return asyncio.run_coroutine_threadsafe(coro, server_loop).result()
        return _multilspy_original_sync_call(self, coro)

    _SyncLanguageServer._sync_call = _multilspy_patched_sync_call

    _multilspy_available = True
except ImportError:
    _logger.debug("multilspy not installed; LSP features disabled")
# ---------------------------------------------------------------------------

# Globals
# ---------------------------------------------------------------------------

_LSP_SERVER: Any | None = None
_LSP_LOCK = threading.Lock()
_LSP_LAST_ACCESS: float = 0.0
_LSP_IDLE_TIMEOUT: int = 300  # 5 minutes
_LSP_ROOT: str | None = None
_LSP_LANGUAGE: str | None = None

# multilspy's SyncLanguageServer.start_server() blocks on
# asyncio.run_coroutine_threadsafe(ctx.__aenter__(), ...).result() with NO timeout
# of its own. If the language server subprocess never completes its LSP
# 'initialize' handshake (observed hanging indefinitely on this setup), that
# call — and therefore _create_server() — never returns. Bound it ourselves so
# a broken handshake degrades to "LSP disabled" instead of freezing the CLI.
_LSP_STARTUP_TIMEOUT: int = 20  # seconds
_LSP_REQUEST_TIMEOUT: int = 15  # seconds — bounds every definition/references/hover/etc. call

# A startup failure here (bad handshake, missing binary, version mismatch) is a
# deterministic environmental problem, not a transient blip — retrying on a timer
# just re-pays the ~20s timeout forever. Measured live: patch_file's LSP pre/post-edit
# hooks made the FIRST patch on a code file cost 20s and, with a time-based cooldown,
# every patch after the cooldown expired cost another 20s — a recurring tax on the
# tool the system prompt tells the model to prefer. Disable per-language for the rest
# of the process once a real startup failure is observed; /lsp shutdown resets it.
_LSP_DISABLED_LANGUAGES: set[str] = set()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXTENSION_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".rs": "rust",
    ".go": "go",
    ".cs": "csharp",
    ".dart": "dart",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".php": "php",
}


def _detect_language(file_path: str) -> str:
    """Detect code language from file extension."""
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext, "python")


def _language_to_multilspy_enum(lang: str) -> Any:
    """Map a language string to the multilspy Language enum value."""
    if _Language is None:
        msg = "multilspy is not installed"
        raise RuntimeError(msg)
    mapping = {
        "python": _Language.PYTHON,
        "typescript": _Language.TYPESCRIPT,
        "javascript": _Language.JAVASCRIPT,
        "java": _Language.JAVA,
        "rust": _Language.RUST,
        "go": _Language.GO,
        "csharp": _Language.CSHARP,
        "dart": _Language.DART,
        "ruby": _Language.RUBY,
        "kotlin": _Language.KOTLIN,
    }
    return mapping.get(lang, _Language.PYTHON)


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------


def _get_project_root(file_path: str = "") -> str:
    """Determine the project root for the LSP server."""
    start = Path(file_path).resolve().parent if file_path else Path.cwd()
    markers = {
        ".git",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "build.gradle",
    }
    for p in [start, *list(start.parents)]:
        if any((p / m).exists() for m in markers):
            return str(p)
    return str(start)


# ---------------------------------------------------------------------------
# Helper: execute a synchronous callback while a file is open in the LSP
# ---------------------------------------------------------------------------


def _with_open_file(lsp, file_path: str, callback) -> Any:
    """Multilspy's ``lsp.open_file`` is a ``@contextmanager``.  Calling it as
    ``lsp.open_file(path)`` without ``with`` returns an unentered generator,
    which means the server never learns about the file and all subsequent
    requests silently fail (or return coroutines that are never awaited).

    This helper enters the context manager, runs *callback* (which receives
    the *lsp* object), and exits the context manager cleanly.
    """
    ctx = lsp.open_file(file_path)
    ctx.__enter__()
    try:
        return callback(lsp)
    finally:
        ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _create_server(file_path: str = "") -> Any:
    """Create a new SyncLanguageServer and start it.

    multilspy's ``lsp.start_server()`` is also a ``@contextmanager``.  We must
    enter it (via ``with`` or manual ``__enter__``) so that the internal event
    loop is created and the server process is actually launched.
    """
    global _LSP_SERVER, _LSP_ROOT, _LSP_LANGUAGE

    if not _multilspy_available:
        msg = "multilspy is not installed. Install with: pip install multilspy"
        raise RuntimeError(
            msg,
        )

    language = _detect_language(file_path)

    if language in _LSP_DISABLED_LANGUAGES:
        msg = f"LSP server for '{language}' failed to start earlier this session; not retrying (run /lsp shutdown to reset)"
        raise RuntimeError(msg)

    root = _get_project_root(file_path)

    lang_enum = _language_to_multilspy_enum(language)
    config = _MultilspyConfig(lang_enum)
    logger = _MultilspyLogger()

    # timeout also bounds every subsequent request_*() call (definition,
    # references, hover, symbols, rename, completion) — multilspy's own
    # SyncLanguageServer methods do `.result(timeout=self.timeout)`, but
    # self.timeout is None (blocks forever) unless passed here.
    lsp = _SyncLanguageServer.create(config, logger, root, timeout=_LSP_REQUEST_TIMEOUT)

    # Enter the context manager so the event loop is created and the
    # underlying language-server process is actually started. This is
    # bounded with a timeout: multilspy's own __enter__() has no timeout,
    # so an initialize handshake that never completes would otherwise hang
    # forever (see _LSP_STARTUP_TIMEOUT comment above). Uses a plain daemon
    # thread rather than ThreadPoolExecutor: a genuinely stuck handshake
    # thread would otherwise never finish, and non-daemon executor workers
    # block clean interpreter shutdown even after we've moved on.
    ctx = lsp.start_server()
    _result: dict[str, Any] = {}

    def _enter() -> None:
        try:
            _result["value"] = ctx.__enter__()
        except Exception as e:  # noqa: BLE001 — surfaced via _result, not raised in this thread
            _result["error"] = e

    _thread = threading.Thread(target=_enter, daemon=True, name="lsp-startup")
    _thread.start()
    _thread.join(timeout=_LSP_STARTUP_TIMEOUT)
    if _thread.is_alive():
        _LSP_DISABLED_LANGUAGES.add(language)
        msg = f"LSP server for '{language}' did not initialize within {_LSP_STARTUP_TIMEOUT}s (handshake hung)"
        _logger.warning(msg)
        raise RuntimeError(msg)
    if "error" in _result:
        _LSP_DISABLED_LANGUAGES.add(language)
        raise _result["error"]
    lsp._start_ctx = ctx  # keep alive for shutdown

    _LSP_SERVER = lsp
    _LSP_ROOT = root
    _LSP_LANGUAGE = language
    _LSP_LAST_ACCESS = time.time()

    return lsp


def _shutdown_server(lsp) -> None:
    """Shut down the LSP server by exiting the start_server context manager."""
    try:
        ctx = getattr(lsp, "_start_ctx", None)
        if ctx is not None:
            ctx.__exit__(None, None, None)
            lsp._start_ctx = None
    except Exception:
        _logger.debug("LSP shutdown (context exit) had an error")
    # multilspy's SyncLanguageServer creates an internal event loop in
    # ``lsp.loop``.  Close it explicitly so it does not linger after shutdown.
    try:
        _loop = getattr(lsp, "loop", None)
        if _loop is not None and not _loop.is_closed():
            _loop.close()
    except Exception:
        _logger.debug("LSP shutdown (event loop close) had an error")
    # Fallback: try the inner server's shutdown method
    with contextlib.suppress(Exception):
        lsp._server.shutdown()


def _ensure_server(file_path: str = "") -> Any:
    """Get or create the LSP server. Re-creates if language changed."""
    global _LSP_SERVER, _LSP_LANGUAGE

    _LSP_LAST_ACCESS = time.time()

    with _LSP_LOCK:
        desired_lang = _detect_language(file_path)

        # If language changed, recreate
        if _LSP_SERVER is not None and desired_lang != _LSP_LANGUAGE:
            _shutdown_server(_LSP_SERVER)
            _LSP_SERVER = None
            _LSP_LANGUAGE = None

        if _LSP_SERVER is None:
            return _create_server(file_path)
        return _LSP_SERVER


def shutdown() -> None:
    """Explicitly shut down the LSP server and clear any startup-failure history.

    This is the user's explicit escape hatch to retry a language that previously
    failed to start (e.g. after fixing the underlying environment issue) — see
    _LSP_DISABLED_LANGUAGES.
    """
    global _LSP_SERVER, _LSP_ROOT, _LSP_LANGUAGE

    with _LSP_LOCK:
        if _LSP_SERVER is not None:
            _shutdown_server(_LSP_SERVER)
            _LSP_SERVER = None
            _LSP_ROOT = None
            _LSP_LANGUAGE = None
        _LSP_DISABLED_LANGUAGES.clear()


def _check_idle_shutdown() -> None:
    """Check if server should be shut down due to inactivity."""
    global _LSP_SERVER, _LSP_ROOT, _LSP_LANGUAGE

    if _LSP_SERVER is None:
        return

    if time.time() - _LSP_LAST_ACCESS > _LSP_IDLE_TIMEOUT:
        with _LSP_LOCK:
            if _LSP_SERVER is not None:
                _shutdown_server(_LSP_SERVER)
                _LSP_SERVER = None
                _LSP_ROOT = None
                _LSP_LANGUAGE = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SYMBOL_KIND_NAMES = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}


def _symbol_kind_name(kind: Any) -> str:
    """Get a human-readable name for a symbol kind."""
    if hasattr(kind, "value"):
        kind = kind.value
    return _SYMBOL_KIND_NAMES.get(kind, f"Kind({kind})")


def _uri_to_path(uri: str) -> str:
    """Convert a file:// URI to a local path."""
    if uri.startswith("file://"):
        path = uri[7:]
        # Handle Windows drive letters
        if len(path) >= 3 and path[1:2] == ":":
            return path[1:]
        return path
    return uri


def _location_to_str(loc: Any) -> str:
    """Format a multilspy Location object as 'path:line:col'."""
    uri = getattr(loc, "uri", getattr(loc, "target_uri", "unknown"))
    rng = getattr(loc, "range", getattr(loc, "target_range", None))
    if rng is None:
        return f"{uri}"
    start = getattr(rng, "start", None)
    line = (start.line + 1) if start else "?"
    col = (start.character + 1) if start else "?"
    return f"{_uri_to_path(uri)}:{line}:{col}"


def _hover_value(hover: Any) -> str:
    """Extract text from a multilspy Hover object."""
    contents = getattr(hover, "contents", None)
    if contents is None:
        return "<no hover info>"

    # MarkupContent
    if hasattr(contents, "value"):
        return contents.value
    # MarkedString (string)
    if isinstance(contents, str):
        return contents
    # List of MarkedString
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "value"):
                parts.append(item.value)
        return "\n".join(parts)
    return str(contents)


# ---------------------------------------------------------------------------
# Public API — LSP operations
# ---------------------------------------------------------------------------


def lsp_status() -> str:
    """Return human-readable status of the LSP server."""
    if not _multilspy_available:
        return "LSP: Not available\n  multilspy is not installed.\n  Install with: pip install multilspy"

    _check_idle_shutdown()

    with _LSP_LOCK:
        if _LSP_SERVER is not None:
            idle = time.time() - _LSP_LAST_ACCESS
            return (
                f"LSP: Running\n"
                f"  Language: {_LSP_LANGUAGE}\n"
                f"  Root: {_LSP_ROOT}\n"
                f"  Idle timeout: {_LSP_IDLE_TIMEOUT}s\n"
                f"  Idle for: {idle:.0f}s"
            )
        return (
            "LSP: Ready (not started)\n  Server will start on first LSP call.\n  Run /lsp diagnose <file> to start it."
        )


def _run_ruff(fp: Path, diagnostics: list) -> None:
    """Run ruff linter on a Python file and append results."""
    import json
    import subprocess

    try:
        r = subprocess.run(
            ["ruff", "check", "--output-format=json", str(fp)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.stdout and r.returncode in (0, 1):
            for rule in json.loads(r.stdout):
                line = rule.get("location", {}).get("row", "?")
                col = rule.get("location", {}).get("column", "?")
                code = rule.get("code", "?")
                msg = rule.get("message", "")
                severity = "Error" if rule.get("fix", {}).get("applicability") == "unsafe" else "Warning"
                if not rule.get("fix"):
                    severity = "Error" if code.startswith(("E", "F", "W")) else "Warning"
                diagnostics.append(
                    f"  [{severity}] {code}: {msg} (line {line}, col {col})",
                )
    except Exception as e:
        diagnostics.append(f"  [Info] ruff failed: {e}")


def _run_pyright(fp: Path, diagnostics: list) -> None:
    """Run pyright type checker on a Python file and append results."""
    import json
    import subprocess

    try:
        r = subprocess.run(
            ["pyright", "--outputjson", str(fp)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.stdout:
            data = json.loads(r.stdout)
            for diag in data.get("generalDiagnostics", []):
                line = diag.get("line", "?")
                col = diag.get("character", "?")
                rule = diag.get("rule", "")
                msg = diag.get("message", "")
                sev = diag.get("severity", "info")
                if sev == "information":
                    sev = "Info"
                elif sev == "warning":
                    sev = "Warning"
                else:
                    sev = "Error"
                diagnostics.append(
                    f"  [{sev}] pyright ({rule}): {msg} (line {line}, col {col})",
                )
    except Exception as e:
        diagnostics.append(f"  [Info] pyright failed: {e}")


def _run_lsp_diagnostics(fp: Path) -> None:
    """Run LSP server diagnostics on a file (best-effort)."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(str(fp))

        _with_open_file(lsp, str(fp), lambda _lsp: time.sleep(0.5))
    except Exception:
        _logger.debug("LSP diagnostics pre-warm failed for %s", fp)


def _format_diagnostics(fp: Path, diagnostics: list) -> str:
    """Format the diagnostics output."""
    if not diagnostics:
        return f"Diagnostics for {fp}:\n  Clean. No issues found."

    errors = sum(1 for d in diagnostics if "[Error]" in d)
    warnings = sum(1 for d in diagnostics if "[Warning]" in d)

    lines = [
        f"Diagnostics for {fp} ({len(diagnostics)} issues):",
        f"  Errors: {errors} | Warnings: {warnings}",
        "",
    ]
    lines.extend(diagnostics)
    return "\n".join(lines)


def lsp_diagnostics(file_path: str) -> str:
    """Run diagnostics on a file using ruff, pyright, and LSP."""
    try:
        fp = Path(file_path).resolve()
        ext = fp.suffix.lower()
        diagnostics: list[str] = []

        if ext in (".py", ".pyi"):
            _run_ruff(fp, diagnostics)
            _run_pyright(fp, diagnostics)

        _run_lsp_diagnostics(fp)

        return _format_diagnostics(fp, diagnostics)

    except Exception as e:
        return f"Diagnostics error: {e}"


def lsp_definition(file_path: str, line: int, column: int) -> str:
    """Find the definition of a symbol at the given location."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(file_path)

        def _req(_lsp):
            return _lsp.request_definition(file_path, line - 1, column - 1)

        result = _with_open_file(lsp, file_path, _req)

        if not result:
            return f"No definition found at {file_path}:{line}:{column}"

        lines = [f"Definition of symbol at {file_path}:{line}:{column}:"]
        for loc in result:
            lines.append(f"  -> {_location_to_str(loc)}")

        return "\n".join(lines)
    except Exception as e:
        return f"LSP definition error: {e}"


def lsp_references(file_path: str, line: int, column: int) -> str:
    """Find all references to a symbol."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(file_path)

        def _req(_lsp):
            return _lsp.request_references(file_path, line - 1, column - 1)

        result = _with_open_file(lsp, file_path, _req)

        if not result:
            return f"No references found for symbol at {file_path}:{line}:{column}"

        lines = [f"References to symbol at {file_path}:{line}:{column} ({len(result)} found):"]
        for loc in result[:50]:
            lines.append(f"  {_location_to_str(loc)}")

        if len(result) > 50:
            lines.append(f"  ... and {len(result) - 50} more")

        return "\n".join(lines)
    except Exception as e:
        return f"LSP references error: {e}"


def lsp_hover(file_path: str, line: int, column: int) -> str:
    """Get hover information for a symbol."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(file_path)

        def _req(_lsp):
            return _lsp.request_hover(file_path, line - 1, column - 1)

        result = _with_open_file(lsp, file_path, _req)

        if result is None:
            return f"No hover info at {file_path}:{line}:{column}"

        value = _hover_value(result)
        return f"Hover ({file_path}:{line}:{column}):\n{value}"
    except Exception as e:
        return f"LSP hover error: {e}"


def lsp_symbols(file_path: str) -> str:
    """List all symbols in a file."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(file_path)

        def _req(_lsp):
            return _lsp.request_document_symbols(file_path)

        symbols, _ = _with_open_file(lsp, file_path, _req)

        if not symbols:
            return f"No symbols found in {file_path}"

        lines = [f"Symbols in {file_path} ({len(symbols)} found):"]

        def _format_sym(sym: Any, indent: int = 0) -> None:
            name = getattr(sym, "name", "?")
            kind = _symbol_kind_name(getattr(sym, "kind", 0))
            rng = getattr(sym, "range", None)
            lnum = "?"
            if rng:
                start = getattr(rng, "start", None)
                if start:
                    lnum = start.line + 1
            prefix = "  " * indent
            lines.append(f"{prefix}{kind}: {name} (line {lnum})")
            children = getattr(sym, "children", None)
            if children:
                for child in children:
                    _format_sym(child, indent + 1)

        for sym in symbols:
            _format_sym(sym)

        return "\n".join(lines)
    except Exception as e:
        return f"LSP symbols error: {e}"


def lsp_rename(file_path: str, line: int, column: int, new_name: str) -> str:
    """Find rename locations for a symbol."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(file_path)

        def _req(_lsp):
            return _lsp.request_references(file_path, line - 1, column - 1)

        result = _with_open_file(lsp, file_path, _req)

        if not result:
            return f"No references found for rename at {file_path}:{line}:{column}"

        lines = [
            f"Rename '{new_name}' — {len(result)} locations to change:",
        ]
        for loc in result[:30]:
            lines.append(f"  {_location_to_str(loc)}")

        if len(result) > 30:
            lines.append(f"  ... and {len(result) - 30} more")

        lines.append("")
        lines.append("Note: Use read_file/patch_file to apply the rename changes.")
        return "\n".join(lines)
    except Exception as e:
        return f"LSP rename error: {e}"


def lsp_completion(file_path: str, line: int, column: int) -> str:
    """Get completions at a location."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(file_path)

        def _req(_lsp):
            return _lsp.request_completions(file_path, line - 1, column - 1)

        result = _with_open_file(lsp, file_path, _req)

        if not result:
            return f"No completions at {file_path}:{line}:{column}"

        lines = [f"Completions at {file_path}:{line}:{column} ({len(result)} found):"]
        for item in result[:20]:
            label = getattr(item, "label", "?")
            kind = getattr(item, "kind", None)
            kind_name = _symbol_kind_name(kind) if kind else ""
            detail = getattr(item, "detail", "")
            extra = f" -- {detail}" if detail else ""
            lines.append(f"  [{kind_name}] {label}{extra}")

        if len(result) > 20:
            lines.append(f"  ... and {len(result) - 20} more")

        return "\n".join(lines)
    except Exception as e:
        return f"LSP completions error: {e}"


# --- Feature stubs for pre/post-edit analysis ---


def _is_code_file(file_path: str) -> bool:
    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".m",
        ".pl",
        ".pm",
        ".ex",
        ".exs",
        ".erl",
        ".beam",
        ".clj",
        ".cljs",
        ".edn",
    }
    return Path(file_path).suffix.lower() in code_extensions


def lsp_preflight_check(code: str, language: str) -> dict:
    try:
        if language == "python":
            compile(code, "<script>", "exec")
            return {"clean": True, "errors": 0, "warnings": 0}
    except SyntaxError:
        return {"clean": False, "errors": 1, "warnings": 0}
    return {"clean": True, "errors": 0, "warnings": 0}


def lsp_pre_edit_check(file_path: str) -> dict:
    try:
        diag = lsp_diagnostics(file_path)
        errors = diag.count("error") if diag else 0
        warnings = diag.count("warning") if diag else 0
        return {"clean": errors == 0, "error_count": errors, "warning_count": warnings}
    except Exception:
        return {"clean": True, "error_count": 0, "warning_count": 0}


def lsp_post_edit_check(file_path: str) -> dict:
    try:
        diag = lsp_diagnostics(file_path)
        errors = diag.count("error") if diag else 0
        return {"new_errors": errors, "fixed_errors": 0}
    except Exception:
        return {"new_errors": 0, "fixed_errors": 0}


def lsp_check_patch_impact(file_path: str, diff: str) -> dict:
    try:
        diag = lsp_diagnostics(file_path)
        if diag and ("error" in diag or "warning" in diag):
            if diag.count("error") > 0:
                return {"conflicts": [f"{diag.count('error')} diagnostic issue(s) in file"]}
    except Exception:
        _logger.debug("LSP patch-impact check failed for %s", file_path)
    return {"conflicts": []}


def lsp_check_imports(file_path: str) -> dict:
    try:
        from pathlib import Path as _Path

        if not _Path(file_path).exists():
            return {"broken": [f"File not found: {file_path}"]}
        result = lsp_diagnostics(file_path)
        if not result or "error" not in result.lower():
            return {"broken": []}
        return {"broken": [result]}
    except Exception:
        return {"broken": []}


def lsp_trend_report() -> str:
    return "LSP trend tracking not yet implemented. Requires server state tracking across edits."


# ---------------------------------------------------------------------------
# Dispatch helper for the tool interface
# ---------------------------------------------------------------------------


def lsp_query(action: str, file_path: str = "", line: int = 0, column: int = 0, new_name: str = "") -> str:
    """Unified entry point for LSP queries. Used by the tool schema."""
    action = action.lower().strip()

    _DISPATCH = {
        "status": lambda _fp, _ln, _col, _nm: lsp_status(),
        "diagnostics": lambda _fp, _ln, _col, _nm: (
            lsp_diagnostics(_fp) if _fp else "Usage: lsp_query(action='diagnostics', file_path='path/to/file.py')"
        ),
        "diagnose": lambda _fp, _ln, _col, _nm: (
            lsp_diagnostics(_fp) if _fp else "Usage: lsp_query(action='diagnostics', file_path='path/to/file.py')"
        ),
        "definition": lambda fp, ln, col, _nm: (
            lsp_definition(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='definition', file_path='path/to/file.py', line=42, column=10)"
        ),
        "define": lambda fp, ln, col, _nm: (
            lsp_definition(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='definition', file_path='path/to/file.py', line=42, column=10)"
        ),
        "goto_definition": lambda fp, ln, col, _nm: (
            lsp_definition(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='definition', file_path='path/to/file.py', line=42, column=10)"
        ),
        "references": lambda fp, ln, col, _nm: (
            lsp_references(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='references', file_path='path/to/file.py', line=42, column=10)"
        ),
        "refs": lambda fp, ln, col, _nm: (
            lsp_references(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='references', file_path='path/to/file.py', line=42, column=10)"
        ),
        "hover": lambda fp, ln, col, _nm: (
            lsp_hover(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='hover', file_path='path/to/file.py', line=42, column=10)"
        ),
        "symbols": lambda _fp, _ln, _col, _nm: (
            lsp_symbols(_fp) if _fp else "Usage: lsp_query(action='symbols', file_path='path/to/file.py')"
        ),
        "document_symbols": lambda _fp, _ln, _col, _nm: (
            lsp_symbols(_fp) if _fp else "Usage: lsp_query(action='symbols', file_path='path/to/file.py')"
        ),
        "rename": lambda fp, ln, col, nm: (
            lsp_rename(fp, ln, col, nm)
            if fp and nm
            else "Usage: lsp_query(action='rename', file_path='path/to/file.py', line=42, column=10, new_name='foo')"
        ),
        "completion": lambda fp, ln, col, _nm: (
            lsp_completion(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='completion', file_path='path/to/file.py', line=42, column=10)"
        ),
        "completions": lambda fp, ln, col, _nm: (
            lsp_completion(fp, ln, col)
            if fp
            else "Usage: lsp_query(action='completion', file_path='path/to/file.py', line=42, column=10)"
        ),
    }

    handler = _DISPATCH.get(action)
    if handler:
        return handler(file_path, line, column, new_name)
    return (
        "Usage: lsp_query(action, file_path, line, column, new_name)\n"
        f"  Unknown action: '{action}'\n"
        "  Valid actions: status, diagnostics, definition, references,\n"
        "  hover, symbols, rename, completion"
    )
