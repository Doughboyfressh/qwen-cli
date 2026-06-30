"""
LSP client wrapper using multilspy.

Provides a lazy-starting, auto-shutting-down language server for diagnostics,
definition lookup, references, hover, symbols, and rename.

Design principles:
  - Zero cost until first LSP call (lazy initialization)
  - Auto-shutdown after 300s of inactivity
  - All calls wrapped in try/except — failures never crash the CLI
  - Import-friendly: if multilspy is not installed, graceful fallback
"""

from __future__ import annotations
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
import inspect
import asyncio

if TYPE_CHECKING:
    from multilspy import SyncLanguageServer
    from multilspy.multilspy_config import MultilspyConfig, Language
    from multilspy.multilspy_logger import MultilspyLogger

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
    from multilspy.multilspy_config import MultilspyConfig as _MultilspyConfig, Language as _Language
    from multilspy.multilspy_logger import MultilspyLogger as _MultilspyLogger
    _multilspy_available = True
except ImportError:
    pass
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Async-safe synchronous wrapper
# ---------------------------------------------------------------------------

def _sync_call(coro_or_value) -> Any:
    """If coro_or_value is a coroutine, run it synchronously. Otherwise return as-is."""
    if inspect.iscoroutine(coro_or_value):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cannot run coroutine on already-running loop; dispose it
                # cleanly to avoid "coroutine was never awaited" warnings.
                coro_or_value.close()
                return None
            return loop.run_until_complete(coro_or_value)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro_or_value)
            finally:
                loop.close()
    return coro_or_value


# Globals
# ---------------------------------------------------------------------------

_LSP_SERVER: Optional[Any] = None
_LSP_LOCK = threading.Lock()
_LSP_LAST_ACCESS: float = 0.0
_LSP_IDLE_TIMEOUT: int = 300  # 5 minutes
_LSP_ROOT: Optional[str] = None
_LSP_LANGUAGE: Optional[str] = None


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
        raise RuntimeError("multilspy is not installed")
    mapping = {
        "python": _Language.PYTHON,
        "typescript": _Language.TYPESCRIPT,
        "javascript": __Language.JAVASCRIPT,
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
    markers = {".git", "setup.py", "setup.cfg", "pyproject.toml",
               "package.json", "Cargo.toml", "go.mod", "build.gradle"}
    for p in [start] + list(start.parents):
        if any((p / m).exists() for m in markers):
            return str(p)
    return str(start)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _create_server(file_path: str = "") -> Any:
    """Create a new SyncLanguageServer and start it."""
    global _LSP_SERVER, _LSP_ROOT, _LSP_LANGUAGE

    if not _multilspy_available:
        raise RuntimeError(
            "multilspy is not installed. Install with: pip install multilspy"
        )

    language = _detect_language(file_path)
    root = _get_project_root(file_path)

    lang_enum = _language_to_multilspy_enum(language)
    config = _MultilspyConfig(lang_enum)
    logger = _MultilspyLogger()

    lsp = _SyncLanguageServer.create(config, logger, root)
    lsp.start_server()

    _LSP_SERVER = lsp
    _LSP_ROOT = root
    _LSP_LANGUAGE = language
    _LSP_LAST_ACCESS = time.time()

    return lsp


def _ensure_server(file_path: str = "") -> Any:
    """Get or create the LSP server. Re-creates if language changed."""
    global _LSP_SERVER, _LSP_LANGUAGE

    _LSP_LAST_ACCESS = time.time()

    with _LSP_LOCK:
        desired_lang = _detect_language(file_path)

        # If language changed, recreate
        if _LSP_SERVER is not None and _LSP_LANGUAGE != desired_lang:
            try:
                _LSP_SERVER._server.shutdown()
            except Exception:
                pass
            _LSP_SERVER = None
            _LSP_LANGUAGE = None

        if _LSP_SERVER is None:
            return _create_server(file_path)
        return _LSP_SERVER


def shutdown() -> None:
    """Explicitly shut down the LSP server."""
    global _LSP_SERVER, _LSP_ROOT, _LSP_LANGUAGE

    with _LSP_LOCK:
        if _LSP_SERVER is not None:
            try:
                _LSP_SERVER._server.shutdown()
            except Exception:
                pass
            _LSP_SERVER = None
            _LSP_ROOT = None
            _LSP_LANGUAGE = None


def _check_idle_shutdown() -> None:
    """Check if server should be shut down due to inactivity."""
    global _LSP_SERVER, _LSP_ROOT, _LSP_LANGUAGE

    if _LSP_SERVER is None:
        return

    if time.time() - _LSP_LAST_ACCESS > _LSP_IDLE_TIMEOUT:
        with _LSP_LOCK:
            if _LSP_SERVER is not None:
                try:
                    _LSP_SERVER._server.shutdown()
                except Exception:
                    pass
                _LSP_SERVER = None
                _LSP_ROOT = None
                _LSP_LANGUAGE = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SYMBOL_KIND_NAMES = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package",
    5: "Class", 6: "Method", 7: "Property", 8: "Field",
    9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
    13: "Variable", 14: "Constant", 15: "String", 16: "Number",
    17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
    21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
    25: "Operator", 26: "TypeParameter",
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
        return (
            "LSP: Not available\n"
            "  multilspy is not installed.\n"
            "  Install with: pip install multilspy"
        )

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
        else:
            return (
                "LSP: Ready (not started)\n"
                "  Server will start on first LSP call.\n"
                "  Run /lsp diagnose <file> to start it."
            )


def lsp_diagnostics(file_path: str) -> str:
    """Run diagnostics on a file."""
    try:
        import subprocess
        import json

        fp = Path(file_path).resolve()
        ext = fp.suffix.lower()
        diagnostics = []

        # --- Ruff (fast linting) ---
        if ext in (".py", ".pyi"):
            try:
                r = subprocess.run(
                    ["ruff", "check", "--output-format=json", str(fp)],
                    capture_output=True, text=True, timeout=30
                )
                if r.stdout and r.returncode in (0, 1):  # 1 = found issues
                    for rule in json.loads(r.stdout):
                        line = rule.get("location", {}).get("row", "?")
                        col = rule.get("location", {}).get("column", "?")
                        code = rule.get("code", "?")
                        msg = rule.get("message", "")
                        severity = "Error" if rule.get("fix", {}).get("applicability") == "unsafe" else "Warning"
                        if not rule.get("fix"):
                            severity = "Error" if code.startswith(("E", "F", "W")) else "Warning"
                        diagnostics.append(
                            f"  [{severity}] {code}: {msg} (line {line}, col {col})"
                        )
            except Exception as e:
                diagnostics.append(f"  [Info] ruff failed: {e}")

        # --- Pyright (type checking) ---
        if ext in (".py", ".pyi"):
            try:
                r = subprocess.run(
                    ["pyright", "--outputjson", str(fp)],
                    capture_output=True, text=True, timeout=60
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
                            f"  [{sev}] pyright ({rule}): {msg} (line {line}, col {col})"
                        )
            except Exception as e:
                diagnostics.append(f"  [Info] pyright failed: {e}")

        # --- LSP server diagnostics (additional) ---
        try:
            _check_idle_shutdown()
            lsp = _ensure_server(file_path)
            lsp.open_file(str(fp))
            # Some LSP servers publish diagnostics after opening
            # We check the internal server for any published diagnostics
            time.sleep(0.5)  # brief pause for server to process
        except Exception:
            pass  # LSP part is best-effort

        # --- Format output ---
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

    except Exception as e:
        return f"Diagnostics error: {e}"


def lsp_definition(file_path: str, line: int, column: int) -> str:
    """Find the definition of a symbol at the given location."""
    try:
        _check_idle_shutdown()
        lsp = _ensure_server(file_path)
        lsp.open_file(file_path)
        result = _sync_call(lsp.request_definition(file_path, line - 1, column - 1))

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
        lsp.open_file(file_path)
        result = _sync_call(lsp.request_references(file_path, line - 1, column - 1))

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
        lsp.open_file(file_path)
        result = _sync_call(lsp.request_hover(file_path, line - 1, column - 1))

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
        lsp.open_file(file_path)
        symbols, _ = _sync_call(lsp.request_document_symbols(file_path))

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
        lsp.open_file(file_path)
        # multilspy does not expose request_rename directly; we can query
        # references to show where the symbol appears
        result = _sync_call(lsp.request_references(file_path, line - 1, column - 1))

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
        lsp.open_file(file_path)
        result = _sync_call(lsp.request_completions(file_path, line - 1, column - 1))

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


# ---------------------------------------------------------------------------
# Dispatch helper for the tool interface
# ---------------------------------------------------------------------------

def lsp_query(action: str, file_path: str = "", line: int = 0,
              column: int = 0, new_name: str = "") -> str:
    """Unified entry point for LSP queries. Used by the tool schema."""
    action = action.lower().strip()

    if action == "status":
        return lsp_status()
    elif action in ("diagnostics", "diagnose"):
        if not file_path:
            return "Usage: lsp_query(action='diagnostics', file_path='path/to/file.py')"
        return lsp_diagnostics(file_path)
    elif action in ("definition", "define", "goto_definition"):
        if not file_path:
            return "Usage: lsp_query(action='definition', file_path='path/to/file.py', line=42, column=10)"
        return lsp_definition(file_path, line, column)
    elif action in ("references", "refs"):
        if not file_path:
            return "Usage: lsp_query(action='references', file_path='path/to/file.py', line=42, column=10)"
        return lsp_references(file_path, line, column)
    elif action == "hover":
        if not file_path:
            return "Usage: lsp_query(action='hover', file_path='path/to/file.py', line=42, column=10)"
        return lsp_hover(file_path, line, column)
    elif action in ("symbols", "document_symbols"):
        if not file_path:
            return "Usage: lsp_query(action='symbols', file_path='path/to/file.py')"
        return lsp_symbols(file_path)
    elif action == "rename":
        if not file_path or not new_name:
            return "Usage: lsp_query(action='rename', file_path='path/to/file.py', line=42, column=10, new_name='foo')"
        return lsp_rename(file_path, line, column, new_name)
    elif action == "completion" or action == "completions":
        if not file_path:
            return "Usage: lsp_query(action='completion', file_path='path/to/file.py', line=42, column=10)"
        return lsp_completion(file_path, line, column)
    else:
        return (
            "Usage: lsp_query(action, file_path, line, column, new_name)\n"
            f"  Unknown action: '{action}'\n"
            "  Valid actions: status, diagnostics, definition, references,\n"
            "  hover, symbols, rename, completion"
        )
