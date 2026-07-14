"""Filesystem and command tool implementations (do_read_file, do_edit_file, ...).

Extracted from main.py as part of the monolith split. The dispatch layer
(_dispatch_interactive, _call_tool_safe, the timeout table) stays in main;
this module is the tools' actual behavior: path resolution, the dangerous-
command gate, audit logging, backups + the /undo stack feed, LSP pre/post
edit reports, and the read/edit/patch/write/move/delete/list/find/search
implementations.

Mutable session state (_session_changes, _turn_read_cache, _auto_approve,
_backup_stack) and runtime-patched paths (AUDIT_LOG_FILE, BACKUPS_DIR) stay
OWNED BY MAIN and are accessed as _main.<name> at call time, as are the
confirmation/backup/LSP hooks that tests monkeypatch on main — the project
convention for split-out modules (see core/turn.py).
"""

import contextlib
import difflib
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from rich.syntax import Syntax

from qwen_cli.core.indexer import IGNORE_DIRS

_logger = logging.getLogger(__name__)


def _resolve(path: str) -> Path:
    """Internal helper: resolve."""
    p = Path(path).expanduser()
    return p.resolve(strict=False) if p.is_absolute() else (Path.cwd() / p).resolve(strict=False)


def _read_raw(p: Path) -> str:
    r"""Read a file with no newline translation — CRLF stays CRLF."""
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


def _write_raw(p: Path, text: str) -> None:
    r"""Write text verbatim, with no newline translation.

    Path.write_text() opens with newline=None, which rewrites every "\n" as
    os.linesep — on Windows that silently converts an LF file to CRLF, so
    writing back a file the model barely touched marks every one of its lines
    as modified in git. do_edit_file already round-tripped endings correctly;
    write_file, patch_file, the backups, /undo and /rollback all still went
    through write_text and corrupted LF files. Everything that writes a user's
    file now goes through here.
    """
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _restore_endings(text_lf: str, uses_crlf: bool) -> str:
    """Re-apply the file's original line-ending style to LF-normalized text."""
    return text_lf.replace("\n", "\r\n") if uses_crlf else text_lf



def _is_dangerous(command: str) -> bool:
    """Check if a command is dangerous, including expanded forms."""
    import qwen_cli.main as _main

    # Check the raw command
    if _main._DANGEROUS_CMD_RE.search(command):
        return True
    # Detect: CMD=rm; $CMD -rf /
    if re.search(r"\$[A-Z_]+\s+(-[a-zA-Z]*r|rm|mkfs|dd|chmod)", command):
        return True
    # Detect: base64 encoded commands
    if re.search(r"base64\s+(-d|--decode)", command):
        return True
    # Detect: eval/exec with suspicious content
    if re.search(r"eval\s+.*\$\(", command):
        return True
    # Detect: command substitution with dangerous commands
    if re.search(r"\$\(.*\b(rm|mkfs|dd|chmod)\b", command):
        return True
    # Detect: pipe to shell/interpreter (curl/wget/echo ... | bash/sh/python/powershell/iex)
    return bool(re.search(r"\|\s*(bash|sh|zsh|fish|python\d*|perl|ruby|pwsh|powershell|iex|cmd)\b", command))


def _confirm_action(prompt: str) -> bool:
    """y/N gate for file mutations. Returns True without asking when /auto is on."""
    import qwen_cli.main as _main

    if _main._auto_approve:
        _main.console.print(f"[dim]  {prompt} — auto-approved (/auto on)[/dim]")
        return True
    try:
        return _main.console.input(f"[bold yellow]  {prompt} [y/N]:[/bold yellow] ").strip().lower() == "y"
    except (KeyboardInterrupt, EOFError):
        return False


_MAX_CMD_OUTPUT = 25_000

_SCRIPT_INTERP: dict[str, tuple[str, str]] = {
    "python": (sys.executable, ".py"),
    "python3": (sys.executable, ".py"),
    "py": (sys.executable, ".py"),
    "javascript": ("node", ".js"),
    "js": ("node", ".js"),
    "node": ("node", ".js"),
    "bash": ("bash", ".sh"),
    "sh": ("sh", ".sh"),
    "powershell": ("pwsh -NoProfile -ExecutionPolicy Bypass -File", ".ps1"),
    "ps1": ("pwsh -NoProfile -ExecutionPolicy Bypass -File", ".ps1"),
    "ruby": ("ruby", ".rb"),
    "php": ("php", ".php"),
    "lua": ("lua", ".lua"),
    "perl": ("perl", ".pl"),
    "go": ("go run", ".go"),
    "r": ("Rscript", ".r"),
    "typescript": ("npx ts-node", ".ts"),
    "ts": ("npx ts-node", ".ts"),
}


def _audit_log(command: str, cwd: Path, outcome: str) -> None:
    """Append a line to ~/.qwen-cli/audit.log recording a shell command execution.

    The only prior trace of what a model actually ran was chat history — no
    persistent, append-only record of what ran, when, or from where. Best-effort:
    a logging failure must never break the command it's recording.
    """
    import qwen_cli.main as _main

    try:
        line = f"{datetime.now().isoformat()} | cwd={cwd} | {outcome} | {command[:500]}\n"
        with _main.AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        _logger.debug("Failed to write audit log entry")


def do_run_command(
    command: str, cwd: str = "", timeout: int = 30, env: dict | None = None, stdin: str = "", quiet: bool = False
) -> str:
    """Run a shell command and return its stdout + stderr in real-time.

    Dangerous commands (rm -rf, format, etc.) require explicit confirmation.
    """
    import qwen_cli.main as _main

    work_dir = _main._resolve(cwd) if cwd else Path.cwd()
    stripped = command.strip()
    if not stripped:
        return "[tool_call_error: empty command — send the complete command to run]"
    if re.search(r"(?:\|\||&&|[|;&])\s*$", stripped):
        _audit_log(command, work_dir, "rejected_malformed")
        return (
            "[tool_call_error: command ends with a dangling operator — "
            "send the complete pipeline in a single call]"
        )
    if _is_dangerous(command):
        if _main._unattended:
            _audit_log(command, work_dir, "denied_unattended")
            return (
                "[blocked: this command needs the user's explicit confirmation, and no user is "
                "attached to this session (piped input or a spawned agent). It was NOT run. "
                "Do not retry it — report what you wanted to run and why, and let the user run it.]"
            )
        _main.console.print(f"[bold red]  [dangerous][/bold red] {command}")
        try:
            answer = _main.console.input("[bold red]  Run anyway? [y/N]:[/bold red] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = ""
        if answer != "y":
            _audit_log(command, work_dir, "declined_by_user")
            return "[command cancelled by user]"
    elif not quiet:
        pass  # silent by default; output still captured in return value

    proc_env = {**os.environ, **env} if env else None
    stdout_buf: list[str] = []
    stderr_buf: list[str] = []
    cancelled = threading.Event()
    t0 = time.monotonic()

    _shell_meta = re.compile(r"[|;&$`<>()\[\]{}!\n]")
    try:
        if sys.platform == "win32":
            # Everything goes through cmd.exe on Windows. dir, echo, type, copy,
            # set, cls and friends are cmd BUILTINS, not .exe files, so a
            # shell=False argv Popen hands them to CreateProcess, which can only
            # launch real executables — every one of them failed with WinError 2.
            # Routing only metacharacter-bearing commands through the shell (the
            # old behavior) made that absurd: `dir` failed while `dir | findstr x`
            # worked. BASE_SYSTEM tells the model run_command runs "via cmd.exe";
            # now that is actually true.
            popen_args: str | list[str] = command
            use_shell = True
        elif _shell_meta.search(command):
            _logger.warning("shell=True for command with metacharacters: %.120s", command)
            popen_args, use_shell = command, True
        else:
            popen_args, use_shell = shlex.split(command), False
        proc = subprocess.Popen(
            popen_args,
            shell=use_shell,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=proc_env,
        )
        if stdin:
            try:
                proc.stdin.write(stdin)
                proc.stdin.close()
            except BrokenPipeError:
                pass

        def _pipe(stream, buf: list[str]) -> None:
            """Internal helper: pipe stream into buffer."""
            for line in stream:
                if cancelled.is_set():
                    break
                buf.append(line)

        t_out = threading.Thread(target=_pipe, args=(proc.stdout, stdout_buf), daemon=True)
        t_err = threading.Thread(target=_pipe, args=(proc.stderr, stderr_buf), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            cancelled.set()
            proc.kill()
            t_out.join(timeout=1)
            t_err.join(timeout=1)
            _audit_log(command, work_dir, f"timed_out({timeout}s)")
            partial = "".join(stdout_buf).strip()
            return f"[timed out after {timeout}s]\n\n{partial}" if partial else f"[timed out after {timeout}s]"
        except KeyboardInterrupt:
            cancelled.set()
            proc.kill()
            t_out.join(timeout=1)
            t_err.join(timeout=1)
            _main.console.print("\n[dim][command cancelled][/dim]")
            _audit_log(command, work_dir, "cancelled_by_ctrl_c")
            partial = "".join(stdout_buf).strip()
            return f"[cancelled]\n\n{partial}" if partial else "[cancelled]"

        t_out.join(timeout=2)
        t_err.join(timeout=2)
        elapsed = time.monotonic() - t0

        full_out = "".join(stdout_buf).strip()
        full_err = "".join(stderr_buf).strip()
        if len(full_out) > _MAX_CMD_OUTPUT:
            full_out = full_out[:_MAX_CMD_OUTPUT] + f"\n... [stdout truncated at {_MAX_CMD_OUTPUT} chars]"
        if len(full_err) > _MAX_CMD_OUTPUT:
            full_err = full_err[:_MAX_CMD_OUTPUT] + f"\n... [stderr truncated at {_MAX_CMD_OUTPUT} chars]"

        parts: list[str] = []
        if full_out:
            parts.append("stdout:\n" + full_out)
        if full_err:
            parts.append("stderr:\n" + full_err)
        rc_note = "✓" if proc.returncode == 0 else "✗"
        parts.append(f"exit code: {proc.returncode} {rc_note}  ({elapsed:.1f}s)")
        _audit_log(command, work_dir, f"exit={proc.returncode}")
        return "\n\n".join(parts)
    except Exception as e:
        _audit_log(command, work_dir, f"error={e}")
        return f"[error: {e}]"


def do_run_script(language: str, code: str, cwd: str = "", timeout: int = 30) -> str:
    """Execute inline code by writing to a temp file and running with the right interpreter."""
    import qwen_cli.main as _main

    import tempfile as _tmpmod

    lang_key = language.lower().strip()
    if lang_key not in _SCRIPT_INTERP:
        supported = ", ".join(sorted(_SCRIPT_INTERP))
        return f"[unsupported language: {language!r} — supported: {supported}]"
    interp, ext = _SCRIPT_INTERP[lang_key]
    # _is_dangerous(), not the bare _main._DANGEROUS_CMD_RE: the regex alone misses
    # pipe-to-shell ("curl x | bash"), eval with command substitution, and
    # variable-expanded rm — all of which run_command catches. Gating the two
    # tools differently meant the model could route around the confirmation
    # simply by picking run_script. Prompt rather than hard-block, so this
    # matches run_command's behavior (and gets the same audit-log entry).
    if _is_dangerous(code):
        if _main._unattended:
            _audit_log(f"[run_script:{lang_key}] {code}", Path(cwd) if cwd else Path.cwd(), "denied_unattended")
            return (
                "[blocked: this script needs the user's explicit confirmation, and no user is "
                "attached to this session (piped input or a spawned agent). It was NOT run. "
                "Do not retry it — report what you wanted to run and why, and let the user run it.]"
            )
        _main.console.print(f"[bold red]  [dangerous script][/bold red] {lang_key}")
        _audit_log(f"[run_script:{lang_key}] {code}", Path(cwd) if cwd else Path.cwd(), "dangerous_prompt")
        try:
            answer = _main.console.input("[bold red]  Run anyway? [y/N]:[/bold red] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = ""
        if answer != "y":
            _audit_log(f"[run_script:{lang_key}] {code}", Path(cwd) if cwd else Path.cwd(), "declined_by_user")
            return "[script cancelled by user]"
    n_lines = len(code.splitlines())
    _main.console.print(f"[bold yellow]  [run_script][/bold yellow] {lang_key}  ({n_lines} lines)")

    # Feature 4: Pre-flight type check for Python scripts
    if lang_key == "python":
        try:
            result = _main._get_lsp().lsp_preflight_check(code, "python")
            if not result["clean"]:
                _main.console.print(
                    f"[dim yellow]  Pre-flight: {result['errors']} error(s), {result['warnings']} warning(s) in script[/dim yellow]"
                )
        except Exception:
            _logger.debug("LSP pre-flight check unavailable", exc_info=True)

    with _tmpmod.NamedTemporaryFile(
        suffix=ext,
        mode="w",
        delete=False,
        encoding="utf-8",
        prefix="qwen_script_",
    ) as tf:
        tf.write(code)
        tmp = tf.name
    interp_parts = interp.split()
    quoted = [f'"{p}"' if " " in p else p for p in interp_parts]
    cmd = " ".join(quoted) + f' "{tmp}"'
    try:
        return do_run_command(cmd, cwd=cwd, timeout=timeout, quiet=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


_READ_PREVIEW_LINES = 40  # _main.console preview cap for full-file reads (model gets everything)


# Per-call read caps. A full read of a big file used to be silently chopped to
# the first TOOL_RESULT_LIMIT chars by _cap_result under a header claiming the
# whole file was returned — the model then "knew" content it never saw. Now
# read_file paginates itself with honest headers and an explicit continuation
# hint, and its results are exempt from _smart_cap (see there).
_MAX_READ_LINES = 400
_MAX_READ_CHARS = 16_000
_MAX_READ_FILE_BYTES = 50_000_000  # refuse to decode beyond this


def do_read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Handle read file operation."""
    import qwen_cli.main as _main

    try:
        p = _main._resolve(path)
        if not p.exists():
            return f"[file not found: {p}]"
        # Dedup identical re-reads within one turn — repeat reads of an unchanged
        # file are pure context waste (the content is already in the conversation).
        # mtime in the key means a file modified between reads is re-read normally.
        read_key = (str(p), offset, limit, p.stat().st_mtime_ns)
        if read_key in _main._turn_read_cache:
            _main.console.print(f"[dim cyan]  {p}  (already read this turn — skipping)[/dim cyan]")
            return f"[{p} lines {offset or 1}+{f' limit {limit}' if limit else ''}: identical content already returned earlier this turn — reuse it]"
        _main._turn_read_cache.add(read_key)
        if p.stat().st_size > _MAX_READ_FILE_BYTES:
            return f"[file too large to read: {p.name} ({p.stat().st_size:,} bytes) — use search_files or run_command]"
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            return f"[binary file not supported: {p.name}]"
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        start = max(0, offset - 1) if offset else 0
        if start >= total > 0:
            return f"[offset {offset} is beyond the end of {p.name} ({total} lines)]"
        want = min(limit, _MAX_READ_LINES) if limit else _MAX_READ_LINES
        end = min(start + want, total)
        chunk = lines[start:end]
        # Char guard for long-line files (minified JS, JSON blobs): shrink the
        # window until it fits, so the range in the header stays truthful.
        while len(chunk) > 1 and sum(len(ln) + 1 for ln in chunk) > _MAX_READ_CHARS:
            chunk.pop()
        snippet = "\n".join(chunk)
        if len(snippet) > _MAX_READ_CHARS:
            snippet = snippet[:_MAX_READ_CHARS] + "\n... [single line truncated]"
        end = start + len(chunk)

        # Evidence for the citation guard: these are the lines the model can now
        # actually see, and the only ones it may cite. See main._unverified_citations.
        _main._turn_seen_lines.setdefault(str(p), set()).update(range(start + 1, end + 1))

        if start == 0 and end >= total:
            header = f"{p}  ({total} lines)"
            _main.console.print(f"[dim]read {p} ({total} lines, {p.stat().st_size} bytes)[/dim]")
        else:
            header = f"{p}  (lines {start + 1}–{end} of {total})"
            _main.console.print(f"[dim]read {p} (lines {start + 1}-{end} of {total})[/dim]")
        footer = (
            f"\n\n[showing lines {start + 1}–{end} of {total}. Read the next part with offset={end + 1}]"
            if end < total
            else ""
        )
        return f"{header}\n\n{snippet}{footer}"
    except Exception as e:
        return f"[error: {e}]"


def _lsp_pre_edit_snapshot(p: Path) -> None:
    """Record a file's current errors before it is written.

    Without this, lsp_post_edit_check has no baseline to diff against and can't
    tell an error the edit INTRODUCED from one that was already there. edit_file
    and write_file previously took no snapshot at all, so their post-edit report
    had nothing to compare with.
    """
    import qwen_cli.main as _main

    try:
        _lsp = _main._get_lsp()
        if p.exists() and _lsp._is_code_file(str(p)):
            _lsp.lsp_pre_edit_check(str(p))
    except Exception:
        _logger.debug("LSP pre-edit snapshot failed for %s (non-critical)", p)


def _lsp_post_edit_report(p: Path) -> None:
    """Print post-edit diagnostics trend + broken-import report for a just-written file."""
    import qwen_cli.main as _main

    try:
        _lsp = _main._get_lsp()
        if not _lsp._is_code_file(str(p)):
            return
        post = _lsp.lsp_post_edit_check(str(p))
        if post["new_errors"] > 0:
            _main.console.print(f"[dim red]  Post-edit: {post['new_errors']} new error(s) introduced[/dim red]")
            for detail in post.get("details", [])[:3]:
                _main.console.print(f"[dim red]    {detail}[/dim red]")
        if post["fixed_errors"] > 0:
            _main.console.print(f"[dim green]  Post-edit: {post['fixed_errors']} error(s) fixed[/dim green]")
        broken = _lsp.lsp_check_imports(str(p))["broken"]
        if broken:
            _main.console.print(f"[dim red]  Imports: {len(broken)} unresolved import(s)[/dim red]")
    except Exception:
        _logger.debug("LSP post-edit check failed for %s (non-critical)", p)


def _recover_old_string(original: str, old_string: str) -> tuple[str | None, str]:
    """Recover what a failed old_string was meant to match. Line endings are
    already normalized by the caller, so the one mismatch class a model
    produces constantly and that is safe to fix mechanically is trailing
    whitespace per line. Returns (exact_block_from_file, note) or (None, "")
    — never an ambiguous match.

    Trailing-whitespace-tolerant sliding window over file lines; leading
    whitespace stays strict (indentation errors are real errors).
    """
    target = [ln.rstrip() for ln in old_string.split("\n")]
    file_lines = original.splitlines(keepends=True)
    stripped = [ln.rstrip() for ln in file_lines]
    n = len(target)
    if not n or n > len(stripped):
        return None, ""
    hits = [i for i in range(len(stripped) - n + 1) if stripped[i : i + n] == target]
    if len(hits) != 1:
        return None, ""
    i = hits[0]
    last = file_lines[i + n - 1]
    block = "".join(file_lines[i : i + n - 1]) + last.rstrip("\r\n")
    if original.count(block) == 1 and block != old_string:
        return block, "matched ignoring trailing whitespace"
    return None, ""


def do_edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string in a file — the model-friendly alternative to unified diffs."""
    import qwen_cli.main as _main

    try:
        p = _main._resolve(path)
        if not p.exists():
            return f"[file not found: {p}]"
        if not old_string:
            return "[edit_file error: old_string is empty — use write_file to create content]"
        if old_string == new_string:
            return "[edit_file error: old_string and new_string are identical — nothing to change]"
        # Read with endings preserved, then match in LF-normalized space (models
        # mix CRLF/LF freely) and write back in the file's own ending style.
        # The old read_text/write_text pair normalized on read but converted
        # every \n to os.linesep on write — every edit on Windows silently
        # rewrote LF files as CRLF.
        raw_text = _read_raw(p)
        uses_crlf = "\r\n" in raw_text
        original = raw_text.replace("\r\n", "\n")
        old_string = old_string.replace("\r\n", "\n")
        new_string = new_string.replace("\r\n", "\n")
        count = original.count(old_string)
        fallback_note = ""
        if count == 0:
            recovered, fallback_note = _recover_old_string(original, old_string)
            if recovered is not None:
                old_string = recovered
                count = original.count(old_string)
        if count == 0:
            return (
                "[edit_file error: old_string not found in the file. Re-read the file and copy the "
                "text EXACTLY as it appears, including all whitespace and indentation.]"
            )
        if count > 1 and not replace_all:
            return (
                f"[edit_file error: old_string occurs {count} times. Include more surrounding lines "
                f"to make it unique, or pass replace_all=true to change every occurrence.]"
            )
        patched = original.replace(old_string, new_string)
        # Store the RAW original (endings intact) — /rollback writes this back
        # verbatim, so an LF-normalized copy here would itself convert the file.
        if str(p) not in _main._session_changes:
            _main._session_changes[str(p)] = raw_text

        changed_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=f"a/{p.name}",
                tofile=f"b/{p.name}",
                lineterm="",
            )
        )
        # Show what the edit will do BEFORE asking. Approving an invisible
        # change isn't approval. Skipped under /auto, where nobody is deciding
        # and _main.console quiet matters (patch_file/write_file style).
        if not _main._auto_approve and changed_lines:
            preview = "\n".join(ln.rstrip("\n") for ln in changed_lines[:60])
            if len(changed_lines) > 60:
                preview += f"\n... ({len(changed_lines) - 60} more diff lines)"
            _main.console.print(Syntax(preview, "diff", theme="monokai"))

        if not _main._confirm_action("Apply edit?"):
            return "[edit cancelled by user]"

        _main._backup_file(p)
        _lsp_pre_edit_snapshot(p)
        _write_raw(p, _restore_endings(patched, uses_crlf))
        lines_changed = sum(
            1 for ln in changed_lines if ln.startswith(("+", "-")) and not ln.startswith(("---", "+++"))
        )
        note = f" [{fallback_note}]" if fallback_note else ""
        _main.console.print(
            f"[bold yellow]  [edit_file][/bold yellow] {p}  "
            f"({count if replace_all else 1} replacement(s), {lines_changed} lines changed){note}"
        )
        _main._lsp_post_edit_report(p)
        return f"[edited: {p}  ({count if replace_all else 1} replacement(s), {lines_changed} lines changed){note}]"
    except Exception as e:
        return f"[error: {e}]"


def do_patch_file(path: str, diff: str) -> str:
    """Handle patch file operation."""
    import qwen_cli.main as _main

    try:
        p = _main._resolve(path)

        # Feature 1: Pre-edit diagnostics + Feature 9: Conflict detection
        try:
            _lsp = _main._get_lsp()
            if _lsp._is_code_file(str(p)):
                pre = _lsp.lsp_pre_edit_check(str(p))
                if not pre["clean"]:
                    _main.console.print(
                        f"[dim yellow]  Pre-edit: {pre['error_count']} error(s), "
                        f"{pre['warning_count']} warning(s) already in file[/dim yellow]",
                    )
                impact = _main._get_lsp().lsp_check_patch_impact(str(p), diff)
                if impact["conflicts"]:
                    _main.console.print(
                        f"[dim red]  Conflict: {len(impact['conflicts'])} error(s) on lines about to be changed[/dim red]"
                    )
        except Exception:
            _logger.debug("LSP pre-edit check failed for %s (non-critical)", p)

        if not p.exists():
            return f"[file not found: {p}]"
        # Match/patch in LF space (models emit LF diffs regardless of the file's
        # endings), write back in the file's own style — same round-trip as
        # do_edit_file. See _write_raw.
        raw_text = _read_raw(p)
        uses_crlf = "\r\n" in raw_text
        original = raw_text.replace("\r\n", "\n")
        if str(p) not in _main._session_changes:
            _main._session_changes[str(p)] = raw_text
        try:
            patched = _main._apply_diff(original, diff)
        except ValueError as e:
            return f"[patch failed: {e}]"

        if patched == original:
            return f"[no changes after applying patch: {p}]"

        # Show diff preview
        preview_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=f"a/{p.name}",
                tofile=f"b/{p.name}",
                lineterm="",
            )
        )
        preview = "".join(preview_lines[:60])
        if len(preview_lines) > 60:
            preview += f"\n... ({len(preview_lines) - 60} more lines)"
        _main.console.print(Syntax(preview, "diff", theme="monokai"))
        if not _main._confirm_action("Apply patch?"):
            return "[patch cancelled by user]"

        # Backup and write. Uses the shared _main._backup_file() helper (same one
        # write_file/move_file/delete_file use) rather than duplicating its
        # logic inline — this path used to skip both the same-second
        # collision fix and the _main.BACKUPS_DIR cleanup cap, so a session
        # dominated by patch_file (the tool the system prompt prefers) could
        # grow backups unbounded and, for rapid same-file retries, silently
        # lose the true original content on disk (the only recovery path if
        # the in-memory _main._backup_stack doesn't survive a crash).
        _main._backup_file(p)

        _write_raw(p, _restore_endings(patched, uses_crlf))
        lines_changed = sum(
            1 for ln in preview_lines if ln.startswith(("+", "-")) and not ln.startswith(("---", "+++"))
        )
        _main.console.print(f"[bold yellow]  [patch_file][/bold yellow] applied to {p}  ({lines_changed} lines changed)")
        # Post-edit trend + import check — must run AFTER the write (it used to run
        # before the patch was applied, so it was reporting on the pre-edit file).
        _main._lsp_post_edit_report(p)
        return f"[patched: {p}  ({lines_changed} lines changed)]"
    except Exception as e:
        return f"[error: {e}]"


def _fmt_size(n: int) -> str:
    """Internal helper: fmt size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}"
        n //= 1024
    return f"{n} TB"


def do_list_directory(path: str, recursive: bool = False) -> str:
    """Handle list directory operation."""
    import qwen_cli.main as _main

    try:
        p = _main._resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        if not p.is_dir():
            return f"[not a directory: {p}]"
        if recursive:
            return f"Tree: {p}\n\n{_main.build_project_tree(p, max_depth=4, max_files=300)}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        nd, nf = len(dirs), len(files)
        lines = [f"{p}:  ({nd} dir{'s' if nd != 1 else ''}, {nf} file{'s' if nf != 1 else ''})\n"]
        for e in dirs:
            lines.append(f"  [dir ]  {e.name}/")
        for e in files:
            st = e.stat()
            size = _fmt_size(st.st_size)
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [file]  {e.name:<45}  {size:>8}  {mtime}")
        return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


def do_find_files(path: str, pattern: str) -> str:
    """Handle find files operation."""
    import qwen_cli.main as _main

    try:
        import fnmatch as _fnmatch
        import os as _os

        p = _main._resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        _name_pat = pattern.rstrip("/").split("/")[-1] if pattern else "*"
        results = []
        for _dp_str, _dns, _fns in _os.walk(p):
            _dns[:] = sorted(d for d in _dns if d not in IGNORE_DIRS)
            _dp = Path(_dp_str)
            for _fn in sorted(_fns):
                if _fnmatch.fnmatch(_fn, _name_pat):
                    results.append(_dp / _fn)
        if not results:
            return f"No files matched '{pattern}' in {p}"
        lines = [f"Matches for '{pattern}' in {p}  ({len(results)} found):\n"]
        for r in results[:500]:
            rel = r.relative_to(p)
            if r.is_file():
                st = r.stat()
                size = _fmt_size(st.st_size)
                mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                lines.append(f"  {rel!s:<60}  {size:>8}  {mtime}")
            else:
                lines.append(f"  {rel}/")
        if len(results) > 500:
            lines.append(f"  ... and {len(results) - 500} more")
        return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


_SEARCH_MAX_FILE_BYTES = 5_000_000
_SEARCH_MAX_MATCHES = 500  # across all files; stop the walk once hit
_SEARCH_MAX_LINES_PER_FILE = 250


def _compile_query(query: str) -> re.Pattern:
    """Compile the query as a regex, falling back to a literal search.

    The model passes both — a bare identifier and a real pattern — and a query
    like `foo(` is a valid literal but an invalid regex.
    """
    try:
        return re.compile(query, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(query), re.IGNORECASE)


def _walk_matching_files(root: Path, name_pat: str) -> Iterator[Path]:
    """Yield files under root whose NAME matches name_pat.

    os.walk with in-place pruning of _dns, not rglob: ignored trees (.venv,
    node_modules) are never descended into at all, which is what keeps a search
    at a repo root from enumerating 10,000+ files and hanging.
    """
    import fnmatch

    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        for fn in sorted(filenames):
            if fnmatch.fnmatch(fn, name_pat):
                yield Path(dirpath) / fn


def _context_ranges(match_idx: list[int], n_lines: int, context: int) -> list[tuple[int, int]]:
    """Line ranges around each match, merging any that touch or overlap."""
    ranges: list[tuple[int, int]] = []
    for mi in match_idx:
        lo, hi = max(0, mi - context), min(n_lines - 1, mi + context)
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], hi))
        else:
            ranges.append((lo, hi))
    return ranges


def _render_matches(file_lines: list[str], match_idx: list[int], context: int) -> list[str]:
    """Format one file's hits: bare match lines, or grep -C style context blocks."""
    if not context:
        return [f"  L{mi + 1:>4}: {file_lines[mi].rstrip()}" for mi in match_idx]

    match_set = set(match_idx)
    out: list[str] = []
    for ri, (lo, hi) in enumerate(_context_ranges(match_idx, len(file_lines), context)):
        if ri:
            out.append("  ···")  # gap between non-adjacent blocks
        for ci in range(lo, hi + 1):
            marker = ">" if ci in match_set else " "
            out.append(f"  {marker} L{ci + 1:>4}: {file_lines[ci].rstrip()}")
    return out


def _search_one_file(fpath: Path, rx: re.Pattern, context: int) -> tuple[list[int], list[str]] | None:
    """Search a single file. None if it has no matches, or isn't searchable text."""
    if fpath.stat().st_size > _SEARCH_MAX_FILE_BYTES:
        return None
    raw = fpath.read_bytes()
    if b"\x00" in raw[:4096]:  # binary
        return None
    file_lines = raw.decode("utf-8", errors="replace").splitlines()
    match_idx = [i for i, ln in enumerate(file_lines) if rx.search(ln)]
    if not match_idx:
        return None
    return match_idx, _render_matches(file_lines, match_idx, context)


def _format_search_report(query: str, root: Path, file_hits: dict, total: int) -> str:
    """Render the grouped-by-file result block."""
    out = [f"Matches for '{query}' in {root} — {total} match(es) in {len(file_hits)} file(s):\n"]
    for fname, (indices, hit_lines) in file_hits.items():
        n = len(indices)
        out.append(f"{fname}  ({n} match{'es' if n != 1 else ''}):")
        out.extend(hit_lines[:_SEARCH_MAX_LINES_PER_FILE])
        if len(hit_lines) > _SEARCH_MAX_LINES_PER_FILE:
            out.append(f"  ... ({len(hit_lines) - _SEARCH_MAX_LINES_PER_FILE} more lines)")
        out.append("")
    if total >= _SEARCH_MAX_MATCHES:
        out.append(f"... (stopped at {_SEARCH_MAX_MATCHES} — narrow your query or glob pattern)")
    return "\n".join(out)


def do_search_files(path: str, query: str, pattern: str = "**/*", context: int = 0) -> str:
    """Handle search files operation."""
    import qwen_cli.main as _main

    try:
        p = _main._resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        rx = _compile_query(query)
        # Only the last segment of the glob is used — the walk handles recursion,
        # so "**/*.py" and "*.py" both mean "any .py file underneath".
        name_pat = pattern.rstrip("/").split("/")[-1] if pattern else "*"

        file_hits: dict[str, tuple[list[int], list[str]]] = {}
        total = 0
        for fpath in _walk_matching_files(p, name_pat):
            if not fpath.is_file():
                continue
            try:
                hit = _search_one_file(fpath, rx, context)
            except Exception:
                _logger.debug("Skipping unreadable file %s during search", fpath)
                continue
            if hit is None:
                continue
            match_idx, lines_out = hit
            file_hits[str(fpath.relative_to(p))] = (match_idx, lines_out)
            # Evidence for the citation guard: a search result DOES show real line
            # numbers, so citing one is legitimate — unlike reasoning about what
            # the code around them does. See main._unverified_citations.
            seen = _main._turn_seen_lines.setdefault(str(fpath), set())
            for mi in match_idx:
                seen.update(range(max(0, mi - context) + 1, mi + context + 2))
            total += len(match_idx)
            if total >= _SEARCH_MAX_MATCHES:
                break

        if not file_hits:
            return f"No matches for '{query}' in {p}"
        return _format_search_report(query, p, file_hits, total)
    except Exception as e:
        return f"[error: {e}]"


def _cleanup_backups(keep: int = 50, keep_per_file: int = 10) -> None:
    """Keep the most recent N backups overall, and at most M per original file.

    The per-file cap stops one hot file (e.g. a test fixture edited hundreds of
    times) from evicting the only backups of everything else.
    """
    import qwen_cli.main as _main

    if not _main.BACKUPS_DIR.exists():
        return
    files = sorted(_main.BACKUPS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    per_file: dict[str, int] = {}
    for i, f in enumerate(files):
        orig = f.name.rsplit(".", 2)[0]  # "<original>.<stamp>.bak" → "<original>"
        per_file[orig] = per_file.get(orig, 0) + 1
        if i >= keep or per_file[orig] > keep_per_file:
            with contextlib.suppress(Exception):
                f.unlink()


def _backup_file(p: Path) -> None:
    """Internal helper: backup file.

    /undo restores from the in-memory _main._backup_stack, not this file, so a
    filename collision here is invisible during normal use — but the on-disk
    .bak is the only recovery path left after a crash (the stack doesn't
    survive one), so a second same-file edit within the same wall-clock
    second must not silently overwrite the first edit's backup.
    """
    import qwen_cli.main as _main

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = _main.BACKUPS_DIR / f"{p.name}.{stamp}.bak"
    if backup.exists():
        n = 1
        while (_main.BACKUPS_DIR / f"{p.name}.{stamp}_{n}.bak").exists():
            n += 1
        backup = _main.BACKUPS_DIR / f"{p.name}.{stamp}_{n}.bak"
    # Byte-faithful: the backup (and the /undo stack entry fed from it) is the
    # only recovery copy, so it must not have its line endings rewritten either.
    content = _read_raw(p)
    _write_raw(backup, content)
    _main._backup_stack.append({"original": p, "backup": backup, "content": content})
    if len(_main._backup_stack) > _main._MAX_BACKUP_STACK:
        _main._backup_stack.pop(0)
    _cleanup_backups()


def do_write_file(path: str, content: str) -> str:
    """Handle write file operation."""
    import qwen_cli.main as _main

    try:
        p = _main._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        # An overwrite keeps the file's existing line-ending style; a new file is
        # written exactly as the model produced it. See _write_raw.
        uses_crlf = False
        content = content.replace("\r\n", "\n")
        if existed:
            raw_old = _read_raw(p)
            uses_crlf = "\r\n" in raw_old
            old = raw_old.replace("\r\n", "\n")
            if str(p) not in _main._session_changes:
                _main._session_changes[str(p)] = raw_old
            if old == content:
                return f"[no changes: {p}]"
            _main._backup_file(p)
            diff = list(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{p.name}",
                    tofile=f"b/{p.name}",
                    lineterm="",
                )
            )
            preview = "".join(diff[:80])
            if len(diff) > 80:
                preview += f"\n... ({len(diff) - 80} more diff lines)"
            _main.console.print(Syntax(preview, "diff", theme="monokai"))
            if not _main._confirm_action("Overwrite?"):
                return "[write cancelled by user]"
            _lsp_pre_edit_snapshot(p)
        _write_raw(p, _restore_endings(content, uses_crlf))
        action = "updated" if existed else "created"
        _main.console.print(f"[bold yellow]  [write_file][/bold yellow] {action}: {p}")
        _main._lsp_post_edit_report(p)
        return f"[{action}: {p}  ({len(content):,} chars)]"
    except Exception as e:
        return f"[error: {e}]"


def do_move_file(src: str, dst: str) -> str:
    """Handle move file operation."""
    import qwen_cli.main as _main

    try:
        s = _main._resolve(src)
        d = _main._resolve(dst)
        if not s.exists():
            return f"[not found: {s}]"
        if d.is_dir():
            d = d / s.name
        if d.exists():
            if not _main._confirm_action(f"{d} already exists. Overwrite?"):
                return "[move cancelled by user]"
            _main._backup_file(d)
        d.parent.mkdir(parents=True, exist_ok=True)
        s.replace(d)  # replace() overwrites dst atomically on Windows; rename() does not
        _main.console.print(f"[bold yellow]  [move_file][/bold yellow] {s} → {d}")
        return f"[moved: {s} → {d}]"
    except Exception as e:
        return f"[error: {e}]"


def do_delete_file(path: str) -> str:
    """Handle delete file operation."""
    import qwen_cli.main as _main

    try:
        p = _main._resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        if p.is_dir():
            return f"[{p} is a directory — use run_command to delete directories]"
        _main.console.print(
            f"[bold red]  [delete_file][/bold red] {p}  ({_fmt_size(p.stat().st_size)})",
        )
        if not _main._confirm_action("Permanently delete?"):
            return "[delete cancelled by user]"
        _main._backup_file(p)
        p.unlink()
        return f"[deleted: {p}]"
    except Exception as e:
        return f"[error: {e}]"


