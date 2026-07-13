"""Session persistence — save/load, autosave, exit/overflow handoffs, export.

Extracted from main.py as part of the monolith split. Command handlers that
*use* sessions (cmd_load_session, cmd_search_sessions, /save, ...) stay in
main/commands; this module owns the storage format and the handoff protocol.

Cross-subsystem calls (console, the live session title, streaming for handoff
summaries) go through a lazy `import qwen_cli.main as _main` at call time —
the project convention for split-out modules (see core/repl.py, core/intel.py,
core/memory.py).

Handoff producers: _save_exit_handoff covers normal exits;
_generate_handoff + _write_handoff cover context overflow (called from
_maybe_autocompact before history is rewritten), including the `next_step`
field that _consume_handoff carries forward.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from qwen_cli.core.config import AUTOSAVE_FILE, EXPORTS_DIR, HANDOFF_FILE, SESSIONS_DIR

_logger = logging.getLogger(__name__)


def _session_meta(path: Path) -> dict:
    """Internal helper: session meta."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        if not meta:
            history = data.get("history", [])
            user_msgs = [
                m
                for m in history
                if m.get("role") == "user" and not (m.get("content") or "").startswith("# Project Context")
            ]
            topic = (user_msgs[0].get("content") or "")[:60].replace("\n", " ") if user_msgs else ""
            turns = sum(1 for m in history if m.get("role") == "assistant")
            meta = {"topic": topic, "turns": turns, "saved_at": ""}
        return meta
    except Exception:
        return {"topic": "", "turns": 0, "saved_at": ""}


def save_session(history: list, system_prompt: str, name: str | None = None) -> None:
    """Save Session."""
    import qwen_cli.main as _main

    user_msgs = [
        m for m in history if m.get("role") == "user" and not (m.get("content") or "").startswith("# Project Context")
    ]
    topic = (user_msgs[0].get("content") or "")[:80].replace("\n", " ") if user_msgs else ""
    turns = sum(1 for m in history if m.get("role") == "assistant")
    fname = (name or datetime.now().strftime("%Y%m%d_%H%M%S")) + ".json"
    path = SESSIONS_DIR / fname
    data = {
        "meta": {"topic": topic, "turns": turns, "saved_at": datetime.now().isoformat()},
        "system_prompt": system_prompt,
        "history": history,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _main.console.print(f"[dim][saved: {path.name}  ({turns} turn{'s' if turns != 1 else ''})][/dim]")


def _silent_autosave(history: list, system_prompt: str) -> None:
    """Rolling checkpoint to autosave.json — no console output, never raises."""
    import qwen_cli.main as _main

    try:
        user_msgs = [
            m
            for m in history
            if m.get("role") == "user" and not (m.get("content") or "").startswith("# Project Context")
        ]
        raw_topic = (user_msgs[0].get("content") or "")[:80].replace("\n", " ") if user_msgs else ""
        topic = _main._session_title or raw_topic
        turns = sum(1 for m in history if m.get("role") == "assistant")
        data = {
            "meta": {"topic": topic, "turns": turns, "saved_at": datetime.now().isoformat()},
            "system_prompt": system_prompt,
            "history": history,
        }
        AUTOSAVE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        _logger.exception("Autosave failed — session data may be lost on crash")


HANDOFF_PROMPT_TEMPLATE = """You are resuming a previous session (it ended via context overflow, a normal exit, \
or a crash). Here is what happened:

{summary}

Use this context to pick up where you left off. Do NOT ask the user to repeat themselves. Check memory.md for additional persistent facts."""


def _generate_handoff(client, history, base_system) -> str:
    """Generate a compact handoff summary for session resumption."""
    import qwen_cli.main as _main

    try:
        chat = [m for m in history if m.get("role") in ("user", "assistant")]
        recent = chat[-16:]
        recent_text = "\n".join(f"{m['role'].upper()}: {(m.get('content') or '')[:800]}" for m in recent)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a session summarizer. Produce a handoff document for an AI assistant "
                    "resuming this conversation. Include: (1) what the session goal is, "
                    "(2) what has been accomplished, (3) what files were created/modified, "
                    "(4) the exact task being worked on and the immediate next step, "
                    "(5) any open questions or blockers. Be precise. Keep it under 800 words."
                ),
            },
            {"role": "user", "content": f"Summarize this session for handoff:\n\n{recent_text}"},
        ]
        summary, _, _ = _main.stream_once(client, prompt, use_tools=False)
        return summary or "(summarization failed - see autosave for raw history)"
    except Exception as e:
        return f"(handoff summary failed: {e})"


def _write_handoff(summary, history, next_step: str = "") -> None:
    """Write a handoff file for the next session to pick up."""
    try:
        chat = [m for m in history if m.get("role") in ("user", "assistant")]
        last_user = ""
        for m in reversed(chat):
            if m.get("role") == "user":
                last_user = m.get("content", "")[:400]
                break
        data = {
            "summary": summary,
            "last_user_message": last_user,
            "turns": sum(1 for m in history if m.get("role") == "assistant"),
            "timestamp": datetime.now().isoformat(),
        }
        if next_step:
            data["next_step"] = next_step[:400]
        HANDOFF_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        _logger.exception("Overflow handoff write failed — session resumption may be degraded")


def _save_exit_handoff(history) -> None:
    """Save a lightweight handoff on normal exit so next session has context."""
    try:
        chat = [m for m in history if m.get("role") in ("user", "assistant")]
        last_user = ""
        for m in reversed(chat):
            if m.get("role") == "user":
                last_user = m.get("content", "")[:400]
                break
        last_assistant = ""
        for m in reversed(chat):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "")[:600]
                break
        data = {
            "summary": (
                f"Session had {sum(1 for m in history if m.get('role') == 'assistant')} turns. "
                f"Last user: {last_user[:200]}. Last assistant: {last_assistant[:200]}"
            ),
            "last_user_message": last_user,
            "last_assistant_message": last_assistant,
            "turns": sum(1 for m in history if m.get("role") == "assistant"),
            "timestamp": datetime.now().isoformat(),
        }
        HANDOFF_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        _logger.exception("Exit handoff write failed — next session will lack context")


def _consume_handoff() -> dict | None:
    """Read and remove the handoff file. Returns dict or None."""
    try:
        if not HANDOFF_FILE.exists():
            return None
        data = json.loads(HANDOFF_FILE.read_text(encoding="utf-8"))
        summary = data.get("summary", "")
        last_user = data.get("last_user_message", "")
        turns = data.get("turns", 0)
        # Also carry forward any explicit next_step from overflow handoffs
        next_step = data.get("next_step", "")
        HANDOFF_FILE.unlink(missing_ok=True)
        if not summary:
            return None
        prompt = HANDOFF_PROMPT_TEMPLATE.format(summary=summary)
        if next_step:
            prompt += f"\n\nThe immediate next step was: {next_step}"
        return {"prompt": prompt, "last_user": last_user, "turns": turns, "next_step": next_step}
    except Exception:
        return None


def load_session(name: str) -> tuple[list | None, str | None]:
    """Load Session."""
    if not name.endswith(".json"):
        name += ".json"
    path = SESSIONS_DIR / name
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("history", []), data.get("system_prompt", "")


def _fuzzy_find_session(query: str) -> list[Path]:
    """Internal helper: fuzzy find session."""
    q = query.lower().removesuffix(".json")
    matches: list[Path] = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        if q in f.stem.lower():
            matches.append(f)
            continue
        if q in _session_meta(f).get("topic", "").lower():
            matches.append(f)
    return matches


def list_sessions() -> None:
    """List Sessions."""
    import qwen_cli.main as _main

    from rich.table import Table

    files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    if not files:
        _main.console.print("[dim][no saved sessions][/dim]")
        return
    t = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    t.add_column("Name", style="cyan", no_wrap=True)
    t.add_column("Turns", justify="right", style="dim")
    t.add_column("Topic", style="white")
    t.add_column("Saved", style="dim", no_wrap=True)
    for f in files:
        meta = _session_meta(f)
        saved = meta.get("saved_at", "")[:16].replace("T", " ") if meta.get("saved_at") else ""
        t.add_row(f.stem, str(meta.get("turns", "?")), meta.get("topic", "")[:60], saved)
    _main.console.print(t)


def export_session(history: list, name: str = "") -> None:
    """Export Session."""
    import qwen_cli.main as _main

    fname = (name or datetime.now().strftime("%Y%m%d_%H%M%S")) + ".md"
    path = EXPORTS_DIR / fname
    lines = [
        "# Qwen Chat Export\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        f"**Model:** {_main.MODEL}\n\n---\n",
    ]
    for msg in history:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role == "user" and not content.startswith("# Project Context"):
            lines.append(f"\n**You:** {content}\n")
        elif role == "assistant":
            lines.append(f"\n**Qwen:**\n\n{content}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    _main.console.print(f"[green][exported: {path}][/green]")
