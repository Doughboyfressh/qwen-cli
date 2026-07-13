"""Persistent memory (memory.md) and pins — load/save, cap, extraction, curation.

Extracted from main.py as part of the monolith split. This module owns the
memory/pins state (_memory_lock, _auto_memory_count, size caps); other modules
reach it through main's re-exports (`_main.load_memory`, ...).

Cross-subsystem calls (aux LLM, session-change tracking, console) go through a
lazy `import qwen_cli.main as _main` at call time — the project convention for
split-out modules (see core/repl.py, core/intel.py). Tests that monkeypatch
`qwen_cli.main._bg_llm`, `_session_changes`, etc. keep working unchanged.
"""

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from qwen_cli.core.config import AUX_LLM_TIMEOUT, MEMORY_FILE, PINS_FILE

_logger = logging.getLogger(__name__)

MEMORY_CURATE_INTERVAL = 10  # consolidate memory.md every N auto-extractions
# _curate_memory()'s periodic LLM-based consolidation is a *soft* mechanism —
# it depends on the model actually compressing well, and only runs every
# MEMORY_CURATE_INTERVAL auto-extractions, not on /remember at all. This is
# the deterministic backstop so memory.md can never grow unbounded regardless.
MEMORY_MAX_CHARS = 8000

_memory_lock = threading.Lock()  # serializes all memory.md read-modify-write
_auto_memory_count: int = 0


# ---------------------------------------------------------------------------
# Pins (persistent per-session reminders injected into every system prompt)
# ---------------------------------------------------------------------------


def load_pins() -> list[str]:
    """Load Pins."""
    if not PINS_FILE.exists():
        return []
    try:
        return json.loads(PINS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_pins(pins: list[str]) -> None:
    """Save Pins."""
    PINS_FILE.write_text(json.dumps(pins, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# memory.md load/save
# ---------------------------------------------------------------------------


def load_memory() -> str:
    """Load Memory."""
    return MEMORY_FILE.read_text(encoding="utf-8").strip() if MEMORY_FILE.exists() else ""


def _enforce_memory_cap(text: str, max_chars: int = MEMORY_MAX_CHARS) -> str:
    """Hard cap on memory.md size — drops the OLDEST entries first (they're
    also the most likely to be superseded), preserving whole entries rather
    than truncating mid-fact. Entries are blank-line-separated blocks; always
    keeps at least one even if it alone exceeds the cap, so a single large
    fact can't wipe out memory entirely.
    """
    if len(text) <= max_chars:
        return text
    blocks = [b for b in text.split("\n\n") if b.strip()]
    kept: list[str] = []
    total = 0
    for block in reversed(blocks):  # walk newest-first, drop oldest once over cap
        block_len = len(block) + 2  # +2 for the "\n\n" separator
        if total + block_len > max_chars and kept:
            break
        kept.append(block)
        total += block_len
    kept.reverse()
    return "\n\n".join(kept)


def save_memory(text: str) -> None:
    """Save Memory."""
    # max_chars passed explicitly (not relying on _enforce_memory_cap's own
    # default) so this always reads the current MEMORY_MAX_CHARS at call
    # time — a default parameter value is bound once at def-time and would
    # never see a later change to the module-level constant.
    MEMORY_FILE.write_text(_enforce_memory_cap(text, max_chars=MEMORY_MAX_CHARS), encoding="utf-8")


# ---------------------------------------------------------------------------
# Automatic extraction, session-change logging, curation
# ---------------------------------------------------------------------------


def record_session_changes_memory(client: object | None = None) -> None:
    """Session-end hook: if files were modified this session, log a dated entry to memory.md.

    Always writes a deterministic entry (file list). If the model is reachable, the
    entry is upgraded to a one-line summary of what actually changed, built from the
    real diffs (original content is kept in _session_changes).
    """
    import qwen_cli.main as _main

    if not _main._session_changes:
        return
    names = sorted({Path(p).name for p in _main._session_changes})
    shown = ", ".join(names[:12]) + (f" (+{len(names) - 12} more)" if len(names) > 12 else "")
    date = datetime.now().strftime("%Y-%m-%d")
    line = f"- {date}: modified {shown}"

    if client is not None:
        try:
            import difflib

            chunks = []
            for path_str, original in list(_main._session_changes.items())[:8]:
                try:
                    current = Path(path_str).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                diff = "\n".join(
                    difflib.unified_diff(
                        original.splitlines(),
                        current.splitlines(),
                        fromfile=Path(path_str).name,
                        tofile=Path(path_str).name,
                        lineterm="",
                        n=1,
                    )
                )
                if diff:
                    chunks.append(diff[:1200])
            if chunks:
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Summarize these code diffs in ONE line under 120 characters: "
                            "what changed, past tense, no filler. Reply with only the line."
                        ),
                    },
                    {"role": "user", "content": "\n\n".join(chunks)[:6000]},
                ]
                bg_client, bg_model = _main._bg_llm(client)
                resp = bg_client.chat.completions.create(
                    model=bg_model, messages=prompt, stream=False, max_tokens=80, timeout=AUX_LLM_TIMEOUT
                )
                summary = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
                if 10 < len(summary) <= 200:
                    line = f"- {date}: {summary} (files: {shown})"
        except Exception:
            _logger.debug("Session-change summary via model failed; using file list")

    marker = "# Recent Changes"
    with _memory_lock:
        mem = load_memory()
        # Dedup: skip if an entry for the same date and file set is already logged
        # (prevents repeated identical entries from tool-test or short sessions).
        prefix = f"- {date}:"
        for existing in mem.splitlines():
            if existing.startswith(prefix) and shown in existing:
                return
        if marker in mem:
            idx = mem.index(marker) + len(marker)
            mem = mem[:idx] + "\n\n" + line + mem[idx:]
        else:
            mem = (mem + f"\n\n{marker}\n\n{line}").strip()
        save_memory(mem.strip())
    _main.console.print(f"[dim][memory: logged session changes — {shown}][/dim]")


# Lines that are just the extractor saying "nothing here" — observed written into
# memory.md as '- SKIP', 'SKIP', '- No', bulleted/bolded variants, etc. The old
# check (facts.upper() != "SKIP") only caught the bare, unbulleted form.
_MEMORY_JUNK_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*)?(?:skip|none|n/?a|no|nothing)(?:\*\*)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
# A memory "fact" containing tool-call or thinking syntax is a malformed model
# reply captured wholesale, not a fact — a live memory.md had a full
# <tool_call><function=browser_action>... block saved as memory.
_MEMORY_TOOL_SYNTAX_RE = re.compile(r"<tool_call>|</?think>|<function[=>]|<parameter[=>]", re.IGNORECASE)
_MEMORY_NEGATION_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?:no|none|nothing)\b", re.IGNORECASE)


def _clean_memory_facts(facts: str, drop_negations: bool = False) -> str:
    """Sanitize an LLM-extracted memory entry before it is persisted.

    Drops sentinel/no-op lines, rejects entries containing tool-call syntax,
    and (for intel entries) drops pure-negation lines ("No critical CVEs
    detected...") that record the absence of information. Returns the cleaned
    entry, or "" if nothing durable remains.
    """
    if not facts or _MEMORY_TOOL_SYNTAX_RE.search(facts):
        return ""
    kept = []
    for line in facts.splitlines():
        if _MEMORY_JUNK_LINE_RE.match(line):
            continue
        if drop_negations and _MEMORY_NEGATION_RE.match(line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    return cleaned if len(cleaned) > 10 else ""


def _auto_extract_memory(client, user_msg: str, assistant_msg: str) -> None:
    """Pull memorable facts from this exchange and append to memory.md."""
    import qwen_cli.main as _main

    global _auto_memory_count
    with _main._main_llm_busy_lock:
        # With an aux backend this work doesn't touch the main model's slot,
        # so there's no reason to skip it while the main model is busy.
        if _main._main_llm_busy and _main._aux_client is None:
            return
    try:
        excerpt = f"User: {user_msg[:600]}\n\nAssistant: {assistant_msg[:1000]}"
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract facts worth remembering long-term from this exchange: "
                    "concrete preferences, decisions, names, configs, or constraints. "
                    "Skip pleasantries and transient state. "
                    "Format as short bullet points. "
                    "If nothing is notable, reply with exactly: NONE"
                ),
            },
            {"role": "user", "content": excerpt},
        ]
        bg_client, bg_model = _main._bg_llm(client)
        resp = bg_client.chat.completions.create(
            model=bg_model,
            messages=prompt,
            stream=False,
            max_tokens=200,
            timeout=AUX_LLM_TIMEOUT,
        )
        facts = _clean_memory_facts((resp.choices[0].message.content or "").strip())
        if facts:
            with _memory_lock:
                mem = load_memory()
                entry = f"\n\n<!-- auto {datetime.now().strftime('%Y-%m-%d')} -->\n{facts}"
                save_memory((mem + entry).strip())
                _auto_memory_count += 1
                if _auto_memory_count % MEMORY_CURATE_INTERVAL == 0:
                    _curate_memory(client, locked=True)
    except Exception:
        _logger.debug("Auto-memory extraction failed")


def _curate_memory(client: object, locked: bool = False) -> None:
    """Background: deduplicate and consolidate memory.md when it grows noisy."""
    import qwen_cli.main as _main

    if not locked:
        lock_ctx = _memory_lock
    else:
        import contextlib

        lock_ctx = contextlib.nullcontext()
    with lock_ctx:
        mem = load_memory()
        if not mem or len(mem) < 400:
            return
        try:
            prompt = [
                {
                    "role": "system",
                    "content": "Consolidate this memory into clean, non-redundant bullet points. Remove duplicates. Preserve all unique facts. Be concise.",
                },
                {"role": "user", "content": mem},
            ]
            bg_client, bg_model = _main._bg_llm(client)
            resp = bg_client.chat.completions.create(
                model=bg_model,
                messages=prompt,
                stream=False,
                max_tokens=600,
                timeout=AUX_LLM_TIMEOUT,
            )
            consolidated = (resp.choices[0].message.content or "").strip()
            if consolidated and len(consolidated) > 20:
                save_memory(consolidated)
        except Exception:
            _logger.debug("Memory curation failed")
