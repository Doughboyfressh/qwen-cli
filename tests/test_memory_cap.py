"""Tests for memory.md's hard size cap.

_curate_memory() consolidates memory.md via an LLM call every
MEMORY_CURATE_INTERVAL auto-extractions, but that's a *soft* mechanism: it
depends on the model actually compressing well, and doesn't run at all for
/remember (which bypasses the auto-extraction counter entirely). These tests
cover the deterministic backstop, _enforce_memory_cap(), plus its wiring
into save_memory().
"""

from types import SimpleNamespace
from unittest.mock import patch


def _block(marker: str, size: int) -> str:
    """Build a single memory "block" (entry) of roughly the given size."""
    return f"<!-- {marker} -->\n" + ("x" * size)


class TestEnforceMemoryCap:
    def test_under_cap_is_unchanged(self, qwen_cli):
        text = "short memory content"
        assert qwen_cli._enforce_memory_cap(text, max_chars=100) == text

    def test_over_cap_drops_oldest_entries(self, qwen_cli):
        oldest = _block("entry1", 50)
        middle = _block("entry2", 50)
        newest = _block("entry3", 50)
        text = "\n\n".join([oldest, middle, newest])

        # Cap small enough that only the most recent entry survives.
        result = qwen_cli._enforce_memory_cap(text, max_chars=70)

        assert "entry3" in result
        assert "entry1" not in result
        assert "entry2" not in result

    def test_keeps_at_least_one_entry_even_if_oversized(self, qwen_cli):
        huge = _block("entry1", 500)

        result = qwen_cli._enforce_memory_cap(huge, max_chars=10)

        assert "entry1" in result  # never wiped out entirely

    def test_keeps_most_recent_entries_in_original_order(self, qwen_cli):
        entries = [_block(f"entry{i}", 30) for i in range(5)]
        text = "\n\n".join(entries)

        result = qwen_cli._enforce_memory_cap(text, max_chars=100)

        # Whichever entries survive, they must stay in chronological order
        # (oldest-kept first, newest last) — not reversed or shuffled.
        kept_indices = [i for i in range(5) if f"entry{i}" in result]
        assert kept_indices == sorted(kept_indices)
        assert kept_indices[-1] == 4  # newest always survives

    def test_empty_text_stays_empty(self, qwen_cli):
        assert qwen_cli._enforce_memory_cap("", max_chars=100) == ""


class TestSaveMemoryEnforcesCapEndToEnd:
    def test_save_memory_applies_the_cap(self, qwen_cli, tmp_path, monkeypatch):
        import qwen_cli.core.memory as memory

        monkeypatch.setattr(memory, "MEMORY_FILE", tmp_path / "memory.md")
        monkeypatch.setattr(memory, "MEMORY_MAX_CHARS", 300)
        entries = [_block(f"entry{i}", 200) for i in range(10)]
        text = "\n\n".join(entries)

        qwen_cli.save_memory(text)

        saved = qwen_cli.load_memory()
        assert len(saved) <= 300 + 50  # allow slack for the "keep at least one entry" rule
        assert "entry9" in saved  # newest survives
        assert "entry0" not in saved  # oldest dropped

    def test_remember_facts_are_individually_droppable(self, qwen_cli, tmp_path, monkeypatch):
        """/remember's blank-line separation (fixed alongside the cap) means
        the cap can drop individual remembered facts instead of lumping every
        /remember call into one block that's either kept or dropped whole."""
        import qwen_cli.core.memory as memory
        from qwen_cli.core.commands import _cmd_remember

        monkeypatch.setattr(memory, "MEMORY_FILE", tmp_path / "memory.md")
        ctx = SimpleNamespace()

        with patch.object(qwen_cli.console, "print"):
            for i in range(5):
                _cmd_remember(ctx, f"fact number {i}")

        mem = qwen_cli.load_memory()
        assert mem.count("\n\n") == 4  # 5 facts joined by 4 blank-line separators
