"""Tests for file-operation tool implementations: do_write_file, do_patch_file,
do_delete_file, do_move_file.

These run against real temp files so the path-resolution and backup logic is
exercised without touching the user's actual workspace.
"""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_confirm(monkeypatch, qwen_cli, answer="y"):
    """Patch console.input so confirmation prompts auto-answer."""
    monkeypatch.setattr(qwen_cli.console, "input", lambda _prompt: answer)


def _patch_backups(monkeypatch, qwen_cli, tmp_path):
    """Redirect BACKUPS_DIR to tmp_path (already exists — no mkdir needed)."""
    monkeypatch.setattr(qwen_cli, "BACKUPS_DIR", tmp_path)


# ---------------------------------------------------------------------------
# do_write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_creates_new_file(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "new.txt"
        result = qwen_cli.do_write_file(str(p), "hello")
        assert "[created:" in result
        assert p.read_text(encoding="utf-8") == "hello"

    def test_overwrites_existing_with_confirmation(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "existing.txt"
        p.write_text("old content", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        result = qwen_cli.do_write_file(str(p), "new content")
        assert "[updated:" in result
        assert p.read_text(encoding="utf-8") == "new content"

    def test_cancels_on_no(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "existing.txt"
        p.write_text("old content", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "n")
        result = qwen_cli.do_write_file(str(p), "new content")
        assert "cancelled" in result
        assert p.read_text(encoding="utf-8") == "old content"

    def test_noop_when_content_identical(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "same.txt"
        p.write_text("unchanged", encoding="utf-8")
        result = qwen_cli.do_write_file(str(p), "unchanged")
        assert "no changes" in result

    def test_creates_parent_dirs(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "deep" / "dir" / "file.txt"
        result = qwen_cli.do_write_file(str(p), "nested")
        assert "[created:" in result
        assert p.exists()


# ---------------------------------------------------------------------------
# do_patch_file
# ---------------------------------------------------------------------------


class TestPatchFile:
    def test_applies_simple_diff(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        p.write_text("alpha\nbravo\ncharlie\n", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        diff = "--- a/code.txt\n+++ b/code.txt\n@@ -2,1 +2,1 @@\n-bravo\n+BRAVO\n"
        result = qwen_cli.do_patch_file(str(p), diff)
        assert "[patched:" in result
        assert p.read_text(encoding="utf-8") == "alpha\nBRAVO\ncharlie\n"

    def test_cancelled_patch_leaves_file_unchanged(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        original = "alpha\nbravo\ncharlie\n"
        p.write_text(original, encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "n")
        diff = "--- a/code.txt\n+++ b/code.txt\n@@ -2,1 +2,1 @@\n-bravo\n+BRAVO\n"
        result = qwen_cli.do_patch_file(str(p), diff)
        assert "cancelled" in result
        assert p.read_text(encoding="utf-8") == original

    def test_mismatched_hunk_returns_error(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        p.write_text("alpha\nbravo\ncharlie\n", encoding="utf-8")
        bad_diff = "--- a/code.txt\n+++ b/code.txt\n@@ -2,1 +2,1 @@\n-NOTHERE\n+X\n"
        result = qwen_cli.do_patch_file(str(p), bad_diff)
        assert "[patch failed:" in result

    def test_nonexistent_file_returns_error(self, qwen_cli, tmp_path):
        result = qwen_cli.do_patch_file(str(tmp_path / "ghost.txt"), "@@ -1 +1 @@\n-x\n+y\n")
        assert "not found" in result

    def test_backup_uses_shared_helper_and_records_original_content(self, qwen_cli, tmp_path, monkeypatch):
        # do_patch_file used to duplicate _backup_file()'s logic inline instead
        # of calling it — verify it now goes through the shared path.
        p = tmp_path / "code.txt"
        p.write_text("alpha\n", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        monkeypatch.setattr(qwen_cli, "_backup_stack", [])
        diff = "--- a/code.txt\n+++ b/code.txt\n@@ -1,1 +1,1 @@\n-alpha\n+beta\n"
        qwen_cli.do_patch_file(str(p), diff)
        assert len(qwen_cli._backup_stack) == 1
        assert qwen_cli._backup_stack[0]["content"] == "alpha\n"

    def test_backups_do_not_collide_within_same_second(self, qwen_cli, tmp_path, monkeypatch):
        # Two patches to the same file in the same wall-clock second used to
        # overwrite each other's on-disk backup — the only recovery path left
        # if the in-memory _backup_stack doesn't survive a crash.
        class _FrozenDatetime:
            @staticmethod
            def now():
                from datetime import datetime as _dt

                return _dt(2026, 1, 1, 12, 0, 0)

        p = tmp_path / "code.txt"
        p.write_text("v1\n", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        monkeypatch.setattr(qwen_cli, "_backup_stack", [])
        monkeypatch.setattr(qwen_cli, "datetime", _FrozenDatetime)

        diff1 = "--- a/code.txt\n+++ b/code.txt\n@@ -1,1 +1,1 @@\n-v1\n+v2\n"
        diff2 = "--- a/code.txt\n+++ b/code.txt\n@@ -1,1 +1,1 @@\n-v2\n+v3\n"
        qwen_cli.do_patch_file(str(p), diff1)
        qwen_cli.do_patch_file(str(p), diff2)

        backups = sorted(tmp_path.glob("code.txt.*.bak"))
        assert len(backups) == 2
        contents = {b.read_text(encoding="utf-8") for b in backups}
        assert contents == {"v1\n", "v2\n"}


# ---------------------------------------------------------------------------
# do_delete_file
# ---------------------------------------------------------------------------


class TestDeleteFile:
    def test_deletes_with_confirmation(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "todelete.txt"
        p.write_text("bye", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        result = qwen_cli.do_delete_file(str(p))
        assert "[deleted:" in result
        assert not p.exists()

    def test_cancels_on_no(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "todelete.txt"
        p.write_text("still here", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "n")
        result = qwen_cli.do_delete_file(str(p))
        assert "cancelled" in result
        assert p.exists()

    def test_nonexistent_returns_error(self, qwen_cli, tmp_path):
        result = qwen_cli.do_delete_file(str(tmp_path / "nope.txt"))
        assert "not found" in result

    def test_directory_is_rejected(self, qwen_cli, tmp_path):
        result = qwen_cli.do_delete_file(str(tmp_path))
        assert "directory" in result.lower()


# ---------------------------------------------------------------------------
# do_move_file
# ---------------------------------------------------------------------------


class TestMoveFile:
    def test_renames_file(self, qwen_cli, tmp_path, monkeypatch):
        src = tmp_path / "a.txt"
        dst = tmp_path / "b.txt"
        src.write_text("content", encoding="utf-8")
        result = qwen_cli.do_move_file(str(src), str(dst))
        assert "[moved:" in result
        assert not src.exists()
        assert dst.read_text(encoding="utf-8") == "content"

    def test_moves_into_directory(self, qwen_cli, tmp_path, monkeypatch):
        src = tmp_path / "file.txt"
        dst_dir = tmp_path / "subdir"
        dst_dir.mkdir()
        src.write_text("hi", encoding="utf-8")
        result = qwen_cli.do_move_file(str(src), str(dst_dir))
        assert "[moved:" in result
        assert (dst_dir / "file.txt").read_text(encoding="utf-8") == "hi"

    def test_nonexistent_source_returns_error(self, qwen_cli, tmp_path):
        result = qwen_cli.do_move_file(str(tmp_path / "ghost.txt"), str(tmp_path / "x.txt"))
        assert "not found" in result

    def test_overwrite_existing_dst_with_confirmation(self, qwen_cli, tmp_path, monkeypatch):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("new", encoding="utf-8")
        dst.write_text("old", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        result = qwen_cli.do_move_file(str(src), str(dst))
        assert "[moved:" in result
        assert dst.read_text(encoding="utf-8") == "new"


class TestProjectContextBudget:
    """load_project_context caps total key-file content at ~25% of the
    context window — a live session opened at 68% context used, one turn
    from auto-trim, before the user typed anything."""

    def test_key_files_respect_total_budget(self, qwen_cli, tmp_path, monkeypatch):
        # Three key files of 12k chars each would blow a small budget.
        for name in ("README.md", "package.json", "pyproject.toml"):
            (tmp_path / name).write_text("x" * 12_000, encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 8_000)  # budget: 8k chars
        history = []
        assert qwen_cli.load_project_context(str(tmp_path), history) is True
        content = history[-1]["content"]
        # All content (tree + key files) must be far below the uncapped 36k.
        assert len(content) < 12_000
        assert "Not inlined" in content  # skipped files are still named

    def test_small_project_is_untouched(self, qwen_cli, tmp_path):
        (tmp_path / "README.md").write_text("hello project", encoding="utf-8")
        history = []
        assert qwen_cli.load_project_context(str(tmp_path), history) is True
        assert "hello project" in history[-1]["content"]
        assert "Not inlined" not in history[-1]["content"]
