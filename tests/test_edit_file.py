"""Tests for do_edit_file (exact-string replacement), the _confirm_action
auto-approve gate, and the prefix-stable system prompt snapshot."""


def _auto_confirm(monkeypatch, qwen_cli, answer="y"):
    """Patch console.input so confirmation prompts auto-answer."""
    monkeypatch.setattr(qwen_cli.console, "input", lambda _prompt: answer)


def _patch_backups(monkeypatch, qwen_cli, tmp_path):
    monkeypatch.setattr(qwen_cli, "BACKUPS_DIR", tmp_path)


# ---------------------------------------------------------------------------
# do_edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_single_replacement(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        p.write_text("def f():\n    return 1\n", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        result = qwen_cli.do_edit_file(str(p), "return 1", "return 2")
        assert "[edited:" in result
        assert p.read_text(encoding="utf-8") == "def f():\n    return 2\n"

    def test_old_string_not_found(self, qwen_cli, tmp_path):
        p = tmp_path / "code.txt"
        p.write_text("alpha\n", encoding="utf-8")
        result = qwen_cli.do_edit_file(str(p), "NOTHERE", "x")
        assert "not found" in result
        assert p.read_text(encoding="utf-8") == "alpha\n"

    def test_ambiguous_without_replace_all(self, qwen_cli, tmp_path):
        p = tmp_path / "code.txt"
        p.write_text("x = 1\nx = 1\n", encoding="utf-8")
        result = qwen_cli.do_edit_file(str(p), "x = 1", "x = 2")
        assert "occurs 2 times" in result
        assert p.read_text(encoding="utf-8") == "x = 1\nx = 1\n"

    def test_replace_all(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        p.write_text("x = 1\nx = 1\n", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        result = qwen_cli.do_edit_file(str(p), "x = 1", "x = 2", replace_all=True)
        assert "2 replacement(s)" in result
        assert p.read_text(encoding="utf-8") == "x = 2\nx = 2\n"

    def test_identical_strings_rejected(self, qwen_cli, tmp_path):
        p = tmp_path / "code.txt"
        p.write_text("alpha\n", encoding="utf-8")
        result = qwen_cli.do_edit_file(str(p), "alpha", "alpha")
        assert "identical" in result

    def test_empty_old_string_rejected(self, qwen_cli, tmp_path):
        p = tmp_path / "code.txt"
        p.write_text("alpha\n", encoding="utf-8")
        result = qwen_cli.do_edit_file(str(p), "", "beta")
        assert "empty" in result

    def test_missing_file(self, qwen_cli, tmp_path):
        result = qwen_cli.do_edit_file(str(tmp_path / "ghost.py"), "a", "b")
        assert "not found" in result

    def test_cancelled_leaves_file_unchanged(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        p.write_text("alpha\n", encoding="utf-8")
        _auto_confirm(monkeypatch, qwen_cli, "n")
        result = qwen_cli.do_edit_file(str(p), "alpha", "beta")
        assert "cancelled" in result
        assert p.read_text(encoding="utf-8") == "alpha\n"

    def test_whitespace_must_match_exactly(self, qwen_cli, tmp_path):
        p = tmp_path / "code.txt"
        p.write_text("    indented\n", encoding="utf-8")
        result = qwen_cli.do_edit_file(str(p), "indented\n ", "x")
        assert "not found" in result

    def test_records_session_change_and_backup(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        p.write_text("v1\n", encoding="utf-8")
        original = p.read_bytes()  # whatever endings this platform actually wrote
        _auto_confirm(monkeypatch, qwen_cli, "y")
        _patch_backups(monkeypatch, qwen_cli, tmp_path)
        monkeypatch.setattr(qwen_cli, "_session_changes", {})
        qwen_cli.do_edit_file(str(p), "v1", "v2")
        # The recorded original is byte-faithful, not newline-normalized: /rollback
        # writes it straight back, so a normalized copy would itself convert the file.
        assert qwen_cli._session_changes[str(p)].encode() == original
        assert list(tmp_path.glob("code.txt.*.bak"))

    def test_registered_as_tool(self, qwen_cli):
        from qwen_cli.core.stream import TOOLS

        names = {t["function"]["name"] for t in TOOLS}
        assert "edit_file" in names
        assert "edit_file" in qwen_cli._TOOL_HANDLERS_INTERACTIVE
        assert "edit_file" in qwen_cli._STATEFUL_TOOLS
        assert "edit_file" in qwen_cli._MUTATING_FILE_TOOLS


# ---------------------------------------------------------------------------
# _confirm_action / auto-approve
# ---------------------------------------------------------------------------


class TestAutoApprove:
    def test_auto_approve_skips_prompt(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "_auto_approve", True)

        def _boom(_prompt):
            raise AssertionError("console.input must not be called in auto-approve mode")

        monkeypatch.setattr(qwen_cli.console, "input", _boom)
        assert qwen_cli._confirm_action("Apply edit?") is True

    def test_prompt_declined(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "_auto_approve", False)
        _auto_confirm(monkeypatch, qwen_cli, "n")
        assert qwen_cli._confirm_action("Apply edit?") is False

    def test_prompt_accepted(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "_auto_approve", False)
        _auto_confirm(monkeypatch, qwen_cli, "y")
        assert qwen_cli._confirm_action("Apply edit?") is True

    def test_edit_applies_without_prompt_when_auto(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "code.txt"
        p.write_text("v1\n", encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "_auto_approve", True)
        _patch_backups(monkeypatch, qwen_cli, tmp_path)

        def _boom(_prompt):
            raise AssertionError("no prompt expected")

        monkeypatch.setattr(qwen_cli.console, "input", _boom)
        result = qwen_cli.do_edit_file(str(p), "v1", "v2")
        assert "[edited:" in result
        assert p.read_text(encoding="utf-8") == "v2\n"

    def test_dangerous_command_still_prompts_when_auto(self, qwen_cli, monkeypatch):
        # Auto-approve covers file edits only — rm -rf must still ask.
        monkeypatch.setattr(qwen_cli, "_auto_approve", True)
        monkeypatch.setattr(qwen_cli.console, "input", lambda _prompt: "n")
        result = qwen_cli.do_run_command("rm -rf /tmp/whatever")
        assert "cancelled" in result


# ---------------------------------------------------------------------------
# Prefix-stable system prompt snapshot
# ---------------------------------------------------------------------------


class TestPromptSnapshot:
    def test_memory_change_not_injected_until_refresh(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "_sysprompt_snapshot", {})
        monkeypatch.setattr(qwen_cli, "load_memory", lambda: "FACT-ALPHA")
        monkeypatch.setattr(qwen_cli, "intel_get_recent", lambda n=6: "")
        first = qwen_cli.build_system_prompt("base")
        assert "FACT-ALPHA" in first

        # Background memory extraction rewrites memory.md mid-session — the
        # prompt must NOT change until an explicit refresh point.
        monkeypatch.setattr(qwen_cli, "load_memory", lambda: "FACT-BETA")
        second = qwen_cli.build_system_prompt("base")
        assert second == first

        qwen_cli.refresh_system_snapshot()
        third = qwen_cli.build_system_prompt("base")
        assert "FACT-BETA" in third
        assert "FACT-ALPHA" not in third

    def test_git_state_not_in_system_prompt(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "_sysprompt_snapshot", {})
        monkeypatch.setattr(qwen_cli, "load_memory", lambda: "")
        monkeypatch.setattr(qwen_cli, "intel_get_recent", lambda n=6: "")
        monkeypatch.setattr(qwen_cli, "get_git_context", lambda: "Branch: main\nStatus: M file.py")
        prompt = qwen_cli.build_system_prompt("base")
        assert "=== Git State ===" not in prompt
        assert "Branch: main" not in prompt

    def test_volatile_tail_appends_git_to_last_user_message(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "get_git_context", lambda: "Branch: main")
        working = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        out = qwen_cli._inject_volatile_tail(working)
        assert out[3]["content"].startswith("second")
        assert "Branch: main" in out[3]["content"]
        # Earlier messages untouched; original list not mutated.
        assert out[1]["content"] == "first"
        assert working[3]["content"] == "second"

    def test_volatile_tail_noop_outside_git_repo(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "get_git_context", lambda: "")
        working = [{"role": "user", "content": "hello"}]
        assert qwen_cli._inject_volatile_tail(working) == working
