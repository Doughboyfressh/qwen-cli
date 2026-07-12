"""Tests for the UX batch: custom slash commands, -c/-r resume flags,
read_file console preview cap, and consolidated /agent history."""

from types import SimpleNamespace
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Custom slash commands (~/.qwen-cli/commands/<name>.md -> /name)
# ---------------------------------------------------------------------------


class TestCustomCommands:
    def _setup_dir(self, monkeypatch, tmp_path):
        import qwen_cli.core.config as cfg
        import qwen_cli.main as _main

        monkeypatch.setattr(cfg, "COMMANDS_DIR", tmp_path)
        monkeypatch.setattr(_main, "COMMANDS_DIR", tmp_path)

    def _capture_prompt(self, monkeypatch):
        import qwen_cli.core.repl as repl_mod

        captured = {}

        def fake_run(ctx, prompt, allow_tools=True):
            captured["prompt"] = prompt

        monkeypatch.setattr(repl_mod, "_run_turn_and_handle_reply", fake_run)
        return captured

    def test_missing_command_returns_false(self, qwen_cli, tmp_path, monkeypatch):
        from qwen_cli.core.commands import _run_custom_command

        self._setup_dir(monkeypatch, tmp_path)
        assert _run_custom_command(SimpleNamespace(), "/nope", "") is False

    def test_arguments_placeholder_substituted(self, qwen_cli, tmp_path, monkeypatch):
        from qwen_cli.core.commands import _run_custom_command

        self._setup_dir(monkeypatch, tmp_path)
        (tmp_path / "deepreview.md").write_text("Review $ARGUMENTS carefully.", encoding="utf-8")
        captured = self._capture_prompt(monkeypatch)
        assert _run_custom_command(SimpleNamespace(), "/deepreview", "main.py") is True
        assert captured["prompt"] == "Review main.py carefully."

    def test_arg_appended_without_placeholder(self, qwen_cli, tmp_path, monkeypatch):
        from qwen_cli.core.commands import _run_custom_command

        self._setup_dir(monkeypatch, tmp_path)
        (tmp_path / "explain.md").write_text("Explain this code.", encoding="utf-8")
        captured = self._capture_prompt(monkeypatch)
        assert _run_custom_command(SimpleNamespace(), "/explain", "foo.py") is True
        assert captured["prompt"] == "Explain this code.\n\nfoo.py"

    def test_no_arg_sends_template_verbatim(self, qwen_cli, tmp_path, monkeypatch):
        from qwen_cli.core.commands import _run_custom_command

        self._setup_dir(monkeypatch, tmp_path)
        (tmp_path / "standup.md").write_text("Summarize today's changes.", encoding="utf-8")
        captured = self._capture_prompt(monkeypatch)
        assert _run_custom_command(SimpleNamespace(), "/standup", "") is True
        assert captured["prompt"] == "Summarize today's changes."

    def test_path_traversal_names_rejected(self, qwen_cli, tmp_path, monkeypatch):
        from qwen_cli.core.commands import _run_custom_command

        self._setup_dir(monkeypatch, tmp_path)
        assert _run_custom_command(SimpleNamespace(), "/../evil", "") is False
        assert _run_custom_command(SimpleNamespace(), "/.hidden", "") is False

    def test_empty_file_is_reported_not_run(self, qwen_cli, tmp_path, monkeypatch):
        from qwen_cli.core.commands import _run_custom_command

        self._setup_dir(monkeypatch, tmp_path)
        (tmp_path / "blank.md").write_text("", encoding="utf-8")
        captured = self._capture_prompt(monkeypatch)
        # Handled (True) but no turn is run.
        assert _run_custom_command(SimpleNamespace(), "/blank", "") is True
        assert "prompt" not in captured

    def test_all_commands_includes_custom(self, qwen_cli, tmp_path, monkeypatch):
        monkeypatch.setattr(qwen_cli, "COMMANDS_DIR", tmp_path)
        (tmp_path / "mycmd.md").write_text("x", encoding="utf-8")
        cmds = qwen_cli._all_commands()
        assert "/mycmd" in cmds
        assert "/help" in cmds  # built-ins still present


# ---------------------------------------------------------------------------
# -c / -r resume flags
# ---------------------------------------------------------------------------


def _run_main_with_argv(qwen_cli, argv, monkeypatch, load_session_result):
    """Drive main() far enough to observe the resume step, then stop."""
    from qwen_cli.core.repl import _ReplContext

    ctx = _ReplContext([], "base", "fake_client")
    seen = {}

    def fake_repl_loop(ctx_arg, history, base_system):
        seen["history"] = history
        seen["base_system"] = base_system
        seen["ctx"] = ctx_arg

    with (
        patch.object(qwen_cli, "make_client", return_value="fake_client"),
        patch.object(qwen_cli, "make_aux_client", return_value=None),
        patch("sys.stdin") as mock_stdin,
        patch("sys.argv", ["qwen-cli.py", *argv]),
        patch.object(qwen_cli, "_repl_setup", return_value=("base", ctx.history, ctx)),
        patch.object(qwen_cli, "_repl_loop", side_effect=fake_repl_loop),
        patch.object(qwen_cli, "load_session", return_value=load_session_result),
        patch("threading.Thread"),
        patch.object(qwen_cli, "_acquire_repl_lock", return_value=True),
    ):
        mock_stdin.isatty.return_value = True
        qwen_cli.main()
    return seen


class TestResumeFlags:
    def test_continue_flag_loads_autosave(self, qwen_cli, monkeypatch):
        loaded = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "turn"}]
        seen = _run_main_with_argv(qwen_cli, ["-c"], monkeypatch, (loaded, "saved-system"))
        assert seen["history"] == loaded
        assert seen["base_system"] == "saved-system"
        assert seen["ctx"].history == loaded  # ctx stays in sync with the list passed to the loop

    def test_continue_flag_missing_autosave_starts_fresh(self, qwen_cli, monkeypatch):
        seen = _run_main_with_argv(qwen_cli, ["--continue"], monkeypatch, (None, None))
        assert seen["history"] == []
        assert seen["base_system"] == "base"

    def test_no_flag_does_not_touch_sessions(self, qwen_cli, monkeypatch):
        seen = _run_main_with_argv(qwen_cli, [], monkeypatch, (None, None))
        assert seen["history"] == []


# ---------------------------------------------------------------------------
# read_file console output — one dim summary line, never file content
# (the preview cap was removed in "Reduce console output" — 274949b)
# ---------------------------------------------------------------------------


class TestReadPreview:
    def test_full_read_returns_everything_but_one_summary_line(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "big.txt"
        content = "\n".join(f"line {i}" for i in range(1, 201))
        p.write_text(content, encoding="utf-8")

        printed = []
        monkeypatch.setattr(qwen_cli.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else ""))
        monkeypatch.setattr(qwen_cli, "_turn_read_cache", set())
        result = qwen_cli.do_read_file(str(p))

        assert "line 200" in result  # model gets the whole file
        assert any("200 lines" in s for s in printed)  # console gets a summary line
        assert not any("line 1" in s and "line 2" in s for s in printed)  # ...not the content

    def test_small_file_summary_only(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "small.txt"
        p.write_text("\n".join(f"line {i}" for i in range(1, 11)), encoding="utf-8")
        printed = []
        monkeypatch.setattr(qwen_cli.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else ""))
        monkeypatch.setattr(qwen_cli, "_turn_read_cache", set())
        result = qwen_cli.do_read_file(str(p))
        assert "line 10" in result
        assert any("10 lines" in s for s in printed)

    def test_explicit_range_returned_in_full(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "big.txt"
        p.write_text("\n".join(f"line {i}" for i in range(1, 201)), encoding="utf-8")
        printed = []
        monkeypatch.setattr(qwen_cli.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else ""))
        monkeypatch.setattr(qwen_cli, "_turn_read_cache", set())
        result = qwen_cli.do_read_file(str(p), offset=1, limit=100)
        assert "line 100" in result


# ---------------------------------------------------------------------------
# /agent history consolidation
# ---------------------------------------------------------------------------


class TestAgentHistory:
    def test_single_consolidated_turn_no_continue_scaffolding(self, qwen_cli, monkeypatch):
        replies = iter(["did step one", "did step two AGENT_DONE"])
        monkeypatch.setattr(qwen_cli, "run_turn", lambda *a, **k: next(replies))
        monkeypatch.setattr(qwen_cli, "_maybe_autocompact", lambda h, b, c: h)
        monkeypatch.setattr(qwen_cli, "build_system_prompt", lambda b: "sys")
        monkeypatch.setattr(qwen_cli.console, "input", lambda _p: "n")  # decline auto-approve
        monkeypatch.setattr(qwen_cli, "_last_turn_tool_names", [])

        history: list = []
        qwen_cli.cmd_agent("fix the bug", history, "base", client=None, max_iter=5)

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "[Agent task] fix the bug"}
        assert "did step one" in history[1]["content"]
        assert "did step two" in history[1]["content"]
        assert not any("[continue]" in (m.get("content") or "") for m in history)

    def test_auto_approve_restored_after_run(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "run_turn", lambda *a, **k: "done AGENT_DONE")
        monkeypatch.setattr(qwen_cli, "_maybe_autocompact", lambda h, b, c: h)
        monkeypatch.setattr(qwen_cli, "build_system_prompt", lambda b: "sys")
        monkeypatch.setattr(qwen_cli.console, "input", lambda _p: "y")  # accept auto-approve
        monkeypatch.setattr(qwen_cli, "_last_turn_tool_names", [])
        monkeypatch.setattr(qwen_cli, "_auto_approve", False)

        qwen_cli.cmd_agent("quick task", [], "base", client=None, max_iter=3)
        assert qwen_cli._auto_approve is False  # granted for the run, restored after
