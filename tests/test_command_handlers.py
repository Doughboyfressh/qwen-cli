"""Tests for the slash-command handlers in core/commands.py.

/cleanup shipped for months quietly deleting teams that had active work, because
no test drove it. These cover the handlers that carry real logic (state
mutation, filesystem writes, dispatch), rather than the thin ones that only
forward to an already-tested function in main.
"""

import json

import pytest

from qwen_cli.core.commands import _REPL_COMMANDS, _run_custom_command
from qwen_cli.core.repl import _dispatch_command, _ReplContext


@pytest.fixture()
def ctx(qwen_cli):
    return _ReplContext([], qwen_cli.BASE_SYSTEM, object())


def call(name, ctx, arg=""):
    """Invoke a handler through the real dispatch table, as the REPL does."""
    return _REPL_COMMANDS[name](ctx, arg)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_exit_breaks_the_repl_loop(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "save_session", lambda *a, **k: None)
    monkeypatch.setattr(qwen_cli, "_save_exit_handoff", lambda *a, **k: None)
    monkeypatch.setattr(qwen_cli, "record_session_changes_memory", lambda *a, **k: None)
    assert _dispatch_command(ctx, "/exit", "") is True


def test_a_crashing_handler_does_not_kill_the_session(qwen_cli, ctx, monkeypatch):
    """One bad command must not end the whole REPL."""

    def boom(_ctx, _arg):
        raise RuntimeError("handler exploded")

    monkeypatch.setitem(_REPL_COMMANDS, "/note", boom)
    assert _dispatch_command(ctx, "/note", "x") is False  # survived


def test_unknown_command_is_reported(qwen_cli, ctx, capsys):
    assert _dispatch_command(ctx, "/definitely-not-a-command", "") is False


# ---------------------------------------------------------------------------
# History: /clear, /note, /system, /retry, /branch
# ---------------------------------------------------------------------------


def test_clear_wipes_history_and_plan(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "refresh_system_snapshot", lambda: None)
    ctx.history.extend([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    qwen_cli._current_plan.append({"text": "step", "status": "pending"})

    call("/clear", ctx)

    assert ctx.history == []
    assert qwen_cli._current_plan == []


def test_clear_n_drops_only_the_last_n_turns(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "refresh_system_snapshot", lambda: None)
    for i in range(4):
        ctx.history.append({"role": "user", "content": f"u{i}"})
        ctx.history.append({"role": "assistant", "content": f"a{i}"})

    call("/clear", ctx, "2")

    assert len(ctx.history) == 4
    assert ctx.history[-1]["content"] == "a1"


def test_clear_resets_the_cached_token_count(qwen_cli, ctx, monkeypatch):
    """History shrank — the last real prompt count no longer describes it."""
    monkeypatch.setattr(qwen_cli, "refresh_system_snapshot", lambda: None)
    monkeypatch.setattr(qwen_cli, "_real_ctx_tokens", 9999)
    call("/clear", ctx)
    assert qwen_cli._real_ctx_tokens == 0


def test_note_appends_without_calling_the_model(qwen_cli, ctx):
    call("/note", ctx, "remember this")
    assert ctx.history[-1] == {"role": "user", "content": "[Note] remember this"}


def test_note_without_text_is_a_noop(qwen_cli, ctx):
    call("/note", ctx, "")
    assert ctx.history == []


def test_system_replaces_prompt_and_clears_history(qwen_cli, ctx):
    ctx.history.append({"role": "user", "content": "old"})
    call("/system", ctx, "you are a pirate")
    assert ctx.base_system == "you are a pirate"
    assert ctx.history == []


def test_retry_pops_the_last_exchange_and_reruns(qwen_cli, ctx, monkeypatch):
    reran = []
    monkeypatch.setattr(qwen_cli, "_last_user_input", "the question")
    monkeypatch.setattr(qwen_cli, "_run_turn_and_handle_reply", lambda c, text: reran.append(text))

    ctx.history.extend(
        [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "a bad answer"},
        ]
    )
    call("/retry", ctx)

    assert ctx.history == [], "the failed exchange should be removed before retrying"
    assert reran == ["the question"]


def test_retry_with_nothing_to_retry(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_last_user_input", "")
    call("/retry", ctx)  # must not raise


def test_branch_save_list_and_restore(qwen_cli, ctx):
    qwen_cli._branches.clear()
    ctx.history.extend([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}])

    call("/branch", ctx, "experiment")
    assert "experiment" in qwen_cli._branches

    ctx.history.append({"role": "user", "content": "diverged"})
    call("/branch", ctx, "restore experiment")

    assert len(ctx.history) == 2, "restore should drop the divergent turn"
    call("/branch", ctx, "list")  # must not raise


def test_branch_restore_unknown_name_is_safe(qwen_cli, ctx):
    qwen_cli._branches.clear()
    ctx.history.append({"role": "user", "content": "keep me"})
    call("/branch", ctx, "restore nope")
    assert len(ctx.history) == 1, "a failed restore must not touch history"


# ---------------------------------------------------------------------------
# /undo — the only recovery path for an overwritten file
# ---------------------------------------------------------------------------


def test_undo_restores_the_overwritten_file(qwen_cli, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_backup_stack", [])
    original = tmp_path / "f.txt"
    original.write_text("clobbered", encoding="utf-8")
    qwen_cli._backup_stack.append(
        {"original": original, "backup": tmp_path / "f.bak", "content": "the good version"}
    )

    call("/undo", ctx)

    assert original.read_text(encoding="utf-8") == "the good version"
    assert qwen_cli._backup_stack == []


def test_undo_is_multi_level(qwen_cli, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_backup_stack", [])
    f = tmp_path / "f.txt"
    f.write_text("v3", encoding="utf-8")
    for version in ("v1", "v2"):
        qwen_cli._backup_stack.append({"original": f, "backup": tmp_path / "b", "content": version})

    call("/undo", ctx)
    assert f.read_text(encoding="utf-8") == "v2", "undo pops the most recent backup"
    call("/undo", ctx)
    assert f.read_text(encoding="utf-8") == "v1"


def test_undo_with_empty_stack(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_backup_stack", [])
    call("/undo", ctx)  # must not raise


# ---------------------------------------------------------------------------
# /pin — persisted across sessions, injected into every system prompt
# ---------------------------------------------------------------------------


@pytest.fixture()
def pins_file(qwen_cli, tmp_path, monkeypatch):
    p = tmp_path / "pins.json"
    monkeypatch.setattr(qwen_cli, "PINS_FILE", p)
    return p


def test_pin_add_and_persist(qwen_cli, ctx, pins_file):
    call("/pin", ctx, "always use tabs")
    assert json.loads(pins_file.read_text(encoding="utf-8")) == ["always use tabs"]


def test_pin_remove_by_number(qwen_cli, ctx, pins_file):
    call("/pin", ctx, "first")
    call("/pin", ctx, "second")
    call("/pin", ctx, "remove 1")
    assert json.loads(pins_file.read_text(encoding="utf-8")) == ["second"]


def test_pin_remove_out_of_range_is_safe(qwen_cli, ctx, pins_file):
    call("/pin", ctx, "only")
    call("/pin", ctx, "remove 99")
    assert json.loads(pins_file.read_text(encoding="utf-8")) == ["only"], "list must be unchanged"


def test_pin_list_with_no_pins(qwen_cli, ctx, pins_file):
    call("/pin", ctx, "list")  # must not raise


# ---------------------------------------------------------------------------
# /auto and /cd
# ---------------------------------------------------------------------------


def test_auto_toggles_approve_mode(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_auto_approve", False)
    call("/auto", ctx, "on")
    assert qwen_cli._auto_approve is True
    call("/auto", ctx, "off")
    assert qwen_cli._auto_approve is False


def test_auto_rejects_garbage_without_changing_state(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_auto_approve", False)
    call("/auto", ctx, "maybe")
    assert qwen_cli._auto_approve is False


def test_cd_changes_directory_and_drops_the_git_cache(qwen_cli, ctx, tmp_path, monkeypatch):
    import os

    invalidated = []
    monkeypatch.setattr(qwen_cli, "_invalidate_git_cache", lambda: invalidated.append(True))
    start = os.getcwd()
    try:
        call("/cd", ctx, str(tmp_path))
        assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))
        assert invalidated, "git context is per-cwd — it must be dropped on /cd"
    finally:
        os.chdir(start)


def test_cd_to_a_file_is_rejected(qwen_cli, ctx, tmp_path):
    import os

    f = tmp_path / "notadir.txt"
    f.write_text("x", encoding="utf-8")
    start = os.getcwd()
    call("/cd", ctx, str(f))
    assert os.getcwd() == start


# ---------------------------------------------------------------------------
# /model — switching between the main and aux backends
# ---------------------------------------------------------------------------


def test_model_switch_to_aux_caps_the_context_budget(qwen_cli, ctx, monkeypatch):
    """The aux server runs a smaller window — chat turns must be capped to fit."""
    aux = object()
    monkeypatch.setattr(qwen_cli, "_aux_client", aux)
    monkeypatch.setattr(qwen_cli, "AUX_MODEL", "aux-model")
    monkeypatch.setattr(qwen_cli, "list_models", lambda c: ["aux-model"] if c is aux else ["main-model"])
    monkeypatch.setattr(qwen_cli, "_TOKEN_LIMIT_BASE", 100_000)
    monkeypatch.setattr(qwen_cli, "_model_params", {})

    call("/model", ctx, "aux-model")

    assert ctx.client is aux
    assert qwen_cli.MODEL == "aux-model"
    assert qwen_cli.TOKEN_LIMIT <= 28_000
    assert qwen_cli._model_params["max_tokens"] == 8192


def test_model_switch_back_to_main_restores_the_budget(qwen_cli, ctx, monkeypatch):
    aux, main = object(), object()
    monkeypatch.setattr(qwen_cli, "_aux_client", aux)
    monkeypatch.setattr(qwen_cli, "_cli_client", main)
    monkeypatch.setattr(qwen_cli, "AUX_MODEL", "aux-model")
    monkeypatch.setattr(qwen_cli, "list_models", lambda c: ["aux-model"] if c is aux else ["main-model"])
    monkeypatch.setattr(qwen_cli, "_TOKEN_LIMIT_BASE", 100_000)
    monkeypatch.setattr(qwen_cli, "_model_params", {"max_tokens": 8192})
    ctx.client = aux

    call("/model", ctx, "main-model")

    assert ctx.client is main
    assert qwen_cli.TOKEN_LIMIT == 100_000
    assert "max_tokens" not in qwen_cli._model_params


# ---------------------------------------------------------------------------
# Teams — /team, /inbox
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_ct(tmp_path, monkeypatch):
    import qwen_cli.core.config as config
    import qwen_cli.main as main
    from qwen_cli.tools import team

    ct = tmp_path / ".clawteam"
    ct.mkdir()
    monkeypatch.setattr(config, "CT_DIR", ct)
    monkeypatch.setattr(main, "CT_DIR", ct)
    monkeypatch.setattr(team._config, "CT_DIR", ct)
    return ct


def test_team_create_and_join(qwen_cli, ctx, isolated_ct):
    from qwen_cli.tools import team

    call("/team", ctx, "create squad a test squad")
    assert "squad" in team._ct_team_list()

    call("/team", ctx, "join squad worker-1")
    members = [m["name"] for m in team._ct_load_team("squad")["members"]]
    assert "worker-1" in members


def test_team_commands_enable_the_team_tool_group(qwen_cli, ctx, isolated_ct):
    """Doing team work by hand should make the team tools available to the model."""
    qwen_cli._enabled_tool_groups.discard("team")
    call("/team", ctx, "list")
    assert "team" in qwen_cli._enabled_tool_groups


def test_inbox_send_then_receive(qwen_cli, ctx, isolated_ct):
    from qwen_cli.tools import team

    team._ct_team_create("squad")
    call("/inbox", ctx, "send squad worker-1 here is your brief")

    msgs = team._ct_inbox_receive("squad", "worker-1", peek=True)
    assert len(msgs) == 1
    assert msgs[0]["body"] == "here is your brief"


def test_inbox_receive_consumes_messages(qwen_cli, ctx, isolated_ct):
    from qwen_cli.tools import team

    team._ct_team_create("squad")
    team._ct_inbox_send("squad", "worker-1", "one-shot")

    call("/inbox", ctx, "squad worker-1")

    assert team._ct_inbox_receive("squad", "worker-1", peek=True) == [], "message should be consumed"


# ---------------------------------------------------------------------------
# /cleanup — backups pass (the teams/tasks passes are in test_audit_20260712)
# ---------------------------------------------------------------------------


def test_cleanup_removes_only_old_backups(qwen_cli, ctx, tmp_path, monkeypatch):
    import os
    import time

    backups = tmp_path / "backups"
    backups.mkdir()
    monkeypatch.setattr(qwen_cli, "BACKUPS_DIR", backups)

    old = backups / "old.txt.bak"
    old.write_text("stale", encoding="utf-8")
    ancient = time.time() - 30 * 86400
    os.utime(old, (ancient, ancient))

    fresh = backups / "fresh.txt.bak"
    fresh.write_text("recent", encoding="utf-8")

    call("/cleanup", ctx, "backups")

    assert not old.exists(), "a 30-day-old backup should be reaped"
    assert fresh.exists(), "a fresh backup must be kept"


def test_cleanup_dry_run_deletes_nothing(qwen_cli, ctx, tmp_path, monkeypatch, isolated_ct):
    import os
    import time

    backups = tmp_path / "backups"
    backups.mkdir()
    monkeypatch.setattr(qwen_cli, "BACKUPS_DIR", backups)
    old = backups / "old.txt.bak"
    old.write_text("stale", encoding="utf-8")
    ancient = time.time() - 30 * 86400
    os.utime(old, (ancient, ancient))

    call("/cleanup", ctx, "dry-run")

    assert old.exists(), "dry-run must not delete anything"


# ---------------------------------------------------------------------------
# /intel
# ---------------------------------------------------------------------------


@pytest.fixture()
def intel_files(qwen_cli, tmp_path, monkeypatch):
    d = tmp_path / "intelligence"
    d.mkdir()
    monkeypatch.setattr(qwen_cli, "INTEL_DIR", d)
    monkeypatch.setattr(qwen_cli, "INTEL_TOPICS", d / "topics.json")
    monkeypatch.setattr(qwen_cli, "INTEL_FEED", d / "feed.md")
    return d


def test_intel_add_and_remove_a_topic(qwen_cli, ctx, intel_files):
    call("/intel", ctx, "add Rust releases|rust language release notes")
    names = [t["name"] for t in qwen_cli._intel_load_topics()]
    assert "Rust releases" in names

    call("/intel", ctx, "remove Rust releases")
    names = [t["name"] for t in qwen_cli._intel_load_topics()]
    assert "Rust releases" not in names


def test_intel_add_is_idempotent(qwen_cli, ctx, intel_files):
    call("/intel", ctx, "add Dupe|a query")
    call("/intel", ctx, "add Dupe|a query")
    names = [t["name"] for t in qwen_cli._intel_load_topics()]
    assert names.count("Dupe") == 1


def test_intel_off_clears_the_flag_without_rebinding_it(qwen_cli, ctx, intel_files):
    """_intel_enabled is a threading.Event shared with crawler threads. Rebinding
    it to a bool crashed every crawler on its next .is_set() call."""
    import threading

    qwen_cli._intel_enabled.set()
    call("/intel", ctx, "off")

    assert isinstance(qwen_cli._intel_enabled, threading.Event)
    assert not qwen_cli._intel_enabled.is_set()


def test_intel_topics_listing(qwen_cli, ctx, intel_files):
    call("/intel", ctx, "add Thing|a query")
    call("/intel", ctx, "topics")  # must not raise


# ---------------------------------------------------------------------------
# /mcp
# ---------------------------------------------------------------------------


def test_mcp_with_nothing_configured(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_mcp_server_configs", dict)
    call("/mcp", ctx, "")  # must not raise


def test_mcp_status_lists_configured_but_unconnected_servers(qwen_cli, ctx, monkeypatch):
    monkeypatch.setattr(
        qwen_cli, "_mcp_server_configs", lambda: {"github": {"command": "npx", "args": []}}
    )
    call("/mcp", ctx, "")  # must not raise, must not try to connect


# ---------------------------------------------------------------------------
# Custom slash commands  (~/.qwen-cli/commands/<name>.md)
# ---------------------------------------------------------------------------


@pytest.fixture()
def commands_dir(tmp_path, monkeypatch):
    import qwen_cli.core.config as config

    d = tmp_path / "commands"
    d.mkdir()
    monkeypatch.setattr(config, "COMMANDS_DIR", d)
    return d


@pytest.fixture()
def captured_turns(qwen_cli, monkeypatch):
    sent = []
    import qwen_cli.core.repl as repl

    monkeypatch.setattr(repl, "_run_turn_and_handle_reply", lambda c, text: sent.append(text))
    return sent


def test_custom_command_substitutes_arguments(ctx, commands_dir, captured_turns):
    (commands_dir / "explain.md").write_text("Explain $ARGUMENTS in one line.", encoding="utf-8")

    assert _run_custom_command(ctx, "/explain", "monads") is True
    assert captured_turns == ["Explain monads in one line."]


def test_custom_command_appends_args_when_no_placeholder(ctx, commands_dir, captured_turns):
    (commands_dir / "review.md").write_text("Review this code.", encoding="utf-8")

    _run_custom_command(ctx, "/review", "main.py")
    assert captured_turns == ["Review this code.\n\nmain.py"]


def test_custom_command_rejects_path_traversal(ctx, commands_dir, captured_turns):
    """A command name comes from user input; it must never escape the commands dir."""
    assert _run_custom_command(ctx, "/../../etc/passwd", "") is False
    assert captured_turns == []


def test_custom_command_missing_file_returns_false(ctx, commands_dir, captured_turns):
    assert _run_custom_command(ctx, "/nosuch", "") is False
    assert captured_turns == []


def test_custom_command_empty_file_is_reported_not_sent(ctx, commands_dir, captured_turns):
    (commands_dir / "blank.md").write_text("   ", encoding="utf-8")

    assert _run_custom_command(ctx, "/blank", "") is True, "handled — do not fall through to 'unknown'"
    assert captured_turns == [], "an empty template must not be sent to the model"


def test_custom_commands_appear_in_tab_completion(qwen_cli, commands_dir, monkeypatch):
    """_all_commands re-globs each call, so a command file written mid-session
    is completable without a restart."""
    monkeypatch.setattr(qwen_cli, "COMMANDS_DIR", commands_dir)
    assert "/mycmd" not in qwen_cli._all_commands()

    (commands_dir / "mycmd.md").write_text("do the thing", encoding="utf-8")
    assert "/mycmd" in qwen_cli._all_commands()


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------


def test_help_query_filters_to_matching_rows(qwen_cli, ctx):
    call("/help", ctx, "rollback")  # must not raise


def test_help_query_with_no_match(qwen_cli, ctx):
    call("/help", ctx, "zzzznotacommand")  # must not raise


def test_help_with_no_argument(qwen_cli, ctx):
    call("/help", ctx, "")  # must not raise
