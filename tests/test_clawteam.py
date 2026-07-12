"""Tests for the ClawTeam coordination layer: teams, tasks, inbox."""

import json
import threading
from pathlib import Path

import pytest


@pytest.fixture()
def ct(qwen_cli, tmp_path, monkeypatch):
    """Patch CT_DIR to a temp location so tests don't touch ~/.qwen-cli/.clawteam."""
    import qwen_cli.core.config as _cfg_mod

    monkeypatch.setattr(_cfg_mod, "CT_DIR", tmp_path / ".clawteam")
    return qwen_cli


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


class TestTeams:
    def test_create_team_produces_config(self, ct, tmp_path):
        result = ct._ct_team_create("alpha")
        assert result["name"] == "alpha"
        assert (tmp_path / ".clawteam" / "teams" / "alpha" / "config.json").exists()

    def test_create_team_is_idempotent(self, ct):
        first = ct._ct_team_create("beta")
        second = ct._ct_team_create("beta")
        assert first["createdAt"] == second["createdAt"]

    def test_team_list_returns_names(self, ct):
        ct._ct_team_create("x")
        ct._ct_team_create("y")
        names = ct._ct_team_list()
        assert "x" in names and "y" in names

    def test_team_list_empty_when_no_teams(self, ct):
        assert ct._ct_team_list() == []

    def test_team_join_adds_member(self, ct):
        ct._ct_team_create("delta")
        member = ct._ct_team_join("delta", "researcher", "general-purpose")
        assert member["name"] == "researcher"
        cfg = ct._ct_load_team("delta")
        assert any(m["name"] == "researcher" for m in cfg["members"])

    def test_team_join_is_idempotent(self, ct):
        ct._ct_team_create("epsilon")
        ct._ct_team_join("epsilon", "agent1")
        ct._ct_team_join("epsilon", "agent1")
        cfg = ct._ct_load_team("epsilon")
        assert len([m for m in cfg["members"] if m["name"] == "agent1"]) == 1


# ---------------------------------------------------------------------------
# Legacy / externally-written team configs — members as bare strings
#
# Team config.json isn't necessarily written by qwen-cli itself (the docs
# advertise ClawTeam CLI compatibility). A real config found on disk had
# "members": ["user", "auditor-arch", ...] — plain strings, not qwen-cli's
# own {"name": ..., "agentId": ..., ...} dicts. Every consumer that did
# m["name"] or m.get("name") crashed immediately: "string indices must be
# integers, not 'str'", confirmed live via team_spawn_agent.
# ---------------------------------------------------------------------------


class TestLegacyStringMembers:
    def _write_legacy_config(self, tmp_path, team="legacy"):
        team_dir = tmp_path / ".clawteam" / "teams" / team
        team_dir.mkdir(parents=True, exist_ok=True)
        (team_dir / "config.json").write_text(
            json.dumps({"name": team, "members": ["user", "auditor-arch", "auditor-security"]}),
            encoding="utf-8",
        )

    def test_load_team_normalizes_string_members_to_dicts(self, ct, tmp_path):
        self._write_legacy_config(tmp_path)
        cfg = ct._ct_load_team("legacy")
        assert cfg["members"][0] == {
            "name": "user",
            "user": "",
            "agentId": "",
            "agentType": "unknown",
            "joinedAt": "",
        }
        assert [m["name"] for m in cfg["members"]] == ["user", "auditor-arch", "auditor-security"]

    def test_team_join_does_not_crash_on_legacy_config(self, ct, tmp_path):
        self._write_legacy_config(tmp_path)
        member = ct._ct_team_join("legacy", "new-agent")
        assert member["name"] == "new-agent"
        cfg = ct._ct_load_team("legacy")
        assert any(m["name"] == "new-agent" for m in cfg["members"])

    def test_board_render_does_not_crash_on_legacy_config(self, ct, tmp_path):
        self._write_legacy_config(tmp_path)
        board = ct._ct_board_render("legacy")
        assert "auditor-arch" in board

    def test_spawn_leader_resolution_does_not_crash_on_legacy_config(self, ct, tmp_path, monkeypatch):
        self._write_legacy_config(tmp_path)

        class _FakeProc:
            pid = 1234

        monkeypatch.setattr("qwen_cli.tools.team.subprocess.Popen", lambda *a, **k: _FakeProc())
        result = ct._ct_spawn("legacy", "worker1", "do the thing")
        assert "Spawned agent 'worker1'" in result

    def test_normalize_member_passes_dicts_through_unchanged(self, ct):
        from qwen_cli.tools import team as team_mod

        d = {"name": "x", "agentId": "abc"}
        assert team_mod._ct_normalize_member(d) is d


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestTasks:
    def test_add_task_creates_file(self, ct, tmp_path):
        ct._ct_team_create("proj")
        task = ct._ct_task_add("proj", "Write tests", owner="dev", priority="high")
        assert task["subject"] == "Write tests"
        assert task["status"] == "pending"
        assert task["owner"] == "dev"
        assert (tmp_path / ".clawteam" / "tasks" / "proj" / f"task-{task['id']}.json").exists()

    def test_task_list_filters_by_status(self, ct):
        ct._ct_team_create("proj")
        ct._ct_task_add("proj", "pending one")
        done = ct._ct_task_add("proj", "done one")
        ct._ct_task_update("proj", done["id"], status="completed")

        pending = ct._ct_task_list("proj", status="pending")
        assert all(t["status"] == "pending" for t in pending)
        completed = ct._ct_task_list("proj", status="completed")
        assert all(t["status"] == "completed" for t in completed)

    def test_task_list_filters_by_owner(self, ct):
        ct._ct_team_create("proj")
        ct._ct_task_add("proj", "alice task", owner="alice")
        ct._ct_task_add("proj", "bob task", owner="bob")
        alices = ct._ct_task_list("proj", owner="alice")
        assert all(t["owner"] == "alice" for t in alices)
        assert len(alices) == 1

    def test_task_update_changes_status(self, ct):
        ct._ct_team_create("proj")
        task = ct._ct_task_add("proj", "do stuff")
        updated = ct._ct_task_update("proj", task["id"], status="in_progress")
        assert updated["status"] == "in_progress"
        assert updated["started_at"] != ""

    def test_task_update_adds_note(self, ct):
        ct._ct_team_create("proj")
        task = ct._ct_task_add("proj", "work item")
        ct._ct_task_update("proj", task["id"], note="halfway done")
        updated = ct._ct_task_update("proj", task["id"], note="finished")
        assert len(updated["notes"]) == 2
        assert updated["notes"][0]["text"] == "halfway done"
        assert updated["notes"][1]["text"] == "finished"

    def test_task_update_unknown_id_returns_none(self, ct):
        ct._ct_team_create("proj")
        result = ct._ct_task_update("proj", "deadbeef")
        assert result is None

    def test_task_update_partial_id_match(self, ct):
        ct._ct_team_create("proj")
        task = ct._ct_task_add("proj", "partial match test")
        short_id = task["id"][:6]
        updated = ct._ct_task_update("proj", short_id, status="completed")
        assert updated is not None
        assert updated["status"] == "completed"


# ---------------------------------------------------------------------------
# Inbox
# ---------------------------------------------------------------------------


class TestInbox:
    def test_send_and_receive(self, ct):
        ct._ct_team_create("chat")
        ct._ct_inbox_send("chat", "alice", "hello alice", from_agent="bob")
        msgs = ct._ct_inbox_receive("chat", "alice")
        assert len(msgs) == 1
        assert msgs[0]["body"] == "hello alice"
        assert msgs[0]["from"] == "bob"

    def test_receive_consumes_messages(self, ct):
        ct._ct_team_create("chat")
        ct._ct_inbox_send("chat", "alice", "msg1")
        ct._ct_inbox_receive("chat", "alice")
        second = ct._ct_inbox_receive("chat", "alice")
        assert second == []

    def test_peek_does_not_consume(self, ct):
        ct._ct_team_create("chat")
        ct._ct_inbox_send("chat", "bob", "peek me")
        first = ct._ct_inbox_receive("chat", "bob", peek=True)
        second = ct._ct_inbox_receive("chat", "bob", peek=True)
        assert len(first) == 1 and len(second) == 1

    def test_multiple_messages_to_same_agent(self, ct):
        ct._ct_team_create("chat")
        ct._ct_inbox_send("chat", "carol", "first")
        ct._ct_inbox_send("chat", "carol", "second")
        msgs = ct._ct_inbox_receive("chat", "carol")
        assert len(msgs) == 2

    def test_messages_isolated_per_agent(self, ct):
        ct._ct_team_create("chat")
        ct._ct_inbox_send("chat", "alice", "for alice")
        assert ct._ct_inbox_receive("chat", "bob") == []

    def test_receive_leaves_no_consuming_files(self, ct):
        ct._ct_team_create("chat")
        ct._ct_inbox_send("chat", "dave", "atomic msg")
        ct._ct_inbox_receive("chat", "dave")
        inbox_dir = ct._ct_inbox_dir("chat", "dave")
        leftover = list(inbox_dir.glob("*.consuming"))
        assert leftover == [], f"leftover .consuming files: {leftover}"

    def test_corrupt_message_skipped_gracefully(self, ct):
        ct._ct_team_create("chat")
        inbox_dir = ct._ct_inbox_dir("chat", "eve")
        inbox_dir.mkdir(parents=True, exist_ok=True)
        bad = inbox_dir / "msg-00000000-bad.json"
        bad.write_text("not valid json", encoding="utf-8")
        ct._ct_inbox_send("chat", "eve", "good msg")
        msgs = ct._ct_inbox_receive("chat", "eve")
        assert len(msgs) == 1
        assert msgs[0]["body"] == "good msg"

    def test_concurrent_reads_each_message_delivered_once(self, ct):
        ct._ct_team_create("chat")
        for i in range(6):
            ct._ct_inbox_send("chat", "frank", f"msg{i}")
        results: list[list] = [[], []]
        barrier = threading.Barrier(2)
        lock = threading.Lock()

        def reader(idx):
            barrier.wait()
            with lock:
                results[idx] = ct._ct_inbox_receive("chat", "frank")

        t0 = threading.Thread(target=reader, args=(0,))
        t1 = threading.Thread(target=reader, args=(1,))
        t0.start()
        t1.start()
        t0.join()
        t1.join()
        combined = results[0] + results[1]
        bodies = sorted(m["body"] for m in combined)
        assert bodies == [f"msg{i}" for i in range(6)], f"expected 6 unique messages, got: {bodies}"


# ---------------------------------------------------------------------------
# Board render
# ---------------------------------------------------------------------------


class TestBoard:
    def test_board_shows_team_name(self, ct):
        ct._ct_team_create("myteam", description="test team")
        ct._ct_team_join("myteam", "dev1")
        ct._ct_task_add("myteam", "first task", owner="dev1")
        board = ct._ct_board_render("myteam")
        assert "myteam" in board
        assert "dev1" in board
        assert "first task" in board

    def test_board_unknown_team(self, ct):
        board = ct._ct_board_render("nosuchteam")
        assert "not found" in board


# ---------------------------------------------------------------------------
# Spawn — team_spawn_agent's process launch
#
# _ct_spawn used to resolve the launcher via shutil.which("qwen") or
# shutil.which("qwen.bat"), falling back to Path(sys.executable).parent /
# "qwen.bat" if neither was found on PATH. Confirmed live (a real self-audit
# run): shutil.which() returned None for both from inside the running
# process even though the interactive shell resolved "qwen" fine, so every
# spawn hit the fallback — which pointed at .venv/Scripts/qwen.bat, a path
# that never exists (qwen.bat lives at the project root, not next to the
# interpreter). Every spawned terminal failed immediately with "not
# recognized as a command" before running any Python; all 4 real tasks
# spawned that way sat at started_at: "" forever.
# ---------------------------------------------------------------------------


class TestSpawnCommand:
    def test_resolves_entry_script_relative_to_this_file(self):
        import sys

        from qwen_cli.tools import team as team_mod

        cmd = team_mod._spawn_command(Path("C:/fake/task.md"))
        assert cmd[0] == sys.executable
        assert cmd[1].endswith("qwen-cli.py")
        assert Path(cmd[1]).exists()
        assert cmd[2:] == ["--task", "@C:\\fake\\task.md"]

    def test_never_falls_back_to_broken_sys_executable_relative_path(self):
        from qwen_cli.tools import team as team_mod

        cmd = team_mod._spawn_command(Path("C:/fake/task.md"))
        # The bug: sys.executable's parent (the venv's Scripts/ dir) was
        # assumed to contain qwen.bat. It never does.
        assert "Scripts" not in cmd[1]

    def test_falls_back_to_path_lookup_when_entry_script_missing(self, monkeypatch):
        from qwen_cli.tools import team as team_mod

        monkeypatch.setattr(team_mod.Path, "exists", lambda self: False)
        monkeypatch.setattr(team_mod.shutil, "which", lambda name: "C:\\found\\qwen.bat" if name == "qwen" else None)
        cmd = team_mod._spawn_command(Path("C:/fake/task.md"))
        assert cmd == ["C:\\found\\qwen.bat", "--task", "@C:\\fake\\task.md"]

    def test_raises_clearly_when_nothing_resolves(self, monkeypatch):
        from qwen_cli.tools import team as team_mod

        monkeypatch.setattr(team_mod.Path, "exists", lambda self: False)
        monkeypatch.setattr(team_mod.shutil, "which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="could not locate qwen-cli.py"):
            team_mod._spawn_command(Path("C:/fake/task.md"))


class TestSpawn:
    def test_ct_spawn_uses_resolved_command(self, ct, monkeypatch):
        # Launches the resolved [python, entry_script, --task, ...] argv directly
        # via CREATE_NEW_CONSOLE rather than through `cmd.exe /c start "title" cmd
        # /k ...` — that wrapper's title argument must be quoted or Windows treats
        # it as the program to run instead of a title ("The system cannot find the
        # file qwen-<name>"), confirmed live, and there's no reliable way to make
        # cmd.exe see a properly quoted title through subprocess's own
        # list-to-command-line quoting (it escapes embedded quotes as literal data).
        captured = {}

        def fake_popen(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

            class _P:
                pid = 1234

            return _P()

        monkeypatch.setattr("qwen_cli.tools.team.subprocess.Popen", fake_popen)
        result = ct._ct_spawn("myteam", "worker1", "do the thing")

        assert "Spawned agent 'worker1'" in result
        args = captured["args"]
        assert args[0].endswith("python.exe")
        assert args[1].endswith("qwen-cli.py")
        assert args[2] == "--task"
        assert "creationflags" in captured["kwargs"]

    def test_ct_spawn_child_gets_all_tool_groups(self, ct, monkeypatch):
        # The agent brief references browser and team tools directly, so the
        # child session must bypass tool-group gating from the start.
        captured = {}

        def fake_popen(args, **kwargs):
            captured["kwargs"] = kwargs

            class _P:
                pid = 1234

            return _P()

        monkeypatch.setattr("qwen_cli.tools.team.subprocess.Popen", fake_popen)
        ct._ct_spawn("envteam", "worker1", "do the thing")
        assert captured["kwargs"]["env"]["QWEN_TOOL_GROUPS"] == "all"

    def test_ct_spawn_records_pid_in_task_metadata(self, ct, monkeypatch):
        class _FakeProc:
            pid = 5678

        monkeypatch.setattr("qwen_cli.tools.team.subprocess.Popen", lambda *a, **k: _FakeProc())
        ct._ct_spawn("myteam", "worker1", "do the thing")

        tasks = ct._ct_task_list("myteam")
        assert len(tasks) == 1
        assert tasks[0]["metadata"]["pid"] == 5678
        assert "spawned_at" in tasks[0]["metadata"]


# ---------------------------------------------------------------------------
# Stale-agent detection — team_spawn_agent's Popen() is fire-and-forget with
# no timeout or heartbeat. A hung or crashed agent otherwise goes unnoticed
# forever; _ct_check_stale() flags tasks whose recorded process has died or
# that haven't been updated in a while.
# ---------------------------------------------------------------------------


class TestCheckStale:
    def test_flags_task_with_dead_process(self, ct):
        ct._ct_team_create("watch")
        task = ct._ct_task_add("watch", "do work", owner="worker1")
        ct._ct_record_spawn_pid("watch", task["id"], 999_999_999)  # essentially guaranteed not to exist

        stale = ct._ct_check_stale("watch")

        assert len(stale) == 1
        assert stale[0]["id"] == task["id"]
        assert "process no longer running" in stale[0]["_stale_reason"]

    def test_does_not_flag_task_with_alive_process(self, ct):
        import os

        ct._ct_team_create("watch")
        task = ct._ct_task_add("watch", "do work", owner="worker1")
        ct._ct_record_spawn_pid("watch", task["id"], os.getpid())  # this test process — definitely alive

        stale = ct._ct_check_stale("watch")

        assert stale == []

    def test_ignores_completed_tasks_regardless_of_pid(self, ct):
        ct._ct_team_create("watch")
        task = ct._ct_task_add("watch", "do work", owner="worker1")
        ct._ct_record_spawn_pid("watch", task["id"], 999_999_999)
        ct._ct_task_update("watch", task["id"], status="completed")

        assert ct._ct_check_stale("watch") == []

    def test_ignores_blocked_tasks_regardless_of_pid(self, ct):
        ct._ct_team_create("watch")
        task = ct._ct_task_add("watch", "do work", owner="worker1")
        ct._ct_record_spawn_pid("watch", task["id"], 999_999_999)
        ct._ct_task_update("watch", task["id"], status="blocked")

        assert ct._ct_check_stale("watch") == []

    def test_flags_task_with_no_pid_based_on_age_alone(self, ct):
        ct._ct_team_create("watch")
        task = ct._ct_task_add("watch", "do work", owner="worker1")
        # No PID recorded at all — simulate an old, never-updated task.
        path = ct._ct_tasks_dir("watch") / f"task-{task['id']}.json"
        stale_task = dict(task)
        stale_task["updated_at"] = "2020-01-01T00:00:00+00:00"
        ct._ct_atomic_write(path, json.dumps(stale_task))

        stale = ct._ct_check_stale("watch", stale_minutes=20)

        assert len(stale) == 1
        assert "no update in" in stale[0]["_stale_reason"]

    def test_board_render_shows_stale_marker(self, ct):
        ct._ct_team_create("watch")
        task = ct._ct_task_add("watch", "do work", owner="worker1")
        ct._ct_record_spawn_pid("watch", task["id"], 999_999_999)

        board = ct._ct_board_render("watch")

        assert "STALE" in board
        assert "process no longer running" in board
