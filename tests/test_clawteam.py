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
