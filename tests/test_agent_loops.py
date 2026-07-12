"""Tests for the autonomous loops: cmd_agent and cmd_task.

cmd_agent's verification gate is the whole point of the mode: an agent that
modifies files must prove the change worked before it is allowed to declare
AGENT_DONE. If that gate breaks, the agent starts claiming success without
checking anything — precisely the failure it exists to prevent — and it fails
open and silent. Nothing covered it.
"""

import pytest


@pytest.fixture()
def agent(qwen_cli, monkeypatch):
    """cmd_agent/cmd_task with the model, prompt-building, and compaction stubbed."""
    monkeypatch.setattr(qwen_cli, "build_system_prompt", lambda base: "SYS")
    monkeypatch.setattr(qwen_cli, "_maybe_autocompact", lambda hist, *a, **k: hist)
    monkeypatch.setattr(qwen_cli, "_auto_approve", False)
    monkeypatch.setattr(qwen_cli, "_current_plan", [])
    return qwen_cli


class ScriptedAgent:
    """Stands in for run_turn. Each entry is (reply, tool_names_used_this_turn)."""

    def __init__(self, qwen_cli, turns):
        self._qwen = qwen_cli
        self._turns = list(turns)
        self.prompts = []  # the last user message handed to each call

    def __call__(self, client, messages, allow_tools=True, presearch=True):
        self.prompts.append(messages[-1].get("content", ""))
        if not self._turns:
            return "AGENT_DONE"
        reply, tools = self._turns.pop(0)
        # run_turn rewrites this list each turn; cmd_agent reads it afterwards.
        del self._qwen._last_turn_tool_names[:]
        self._qwen._last_turn_tool_names.extend(tools)
        return reply


def script(agent, monkeypatch, turns):
    s = ScriptedAgent(agent, turns)
    monkeypatch.setattr(agent, "run_turn", s)
    return s


def answers(agent, monkeypatch, *replies):
    """Queue console.input responses (auto-approve prompt, step prompts, ...)."""
    q = list(replies)
    monkeypatch.setattr(agent.console, "input", lambda *a, **k: q.pop(0) if q else "")
    return q


# ---------------------------------------------------------------------------
# _verification_pending — the rule the gate is built on
# ---------------------------------------------------------------------------


def test_mutation_sets_pending(qwen_cli):
    assert qwen_cli._verification_pending(["edit_file"]) is True


def test_a_verifying_tool_after_the_mutation_clears_it(qwen_cli):
    assert qwen_cli._verification_pending(["edit_file", "run_command"]) is False


def test_a_verifying_tool_BEFORE_the_mutation_does_not_clear_it(qwen_cli):
    """Reading a file and then editing it is not verification of the edit."""
    assert qwen_cli._verification_pending(["read_file", "edit_file"]) is True


def test_a_later_mutation_re_arms_the_gate(qwen_cli):
    assert qwen_cli._verification_pending(["edit_file", "read_file", "write_file"]) is True


def test_pending_state_carries_across_turns(qwen_cli):
    pending = qwen_cli._verification_pending(["edit_file"])
    assert qwen_cli._verification_pending(["web_search"], pending) is True
    assert qwen_cli._verification_pending(["read_file"], pending) is False


def test_read_only_work_never_arms_the_gate(qwen_cli):
    assert qwen_cli._verification_pending(["web_search", "read_file"]) is False


# ---------------------------------------------------------------------------
# cmd_agent — the enforcement gate
# ---------------------------------------------------------------------------


def test_agent_stops_cleanly_on_agent_done(agent, monkeypatch):
    s = script(agent, monkeypatch, [("all finished. AGENT_DONE", [])])
    answers(agent, monkeypatch, "y")
    history = []

    agent.cmd_agent("do a thing", history, "SYS", object())

    assert len(s.prompts) == 1, "should not iterate past AGENT_DONE"


def test_agent_done_is_rejected_after_an_unverified_edit(agent, monkeypatch):
    """The core guarantee: you changed a file, you did not check it, you are not done."""
    s = script(
        agent,
        monkeypatch,
        [
            ("edited it. AGENT_DONE", ["edit_file"]),   # mutated, never verified -> rejected
            ("re-read it, looks right. AGENT_DONE", ["read_file"]),  # verified -> accepted
        ],
    )
    answers(agent, monkeypatch, "y")
    history = []

    agent.cmd_agent("fix the bug", history, "SYS", object())

    assert len(s.prompts) == 2, "the first AGENT_DONE should have been rejected"
    assert "REJECTED" in s.prompts[1], "the agent must be told why, and told to verify"
    assert "verify" in s.prompts[1].lower()


def test_agent_done_is_accepted_when_the_edit_was_verified(agent, monkeypatch):
    s = script(agent, monkeypatch, [("edited and ran the tests. AGENT_DONE", ["edit_file", "run_command"])])
    answers(agent, monkeypatch, "y")

    agent.cmd_agent("fix it", [], "SYS", object())

    assert len(s.prompts) == 1, "a verified edit must not be rejected"


def test_agent_done_needs_no_verification_when_nothing_was_touched(agent, monkeypatch):
    s = script(agent, monkeypatch, [("here is the answer. AGENT_DONE", ["read_file", "web_search"])])
    answers(agent, monkeypatch, "y")

    agent.cmd_agent("explain something", [], "SYS", object())

    assert len(s.prompts) == 1


def test_rejection_is_capped_so_a_stubborn_agent_cannot_spin(agent, monkeypatch):
    """After 3 rejections it gives up and warns, rather than looping forever."""
    s = script(agent, monkeypatch, [("done. AGENT_DONE", ["write_file"])] * 10)
    answers(agent, monkeypatch, "y")

    agent.cmd_agent("write a file", [], "SYS", object(), max_iter=20)

    # initial + 3 rejections, then it stops fighting
    assert len(s.prompts) == 4


# ---------------------------------------------------------------------------
# cmd_agent — loop control and state hygiene
# ---------------------------------------------------------------------------


def test_agent_respects_max_iter(agent, monkeypatch):
    s = script(agent, monkeypatch, [("still working...", [])] * 50)
    answers(agent, monkeypatch, "y")

    agent.cmd_agent("endless task", [], "SYS", object(), max_iter=4)

    assert len(s.prompts) == 4


def test_agent_stops_when_the_turn_fails(agent, monkeypatch):
    s = script(agent, monkeypatch, [("", [])])  # run_turn returned falsy = error/cancel
    answers(agent, monkeypatch, "y")

    agent.cmd_agent("task", [], "SYS", object(), max_iter=10)

    assert len(s.prompts) == 1


def test_agent_records_one_consolidated_turn_not_per_iteration_scaffolding(agent, monkeypatch):
    """Saved sessions used to replay '[continue]'/reply pairs as real conversation."""
    script(
        agent,
        monkeypatch,
        [("step one done", []), ("step two done", []), ("finished. AGENT_DONE", [])],
    )
    answers(agent, monkeypatch, "y")
    history = []

    agent.cmd_agent("multi-step goal", history, "SYS", object())

    assert len(history) == 2, "three iterations must collapse to one user/assistant pair"
    assert history[0]["role"] == "user"
    assert "multi-step goal" in history[0]["content"]
    # but no reply may be lost
    assert "step one done" in history[1]["content"]
    assert "step two done" in history[1]["content"]
    assert "finished" in history[1]["content"]


def test_agent_leaves_no_history_when_it_produced_nothing(agent, monkeypatch):
    script(agent, monkeypatch, [("", [])])
    answers(agent, monkeypatch, "y")
    history = []

    agent.cmd_agent("task", history, "SYS", object())

    assert history == []


def test_auto_approve_does_not_leak_into_the_session(agent, monkeypatch):
    """Auto-approve is granted for the agent RUN only. Leaking it would mean the
    next interactive edit applies with no prompt."""
    script(agent, monkeypatch, [("done. AGENT_DONE", [])])
    answers(agent, monkeypatch, "y")
    assert agent._auto_approve is False

    agent.cmd_agent("task", [], "SYS", object())

    assert agent._auto_approve is False, "auto-approve leaked out of the agent run"


def test_auto_approve_is_restored_even_when_the_run_blows_up(agent, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("model died")

    monkeypatch.setattr(agent, "run_turn", explode)
    answers(agent, monkeypatch, "y")

    with pytest.raises(RuntimeError):
        agent.cmd_agent("task", [], "SYS", object())

    assert agent._auto_approve is False, "a crash must not leave auto-approve on"


def test_declining_auto_approve_leaves_it_off(agent, monkeypatch):
    script(agent, monkeypatch, [("done. AGENT_DONE", [])])
    answers(agent, monkeypatch, "n")

    agent.cmd_agent("task", [], "SYS", object())

    assert agent._auto_approve is False


def test_agent_resets_the_plan_at_the_start_of_a_run(agent, monkeypatch):
    agent._current_plan.append({"text": "stale step from a previous run", "status": "completed"})
    script(agent, monkeypatch, [("done. AGENT_DONE", [])])
    answers(agent, monkeypatch, "y")

    agent.cmd_agent("new goal", [], "SYS", object())

    assert not any("stale" in s["text"] for s in agent._current_plan)


# ---------------------------------------------------------------------------
# _detect_test_command — what /task runs after a code change
# ---------------------------------------------------------------------------


def test_detect_pytest_from_pytest_ini(qwen_cli, tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")
    assert qwen_cli._detect_test_command(tmp_path) == "pytest"


def test_detect_pytest_from_a_tests_dir(qwen_cli, tmp_path):
    (tmp_path / "tests").mkdir()
    assert qwen_cli._detect_test_command(tmp_path) == "pytest"


def test_detect_npm_test_from_package_json(qwen_cli, tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    assert qwen_cli._detect_test_command(tmp_path) == "npm test"


def test_detect_cargo_and_go(qwen_cli, tmp_path):
    (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
    assert qwen_cli._detect_test_command(tmp_path) == "cargo test"


def test_detect_nothing_in_an_empty_dir(qwen_cli, tmp_path):
    assert qwen_cli._detect_test_command(tmp_path) is None


def test_detect_survives_malformed_package_json(qwen_cli, tmp_path):
    (tmp_path / "package.json").write_text("{ not json", encoding="utf-8")
    assert qwen_cli._detect_test_command(tmp_path) is None


# ---------------------------------------------------------------------------
# cmd_task — plan / approve / execute
# ---------------------------------------------------------------------------


def fake_planner(qwen_cli, monkeypatch, plan_text):
    """Stub the non-streaming completions call cmd_task uses to build its plan."""

    class _Msg:
        content = plan_text

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Comp:
        def create(self, **kw):
            return _Resp()

    class _Client:
        chat = type("C", (), {"completions": _Comp()})()

    return _Client()


def test_task_cancelled_at_the_approval_prompt_runs_nothing(agent, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = script(agent, monkeypatch, [("should never run", [])])
    answers(agent, monkeypatch, "n")
    client = fake_planner(agent, monkeypatch, "1. do a thing\n2. do another")

    agent.cmd_task("a goal", [], "SYS", client)

    assert s.prompts == [], "declining the plan must not execute any step"


def test_task_seeds_the_visible_plan_from_the_parsed_steps(agent, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    script(agent, monkeypatch, [("ok", []), ("ok", [])])
    answers(agent, monkeypatch, "y", "s")  # approve, then stop after step 1
    client = fake_planner(agent, monkeypatch, "1. read the file\n2. patch the file")

    agent.cmd_task("a goal", [], "SYS", client)

    texts = [s["text"] for s in agent._current_plan]
    assert len(texts) == 2
    assert "read the file" in texts[0]
    assert "patch the file" in texts[1]


def test_task_executes_each_step_in_order(agent, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = script(agent, monkeypatch, [("did 1", []), ("did 2", [])])
    answers(agent, monkeypatch, "y", "")  # approve, Enter to continue past step 1
    client = fake_planner(agent, monkeypatch, "1. first step\n2. second step")

    agent.cmd_task("a goal", [], "SYS", client)

    assert len(s.prompts) == 2
    assert "first step" in s.prompts[0]
    assert "second step" in s.prompts[1]


def test_task_stops_when_a_step_fails(agent, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = script(agent, monkeypatch, [("", [])])  # step 1 errors out
    answers(agent, monkeypatch, "y")
    client = fake_planner(agent, monkeypatch, "1. first step\n2. second step")

    agent.cmd_task("a goal", [], "SYS", client)

    assert len(s.prompts) == 1, "a failed step must not fall through to the next"


def test_task_retries_a_step_whose_tests_fail(agent, monkeypatch, tmp_path):
    """A step that edits code and breaks the tests gets retried with the failure."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")

    runs = []

    def fake_tests(cmd, **kw):
        runs.append(cmd)
        return "FAILED tests/test_x.py::test_y" if len(runs) == 1 else "3 passed"

    monkeypatch.setattr(agent, "do_run_command", fake_tests)
    s = script(agent, monkeypatch, [("edited", ["edit_file"]), ("fixed it", ["edit_file"])])
    answers(agent, monkeypatch, "y")
    client = fake_planner(agent, monkeypatch, "1. only step")

    agent.cmd_task("a goal", [], "SYS", client)

    assert len(s.prompts) == 2, "a failing test run should trigger one retry"
    assert "failed tests" in s.prompts[1].lower()
    assert len(runs) == 2


def test_task_does_not_run_tests_when_no_file_changed(agent, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")

    runs = []
    monkeypatch.setattr(agent, "do_run_command", lambda cmd, **kw: runs.append(cmd) or "ok")
    script(agent, monkeypatch, [("just looked around", ["read_file"])])
    answers(agent, monkeypatch, "y")
    client = fake_planner(agent, monkeypatch, "1. investigate")

    agent.cmd_task("a goal", [], "SYS", client)

    assert runs == [], "read-only steps should not trigger a test run"
