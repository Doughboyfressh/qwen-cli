"""Tests for main.run_turn — the core tool loop.

This is the highest-traffic code in the project (every model turn goes through
it) and had no coverage at all. It drives tool dispatch, the round cap, the
auto-continue on a cut-off answer, the empty-reply nudge, mid-run compaction,
and the turn ledger — all of which only surface at runtime.

The model is scripted rather than mocked loosely: each test hands run_turn an
exact sequence of (text, tool_calls, usage) triples and asserts on what the loop
did with them.
"""

import json

import pytest


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def usage(prompt=10, completion=5, truncated=False, finish_reason="stop"):
    return {
        "prompt": prompt,
        "completion": completion,
        "truncated": truncated,
        "finish_reason": finish_reason,
    }


def tool_call(name, args, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptedModel:
    """Stands in for stream_once. Returns each scripted turn in order."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.rounds = []  # one entry per call: the messages + use_tools it saw

    def __call__(self, client, messages, use_tools, update_fn=None):
        self.rounds.append({"messages": list(messages), "use_tools": use_tools})
        if not self._turns:
            return "fallback final answer", [], usage()
        return self._turns.pop(0)

    @property
    def exhausted(self):
        return not self._turns


@pytest.fixture()
def loop(qwen_cli, monkeypatch):
    """run_turn with the network, git, and presearch stubbed out."""
    # Auto-presearch would hit the live web; git context shells out per turn.
    monkeypatch.setattr(qwen_cli, "AUTO_SEARCH_MODE", "off")
    monkeypatch.setattr(qwen_cli, "get_git_context", lambda: "")
    return qwen_cli


def script(qwen_cli, monkeypatch, turns):
    model = ScriptedModel(turns)
    monkeypatch.setattr(qwen_cli, "stream_once", model)
    return model


def run(qwen_cli, msgs=None, **kw):
    return qwen_cli.run_turn(object(), msgs or [{"role": "user", "content": "hi"}], **kw)


# ---------------------------------------------------------------------------
# The straight-through path
# ---------------------------------------------------------------------------


def test_plain_answer_with_no_tools(loop, monkeypatch):
    script(loop, monkeypatch, [("the answer is 42", [], usage())])
    assert run(loop) == "the answer is 42"


def test_tool_result_is_fed_back_to_the_model(loop, monkeypatch, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("file contents here", encoding="utf-8")

    model = script(
        loop,
        monkeypatch,
        [
            ("", [tool_call("read_file", {"path": str(f)})], usage()),
            ("I read it.", [], usage()),
        ],
    )
    reply = run(loop)

    assert reply == "I read it."
    # The second round must carry the tool result back.
    second = model.rounds[1]["messages"]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "file contents here" in tool_msgs[0]["content"]


def test_tool_names_are_recorded_for_agent_verification(loop, monkeypatch, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    script(
        loop,
        monkeypatch,
        [
            ("", [tool_call("read_file", {"path": str(f)})], usage()),
            ("done", [], usage()),
        ],
    )
    run(loop)
    # /agent reads this to decide whether a mutation was verified.
    assert "read_file" in loop._last_turn_tool_names


def test_allow_tools_false_never_sends_schemas(loop, monkeypatch):
    model = script(loop, monkeypatch, [("no tools for me", [], usage())])
    run(loop, allow_tools=False)
    assert model.rounds[0]["use_tools"] is False


# ---------------------------------------------------------------------------
# Malformed / unknown tool calls
# ---------------------------------------------------------------------------


def test_malformed_json_arguments_do_not_crash_the_turn(loop, monkeypatch):
    bad = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{not valid json"},
    }
    model = script(
        loop,
        monkeypatch,
        [
            ("", [bad], usage()),
            ("recovered", [], usage()),
        ],
    )
    assert run(loop) == "recovered"

    tool_msg = next(m for m in model.rounds[1]["messages"] if m.get("role") == "tool")
    assert "tool_call_error" in tool_msg["content"]
    assert "valid JSON" in tool_msg["content"]


def test_unknown_tool_is_reported_not_raised(loop, monkeypatch):
    model = script(
        loop,
        monkeypatch,
        [
            ("", [tool_call("no_such_tool", {})], usage()),
            ("ok", [], usage()),
        ],
    )
    assert run(loop) == "ok"
    tool_msg = next(m for m in model.rounds[1]["messages"] if m.get("role") == "tool")
    assert "unknown tool" in tool_msg["content"]


# ---------------------------------------------------------------------------
# Round cap -> forced synthesis
# ---------------------------------------------------------------------------


def test_round_cap_forces_a_final_no_tools_synthesis(loop, monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "MAX_TOOL_DEPTH", 3)
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")

    # The model keeps calling tools for exactly MAX_TOOL_DEPTH rounds; the loop
    # must then cut it off and spend its last call on a tool-free synthesis.
    greedy = [("", [tool_call("read_file", {"path": str(f)})], usage())] * 3
    model = script(loop, monkeypatch, [*greedy, ("synthesized answer", [], usage())])

    reply = run(loop)

    assert "synthesized answer" in reply
    assert loop._turn_hit_round_cap is True, "REPL auto-continue depends on this flag"
    # The synthesis round must be tool-free, or the model just keeps going.
    assert model.rounds[-1]["use_tools"] is False
    assert any(
        "synthesize what you have" in (m.get("content") or "")
        for m in model.rounds[-1]["messages"]
    )


def test_round_cap_flag_is_reset_each_turn(loop, monkeypatch):
    monkeypatch.setattr(loop, "_turn_hit_round_cap", True)
    script(loop, monkeypatch, [("clean turn", [], usage())])
    run(loop)
    assert loop._turn_hit_round_cap is False


# ---------------------------------------------------------------------------
# Auto-continue on a cut-off answer
# ---------------------------------------------------------------------------


def test_truncated_answer_is_resumed_and_segments_joined(loop, monkeypatch):
    script(
        loop,
        monkeypatch,
        [
            ("first half", [], usage(truncated=True, finish_reason="length")),
            ("second half", [], usage()),
        ],
    )
    reply = run(loop)
    assert reply == "first halfsecond half"


def test_auto_continue_asks_the_model_not_to_repeat_itself(loop, monkeypatch):
    model = script(
        loop,
        monkeypatch,
        [
            ("cut off", [], usage(truncated=True, finish_reason="length")),
            ("rest", [], usage()),
        ],
    )
    run(loop)
    nudge = model.rounds[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "cut off" in nudge["content"]
    assert "do not repeat" in nudge["content"].lower()


def test_auto_continue_is_bounded(loop, monkeypatch):
    monkeypatch.setattr(loop, "MAX_AUTO_CONTINUE", 2)
    # Always truncated — the loop must give up rather than spin.
    model = script(
        loop,
        monkeypatch,
        [("chunk", [], usage(truncated=True, finish_reason="length"))] * 12,
    )
    reply = run(loop)
    # 1 initial + MAX_AUTO_CONTINUE resumptions
    assert len(model.rounds) == 3
    assert reply == "chunkchunkchunk"


def test_a_complete_answer_is_not_resumed(loop, monkeypatch):
    model = script(loop, monkeypatch, [("all done", [], usage(truncated=False))])
    assert run(loop) == "all done"
    assert len(model.rounds) == 1


# ---------------------------------------------------------------------------
# Empty-reply nudge
# ---------------------------------------------------------------------------


def test_empty_reply_is_nudged_once(loop, monkeypatch):
    model = script(
        loop,
        monkeypatch,
        [
            ("", [], usage()),          # model said nothing at all
            ("actual answer", [], usage()),
        ],
    )
    assert run(loop) == "actual answer"
    assert len(model.rounds) == 2
    assert "empty" in model.rounds[1]["messages"][-1]["content"].lower()


def test_empty_reply_nudge_does_not_loop_forever(loop, monkeypatch):
    model = script(loop, monkeypatch, [("", [], usage())] * 6)
    run(loop)
    # One nudge, then it accepts the empty answer rather than spinning.
    assert len(model.rounds) == 2


# ---------------------------------------------------------------------------
# Turn ledger — what the next turn learns about this one
# ---------------------------------------------------------------------------


def test_ledger_records_reads_and_mutations(loop, monkeypatch, tmp_path):
    f = tmp_path / "led.txt"
    f.write_text("one\ntwo\n", encoding="utf-8")
    monkeypatch.setattr(loop, "_auto_approve", True)  # skip the y/N on the edit

    script(
        loop,
        monkeypatch,
        [
            ("", [tool_call("read_file", {"path": str(f)}, "c1")], usage()),
            (
                "",
                [tool_call("edit_file", {"path": str(f), "old_string": "one", "new_string": "1"}, "c2")],
                usage(),
            ),
            ("done", [], usage()),
        ],
    )
    run(loop)

    ledger = " ; ".join(loop._turn_ledger)
    assert "read led.txt" in ledger
    assert "edited" in ledger or "edit_file" in ledger


def test_ledger_is_cleared_between_turns(loop, monkeypatch):
    script(loop, monkeypatch, [("a", [], usage())])
    run(loop)
    first = list(loop._turn_ledger)
    script(loop, monkeypatch, [("b", [], usage())])
    run(loop)
    assert loop._turn_ledger == [] or loop._turn_ledger != first


def test_format_turn_ledger_is_empty_when_nothing_happened(loop, monkeypatch):
    script(loop, monkeypatch, [("just chatting", [], usage())])
    run(loop)
    assert loop._format_turn_ledger() == ""


# ---------------------------------------------------------------------------
# Read dedup within a turn
# ---------------------------------------------------------------------------


def test_identical_reads_in_one_turn_are_deduped(loop, monkeypatch, tmp_path):
    f = tmp_path / "same.txt"
    f.write_text("payload", encoding="utf-8")

    model = script(
        loop,
        monkeypatch,
        [
            ("", [tool_call("read_file", {"path": str(f)}, "c1")], usage()),
            ("", [tool_call("read_file", {"path": str(f)}, "c2")], usage()),
            ("done", [], usage()),
        ],
    )
    run(loop)

    tool_msgs = [m for m in model.rounds[2]["messages"] if m.get("role") == "tool"]
    assert "payload" in tool_msgs[0]["content"]
    # The second identical read must not re-send the whole file.
    assert "already returned earlier this turn" in tool_msgs[1]["content"]


def test_read_dedup_resets_between_turns(loop, monkeypatch, tmp_path):
    f = tmp_path / "again.txt"
    f.write_text("payload", encoding="utf-8")
    turns = [
        ("", [tool_call("read_file", {"path": str(f)})], usage()),
        ("done", [], usage()),
    ]
    script(loop, monkeypatch, list(turns))
    run(loop)

    model = script(loop, monkeypatch, list(turns))
    run(loop)
    tool_msg = next(m for m in model.rounds[1]["messages"] if m.get("role") == "tool")
    assert "payload" in tool_msg["content"], "a new turn must be able to read the file again"


# ---------------------------------------------------------------------------
# Batching: stateful tools must never run in parallel
# ---------------------------------------------------------------------------


def test_stateful_tools_are_never_batched_together(qwen_cli):
    calls = [
        {"function": {"name": "write_file"}},
        {"function": {"name": "write_file"}},
        {"function": {"name": "read_file"}},
    ]
    batches = qwen_cli._classify_tool_batch(calls, [{}, {}, {}])
    for batch in batches:
        stateful = [i for i in batch if calls[i]["function"]["name"] in qwen_cli._STATEFUL_TOOLS]
        assert len(stateful) <= 1, "two stateful tools shared a parallel batch"


def test_same_domain_fetches_are_serialized(qwen_cli):
    calls = [
        {"function": {"name": "fetch_url"}},
        {"function": {"name": "fetch_url"}},
    ]
    args = [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]
    batches = qwen_cli._classify_tool_batch(calls, args)
    assert any(len(b) == 2 for b in batches), "same-domain fetches should share one serialized batch"


def test_read_only_tools_can_batch_together(qwen_cli):
    calls = [{"function": {"name": "read_file"}}, {"function": {"name": "list_directory"}}]
    batches = qwen_cli._classify_tool_batch(calls, [{}, {}])
    assert len(batches) == 1 and len(batches[0]) == 2


# ---------------------------------------------------------------------------
# Mid-run compaction
# ---------------------------------------------------------------------------


def test_compact_tool_loop_shrinks_only_older_results(qwen_cli):
    working = [{"role": "user", "content": "task"}]
    for i in range(8):
        working.append({"role": "assistant", "content": f"step {i}"})
        working.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * 5000})

    out = qwen_cli._compact_tool_loop(working, keep_recent_tools=3)

    tools = [m for m in out if m["role"] == "tool"]
    assert len(tools) == 8, "no message may be dropped — it would orphan a tool_call"
    assert all("condensed to fit context" in t["content"] for t in tools[:-3])
    assert all(len(t["content"]) == 5000 for t in tools[-3:]), "recent results must stay intact"


def test_compact_tool_loop_is_a_noop_when_short(qwen_cli):
    working = [
        {"role": "user", "content": "task"},
        {"role": "tool", "tool_call_id": "c0", "content": "Y" * 5000},
    ]
    assert qwen_cli._compact_tool_loop(working, keep_recent_tools=4) == working


def test_task_anchor_is_appended_at_the_tail(qwen_cli):
    working = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "Hey! What can I help you with?"},
        {"role": "user", "content": "audit yourself"},
        {"role": "tool", "tool_call_id": "c0", "content": "result"},
    ]
    out = qwen_cli._refresh_task_anchor(working, "audit yourself")
    assert out[-1]["role"] == "user"
    assert out[-1]["content"].startswith(qwen_cli._TASK_ANCHOR_PREFIX)
    assert "audit yourself" in out[-1]["content"]
    assert out[:-1] == working  # nothing else touched


def test_task_anchor_is_idempotent_across_repeated_compactions(qwen_cli):
    working = [{"role": "user", "content": "the task"}]
    out = qwen_cli._refresh_task_anchor(working, "the task")
    out.append({"role": "tool", "tool_call_id": "c1", "content": "more results"})
    out = qwen_cli._refresh_task_anchor(out, "the task")

    anchors = [m for m in out if (m.get("content") or "").startswith(qwen_cli._TASK_ANCHOR_PREFIX)]
    assert len(anchors) == 1, "repeated compactions must not stack anchors"
    assert out[-1] is anchors[0], "the anchor must move back to the tail"


def test_task_anchor_noop_without_a_task(qwen_cli):
    working = [{"role": "tool", "tool_call_id": "c0", "content": "r"}]
    assert qwen_cli._refresh_task_anchor(working, "   ") == working


def test_mid_run_compaction_reanchors_the_active_task(loop, monkeypatch, tmp_path):
    """Regression: with the window nearly full, compaction condensed old tool
    results but nothing re-stated the request — the model then answered the
    stale greeting at the top of the window ("Hey! What's up?") instead of
    finishing the task."""
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    model = script(
        loop,
        monkeypatch,
        [
            # Real prompt count at the window limit forces compaction next round.
            ("", [tool_call("read_file", {"path": str(f)})], usage(prompt=loop.TOKEN_LIMIT)),
            ("audit finished", [], usage()),
        ],
    )
    msgs = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "Hey! What can I help you with?"},
        {"role": "user", "content": "audit yourself and tell me how we can improve you."},
    ]
    assert loop.run_turn(object(), msgs) == "audit finished"

    second = model.rounds[1]["messages"]
    assert second[-1]["content"].startswith(loop._TASK_ANCHOR_PREFIX)
    assert "audit yourself" in second[-1]["content"]


# ---------------------------------------------------------------------------
# Presearch
# ---------------------------------------------------------------------------


def test_presearch_false_skips_the_grounding_search(qwen_cli, monkeypatch):
    """Synthetic follow-ups already tell the model to search; presearching them
    just web-searches the critique text itself."""
    monkeypatch.setattr(qwen_cli, "AUTO_SEARCH_MODE", "aggressive")
    monkeypatch.setattr(qwen_cli, "get_git_context", lambda: "")

    searched = []
    monkeypatch.setattr(qwen_cli, "do_web_search", lambda *a, **k: searched.append(a) or "results")
    script(qwen_cli, monkeypatch, [("answer", [], usage())])

    qwen_cli.run_turn(object(), [{"role": "user", "content": "what is the capital of France?"}], presearch=False)
    assert searched == []
