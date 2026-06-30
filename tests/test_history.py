"""Tests for context-management helpers: truncate_middle, approx_tokens, _fmt_size."""


def _chat_history(n):
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(n):
        msgs.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"m{i}",
        })
    return msgs


def test_truncate_middle_drops_the_middle(qwen_cli):
    history = _chat_history(40)
    out = qwen_cli.truncate_middle(history, keep_first=6, keep_last=20)
    # system + first 6 + 1 marker + last 20
    assert len(out) == 28
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1]["content"] == "m0"          # first kept turn
    assert out[-1]["content"] == "m39"        # last kept turn
    markers = [m for m in out if "removed" in (m.get("content") or "")]
    assert len(markers) == 1
    assert "14 messages" in markers[0]["content"]   # 40 - 6 - 20 = 14 dropped


def test_truncate_middle_noop_when_short(qwen_cli):
    history = _chat_history(10)
    out = qwen_cli.truncate_middle(history, keep_first=6, keep_last=20)
    assert out is history     # returned unchanged


def test_approx_tokens(qwen_cli):
    assert qwen_cli.approx_tokens([]) == 0
    assert qwen_cli.approx_tokens([{"role": "user", "content": "x" * 40}]) == 10
    # None content must not blow up
    assert qwen_cli.approx_tokens([{"role": "assistant", "content": None}]) == 0


def test_approx_tokens_monotonic(qwen_cli):
    small = qwen_cli.approx_tokens([{"content": "abc"}])
    big = qwen_cli.approx_tokens([{"content": "abc" * 100}])
    assert big > small


def test_fmt_size(qwen_cli):
    assert qwen_cli._fmt_size(0) == "0 B"
    assert qwen_cli._fmt_size(500) == "500 B"
    assert qwen_cli._fmt_size(1023) == "1023 B"
    assert qwen_cli._fmt_size(1024) == "1 KB"
    assert qwen_cli._fmt_size(1536) == "1 KB"     # integer floor
    assert qwen_cli._fmt_size(1048576) == "1 MB"
    assert qwen_cli._fmt_size(1073741824) == "1 GB"


# --- auto-compact (context-limit handling) ----------------------------------

def test_maybe_autocompact_noop_when_low(qwen_cli, monkeypatch):
    monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 1000)
    monkeypatch.setattr(qwen_cli, "_real_ctx_tokens", 100)   # 10% — well under
    hist = _chat_history(40)
    out = qwen_cli._maybe_autocompact(hist, "BASE", client=None)
    assert out is hist                                       # untouched


def test_maybe_autocompact_summarizes_at_80(qwen_cli, monkeypatch):
    monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 1000)
    monkeypatch.setattr(qwen_cli, "_real_ctx_tokens", 850)   # 85%
    sentinel = [{"role": "system", "content": "SUMMARY"}]
    called = {}

    def fake_trim(history, client):
        called["trim"] = True
        return sentinel

    monkeypatch.setattr(qwen_cli, "cmd_trim", fake_trim)
    out = qwen_cli._maybe_autocompact(_chat_history(40), "BASE", client="C")
    assert called.get("trim") is True       # summarized, not blunt-truncated
    assert out is sentinel


def test_maybe_autocompact_falls_back_to_truncate_on_failure(qwen_cli, monkeypatch):
    monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 1000)
    monkeypatch.setattr(qwen_cli, "_real_ctx_tokens", 950)   # 95% — critically full
    monkeypatch.setattr(qwen_cli, "cmd_trim",
                        lambda h, c: (_ for _ in ()).throw(RuntimeError("server down")))
    hist = _chat_history(40)
    out = qwen_cli._maybe_autocompact(hist, "BASE", client="C")
    # summarization failed -> blunt truncate-middle kept the session alive
    assert len(out) < len(hist)
    assert any("removed" in (m.get("content") or "") for m in out)


def test_cmd_trim_preserves_current_task_and_work_turns(qwen_cli, monkeypatch):
    # A file-editing ("work") turn early, then enough chat to trigger summarizing.
    history = [
        {"role": "user",      "content": "please edit config"},
        {"role": "assistant", "content": "[patched: config.toml (2 lines changed)]"},
    ]
    for i in range(16):
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"chat {i}"})

    monkeypatch.setattr(qwen_cli, "stream_once",
                        lambda *a, **k: ("Edited config. CURRENT TASK: wire up the parser.", {}, []))
    out = qwen_cli.cmd_trim(history, client="C")

    # Leading summary instructs the model to keep going on the current task.
    assert out[0]["role"] == "system"
    assert "CURRENT TASK" in out[0]["content"]
    assert "without asking the user" in out[0]["content"]
    # The file-editing turn survives verbatim (work is not lost to summarization).
    assert any("[patched: config.toml" in (m.get("content") or "") for m in out)


# --- mid-run (tool-loop) compaction -----------------------------------------

def _tool_round(rid, result):
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": rid, "type": "function",
                         "function": {"name": "fetch_url", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": rid, "content": result},
    ]


def test_compact_tool_loop_shrinks_old_keeps_recent_and_never_orphans(qwen_cli):
    big = "X" * 5000
    working = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "do it"}]
    for r in range(6):
        working += _tool_round(f"t{r}", big)

    out = qwen_cli._compact_tool_loop(working, keep_recent_tools=2, head_chars=100)

    # No message removed → every tool_call still has its matching result, in order.
    assert len(out) == len(working)
    call_ids = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    tool_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert call_ids == tool_ids == [f"t{r}" for r in range(6)]

    tools = [m for m in out if m.get("role") == "tool"]
    assert all("condensed to fit context" in t["content"] for t in tools[:4])  # old shrunk
    assert all(t["content"] == big for t in tools[-2:])                        # recent intact


def test_compact_tool_loop_noop_when_few_tools(qwen_cli):
    working = [{"role": "user", "content": "x"}] + _tool_round("a", "Y" * 9999)
    out = qwen_cli._compact_tool_loop(working, keep_recent_tools=4)
    assert out is working   # nothing to shrink → unchanged object
