"""Tests for tool-call handling in core/stream.py.

The XML fallback is failure-recovery code for a failure that provably happens:
Qwen3 intermittently emits tool calls as <tool_call> markup in the text body
instead of as proper API tool_calls, and drifts between three different markup
formats (format C specifically shows up on long contexts). If this silently
stops working, turns just quietly lose their tool calls — the model looks like
it answered, having never run anything.

Existing coverage (test_stream.py) is about mid-stream resilience. This is about
what the loop extracts from a stream once it has one.
"""

import json

import pytest

from qwen_cli.core.stream import (
    _create_with_retry,
    _parse_xml_tool_calls,
    _short_args,
    _strip_think,
    stream_once,
)


# ---------------------------------------------------------------------------
# Fake stream plumbing
# ---------------------------------------------------------------------------


class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _TcDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.delta = _Delta(content, tool_calls)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt=0, completion=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Chunk:
    def __init__(self, content=None, tool_calls=None, finish_reason=None, usage=None):
        self.choices = [] if usage is not None else [_Choice(content, tool_calls, finish_reason)]
        self.usage = usage


def fake_client(chunks, capture=None):
    class _Comp:
        def create(self, **kw):
            if capture is not None:
                capture.append(kw)
            return iter(chunks)

    class _Chat:
        completions = _Comp()

    class _Client:
        chat = _Chat()

    return _Client()


@pytest.fixture(autouse=True)
def quiet_tools(qwen_cli, monkeypatch):
    """stream_once reads active_tools()/preset state off main."""
    monkeypatch.setattr(qwen_cli, "active_tools", list)
    monkeypatch.setattr(qwen_cli, "_model_params", {})


# ---------------------------------------------------------------------------
# _parse_xml_tool_calls — format A: JSON body
# ---------------------------------------------------------------------------


def test_format_a_json_body():
    text = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    clean, calls = _parse_xml_tool_calls(text)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}
    assert clean == ""


def test_format_a_accepts_tool_and_parameters_aliases():
    text = '<tool_call>{"tool": "web_search", "parameters": {"query": "x"}}</tool_call>'
    _, calls = _parse_xml_tool_calls(text)
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "x"}


# ---------------------------------------------------------------------------
# format B: <function_name> + <parameter name="k">
# ---------------------------------------------------------------------------


def test_format_b_function_name_and_named_parameters():
    text = (
        "<tool_call><function_name>edit_file</function_name>"
        '<parameter name="path">x.py</parameter>'
        '<parameter name="old_string">a</parameter></tool_call>'
    )
    _, calls = _parse_xml_tool_calls(text)

    assert calls[0]["function"]["name"] == "edit_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "x.py", "old_string": "a"}


# ---------------------------------------------------------------------------
# format C: <function=name> <parameter=key>  (the long-context drift)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "body"),
    [
        # Both spellings occur in the wild. The closed one is the common
        # Llama/Qwen shape, and its closing tag used to be swallowed into the
        # value: path became "main.py</parameter>", a guaranteed file-not-found.
        ("closed", "<function=read_file><parameter=path>main.py</parameter></function>"),
        ("unclosed", "<function=read_file><parameter=path>main.py</function>"),
    ],
)
def test_format_c_function_eq_and_parameter_eq(label, body):
    _, calls = _parse_xml_tool_calls(f"<tool_call>{body}</tool_call>")

    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "main.py"


def test_format_c_closing_tag_is_not_swallowed_into_the_value():
    """Regression: the </parameter> tag ended up inside every argument value."""
    text = "<tool_call><function=run_command><parameter=command>ls -la</parameter></function></tool_call>"
    _, calls = _parse_xml_tool_calls(text)

    args = json.loads(calls[0]["function"]["arguments"])
    assert "</parameter>" not in args["command"]
    assert args["command"] == "ls -la"


def test_format_c_multiple_parameters():
    text = (
        "<tool_call><function=edit_file>"
        "<parameter=path>a.py</parameter>"
        "<parameter=old_string>foo</parameter>"
        "<parameter=new_string>bar</parameter>"
        "</function></tool_call>"
    )
    _, calls = _parse_xml_tool_calls(text)

    assert json.loads(calls[0]["function"]["arguments"]) == {
        "path": "a.py",
        "old_string": "foo",
        "new_string": "bar",
    }


def test_format_c_decodes_json_typed_values():
    """update_plan's `steps` is an ARRAY. Passed through as a bare string it
    fails the tool's isinstance(steps, list) check and the plan is silently lost."""
    steps = [{"text": "do the thing", "status": "pending"}]
    text = (
        "<tool_call><function=update_plan>"
        f"<parameter=steps>{json.dumps(steps)}</parameter>"
        "</function></tool_call>"
    )
    _, calls = _parse_xml_tool_calls(text)

    parsed = json.loads(calls[0]["function"]["arguments"])
    assert parsed["steps"] == steps
    assert isinstance(parsed["steps"], list), "a JSON array must not arrive as a string"


def test_format_c_keeps_plain_text_values_as_strings():
    text = (
        "<tool_call><function=run_command>"
        "<parameter=command>git status --short</parameter>"
        "</function></tool_call>"
    )
    _, calls = _parse_xml_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"])["command"] == "git status --short"


# ---------------------------------------------------------------------------
# General parsing behavior
# ---------------------------------------------------------------------------


def test_prose_around_the_call_is_preserved():
    text = 'Let me look.\n<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>\nDone.'
    clean, calls = _parse_xml_tool_calls(text)

    assert len(calls) == 1
    assert "Let me look." in clean
    assert "Done." in clean
    assert "<tool_call>" not in clean


def test_multiple_calls_get_distinct_ids():
    text = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "b"}}</tool_call>'
    )
    _, calls = _parse_xml_tool_calls(text)

    assert len(calls) == 2
    assert calls[0]["id"] != calls[1]["id"], "duplicate ids would collide in the tool-result map"


def test_unparseable_body_yields_no_calls():
    clean, calls = _parse_xml_tool_calls("<tool_call>total gibberish</tool_call>")
    assert calls == []


def test_text_with_no_tool_call_is_untouched():
    clean, calls = _parse_xml_tool_calls("just a normal answer")
    assert calls == []
    assert clean == "just a normal answer"


# ---------------------------------------------------------------------------
# stream_once: assembling tool calls from streamed deltas
# ---------------------------------------------------------------------------


def test_tool_call_is_reassembled_from_fragmented_deltas(qwen_cli):
    """Name and arguments arrive split across chunks and must be concatenated."""
    chunks = [
        _Chunk(tool_calls=[_TcDelta(0, id="call_1", name="read_")]),
        _Chunk(tool_calls=[_TcDelta(0, name="file", arguments='{"pa')]),
        _Chunk(tool_calls=[_TcDelta(0, arguments='th": "a.py"}')], finish_reason="tool_calls"),
    ]
    text, calls, usage = stream_once(fake_client(chunks), [], use_tools=True)

    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}


def test_parallel_tool_calls_are_kept_separate(qwen_cli):
    chunks = [
        _Chunk(tool_calls=[_TcDelta(0, id="c0", name="read_file", arguments='{"path":"a"}')]),
        _Chunk(tool_calls=[_TcDelta(1, id="c1", name="read_file", arguments='{"path":"b"}')]),
    ]
    _, calls, _ = stream_once(fake_client(chunks), [], use_tools=True)

    assert [c["id"] for c in calls] == ["c0", "c1"]


def test_usage_is_captured_from_the_usage_chunk(qwen_cli):
    chunks = [_Chunk(content="hi"), _Chunk(usage=_Usage(prompt=120, completion=7))]
    _, _, usage = stream_once(fake_client(chunks), [], use_tools=False)

    assert usage["prompt"] == 120
    assert usage["completion"] == 7


# ---------------------------------------------------------------------------
# stream_once: the XML fallback path
# ---------------------------------------------------------------------------


def test_xml_tool_call_in_text_is_rescued(qwen_cli):
    """No API tool_calls, but markup in the body — the turn must not lose the call."""
    body = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    chunks = [_Chunk(content=body, finish_reason="stop")]

    text, calls, _ = stream_once(fake_client(chunks), [], use_tools=True)

    assert len(calls) == 1, "a tool call emitted as text was silently dropped"
    assert calls[0]["function"]["name"] == "read_file"


def test_real_api_tool_calls_win_over_xml_parsing(qwen_cli):
    """If the API gave us proper tool_calls, don't also re-parse the body."""
    chunks = [
        _Chunk(
            content='<tool_call>{"name": "ignored", "arguments": {}}</tool_call>',
            tool_calls=[_TcDelta(0, id="c0", name="read_file", arguments="{}")],
        )
    ]
    _, calls, _ = stream_once(fake_client(chunks), [], use_tools=True)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"


def test_tool_markup_is_stripped_when_tools_are_disabled(qwen_cli):
    """The forced no-tools synthesis round: the model may still emit markup, and
    it must not leak into the user's answer or into saved history."""
    body = 'Here you go.<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
    chunks = [_Chunk(content=body, finish_reason="stop")]

    text, calls, _ = stream_once(fake_client(chunks), [], use_tools=False)

    assert calls == []
    assert "<tool_call>" not in text
    assert "Here you go." in text


def test_tools_disabled_and_only_markup_leaves_an_explanatory_note(qwen_cli):
    body = '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
    chunks = [_Chunk(content=body, finish_reason="stop")]

    text, calls, _ = stream_once(fake_client(chunks), [], use_tools=False)

    assert calls == []
    assert "<tool_call>" not in text
    assert text.strip(), "an empty reply here would look like the model said nothing"
    assert "web_search" in text


# ---------------------------------------------------------------------------
# _create_with_retry — server capability probing
# ---------------------------------------------------------------------------


def test_unsupported_stream_options_is_stripped_and_retried(qwen_cli, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_stream_usage_supported", True)
    seen = []

    class _Comp:
        def create(self, **kw):
            seen.append(dict(kw))
            if "stream_options" in kw:
                raise ValueError("unknown field: stream_options")
            return iter([])

    class _Client:
        chat = type("C", (), {"completions": _Comp()})()

    _create_with_retry(_Client(), model="m", messages=[], stream_options={"include_usage": True})

    assert len(seen) == 2, "should retry once without the offending field"
    assert "stream_options" not in seen[1]
    assert qwen_cli._stream_usage_supported is False, "the probe result must be remembered"


def test_connection_error_is_retried_with_backoff(qwen_cli, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    attempts = []

    class _Comp:
        def create(self, **kw):
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("connection refused")
            return iter([])

    class _Client:
        chat = type("C", (), {"completions": _Comp()})()

    _create_with_retry(_Client(), model="m", messages=[])
    assert len(attempts) == 3


def test_a_non_retryable_error_is_raised(qwen_cli):
    class _Comp:
        def create(self, **kw):
            raise RuntimeError("model not found")

    class _Client:
        chat = type("C", (), {"completions": _Comp()})()

    with pytest.raises(RuntimeError, match="model not found"):
        _create_with_retry(_Client(), model="m", messages=[])


# ---------------------------------------------------------------------------
# <think> stripping and the tool-chain label
# ---------------------------------------------------------------------------


def test_strip_think_separates_reasoning_from_the_answer():
    clean, think = _strip_think("<think>reasoning here</think>The answer.")
    assert clean == "The answer."
    assert think == "reasoning here"


def test_strip_think_joins_multiple_blocks():
    clean, think = _strip_think("<think>one</think>A<think>two</think>B")
    assert "one" in think and "two" in think
    assert "<think>" not in clean


def test_strip_think_leaves_plain_text_alone():
    clean, think = _strip_think("no reasoning here")
    assert clean == "no reasoning here"
    assert think == ""


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("web_search", {"query": "python 3.14"}, "python 3.14"),
        ("read_file", {"path": "main.py"}, "main.py"),
        ("run_command", {"command": "git status"}, "git status"),
    ],
)
def test_short_args_labels_the_tool_chain(name, args, expected):
    assert expected in _short_args(name, args)


def test_short_args_is_empty_for_unlabelled_tools():
    assert _short_args("update_plan", {"steps": []}) == ""
