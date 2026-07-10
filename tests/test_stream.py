"""Tests for stream_once mid-stream resilience (salvage partial reply on a dropped stream)."""

import pytest


class _Delta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, content, finish_reason=None):
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, content, finish_reason=None):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = None


def _client(contents, raise_at_end=False, raise_immediately=False, finish_reason="stop"):
    """Fake OpenAI client whose stream yields `contents` then optionally raises.

    The final chunk carries `finish_reason` (the real SDK reports it on the last
    chunk) so we can exercise truncation detection.
    """

    def gen():
        if raise_immediately:
            raise ConnectionError("reset by peer")
        for i, c in enumerate(contents):
            last = i == len(contents) - 1
            yield _Chunk(c, finish_reason if last else None)
        if raise_at_end:
            raise ConnectionError("reset by peer")

    class _Comp:
        def create(self, **kw):
            return gen()

    class _Chat:
        completions = _Comp()

    class _Client:
        chat = _Chat()

    return _Client()


def test_stream_once_normal_completion(qwen_cli):
    text, calls, _usage = qwen_cli.stream_once(
        _client(["Hello ", "world"]), [{"role": "user", "content": "hi"}], use_tools=False
    )
    assert text == "Hello world"
    assert calls == []


def test_stream_once_salvages_partial_on_drop(qwen_cli):
    # Stream yields two chunks then the connection drops — we keep the partial text
    # instead of losing the whole turn, and drop any (truncated) tool call.
    text, calls, _usage = qwen_cli.stream_once(
        _client(["Partial ", "answer"], raise_at_end=True),
        [{"role": "user", "content": "hi"}],
        use_tools=False,
    )
    assert text == "Partial answer"
    assert calls == []


def test_stream_once_reraises_when_nothing_received(qwen_cli):
    # Drop before any content arrives -> nothing to salvage, so the error propagates
    # (the caller's retry/refresh path handles it).
    with pytest.raises(Exception):
        qwen_cli.stream_once(
            _client([], raise_immediately=True),
            [{"role": "user", "content": "hi"}],
            use_tools=False,
        )


def test_stream_once_reports_complete(qwen_cli):
    # finish_reason == "stop" -> the answer is whole, not truncated.
    _text, _calls, usage = qwen_cli.stream_once(
        _client(["all done"], finish_reason="stop"),
        [{"role": "user", "content": "hi"}],
        use_tools=False,
    )
    assert usage["finish_reason"] == "stop"
    assert usage["truncated"] is False


def test_stream_once_flags_length_truncation(qwen_cli):
    # finish_reason == "length" -> the token cap cut the model off mid-output.
    _text, _calls, usage = qwen_cli.stream_once(
        _client(["this got cut o"], finish_reason="length"),
        [{"role": "user", "content": "hi"}],
        use_tools=False,
    )
    assert usage["finish_reason"] == "length"
    assert usage["truncated"] is True


def test_stream_once_flags_dropped_stream_as_truncated(qwen_cli):
    # A dropped connection mid-stream is also an incomplete answer.
    _text, _calls, usage = qwen_cli.stream_once(
        _client(["partial ", "answer"], raise_at_end=True),
        [{"role": "user", "content": "hi"}],
        use_tools=False,
    )
    assert usage["truncated"] is True
    assert usage["finish_reason"] == "interrupted"
