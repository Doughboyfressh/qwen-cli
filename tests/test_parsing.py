"""Tests for model-output parsing: XML tool calls and <think> block stripping."""

import json

from qwen_cli.core.stream import _parse_xml_tool_calls, _strip_think


def test_parse_xml_tool_call_json_body(qwen_cli):
    text = '<tool_call>{"name": "web_search", "arguments": {"query": "cats"}}</tool_call>'
    clean, calls = _parse_xml_tool_calls(text)
    assert clean == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}


def test_parse_xml_tool_call_param_body(qwen_cli):
    text = '<tool_call><function_name>read_file</function_name><parameter name="path">notes.txt</parameter></tool_call>'
    _clean, calls = _parse_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "notes.txt"}


def test_parse_xml_no_tool_call_is_passthrough(qwen_cli):
    clean, calls = _parse_xml_tool_calls("just a normal answer")
    assert clean == "just a normal answer"
    assert calls == []


def test_parse_xml_keeps_surrounding_prose(qwen_cli):
    text = 'before <tool_call>{"name": "x", "arguments": {}}</tool_call> after'
    clean, calls = _parse_xml_tool_calls(text)
    assert "before" in clean and "after" in clean
    assert len(calls) == 1


def test_strip_think_no_block(qwen_cli):
    clean, think = _strip_think("plain answer")
    assert clean == "plain answer"
    assert think == ""


def test_strip_think_extracts_block(qwen_cli):
    clean, think = _strip_think("<think>reasoning here</think>the answer")
    assert clean == "the answer"
    assert think == "reasoning here"


def test_strip_think_multiple_blocks_joined(qwen_cli):
    clean, think = _strip_think("a<think>r1</think>b<think>r2</think>c")
    assert clean == "abc"
    assert "r1" in think and "r2" in think


def test_extract_first_code_block_any_language():
    text = 'intro\n```json\n{"a": 1}\n```\ntrailing'
    from qwen_cli.main import _extract_first_code_block

    assert _extract_first_code_block(text) == ("json", '{"a": 1}')


def test_extract_first_code_block_none_when_absent():
    from qwen_cli.main import _extract_first_code_block

    assert _extract_first_code_block("just prose, no fences") is None


def test_runnable_vs_first_block_differ():
    from qwen_cli.main import _extract_first_code_block, _extract_runnable_code

    # First block is non-runnable (json); the runnable extractor skips to python.
    text = "```json\n{}\n```\n```python\nprint(1)\n```"
    assert _extract_first_code_block(text) == ("json", "{}")
    assert _extract_runnable_code(text) == ("python", "print(1)")
