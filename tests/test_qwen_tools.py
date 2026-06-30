"""Tests for the shared qwen_tools module: _resolve and _apply_diff."""
from pathlib import Path

import pytest


def test_resolve_relative_is_anchored_to_cwd(qwen_tools):
    resolved = qwen_tools._resolve("notes.txt")
    assert resolved.is_absolute()
    assert resolved == (Path.cwd() / "notes.txt").resolve()


def test_resolve_absolute_is_preserved(qwen_tools, tmp_path):
    target = tmp_path / "x.txt"
    assert qwen_tools._resolve(str(target)) == target.resolve()


def test_apply_diff_replaces_a_line(qwen_tools):
    original = "alpha\nbravo\ncharlie\n"
    diff = "@@ -2,1 +2,1 @@\n-bravo\n+BRAVO\n"
    assert qwen_tools._apply_diff(original, diff) == "alpha\nBRAVO\ncharlie\n"


def test_apply_diff_add_and_remove(qwen_tools):
    original = "one\ntwo\nthree\n"
    diff = "@@ -2,1 +2,2 @@\n two\n+inserted\n"
    assert qwen_tools._apply_diff(original, diff) == "one\ntwo\ninserted\nthree\n"


def test_apply_diff_empty_is_noop(qwen_tools):
    original = "unchanged\n"
    assert qwen_tools._apply_diff(original, "") == original
    assert qwen_tools._apply_diff(original, "   \n") == original


def test_apply_diff_mismatch_raises(qwen_tools):
    original = "alpha\nbravo\ncharlie\n"
    bad = "@@ -2,1 +2,1 @@\n-NOTHERE\n+X\n"
    with pytest.raises(ValueError):
        qwen_tools._apply_diff(original, bad)
