"""Tests for the tools the model uses to FIND things.

do_search_files is the model's grep — it reaches for it constantly, and it was
85% uncovered. A regression here doesn't crash: it returns "No matches", the
model concludes the code isn't there, and confidently reasons from a false
negative. Same for find_files and @file expansion.
"""

import pytest


@pytest.fixture()
def tree(tmp_path):
    """A small project with the shapes that trip search: ignored dirs, binaries,
    nested sources, a big file."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\n\ndef connect():\n    return os.environ['DB_URL']\n\n\ndef disconnect():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "util.py").write_text("def connect():\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("call connect() to start\n", encoding="utf-8")

    # Must never be searched — this is what made searches hang on real projects.
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "vendored.py").write_text("def connect():\n    pass\n", encoding="utf-8")

    node = tmp_path / "node_modules"
    node.mkdir()
    (node / "dep.js").write_text("function connect() {}\n", encoding="utf-8")

    (tmp_path / "image.bin").write_bytes(b"\x00\x01connect\x00\x02")
    return tmp_path


# ---------------------------------------------------------------------------
# do_search_files
# ---------------------------------------------------------------------------


def test_search_finds_matches_with_line_numbers(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree), "def connect")

    assert "app.py" in out
    assert "util.py" in out
    assert "L" in out, "line numbers must be reported so the model can navigate"


def test_search_never_descends_into_ignored_dirs(qwen_cli, tree):
    """.venv/node_modules are why an unpruned walk hung on 10k+ files."""
    out = qwen_cli.do_search_files(str(tree), "connect")

    assert "vendored.py" not in out
    assert "dep.js" not in out


def test_search_skips_binary_files(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree), "connect")
    assert "image.bin" not in out


def test_search_reports_no_matches_honestly(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree), "zzz_definitely_absent")
    assert "No matches" in out


def test_search_treats_the_query_as_a_regex(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree), r"def (connect|disconnect)")
    assert "disconnect" in out


def test_an_invalid_regex_falls_back_to_a_literal_search(qwen_cli, tree):
    """A model writing `connect(` must get results, not a crash."""
    (tree / "src" / "call.py").write_text("connect(host)\n", encoding="utf-8")

    out = qwen_cli.do_search_files(str(tree), "connect(")

    assert "[error" not in out
    assert "call.py" in out


def test_search_glob_filters_which_files_are_read(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree), "connect", pattern="*.md")

    assert "README.md" in out
    assert "app.py" not in out


def test_context_lines_are_included_and_the_match_is_marked(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree), "DB_URL", context=2)

    assert "def connect" in out, "context should pull in the surrounding lines"
    assert ">" in out, "the matching line must be distinguishable from its context"


def test_separate_context_ranges_are_delimited(qwen_cli, tmp_path):
    """Two far-apart hits in one file must not read as one contiguous block."""
    f = tmp_path / "wide.py"
    f.write_text("hit\n" + "\n".join(f"filler{i}" for i in range(40)) + "\nhit\n", encoding="utf-8")

    out = qwen_cli.do_search_files(str(tmp_path), "hit", context=1)

    assert "···" in out


def test_adjacent_context_ranges_are_merged(qwen_cli, tmp_path):
    f = tmp_path / "near.py"
    f.write_text("hit\nfiller\nhit\n", encoding="utf-8")

    out = qwen_cli.do_search_files(str(tmp_path), "hit", context=2)

    assert "···" not in out, "overlapping context windows should merge, not repeat"


def test_search_a_single_file_directly(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree / "src" / "app.py"), "connect")
    assert "L" in out


def test_search_missing_path_is_reported(qwen_cli, tmp_path):
    out = qwen_cli.do_search_files(str(tmp_path / "nope"), "x")
    assert "not found" in out


def test_search_caps_runaway_result_counts(qwen_cli, tmp_path):
    """A pathological query must not dump the whole repo into the context window."""
    for i in range(12):
        (tmp_path / f"f{i}.txt").write_text("match\n" * 100, encoding="utf-8")

    out = qwen_cli.do_search_files(str(tmp_path), "match")

    assert "stopped at 500" in out, "the 500-match cap must be reported, not silently applied"


def test_search_is_case_insensitive(qwen_cli, tree):
    out = qwen_cli.do_search_files(str(tree), "DEF CONNECT")
    assert "app.py" in out


# ---------------------------------------------------------------------------
# do_find_files
# ---------------------------------------------------------------------------


def test_find_matches_by_glob(qwen_cli, tree):
    out = qwen_cli.do_find_files(str(tree), "*.py")

    assert "app.py" in out
    assert "util.py" in out
    assert "README.md" not in out


def test_find_prunes_ignored_dirs(qwen_cli, tree):
    out = qwen_cli.do_find_files(str(tree), "*.py")
    assert "vendored.py" not in out


def test_find_accepts_a_recursive_glob_prefix(qwen_cli, tree):
    """The model habitually writes '**/*.py'; only the basename pattern matters."""
    out = qwen_cli.do_find_files(str(tree), "**/*.py")
    assert "app.py" in out


def test_find_reports_no_matches(qwen_cli, tree):
    out = qwen_cli.do_find_files(str(tree), "*.rs")
    assert "No files matched" in out


def test_find_missing_path_is_reported(qwen_cli, tmp_path):
    assert "not found" in qwen_cli.do_find_files(str(tmp_path / "nope"), "*")


# ---------------------------------------------------------------------------
# expand_at_refs — @file and @file::symbol
# ---------------------------------------------------------------------------


def test_at_ref_injects_the_file(qwen_cli, tree, monkeypatch):
    monkeypatch.chdir(tree)
    out = qwen_cli.expand_at_refs("look at @src/app.py please")

    assert "look at" in out
    assert "def connect" in out, "the file body must be injected"
    assert "```python" in out, "language should be tagged for the model"


def test_at_ref_symbol_injects_only_that_symbol(qwen_cli, tree, monkeypatch):
    """The whole point is saving context — the rest of the file must not come too."""
    monkeypatch.chdir(tree)
    out = qwen_cli.expand_at_refs("explain @src/app.py::connect")

    assert "def connect" in out
    assert "def disconnect" not in out


def test_at_ref_unknown_symbol_injects_nothing(qwen_cli, tree, monkeypatch):
    monkeypatch.chdir(tree)
    out = qwen_cli.expand_at_refs("explain @src/app.py::nosuchfunc")

    assert "def connect" not in out


def test_at_ref_to_a_missing_file_is_left_alone(qwen_cli, tree, monkeypatch):
    monkeypatch.chdir(tree)
    text = "what about @src/ghost.py"
    assert qwen_cli.expand_at_refs(text) == text


def test_at_ref_skips_binaries(qwen_cli, tree, monkeypatch):
    monkeypatch.chdir(tree)
    out = qwen_cli.expand_at_refs("check @image.bin")
    assert "```" not in out


@pytest.mark.parametrize("suffix", ["?", "!", ".", ",", ")", ":", ";"])
def test_at_ref_strips_trailing_punctuation(qwen_cli, tree, monkeypatch, suffix):
    """'what does @src/app.py do?' written as '@src/app.py?' must still resolve.

    '?' and '!' used to be missing from the strip set, so the most natural way to
    ask about a file silently injected nothing and the model answered about a
    file it had never seen.
    """
    monkeypatch.chdir(tree)
    out = qwen_cli.expand_at_refs(f"have you seen @src/app.py{suffix}")

    assert "def connect" in out, f"@path{suffix} failed to resolve"


def test_at_ref_symbol_strips_trailing_punctuation(qwen_cli, tree, monkeypatch):
    monkeypatch.chdir(tree)
    out = qwen_cli.expand_at_refs("what does @src/app.py::connect do?")

    assert "def connect" in out
    assert "def disconnect" not in out


def test_multiple_at_refs_all_expand(qwen_cli, tree, monkeypatch):
    monkeypatch.chdir(tree)
    out = qwen_cli.expand_at_refs("compare @src/app.py and @src/util.py")

    assert out.count("```python") == 2


def test_text_without_at_refs_is_unchanged(qwen_cli):
    assert qwen_cli.expand_at_refs("just a plain message") == "just a plain message"


# ---------------------------------------------------------------------------
# _smart_cap — what reaches the context window
# ---------------------------------------------------------------------------


def test_read_file_results_are_never_capped(qwen_cli):
    """read_file paginates itself with an honest range header and a continuation
    hint. Head-truncating it here would cut the footer and put the model back to
    believing it read a whole file it only saw part of."""
    huge = "line\n" * 100_000
    assert qwen_cli._smart_cap(None, huge, "read_file") is huge


def test_small_results_pass_through_untouched(qwen_cli):
    assert qwen_cli._smart_cap(None, "short", "run_command") == "short"


def test_oversized_non_summarizable_results_are_truncated_with_a_notice(qwen_cli):
    big = "x" * (qwen_cli.TOOL_RESULT_LIMIT + 5_000)

    out = qwen_cli._smart_cap(None, big, "run_command")

    assert len(out) < len(big)
    assert "truncated" in out, "silent truncation would let the model assume it saw everything"


def test_web_results_are_condensed_by_key_facts_without_an_llm_call(qwen_cli):
    """The cheap path: pull URLs/numbers/short lines rather than paying for a
    summarization round-trip. client=None proves no LLM call is made."""
    # Long, digit-free filler lines are neither short nor fact-bearing, so only
    # the URL line survives. Must exceed TOOL_RESULT_LIMIT to trigger capping.
    body = "\n".join("filler line " + "y" * 200 for _ in range(120))
    result = "https://example.com/found\n" + body
    assert len(result) > qwen_cli.TOOL_RESULT_LIMIT

    out = qwen_cli._smart_cap(None, result, "web_search")

    assert "key facts extracted" in out
    assert "https://example.com/found" in out, "URLs must survive condensing"
    assert len(out) < len(result)
