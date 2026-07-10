"""Tests for qwen_tools.presearch_decision — the auto-web-search trigger
shared by the CLI and the web UI."""


def _search(qwen_tools, text, mode):
    return qwen_tools.presearch_decision(text, mode)[0]


def test_off_mode_never_searches(qwen_tools):
    assert _search(qwen_tools, "what is the latest python version", "off") is False


def test_too_short_is_skipped(qwen_tools):
    assert _search(qwen_tools, "hi", "aggressive") is False


def test_conversational_is_skipped(qwen_tools):
    for ack in ("thanks!", "ok", "sounds good", "go ahead"):
        assert _search(qwen_tools, ack, "aggressive") is False


def test_code_block_is_skipped(qwen_tools):
    assert _search(qwen_tools, "```python\nprint(1)\n```", "aggressive") is False


def test_text_transform_is_skipped(qwen_tools):
    for t in (
        "rewrite this paragraph to be more formal",
        "translate the following to French",
        "proofread this for me",
    ):
        assert _search(qwen_tools, t, "aggressive") is False


def test_meta_question_about_app_is_skipped(qwen_tools):
    assert _search(qwen_tools, "how can we improve the cli", "aggressive") is False
    assert _search(qwen_tools, "what are your capabilities", "aggressive") is False


def test_file_op_is_skipped(qwen_tools):
    assert _search(qwen_tools, "write the file config.py", "aggressive") is False
    assert _search(qwen_tools, "/help", "aggressive") is False


def test_smart_requires_factual_intent(qwen_tools):
    # No question/factual signal -> smart skips, aggressive still fires.
    assert _search(qwen_tools, "center a div in css please", "smart") is False
    assert _search(qwen_tools, "center a div in css please", "aggressive") is True


def test_smart_fires_on_questions(qwen_tools):
    for t in (
        "what is the latest python version",
        "latest LM Studio release?",
        "is qwen 3.6 out yet",
        "tell me about the Tokyo housing market in 2026",
    ):
        assert _search(qwen_tools, t, "smart") is True


def test_unknown_mode_defaults_to_aggressive(qwen_tools):
    assert _search(qwen_tools, "center a div in css please", "wat") is True


def test_query_extracted_from_long_message(qwen_tools):
    long = (
        "I was reading through a lot of documentation today and there is a great "
        "deal of background that I want to give you first so that you really "
        "understand the full situation before you answer. "
        "What is the current price of an RTX 5090?"
    )
    do_search, query = qwen_tools.presearch_decision(long, "aggressive")
    assert do_search is True
    import datetime

    expected = f"What is the current price of an RTX 5090? {datetime.datetime.now().year}"
    assert query == expected
    assert len(query) <= 200
