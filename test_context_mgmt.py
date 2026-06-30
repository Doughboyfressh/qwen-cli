"""Tests for context window management improvements."""
import sys
import importlib.util

ROOT = r"C:\Users\Dough\.qwen-cli"
sys.path.insert(0, r"C:\Users\Dough\.qwen-cli")

# Load qwen-cli.py via importlib (hyphen in filename prevents direct import)
spec = importlib.util.spec_from_file_location("qwen_cli", ROOT + "\\qwen-cli.py")
_qwen_cli = importlib.util.module_from_spec(spec)
sys.modules["qwen_cli"] = _qwen_cli
spec.loader.exec_module(_qwen_cli)

_track_context_growth = _qwen_cli._track_context_growth
_estimate_turns_remaining = _qwen_cli._estimate_turns_remaining
_detect_session_type = _qwen_cli._detect_session_type
_adaptive_compaction_threshold = _qwen_cli._adaptive_compaction_threshold
_context_growth_history = _qwen_cli._context_growth_history

def test_track_context_growth_records_values():
    while _context_growth_history:
        _context_growth_history.pop()
    _track_context_growth(1000)
    _track_context_growth(2000)
    _track_context_growth(3000)
    assert len(_context_growth_history) == 3
    assert _context_growth_history == [1000, 2000, 3000]

def test_track_context_growth_caps_at_10():
    while _context_growth_history:
        _context_growth_history.pop()
    for i in range(15):
        _track_context_growth(i * 100)
    assert len(_context_growth_history) == 10

def test_estimate_turns_remaining_insufficient_data():
    while _context_growth_history:
        _context_growth_history.pop()
    _track_context_growth(1000)
    assert _estimate_turns_remaining(1000) == -1

def test_estimate_turns_remaining_positive():
    while _context_growth_history:
        _context_growth_history.pop()
    for v in [10000, 12000, 14000, 16000, 18000]:
        _track_context_growth(v)
    token_limit = _qwen_cli.TOKEN_LIMIT
    threshold = token_limit * 80 // 100
    # remaining = threshold - current = 5000, growth_rate = 2000, turns = 5000 // 2000 = 2
    current = threshold - 5000
    turns = _estimate_turns_remaining(current, 80)
    # growth_rate is 2000, remaining is exactly 5000
    expected = 5000 // 2000
    assert turns == expected, f"Expected {expected}, got {turns} (TOKEN_LIMIT={token_limit}, threshold={threshold}, remaining={threshold - current})"

def test_estimate_turns_remaining_no_growth():
    while _context_growth_history:
        _context_growth_history.pop()
    for v in [50000, 50000, 50000]:
        _track_context_growth(v)
    assert _estimate_turns_remaining(50000) == -1

def test_estimate_turns_remaining_shrinking():
    while _context_growth_history:
        _context_growth_history.pop()
    for v in [50000, 48000, 46000]:
        _track_context_growth(v)
    assert _estimate_turns_remaining(46000) == -1

def test_detect_session_type_normal():
    history = [{"role": "user", "content": "Hello"}]
    assert _detect_session_type(history) == "normal"

def test_detect_session_type_heavy():
    history = [
        {"role": "user", "content": "x" * 10000},
        {"role": "assistant", "content": "x" * 10000},
    ]
    assert _detect_session_type(history) == "heavy"

def test_detect_session_type_chatty():
    history = []
    for i in range(30):
        history.append({"role": "user", "content": f"msg {i}"})
        history.append({"role": "assistant", "content": f"reply {i}"})
    assert _detect_session_type(history) == "chatty"

def test_detect_session_type_empty():
    assert _detect_session_type([]) == "normal"

def test_detect_session_type_no_text_roles():
    history = [{"role": "system", "content": "be nice"}]
    assert _detect_session_type(history) == "normal"

def test_adaptive_compaction_threshold_heavy():
    assert _adaptive_compaction_threshold("heavy") == 70

def test_adaptive_compaction_threshold_chatty():
    assert _adaptive_compaction_threshold("chatty") == 85

def test_adaptive_compaction_threshold_normal():
    assert _adaptive_compaction_threshold("normal") == 80

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
