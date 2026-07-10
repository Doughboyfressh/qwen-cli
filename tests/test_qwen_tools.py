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


# =============================================================================
# _merge_results
# =============================================================================


def test_merge_results_empty(qwen_tools):
    sources, merged = qwen_tools._merge_results([], 10)
    assert sources == ""
    assert merged == []


def test_merge_results_single_engine(qwen_tools):
    items = [
        {"href": "http://a.com", "title": "A"},
        {"href": "http://b.com", "title": "B"},
    ]
    sources, merged = qwen_tools._merge_results([("Google", items)], 10)
    assert sources == "Google"
    assert len(merged) == 2
    assert merged[0]["title"] == "A"
    assert merged[1]["title"] == "B"


def test_merge_results_dedup(qwen_tools):
    google = [{"href": "http://x.com/?q=1", "title": "X from Google"}]
    brave = [{"href": "http://x.com/", "title": "X from Brave"}]
    sources, merged = qwen_tools._merge_results(
        [("Google", google), ("Brave", brave)],
        10,
    )
    assert sources == "Google+Brave"
    assert len(merged) == 1
    assert merged[0]["title"] == "X from Google"


def test_merge_results_max_results(qwen_tools):
    items = [{"href": f"http://example.com/{i}"} for i in range(10)]
    sources, merged = qwen_tools._merge_results([("E", items)], 3)
    assert len(merged) == 3


def test_merge_results_skips_empty_engine(qwen_tools):
    sources, merged = qwen_tools._merge_results(
        [("Google", []), ("Brave", [{"href": "http://a.com"}])],
        10,
    )
    assert sources == "Brave"
    assert len(merged) == 1


# =============================================================================
# _format_search_results
# =============================================================================


def test_format_search_results_basic(qwen_tools):
    results = [{"title": "Example", "href": "http://ex.com", "body": "A page"}]
    out = qwen_tools._format_search_results("test query", results)
    assert 'for: "test query"' in out
    assert "1. Example" in out
    assert "URL: http://ex.com" in out
    assert "A page" in out


def test_format_search_results_with_source(qwen_tools):
    results = [{"title": "X", "href": "http://x.com"}]
    out = qwen_tools._format_search_results("q", results, source="Google")
    assert "(via Google)" in out


def test_format_search_results_empty(qwen_tools):
    out = qwen_tools._format_search_results("q", [])
    assert 'for: "q"' in out
    assert out.count("\n") == 1


def test_format_search_results_missing_fields(qwen_tools):
    results = [
        {"name": "Named Result", "url": "http://named.com", "snippet": "SNIPPET"},
        {"body": "Has body but no title/url"},
    ]
    out = qwen_tools._format_search_results("q", results)
    assert "1. Named Result" in out
    assert "URL: http://named.com" in out
    assert "SNIPPET" in out
    assert "2. Result 2" in out
    assert "Has body but no title/url" in out


# =============================================================================
# _smart_truncate
# =============================================================================


def test_smart_truncate_shorter_than_limit(qwen_tools):
    text = "Short text with sentence. "
    assert qwen_tools._smart_truncate(text, 100) == text


def test_smart_truncate_sentence_boundary(qwen_tools):
    text = "First sentence. Second sentence. Third sentence."
    result = qwen_tools._smart_truncate(text, 25)
    assert result.startswith("First sentence.")
    assert "truncated" in result
    assert len(result) < len(text)


def test_smart_truncate_hard_cut(qwen_tools):
    text = "NoBoundaries" * 30
    result = qwen_tools._smart_truncate(text, 50)
    expected = "NoBoundaries" * 4 + "No" + "\n\n... [truncated at 50 chars]"
    assert result == expected
    assert len(result) < len(text)


def test_smart_truncate_very_long(qwen_tools):
    text = "A" * 400 + ". " + "B" * 2000
    result = qwen_tools._smart_truncate(text, 500)
    assert result.startswith("A" * 400 + ".")
    assert "truncated" in result
    assert len(result) < len(text)


# =============================================================================
# _backup_file
# =============================================================================


def test_backup_file_creates_dir_and_backup(qwen_tools, tmp_path, monkeypatch):
    backup_dir = tmp_path / "my_backups"
    monkeypatch.setattr(qwen_tools, "BACKUPS_DIR", backup_dir)
    src = tmp_path / "doc.txt"
    src.write_text("important data", encoding="utf-8")
    qwen_tools._backup_file(src)
    assert backup_dir.is_dir()
    backups = list(backup_dir.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "important data"


def test_backup_file_preserves_original(qwen_tools, tmp_path, monkeypatch):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(qwen_tools, "BACKUPS_DIR", backup_dir)
    src = tmp_path / "keep.txt"
    src.write_text("original", encoding="utf-8")
    qwen_tools._backup_file(src)
    assert src.read_text(encoding="utf-8") == "original"


# =============================================================================
# _decompress
# =============================================================================


def test_decompress_gzip(qwen_tools):
    import gzip

    raw = gzip.compress(b"hello gzip")
    assert qwen_tools._decompress(raw, "gzip") == b"hello gzip"


def test_decompress_deflate(qwen_tools):
    import zlib

    raw = zlib.compress(b"hello deflate")
    assert qwen_tools._decompress(raw, "deflate") == b"hello deflate"


def test_decompress_raw_deflate(qwen_tools):
    import zlib

    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw_deflate = compressor.compress(b"raw deflate") + compressor.flush()
    result = qwen_tools._decompress(raw_deflate, "deflate")
    assert result == b"raw deflate"


def test_decompress_unknown_encoding(qwen_tools):
    raw = b"raw bytes"
    assert qwen_tools._decompress(raw, "identity") is raw
    assert qwen_tools._decompress(raw, "unknown") == b"raw bytes"


def test_decompress_corrupted_fallback(qwen_tools):
    result = qwen_tools._decompress(b"not valid gzip data", "gzip")
    assert result == b"not valid gzip data"


# =============================================================================
# _pdf_to_text
# =============================================================================


def test_pdf_to_text_success(qwen_tools, monkeypatch):
    import sys

    mock_page = type("MockPage", (), {"extract_text": lambda self: "Extracted PDF content"})
    mock_reader = type("MockReader", (), {"pages": [mock_page()]})
    fake = type(sys)("pypdf")
    fake.PdfReader = lambda stream: mock_reader
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    result = qwen_tools._pdf_to_text(b"fake pdf", 1000)
    assert "Extracted PDF content" in result


def test_pdf_to_text_missing_pypdf(qwen_tools, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    result = qwen_tools._pdf_to_text(b"x", 100)
    assert result == "[PDF support requires: pip install pypdf]"


def test_pdf_to_text_error(qwen_tools, monkeypatch):
    import sys

    class FailingReader:
        def __init__(self, stream):
            raise ValueError("corrupt PDF")

    fake = type(sys)("pypdf")
    fake.PdfReader = FailingReader
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    result = qwen_tools._pdf_to_text(b"garbage", 100)
    assert "PDF extraction error" in result
