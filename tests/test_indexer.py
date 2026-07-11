"""Tests for qwen_cli.core.indexer — symbol extraction, ignore rules, index cache."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import qwen_cli.core.indexer as idx


# ---------------------------------------------------------------------------
# Python symbol extraction (AST-based)
# ---------------------------------------------------------------------------


class TestExtractPySymbols:
    def test_functions_classes_and_methods(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "def top():\n    pass\n\n"
            "async def atop():\n    pass\n\n"
            "class Foo:\n"
            "    def method_a(self):\n        pass\n"
            "    async def method_b(self):\n        pass\n",
            encoding="utf-8",
        )
        syms = idx._extract_py_symbols(f)
        assert syms["functions"] == ["top", "atop"]
        assert syms["classes"] == [{"name": "Foo", "methods": ["method_a", "method_b"]}]

    def test_nested_functions_not_included(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def outer():\n    def inner():\n        pass\n", encoding="utf-8")
        assert idx._extract_py_symbols(f)["functions"] == ["outer"]

    def test_syntax_error_returns_empty(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def broken(:\n", encoding="utf-8")
        assert idx._extract_py_symbols(f) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert idx._extract_py_symbols(tmp_path / "nope.py") == {}


# ---------------------------------------------------------------------------
# Generic (regex-based) symbol extraction
# ---------------------------------------------------------------------------


class TestExtractGenericSymbols:
    def test_javascript(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text(
            "export function main() {}\n"
            "const handler = async (req) => {}\n"
            "export default class Widget {}\n",
            encoding="utf-8",
        )
        syms = idx._extract_generic_symbols(f, ".js")
        assert "main" in syms["functions"]
        assert "handler" in syms["functions"]
        assert "Widget" in syms["classes"]

    def test_go(self, tmp_path):
        f = tmp_path / "main.go"
        f.write_text(
            "func Run() {\n}\n"
            "func (s *Server) Handle(w http.ResponseWriter) {\n}\n"
            "type Server struct {\n}\n",
            encoding="utf-8",
        )
        syms = idx._extract_generic_symbols(f, ".go")
        assert "Run" in syms["functions"]
        assert "Handle" in syms["functions"]
        assert "Server" in syms["classes"]

    def test_rust(self, tmp_path):
        f = tmp_path / "lib.rs"
        f.write_text(
            "pub async fn fetch() {}\nfn private_helper() {}\npub struct Client {}\npub trait Runner {}\n",
            encoding="utf-8",
        )
        syms = idx._extract_generic_symbols(f, ".rs")
        assert {"fetch", "private_helper"} <= set(syms["functions"])
        assert {"Client", "Runner"} <= set(syms["classes"])

    def test_java_keywords_not_treated_as_functions(self, tmp_path):
        f = tmp_path / "A.java"
        f.write_text(
            "public class A {\n"
            "    public void doWork() {\n"
            "        if (x) {\n"
            "        while (y) {\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        syms = idx._extract_generic_symbols(f, ".java")
        assert "doWork" in syms["functions"]
        assert "if" not in syms["functions"]
        assert "while" not in syms["functions"]
        assert "A" in syms["classes"]

    def test_duplicates_removed(self, tmp_path):
        f = tmp_path / "dup.js"
        f.write_text("function go() {}\nfunction go() {}\n", encoding="utf-8")
        assert idx._extract_generic_symbols(f, ".js")["functions"] == ["go"]


# ---------------------------------------------------------------------------
# Single-symbol source extraction
# ---------------------------------------------------------------------------


class TestExtractSymbolFromFile:
    def test_python_function_body(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n", encoding="utf-8")
        src = idx._extract_symbol_from_file(f, "b")
        assert src is not None
        assert src.startswith("def b():")
        assert "return 2" in src
        assert "return 1" not in src

    def test_python_class_body(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("class C:\n    x = 1\n\n    def m(self):\n        pass\n", encoding="utf-8")
        src = idx._extract_symbol_from_file(f, "C")
        assert src is not None and "class C:" in src and "def m" in src

    def test_unknown_symbol_returns_none(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def a():\n    pass\n", encoding="utf-8")
        assert idx._extract_symbol_from_file(f, "does_not_exist") is None

    def test_js_symbol_regex_fallback(self, tmp_path):
        f = tmp_path / "m.js"
        f.write_text("function target() {\n  return 42;\n}\nlet after = 1;\n", encoding="utf-8")
        src = idx._extract_symbol_from_file(f, "target")
        assert src is not None and "return 42" in src


# ---------------------------------------------------------------------------
# .qwenignore
# ---------------------------------------------------------------------------


class TestQwenignore:
    def test_missing_file_returns_empty_set(self, tmp_path):
        assert idx.load_qwenignore(tmp_path) == set()

    def test_parses_patterns_and_skips_comments(self, tmp_path):
        (tmp_path / ".qwenignore").write_text("# comment\n*.log\nsecrets/\n\n", encoding="utf-8")
        assert idx.load_qwenignore(tmp_path) == {"*.log", "secrets/"}

    def test_qwen_ignored_matches_name_and_relpath(self, tmp_path):
        target = tmp_path / "sub" / "app.log"
        target.parent.mkdir()
        target.write_text("", encoding="utf-8")
        assert idx._qwen_ignored(target, tmp_path, {"*.log"}) is True
        assert idx._qwen_ignored(target, tmp_path, {"sub/app.log"}) is True
        assert idx._qwen_ignored(target, tmp_path, {"*.py"}) is False


# ---------------------------------------------------------------------------
# build_symbol_index
# ---------------------------------------------------------------------------


class TestBuildSymbolIndex:
    def _make_project(self, tmp_path: Path) -> Path:
        (tmp_path / "app.py").write_text("def entry():\n    pass\n", encoding="utf-8")
        (tmp_path / "util.js").write_text("function helper() {}\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("# doc\n", encoding="utf-8")  # not indexed ext
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("function hidden() {}\n", encoding="utf-8")
        return tmp_path

    def test_indexes_code_and_skips_ignored_dirs(self, tmp_path):
        root = self._make_project(tmp_path)
        index = idx.build_symbol_index(root)
        assert "app.py" in index
        assert index["app.py"]["functions"] == ["entry"]
        assert "util.js" in index
        assert not any("node_modules" in k for k in index)
        assert not any(k.endswith(".md") for k in index)

    def test_respects_qwenignore(self, tmp_path):
        root = self._make_project(tmp_path)
        (root / ".qwenignore").write_text("util.js\n", encoding="utf-8")
        index = idx.build_symbol_index(root)
        assert "util.js" not in index
        assert "app.py" in index

    def test_max_files_cap(self, tmp_path):
        for i in range(5):
            (tmp_path / f"m{i}.py").write_text(f"def f{i}():\n    pass\n", encoding="utf-8")
        index = idx.build_symbol_index(tmp_path, max_files=2)
        assert len(index) == 2


# ---------------------------------------------------------------------------
# Index cache save/load
# ---------------------------------------------------------------------------


class TestIndexCache:
    @pytest.fixture(autouse=True)
    def _index_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(idx, "INDEX_DIR", tmp_path / "indices")
        (tmp_path / "indices").mkdir()
        self.tmp_path = tmp_path

    def test_roundtrip(self):
        root = self.tmp_path / "proj"
        root.mkdir()
        index = {"a.py": {"functions": ["f"], "classes": []}}
        idx._save_symbol_index(root, index)
        assert idx._load_symbol_index(root) == index

    def test_stale_index_returns_none(self):
        root = self.tmp_path / "proj"
        root.mkdir()
        idx._save_symbol_index(root, {"a.py": {"functions": ["f"], "classes": []}})
        p = idx._project_index_path(root)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["built_at"] = (datetime.now() - timedelta(hours=48)).isoformat()
        p.write_text(json.dumps(data), encoding="utf-8")
        assert idx._load_symbol_index(root, max_age_hours=24) is None

    def test_missing_index_returns_none(self):
        assert idx._load_symbol_index(self.tmp_path / "never-indexed") is None

    def test_corrupt_index_returns_none(self):
        root = self.tmp_path / "proj"
        root.mkdir()
        idx._project_index_path(root).write_text("{not json", encoding="utf-8")
        assert idx._load_symbol_index(root) is None


def test_format_symbol_index_handles_both_class_shapes():
    index = {
        "a.py": {"functions": ["f"], "classes": [{"name": "C", "methods": ["m"]}]},
        "b.go": {"functions": ["Run"], "classes": ["Server"]},
    }
    out = idx._format_symbol_index(index)
    assert "a.py: class C, f()" in out
    assert "b.go: class Server, Run()" in out
