"""Project symbol indexer — builds/loads/saves symbol indices for project files."""

import ast
import fnmatch
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

_logger = logging.getLogger(__name__)

from qwen_cli.core.config import INDEX_DIR  # noqa: E402

_INDEX_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".mjs",
    ".go",
    ".rs",
    ".java",
    ".cs",
    ".cpp",
    ".c",
    ".rb",
    ".php",
}

IGNORE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    "dist",
    "build",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    "coverage",
    ".tox",
    "env",
    ".eggs",
    "target",
    "out",
    ".nuxt",
    ".output",
    "models",
}


def _project_index_path(root: Path) -> Path:
    slug = hashlib.md5(str(root.resolve()).encode()).hexdigest()[:12]
    return INDEX_DIR / f"{slug}.json"


def _extract_py_symbols(filepath: Path) -> dict:
    try:
        tree = ast.parse(filepath.read_bytes(), filename=str(filepath))
    except Exception:
        return {}
    functions: list[str] = []
    classes: list[dict] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in ast.iter_child_nodes(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append({"name": node.name, "methods": methods})
    return {"functions": functions, "classes": classes}


def _extract_generic_symbols(filepath: Path, ext: str) -> dict:
    functions: list[str] = []
    classes: list[str] = []
    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    for line in lines:
        s = line.strip()
        if ext in (".js", ".ts", ".tsx", ".jsx", ".mjs"):
            m = re.match(r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)", s)
            if m:
                classes.append(m.group(1))
        elif ext == ".go":
            m = re.match(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"type\s+(\w+)\s+struct", s)
            if m:
                classes.append(m.group(1))
        elif ext == ".rs":
            m = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", s)
            if m:
                classes.append(m.group(1))
        elif ext in (".java", ".cs"):
            m = re.match(
                r"(?:(?:public|private|protected|static|final|abstract|synchronized|async|void|int|long|float|double|bool|boolean|string|char|byte|short|var|auto|Task<\w+>|List<\w+>|Map<\w+,\s*\w+>|Set<\w+>|Dictionary<\w+,\s*\w+>)\s+)*"
                r"(\w+)\s*\(",
                s,
            )
            if m and m.group(1) not in (
                "if",
                "for",
                "while",
                "switch",
                "catch",
                "using",
                "new",
                "return",
                "throw",
                "else",
                "try",
            ):
                functions.append(m.group(1))
            m = re.match(
                r"(?:(?:public|private|protected|abstract|static|sealed|partial|internal|friend)\s+)*"
                r"(?:class|struct|interface|enum)\s+(\w+)",
                s,
            )
            if m:
                classes.append(m.group(1))
    return {
        "functions": list(dict.fromkeys(functions)),
        "classes": list(dict.fromkeys(classes)),
    }


def _extract_symbol_from_file(p: Path, symbol: str) -> str | None:
    ext = p.suffix.lower()
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if ext == ".py":
        try:
            tree = ast.parse(source.encode(), filename=str(p))
            lines = source.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == symbol and hasattr(node, "end_lineno"):
                        return "\n".join(lines[node.lineno - 1 : node.end_lineno])
        except Exception:
            _logger.debug("AST extraction failed for %s", p)
        return None
    lines = source.splitlines()
    patterns = [
        re.compile(rf"(?:async\s+)?(?:def|function|func|fn)\s+{re.escape(symbol)}\s*[\(\{{<]"),
        re.compile(rf"(?:class|struct|enum|trait|interface)\s+{re.escape(symbol)}\b"),
        re.compile(rf"\b{re.escape(symbol)}\s*[:=]\s*(?:async\s+)?\(?(?:function)?\s*\(?"),
    ]
    start = None
    for i, line in enumerate(lines):
        if any(rx.search(line) for rx in patterns):
            start = i
            break
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and end - start < 150:
        line = lines[end]
        if line and not line[0].isspace() and end > start + 2:
            break
        end += 1
    return "\n".join(lines[start:end])


def load_qwenignore(root: Path) -> set[str]:
    f = root / ".qwenignore"
    if not f.exists():
        return set()
    patterns: set[str] = set()
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.add(line)
    return patterns


def _qwen_ignored(entry: Path, root: Path, extra: set[str]) -> bool:
    name = entry.name
    rel = str(entry.relative_to(root)).replace("\\", "/")
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in extra)


def build_symbol_index(root: Path, max_files: int = 500) -> dict:
    extra_ignore = load_qwenignore(root)
    index: dict[str, dict] = {}
    for fpath in sorted(root.rglob("*")):
        if len(index) >= max_files:
            break
        if not fpath.is_file():
            continue
        if any(part in IGNORE_DIRS for part in fpath.parts):
            continue
        if _qwen_ignored(fpath, root, extra_ignore):
            continue
        ext = fpath.suffix.lower()
        if ext not in _INDEX_EXTS or fpath.stat().st_size > 500_000:
            continue
        rel = str(fpath.relative_to(root)).replace("\\", "/")
        syms = _extract_py_symbols(fpath) if ext == ".py" else _extract_generic_symbols(fpath, ext)
        if syms.get("functions") or syms.get("classes"):
            index[rel] = syms
    return index


def _save_symbol_index(root: Path, index: dict) -> None:
    data = {"root": str(root), "built_at": datetime.now().isoformat(), "index": index}
    _project_index_path(root).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_symbol_index(root: Path, max_age_hours: int = 24) -> dict | None:
    p = _project_index_path(root)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        age_h = (datetime.now() - datetime.fromisoformat(data["built_at"])).total_seconds() / 3600
        return data["index"] if age_h <= max_age_hours else None
    except Exception:
        return None


def _format_symbol_index(index: dict) -> str:
    lines = []
    for rel, syms in sorted(index.items()):
        parts = [f"class {c['name']}" if isinstance(c, dict) else f"class {c}" for c in syms.get("classes", [])]
        parts += [f"{f}()" for f in syms.get("functions", [])]
        if parts:
            lines.append(f"{rel}: {', '.join(parts)}")
    return "\n".join(lines)
