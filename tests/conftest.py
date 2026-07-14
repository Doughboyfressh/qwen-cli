"""Shared pytest fixtures.

qwen_cli is now a proper package; tests import it normally.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Redirect qwen_cli.main's file logging away from the production qwen.log.
# Tests deliberately exercise failure paths, and their tracebacks otherwise
# pile up in the real log. Must be set before qwen_cli.main is imported —
# conftest loads before any test module, so this top-level assignment is safe.
os.environ.setdefault("QWEN_LOG_FILE", str(Path(tempfile.gettempdir()) / "qwen-cli-test.log"))

# Redirect the entire data dir (memory.md, sessions/, backups/, ...) to a
# throwaway location: the suite used to write real entries into the user's
# memory.md ("modified code.txt, existing.txt, same.txt") and drop snapshot
# files into the real context_snapshots/. Same before-import constraint.
os.environ.setdefault("QWEN_DATA_DIR", tempfile.mkdtemp(prefix="qwen-cli-test-data-"))


def _import_module(name: str):
    """Import a top-level module by name (handles hyphen-free names)."""
    __import__(name)
    return sys.modules[name]


@pytest.fixture(scope="session")
def qwen_cli():
    """The qwen_cli.main module, loaded once per test session."""
    _import_module("qwen_cli")  # ensure package is loaded first
    _import_module("qwen_cli.main")  # ensure submodule is loaded
    return sys.modules["qwen_cli.main"]


@pytest.fixture(autouse=True)
def _reset_permission_flags():
    """Reset the process-wide permission flags before every test.

    main() legitimately latches _auto_approve and _unattended on for a spawned
    --task agent, and run_piped does the same — in production each is the last
    thing the process does, so nothing ever unsets them. In-process they leak:
    qwen_cli.main is imported once per session, so any test that drives main()
    or run_piped() left the flags on for every test that ran afterwards. That
    silently turned later dangerous-command prompts into auto-denials and made
    an unrelated audit-log test fail only under the full suite, never alone.
    Fail loudly on a leak instead of letting it change another test's meaning.
    """
    main = sys.modules.get("qwen_cli.main")
    if main is not None:
        main._auto_approve = False
        main._unattended = False
    yield


@pytest.fixture(scope="session")
def qwen_tools():
    """The shared qwen_cli.tools.shared module."""
    _import_module("qwen_cli.tools.shared")
    return sys.modules["qwen_cli.tools.shared"]


@pytest.fixture(scope="session")
def lsp_client():
    """The qwen_cli.tools.lsp module."""
    _import_module("qwen_cli.tools.lsp")
    return sys.modules["qwen_cli.tools.lsp"]
