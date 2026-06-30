"""Shared pytest fixtures.

qwen-cli.py has a hyphen in its name, so it can't be imported with a normal
`import` statement — we load it via importlib. Module-level code in qwen-cli.py
is side-effect free (the network health check lives inside main()), so importing
it for unit tests is safe and fast.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, filename: str):
    """Load a Python file as a module using importlib."""
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def qwen_cli():
    """The qwen-cli.py module, loaded once per test session."""
    # qwen_tools must be importable before qwen-cli loads (it does `import qwen_tools`)
    if "qwen_tools" not in sys.modules:
        _load_module("qwen_tools", "qwen_tools.py")
    return _load_module("qwen_cli", "qwen-cli.py")


@pytest.fixture(scope="session")
def qwen_tools():
    """The shared qwen_tools.py module."""
    if "qwen_tools" not in sys.modules:
        _load_module("qwen_tools", "qwen_tools.py")
    return sys.modules["qwen_tools"]


@pytest.fixture(scope="session")
def lsp_client():
    """The lsp_client.py module."""
    if "lsp_client" not in sys.modules:
        _load_module("lsp_client", "lsp_client.py")
    return sys.modules["lsp_client"]
