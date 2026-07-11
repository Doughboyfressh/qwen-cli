"""Qwen CLI package."""

__all__: list[str] = []


def __getattr__(name):
    """Lazy-load symbols from qwen_cli.main on first access.

    Avoids pulling in the entire (large) main module at package import time.
    """
    import importlib

    mod = importlib.import_module("qwen_cli.main")
    return getattr(mod, name)
