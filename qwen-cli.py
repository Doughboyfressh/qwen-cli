#!/usr/bin/env python3
"""Entry point — delegates to the qwen_cli package."""

if __name__ == "__main__":
    from qwen_cli.main import main as _main

    _main()
