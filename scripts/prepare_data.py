"""Thin compatibility entry point for pinned dataset-manifest preparation."""

from __future__ import annotations

import sys

from rahc_lora.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare-data", *sys.argv[1:]]))
