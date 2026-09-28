"""Thin compatibility entry point for the canonical doctor command."""

from rahc_lora.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["doctor"]))
