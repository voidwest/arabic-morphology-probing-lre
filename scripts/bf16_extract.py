#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_stack.bf16_capture.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["extract", *__import__("sys").argv[1:]]))
