"""
scripts/reproduce.py: rerun every number in the README from the shipped data.
Offline, no keys, no network.

    python3 -I scripts/reproduce.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flip_test_uma  # noqa: E402
import missing_greek  # noqa: E402
import report_uma_conflict  # noqa: E402


def main() -> None:
    print("#" * 72)
    print("# 1. THE MISSING GREEK: settlement risk on the classic UMA book")
    print("#" * 72)
    missing_greek.main()
    print()
    print("#" * 72)
    print("# 2. DVM VOTERS AND THEIR OWN POLYMARKET POSITIONS")
    print("#" * 72)
    report_uma_conflict.main()
    print()
    flip_test_uma.main()


if __name__ == "__main__":
    main()
