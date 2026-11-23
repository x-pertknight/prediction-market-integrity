"""
scripts/reproduce.py: rerun every number in the README from the shipped data.
Offline, no keys, no network. After printing, the headline figures are compared with
the values published in README.md (scripts/check_readme.py); any mismatch exits 1.

    python3 -I scripts/reproduce.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_readme  # noqa: E402
import flip_test_uma  # noqa: E402
import missing_greek  # noqa: E402
import report_uma_conflict  # noqa: E402


def main() -> None:
    print("#" * 72)
    print("# 1. THE MISSING GREEK: settlement risk on the classic UMA book")
    print("#" * 72)
    figures = dict(missing_greek.main())
    print()
    print("#" * 72)
    print("# 2. DVM VOTERS AND THEIR OWN POLYMARKET POSITIONS")
    print("#" * 72)
    figures.update(report_uma_conflict.main())
    print()
    figures.update(flip_test_uma.main())
    sys.exit(check_readme.check(figures))


if __name__ == "__main__":
    main()
