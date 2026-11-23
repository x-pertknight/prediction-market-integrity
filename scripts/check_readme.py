"""
scripts/check_readme.py: compare the figures computed by scripts/reproduce.py with the
values published in README.md. The expected values are read from the README text itself,
so they are written down in one place only. Each figure is formatted the way the README
writes it (same number of decimals) before the comparison.

Called by scripts/reproduce.py; returns the number of mismatches.
"""
from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
NUM = r"([\d,]+(?:\.\d+)?)"

# (README pattern with one group per figure, computed keys in group order).
# Keys ending in "p_", "rate_" or "stake_" are shares and are compared as percent.
PATTERNS = [
    (rf"differs from the original proposal {NUM}% of the time \({NUM} of {NUM}\)",
     ["p_change", "changed", "n"]),
    (rf'"the disputer wins {NUM}% of the time" \({NUM} of {NUM}\)', ["p_overturned", "overturned", "n"]),
    (rf"That count treats {NUM} too early resets", ["too_early"]),
    (rf"{NUM} bps of notional in the current regime \(DVM dispute rate {NUM}%, {NUM} of {NUM} resolutions",
     ["bps_post", "rate_post", "post_disputes", "post_resolutions"]),
    (rf"and {NUM} bps before the August 2025 move to whitelisted proposers \({NUM}%, {NUM} of {NUM},",
     ["bps_pre", "rate_pre", "pre_disputes", "pre_resolutions"]),
    (rf"voted their own book {NUM}% of the time \({NUM} of {NUM}\)", ["p_own_book", "own_book", "directional"]),
    (rf"at most {NUM}% of the final round stake", ["stake_max"]),
    (rf"smallest winning margin in any final round is {NUM}% of the stake", ["margin_min"]),
]


def formatted(key: str, value, published: str) -> str:
    """Format a computed value like the published string: decimals, thousands separator, percent."""
    decimals = len(published.split(".")[1]) if "." in published else 0
    if key.startswith(("p_", "rate_", "stake_")) or key == "margin_min":
        value = 100 * value
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def check(figures: dict) -> int:
    text = " ".join(README.read_text().split())
    print("\n" + "=" * 72)
    print("README FIGURES vs RECOMPUTE")
    print("=" * 72)
    bad = 0
    for pattern, keys in PATTERNS:
        found = re.findall(pattern, text)
        if len(found) != 1:
            print(f"  [FAIL] README.md has {len(found)} matches for /{pattern}/, expected 1")
            bad += 1
            continue
        groups = found[0] if isinstance(found[0], tuple) else (found[0],)
        for key, published in zip(keys, groups):
            if key not in figures:
                print(f"  [FAIL] {key}: README says {published}, the recompute did not produce it")
                bad += 1
                continue
            got = formatted(key, figures[key], published)
            ok = got == published
            bad += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {key}: README {published}, recompute {got}")
    print(f"\n  {'FAIL: %d mismatch(es)' % bad if bad else 'OK: every README figure matches the recompute'}")
    return 1 if bad else 0
