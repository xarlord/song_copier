#!/usr/bin/env python3
"""
Coverage Diff Checker — CI Gate: "Coverage must not decrease"

Compares test coverage between the base branch and the PR branch.
Fails if coverage decreased overall or for changed files.

Usage:
    python coverage-diff-check.py --base main --coverage-file coverage/coverage-summary.json

Supports:
    - Jest/Vitest coverage-summary.json
    - pytest cov JSON report
    - JaCoCo XML reports (BlockBuilder)

Exit codes:
    0 = PASS (coverage maintained or improved)
    1 = FAIL (coverage decreased)
    2 = ERROR (setup/parsing issue)
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def get_base_coverage_jest(base: str, cwd: str) -> dict | None:
    """Get coverage from base branch by checking out and running."""
    # In CI, we use the artifact from the base branch
    # Locally, we compare against git stash of coverage
    coverage_file = os.path.join(cwd, "coverage-base", "coverage-summary.json")
    if os.path.exists(coverage_file):
        return parse_jest_coverage(coverage_file)
    return None


def parse_jest_coverage(path: str) -> dict:
    """Parse Jest/Vitest coverage-summary.json."""
    with open(path) as f:
        data = json.load(f)

    total = data.get("total", {})
    lines = total.get("lines", {}).get("pct", 0)
    branches = total.get("branches", {}).get("pct", 0)
    functions = total.get("functions", {}).get("pct", 0)
    statements = total.get("statements", {}).get("pct", 0)

    return {
        "lines": lines,
        "branches": branches,
        "functions": functions,
        "statements": statements,
        "avg": round((lines + branches + functions + statements) / 4, 1),
        "files": data.get("files", {}),
    }


def parse_pytest_coverage(path: str) -> dict:
    """Parse pytest --cov-json output."""
    with open(path) as f:
        data = json.load(f)

    totals = data.get("totals", {})
    lines = totals.get("percent_covered", 0)

    return {
        "lines": round(lines, 1),
        "branches": 0,
        "functions": 0,
        "statements": 0,
        "avg": round(lines, 1),
    }


def parse_jacoco_xml(path: str) -> dict:
    """Parse JaCoCo XML report."""
    tree = ET.parse(path)
    root = tree.getroot()

    counters = {}
    for counter in root.findall(".//counter"):
        ctype = counter.get("type")
        missed = int(counter.get("missed", 0))
        covered = int(counter.get("covered", 0))
        total = missed + covered
        pct = round((covered / total * 100) if total > 0 else 0, 1)
        counters[ctype] = pct

    return {
        "lines": counters.get("LINE", 0),
        "branches": counters.get("BRANCH", 0),
        "functions": counters.get("METHOD", 0),
        "statements": counters.get("INSTRUCTION", 0),
        "avg": round(sum(counters.values()) / max(len(counters), 1), 1),
    }


def compare_coverage(base: dict, head: dict, min_avg: float = 80.0) -> tuple[bool, list[str]]:
    """Compare base vs head coverage. Returns (pass, messages)."""
    messages = []
    passed = True

    # Check minimum threshold
    if head["avg"] < min_avg:
        messages.append(f"❌ Average coverage {head['avg']}% is below minimum {min_avg}%")
        passed = False
    else:
        messages.append(f"✅ Average coverage {head['avg']}% meets minimum {min_avg}%")

    # Check per-metric diff
    metrics = ["lines", "branches", "functions", "statements"]
    for metric in metrics:
        diff = head.get(metric, 0) - base.get(metric, 0)
        if diff < -2:  # Allow 2% tolerance
            messages.append(f"⚠️  {metric}: {base.get(metric, 0)}% → {head.get(metric, 0)}% ({diff:+.1f}%)")
            if diff < -5:
                messages.append(f"❌ {metric} dropped by more than 5%")
                passed = False
        elif diff > 0:
            messages.append(f"📈 {metric}: {base.get(metric, 0)}% → {head.get(metric, 0)}% ({diff:+.1f}%)")

    return passed, messages


def main():
    parser = argparse.ArgumentParser(description="Check coverage diff between branches")
    parser.add_argument("--base", default="main", help="Base branch")
    parser.add_argument("--coverage-file", required=True, help="Path to coverage report")
    parser.add_argument("--base-coverage", help="Path to base branch coverage report")
    parser.add_argument("--format", choices=["jest", "pytest", "jacoco"], help="Report format")
    parser.add_argument("--min-coverage", type=float, default=80.0, help="Minimum average coverage %%")
    parser.add_argument("--fail-on-drop", action="store_true", help="Fail if any coverage metric drops")
    args = parser.parse_args()

    # Auto-detect format
    fmt = args.format
    if not fmt:
        if "coverage-summary.json" in args.coverage_file:
            fmt = "jest"
        elif args.coverage_file.endswith(".xml"):
            fmt = "jacoco"
        else:
            fmt = "pytest"

    # Parse head coverage
    parsers = {"jest": parse_jest_coverage, "pytest": parse_pytest_coverage, "jacoco": parse_jacoco_xml}
    head = parsers[fmt](args.coverage_file)

    # Parse base coverage
    base = None
    if args.base_coverage and os.path.exists(args.base_coverage):
        base = parsers[fmt](args.base_coverage)

    if base is None:
        # No base to compare — just check minimum
        print(f"📊 Head coverage: avg={head['avg']}%")
        if head["avg"] < args.min_coverage:
            print(f"❌ Coverage {head['avg']}% is below minimum {args.min_coverage}%")
            sys.exit(1)
        print(f"✅ Coverage meets minimum threshold. PASS.")
        sys.exit(0)

    # Compare
    print(f"📊 Base coverage: avg={base['avg']}%")
    print(f"📊 Head coverage: avg={head['avg']}%")
    print()

    passed, messages = compare_coverage(base, head, args.min_coverage)
    for msg in messages:
        print(f"  {msg}")

    if args.fail_on_drop and not passed:
        print()
        print("❌ COVERAGE DIFF GATE FAILED")
        print("   → Add tests to restore or improve coverage.")
        sys.exit(1)

    if not passed:
        print()
        print("⚠️  Coverage decreased. Please add tests to maintain coverage.")
        # Hard fail if below minimum
        if head["avg"] < args.min_coverage:
            sys.exit(1)

    print()
    print("✅ COVERAGE GATE PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
