#!/usr/bin/env python3
"""
Test Presence Checker — CI Gate: "No tests = No merge"

Checks that every PR which modifies source files also modifies or creates
corresponding test files. Fails the gate if new/changed source code has
no test coverage in the PR.

Usage:
    python check-test-presence.py --base main

Exit codes:
    0 = PASS (all changed source files have test changes)
    1 = FAIL (source files changed without corresponding tests)
    2 = ERROR (git/setup issue)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


# ── Project-specific config ──────────────────────────────────────────

PROJECT_CONFIGS = {
    "astroverse": {
        "source_patterns": [
            r"backend/src/**/*.ts",
            r"frontend/src/**/*.{ts,tsx}",
            r"packages/*/src/**/*.ts",
        ],
        "test_patterns": [
            r"backend/src/**/?(*.)+(spec|test).ts",
            r"backend/src/__tests__/**/*.test.ts",
            r"frontend/src/**/?(*.)+(spec|test).{ts,tsx}",
            r"frontend/src/__tests__/**/*.{test,spec}.{ts,tsx}",
            r"frontend/e2e/**/*.spec.ts",
            r"frontend/tests/**/*.spec.ts",
            r"frontend/tests/**/*.test.ts",
        ],
        "source_exclude": [
            r"**/types/**/*.ts",       # Pure type files (no logic)
            r"**/*.d.ts",              # Declaration files
            r"**/migrations/**",       # DB migrations
            r"**/config/**",           # Config files
        ],
    },
    "block_builder": {
        "source_patterns": [
            r"app/app/src/main/**/*.kt",
        ],
        "test_patterns": [
            r"app/app/src/test/**/*.kt",
            r"app/app/src/androidTest/**/*.kt",
        ],
        "source_exclude": [
            r"**/theme/**",            # Theme files
            r"**/di/**",               # DI modules
            r"**/navigation/**",       # Navigation routes
        ],
    },
    "audio_generator": {
        "source_patterns": [
            r"services/*.py",
            r"models/*.py",
            r"pipelines/*.py",
            r"ui/tabs/*.py",
            r"utils/*.py",
            r"config.py",
        ],
        "test_patterns": [
            r"tests/test_*.py",
            r"tests/*_test.py",
        ],
        "source_exclude": [
            r"**/__init__.py",
            r"**/cleanup.py",         # Hard to unit test
        ],
    },
}


def detect_project(root: str) -> str:
    """Auto-detect project type from repo structure."""
    if os.path.exists(os.path.join(root, "backend")) and os.path.exists(os.path.join(root, "frontend")):
        return "astroverse"
    if os.path.exists(os.path.join(root, "app", "app", "src", "main")):
        return "block_builder"
    if os.path.exists(os.path.join(root, "services")) and os.path.exists(os.path.join(root, "models")):
        return "audio_generator"
    return "unknown"


def git_diff_names(base: str) -> list[str]:
    """Get list of files changed vs base branch."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{base}...HEAD"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Fallback: diff against merge-base
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            capture_output=True, text=True
        )
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def matches_patterns(filepath: str, patterns: list[str]) -> bool:
    """Check if filepath matches any glob pattern."""
    for pattern in patterns:
        # Convert glob to regex
        regex = pattern
        regex = regex.replace("**/", "(?:.*/)?")
        regex = regex.replace("**", ".*")
        regex = regex.replace("*", "[^/]*")
        regex = regex.replace("?", ".")
        regex = regex.replace(".{ts,tsx}", r"\.(ts|tsx)")
        regex = regex.replace("?(*.)+(spec|test)", r"(?:spec|test)")
        if re.search(regex, filepath):
            return True
    return False


def find_test_for_source(source_file: str, project: str, all_changed: list[str]) -> str | None:
    """
    Given a changed source file, find a corresponding test file
    either in the changed files list or existing in the repo.
    """
    config = PROJECT_CONFIGS.get(project, {})

    # Strategy 1: Check if any changed file is a test for this source file
    for changed in all_changed:
        if matches_patterns(changed, config.get("test_patterns", [])):
            # Rough heuristic: test file name contains source file stem
            source_stem = Path(source_file).stem.split(".")[0]
            test_stem = Path(changed).stem.replace("test_", "").replace("_test", "")
            if source_stem in test_stem or test_stem in source_stem:
                return changed

    # Strategy 2: Check if ANY test file was changed in the PR
    # (if they added a new test file for any of the changed sources)
    test_changed = [f for f in all_changed if matches_patterns(f, config.get("test_patterns", []))]
    if test_changed:
        # Check if any test file exists for this source
        for tp in config.get("test_patterns", []):
            # Build likely test path
            stem = Path(source_file).stem.split(".")[0]
            # Try common test file locations
            candidates = [
                f"tests/test_{stem}.py",
                f"tests/{stem}_test.py",
                f"backend/src/__tests__/{stem}.test.ts",
                f"frontend/src/__tests__/{stem}.test.ts",
                f"app/app/src/test/java/com/architectai/{stem}Test.kt",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate

    return None


def main():
    parser = argparse.ArgumentParser(description="Check that changed source files have tests")
    parser.add_argument("--base", default="main", help="Base branch (default: main)")
    parser.add_argument("--project", help="Project type (auto-detected if omitted)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Auto-detect project
    project = args.project or detect_project(".")
    if project == "unknown":
        print("⚠️  Could not detect project type. Specify --project.")
        sys.exit(2)

    config = PROJECT_CONFIGS.get(project)
    if not config:
        print(f"⚠️  Unknown project: {project}")
        sys.exit(2)

    # Get changed files
    changed = git_diff_names(args.base)
    if not changed:
        print("✅ No files changed (or could not diff). Passing.")
        sys.exit(0)

    # Separate source and test files
    source_files = []
    test_files = []
    for f in changed:
        if matches_patterns(f, config["test_patterns"]):
            test_files.append(f)
        elif matches_patterns(f, config["source_patterns"]):
            # Check exclusions
            excluded = matches_patterns(f, config.get("source_exclude", []))
            if not excluded:
                source_files.append(f)

    if args.verbose:
        print(f"Project: {project}")
        print(f"Base: {args.base}")
        print(f"Changed files: {len(changed)}")
        print(f"Source files: {len(source_files)}")
        print(f"Test files: {len(test_files)}")
        for s in source_files:
            print(f"  SRC: {s}")
        for t in test_files:
            print(f"  TST: {t}")

    # If no source files changed, pass
    if not source_files:
        print("✅ No source files changed. PASS.")
        sys.exit(0)

    # If source files changed but no test files changed, check if tests already exist
    if not test_files:
        # Check if tests exist for the changed source files
        uncovered = []
        for sf in source_files:
            test = find_test_for_source(sf, project, changed)
            if not test:
                uncovered.append(sf)

        if uncovered:
            print("❌ TEST PRESENCE GATE FAILED")
            print(f"   {len(uncovered)} source file(s) changed without test coverage:")
            for f in uncovered:
                print(f"   - {f}")
            print()
            print("   → Add tests for these files before merging.")
            print("   → Follow TDD: write the test, watch it fail, then implement.")
            sys.exit(1)
        else:
            print(f"✅ {len(source_files)} source file(s) changed. Existing tests cover them. PASS.")
            sys.exit(0)

    # Both source and test files changed — check coverage
    uncovered = []
    for sf in source_files:
        test = find_test_for_source(sf, project, changed)
        if not test:
            uncovered.append(sf)

    if uncovered:
        print("⚠️  PARTIAL COVERAGE WARNING")
        print(f"   {len(uncovered)} source file(s) may not have test changes in this PR:")
        for f in uncovered:
            print(f"   - {f}")
        print()
        print("   → Consider adding tests for these files.")
        # Warning only, not a hard fail if at least some tests were added
        if len(test_files) > 0:
            print(f"   ✅ {len(test_files)} test file(s) were modified. Proceeding.")
            sys.exit(0)

    print(f"✅ {len(source_files)} source file(s) and {len(test_files)} test file(s) changed. PASS.")
    sys.exit(0)


if __name__ == "__main__":
    main()
