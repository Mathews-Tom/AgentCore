#!/usr/bin/env python3
"""
Validate that LLM provider SDK versions are correctly pinned in pyproject.toml.

This script is used as a pre-commit hook to ensure version pinning discipline.
"""

import re
import sys
from pathlib import Path


def main() -> int:
    """Validate pinned versions in pyproject.toml."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print("❌ ERROR: pyproject.toml not found")
        return 1

    content = pyproject_path.read_text()

    # Define expected pinned versions
    expected_pins = {
        "openai": "1.54.0",
        "anthropic": "0.40.0",
        "google-generativeai": "0.2.0",
    }

    errors = []

    for package, expected_version in expected_pins.items():
        # Look for the package in dependencies
        # Pattern: "package==version" or "package>=version"
        pattern = rf'"{re.escape(package)}(==|>=)([^"]+)"'
        match = re.search(pattern, content)

        if not match:
            errors.append(f"❌ {package}: not found in dependencies")
            continue

        operator, version = match.groups()

        # Check for exact pin
        if operator != "==":
            errors.append(
                f"❌ {package}: must use exact pin (==) not {operator}"
                f"\n   Found: {package}{operator}{version}"
                f"\n   Expected: {package}=={expected_version}"
            )
            continue

        # Check version matches expected
        if version != expected_version:
            errors.append(
                f"❌ {package}: version mismatch"
                f"\n   Found: {version}"
                f"\n   Expected: {expected_version}"
            )
            continue

        print(f"✅ {package}=={version}")

    if errors:
        print("\n" + "\n".join(errors))
        print(
            "\nPlease update pyproject.toml with correct pinned versions."
            "\nSee DEPENDENCIES.md for current version requirements."
        )
        return 1

    print("\n✅ All LLM SDK versions are correctly pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
