"""Keep the runtime bots independent from CFR training code."""

from __future__ import annotations

import subprocess
import tomllib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    failures: list[str] = []
    for manifest_path in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            dependencies = manifest.get(section, {})
            if "truco-solver" in dependencies:
                failures.append(
                    f"{manifest_path.relative_to(ROOT)} {section} imports truco-solver"
                )

    tree = subprocess.run(
        [
            "cargo",
            "tree",
            "-p",
            "truco-policy-bot",
            "--edges",
            "normal",
            "--locked",
            "--offline",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "truco-policy-format" not in tree:
        failures.append("runtime policy bot does not contain truco-policy-format")
    if re.search(r"(?m)^(?:[│ ]*[├└]── )?truco-solver v", tree):
        failures.append("runtime policy bot transitively contains truco-solver")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("Runtime policy bot depends on the format contract, not the solver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
