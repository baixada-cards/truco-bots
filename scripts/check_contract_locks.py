"""Verify Cargo dependencies match the immutable public contract lock."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT_MANIFEST = ROOT / "Cargo.toml"
LOCK = ROOT / "contracts.lock.json"
CRATE_MANIFESTS = {
    ROOT / "crates/truco-bot-core/Cargo.toml": {"truco-engine"},
    ROOT / "crates/truco-bots/Cargo.toml": {"truco-engine"},
    ROOT / "crates/truco-policy-bot/Cargo.toml": {
        "truco-engine",
        "truco-policy-format",
    },
}


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    root = tomllib.loads(ROOT_MANIFEST.read_text(encoding="utf-8"))
    workspace_dependencies = root["workspace"]["dependencies"]
    failures: list[str] = []

    if lock.get("format") != "baixada-bot-contract-lock/v1":
        failures.append("contracts.lock.json has an unsupported format")

    for lock_name, dependency_name in (
        ("engine", "truco-engine"),
        ("policy_format", "truco-policy-format"),
    ):
        locked = lock.get(lock_name, {})
        dependency = workspace_dependencies.get(dependency_name, {})
        if dependency.get("git") != locked.get("repository"):
            failures.append(f"{dependency_name} Git URL differs from the contract lock")
        if dependency.get("rev") != locked.get("revision"):
            failures.append(f"{dependency_name} revision differs from the contract lock")

    for manifest_path, dependencies in CRATE_MANIFESTS.items():
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        declared_dependencies = manifest.get("dependencies", {})
        for dependency_name in dependencies:
            if declared_dependencies.get(dependency_name) != {"workspace": True}:
                failures.append(
                    f"{manifest_path.relative_to(ROOT)} must inherit "
                    f"{dependency_name} from workspace dependencies"
                )

    cargo_lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    for lock_name in ("engine", "policy_format"):
        locked = lock[lock_name]
        expected_source = (
            f'git+{locked["repository"]}?rev={locked["revision"]}'
            f'#{locked["revision"]}'
        )
        if expected_source not in cargo_lock:
            failures.append(
                f"Cargo.lock does not contain the exact {locked['package']} revision"
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("Engine and policy-format dependencies match contracts.lock.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
