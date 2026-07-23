.PHONY: check sync locks rust boundary

sync:
	sfw cargo fetch --locked

locks:
	python3 scripts/check_action_pins.py
	python3 scripts/check_contract_locks.py

rust:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
	cargo test --workspace --all-targets --locked --offline

boundary:
	cargo tree -p truco-policy-bot --edges normal --locked --offline
	python3 scripts/check_runtime_boundary.py

check: locks rust boundary
