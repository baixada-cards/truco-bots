# Agent Instructions

## Repository Purpose

This repository owns Baixada's Truco runtime bots: shared gameplay behavior,
provider-backed integrations, and the policy-bundle reader. It consumes the
authoritative public engine and policy-format contracts.

## Boundaries

- `crates/truco-bot-core` owns common bot types and local gameplay bots.
- `crates/truco-bots` owns provider HTTP transports and model catalogs.
- `crates/truco-policy-bot` owns runtime TPB1 loading, live information-set
  reconstruction, action mapping, seeded hands, and heuristic fallback.
- `baixada-cards/truco-engine` owns game rules.
- `baixada-cards/truco-solver` owns CFR training and publishes only the small
  `truco-policy-format` contract consumed here.
- Full solver code, checkpoints, policy artifacts, service sessions, product
  UI, deployment state, and credentials do not belong here.

## Workflow

- Run `make check` before wrapping up a change.
- Use `sfw` for public-registry dependency fetches.
- Keep `Cargo.lock` current and use locked installs in CI.
- Sign commits.
- Keep Git dependencies pinned to full commits and synchronize them with
  `contracts.lock.json`.
- Never commit credentials, `.env` files, policy bundles, or cloud inventory.

## Compatibility

- Production dependencies must never include `truco-solver`.
- Preserve TPB1 v1 and deterministic information-set compatibility through
  `truco-policy-format`; format changes begin in the solver repository.
- Provider integration tests must use local mocks and must not make billable
  or credentialed network calls.
- Cross-repository solve/runtime parity belongs in the integration repository;
  this repository verifies runtime behavior against contract-generated
  fixtures without linking the solver.

## Verification

- Rust formatting and Clippy with warnings denied.
- All workspace tests and targets.
- Exact engine/policy contract locks.
- `cargo tree -p truco-policy-bot --edges normal` contains no full solver.
- All GitHub Actions references use immutable full SHAs.
