# Truco Bots

Gameplay and provider-backed bots for Baixada's two-player Brazilian Truco.

This repository is the runtime bot boundary. It consumes immutable public
contracts and deliberately contains no game-rules implementation or CFR
training code:

- [`truco-bot-core`](crates/truco-bot-core) owns shared bot types plus random,
  simple, heuristic, and generic LLM gameplay.
- [`truco-bots`](crates/truco-bots) owns HTTP integrations and model catalogs
  for OpenAI, Anthropic, and OpenRouter.
- [`truco-policy-bot`](crates/truco-policy-bot) loads TPB1 policy bundles and
  maps their abstract actions onto live engine actions, with a heuristic
  fallback for uncovered states.

The authoritative rules engine lives in
[`baixada-cards/truco-engine`](https://github.com/baixada-cards/truco-engine).
The policy interchange contract lives in
[`baixada-cards/truco-solver`](https://github.com/baixada-cards/truco-solver)
as the standalone `truco-policy-format` crate. This workspace pins both exact
commits in [`Cargo.toml`](Cargo.toml) and
[`contracts.lock.json`](contracts.lock.json); it never imports the full solver.

## Development

Prerequisites are stable Rust with Clippy and rustfmt, plus
[Socket Firewall Free](https://docs.socket.dev/docs/socket-firewall-free) for
public-registry fetches.

```sh
make sync
make check
```

The complete gate verifies immutable contract pins, full-SHA GitHub Actions,
formatting, Clippy, all tests, and the runtime dependency boundary. In
particular:

```sh
cargo tree -p truco-policy-bot --edges normal --locked --offline
```

must contain `truco-policy-format` but not `truco-solver`.

## Provider credentials

Provider-backed bots read ordinary environment variables at runtime:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`

Optional model and endpoint overrides are documented in
[`crates/truco-bots/README.md`](crates/truco-bots/README.md). Never commit
credentials or real `.env` files. GitHub Actions for this repository require
no provider secrets because tests use local fixtures and mock HTTP responses.

For local use, keep 1Password references in an ignored `.env` file under the
final variable names, then resolve them only at the process boundary:

```sh
op run --env-file=.env -- your-command
```

## Policies and artifacts

This repository contains readers and compatibility tests, not trained policy
bundles. TPB1 files, manifests for live deployments, checkpoints, and provider
credentials remain outside public Git. Production retrieves immutable,
checksum-verified policy bundles through the private operations boundary.

## Versioning

Repository releases follow Semantic Versioning. Runtime compatibility is
defined by the pinned engine API and policy wire-format version. Updating
either contract requires a lock update and a green full gate.

## License

MIT. See [`LICENSE`](LICENSE).
