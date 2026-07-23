# truco-bot-core

Shared bot crate for gameplay-facing Truco bots.

## Purpose

- define shared bot turn/decision types and helper utilities
- host gameplay bot implementations such as random, simple, heuristic, and generic LLM bots
- depend only on the public `truco-engine` API surface

## Boundaries

- `truco-engine` owns rules, state, and tactical hidden-information analysis
- `truco-bots` owns provider-backed transport and catalog integrations
- this crate owns gameplay bot behavior layered on top of engine views and legal actions
