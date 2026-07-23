# truco-bots

Provider integration crate for network-backed Truco bots.

## Purpose

- wrap provider-specific HTTP APIs for OpenAI and Anthropic bot play
- fetch provider model catalogs for the service/frontend
- convert provider responses into `truco-bot-core` bot decisions

## Runtime configuration

Credentials use the ordinary environment variables `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY`. Model selection and endpoint
overrides use the `TRUCO_OPENAI_*`, `TRUCO_ANTHROPIC_*`, and
`TRUCO_OPENROUTER_*` variables defined in `src/provider.rs`.

No provider credential is needed for tests. Tests must use local mock servers
and must never call a billable provider endpoint.
