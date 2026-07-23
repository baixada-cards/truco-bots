# truco-policy-bot

Runtime adapter from solver-produced policy bundles to live Truco bot
decisions.

The crate:

- loads and memory-maps TPB1 policy profiles;
- validates `truco-policy-bot/v1` manifests;
- reconstructs deterministic information-set keys from a live match and its
  exact action log;
- maps abstract policy actions onto legal concrete engine actions;
- samples the stored mixed strategy;
- falls back visibly to the heuristic bot when a state is uncovered.

It depends on the small `truco-policy-format` contract published by
`baixada-cards/truco-solver`, but never on the full solver. Policy generation,
training, certification, checkpoints, and policy artifacts belong outside this
runtime repository.
