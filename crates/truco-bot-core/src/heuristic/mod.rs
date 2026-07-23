pub(crate) mod cards;
pub(crate) mod play;
pub(crate) mod raise;
pub(crate) mod raise_response;
pub(crate) mod reasoning;
pub(crate) mod tactical;

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use truco_engine::{bot_analysis::analyze_tactical_turn_sampled, Action, Card, Match};

use crate::{Bot, BotDecision, BotDecisionSource, BotError, BotTurn};

use play::{choose_opening_raise, choose_play, should_raise_before_showing_killer_card};
use raise::raise_action;
use raise_response::{choose_mao_de_onze_response, choose_raise_response};
use reasoning::{single_choice_decision, tactical_reasoning};
use tactical::{
    best_tactical_summaries, choose_tactical_action, choose_tactical_raise_response,
    forced_raise_summary, should_use_tactical_oracle,
};

/// Upper bound on determinizations visited per tactical analysis. Exhaustive
/// enumeration still runs when the compatible determinization count is at or
/// below this budget; beyond it we fall back to Monte Carlo sampling so the
/// bot can respond in well under a second even on the widest game states
/// (e.g. round 1 with a full three-card opponent hand).
const TACTICAL_SAMPLE_BUDGET: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BotProfile {
    Conservative,
    #[default]
    Balanced,
    Aggressive,
    Tricky,
}

#[derive(Debug, Clone)]
pub struct HeuristicBot<R = StdRng> {
    rng: R,
    profile: BotProfile,
}

impl HeuristicBot<StdRng> {
    pub fn from_seed(seed: u64) -> Self {
        Self::from_seed_with_profile(seed, BotProfile::Balanced)
    }

    pub fn from_seed_with_profile(seed: u64, profile: BotProfile) -> Self {
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        Self {
            rng: StdRng::from_seed(bytes),
            profile,
        }
    }

    pub fn from_entropy() -> Self {
        Self::from_entropy_with_profile(BotProfile::Balanced)
    }

    pub fn from_entropy_with_profile(profile: BotProfile) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            profile,
        }
    }
}

impl<R> HeuristicBot<R> {
    pub fn new(rng: R, profile: BotProfile) -> Self {
        Self { rng, profile }
    }

    pub fn profile(&self) -> BotProfile {
        self.profile
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CandidateAction {
    pub(crate) card: Card,
    pub(crate) strength: u8,
    pub(crate) up: Option<Action>,
    pub(crate) down: Option<Action>,
}

impl<R: Rng> BotDecisionSource for HeuristicBot<R> {
    fn choose_decision(&mut self, turn: &BotTurn) -> Result<BotDecision, BotError> {
        if turn.legal_actions.is_empty() {
            return Err(BotError::NoLegalActions);
        }

        if should_use_tactical_oracle(turn) {
            let game = Match::from_state(turn.match_state.clone())?;
            let tactical = analyze_tactical_turn_sampled(
                &game,
                turn.player,
                TACTICAL_SAMPLE_BUDGET,
                &mut self.rng,
            )?;
            if let Some(summary) = forced_raise_summary(turn, &tactical.action_summaries) {
                return Ok(single_choice_decision(
                    summary.action.clone(),
                    tactical_reasoning(self.profile, summary),
                ));
            }
            if let Some((action, reasoning)) = choose_tactical_raise_response(
                turn,
                self.profile,
                &tactical.action_summaries,
                &mut self.rng,
            ) {
                return Ok(single_choice_decision(action, reasoning));
            }
            let best_summaries = best_tactical_summaries(&tactical.action_summaries);
            let chosen_action =
                choose_tactical_action(turn, self.profile, &best_summaries, &mut self.rng)
                    .ok_or_else(|| {
                        BotError::InvalidDecision(
                            "heuristic bot could not choose from tactical summaries".to_string(),
                        )
                    })?;
            let chosen_summary = best_summaries
                .iter()
                .copied()
                .find(|summary| summary.action == chosen_action)
                .ok_or_else(|| {
                    BotError::InvalidDecision(
                        "heuristic bot chose an action outside the best tactical summaries"
                            .to_string(),
                    )
                })?;

            return Ok(single_choice_decision(
                chosen_action,
                tactical_reasoning(self.profile, chosen_summary),
            ));
        }

        if let Some(action) = choose_raise_response(turn, self.profile, &mut self.rng) {
            return Ok(single_choice_decision(
                action,
                format!(
                    "{:?} heuristic profile resolved the pending raise response.",
                    self.profile
                ),
            ));
        }

        if let Some(action) = choose_mao_de_onze_response(turn, self.profile, &mut self.rng) {
            return Ok(single_choice_decision(
                action,
                format!(
                    "{:?} heuristic profile resolved the mão de onze decision.",
                    self.profile
                ),
            ));
        }

        let chosen_play = choose_play(turn, self.profile, &mut self.rng);
        if should_raise_before_showing_killer_card(turn, chosen_play.as_ref()) {
            if let Some(action) = raise_action(turn) {
                return Ok(single_choice_decision(
                    action,
                    format!(
                        "{:?} heuristic profile raised before exposing the strongest manilha.",
                        self.profile
                    ),
                ));
            }
        }

        if let Some(action) = choose_opening_raise(turn, self.profile, &mut self.rng) {
            return Ok(single_choice_decision(
                action,
                format!(
                    "{:?} heuristic profile chose an opening raise from pressure and hand texture.",
                    self.profile
                ),
            ));
        }

        if let Some(action) = chosen_play {
            return Ok(single_choice_decision(
                action,
                format!(
                    "{:?} heuristic profile selected a tactical card play.",
                    self.profile
                ),
            ));
        }

        let index = self.rng.gen_range(0..turn.legal_actions.len());
        Ok(single_choice_decision(
            turn.legal_actions[index].clone(),
            format!(
                "{:?} heuristic profile fell back to a random legal action.",
                self.profile
            ),
        ))
    }
}

impl<R: Rng> Bot for HeuristicBot<R> {
    fn choose_action(&mut self, turn: &BotTurn) -> Result<Action, BotError> {
        Ok(self.choose_decision(turn)?.action)
    }
}
