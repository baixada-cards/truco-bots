use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use truco_engine::{
    bot_analysis::BotAnalysisError, Action, EngineError, Match, MatchState, Player,
    PlayerMatchView, Turnup,
};

pub mod heuristic;
pub mod llm;
pub mod random;
pub mod simple;
pub use heuristic::{BotProfile, HeuristicBot};
pub use llm::{
    build_request as build_llm_request, build_turn_payload as build_llm_turn_payload,
    parse_llm_response, LlmBot, LlmBotError, LlmBotRequest, LlmBotResponse,
};
pub use random::UniformRandomBot;
pub use simple::SimpleTrainerBot;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BotTurn {
    pub player: Player,
    pub turnup: Option<Turnup>,
    pub match_state: MatchState,
    pub view: PlayerMatchView,
    pub legal_actions: Vec<Action>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedActionChoice {
    pub action: Action,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BotPlan {
    pub choices: Vec<WeightedActionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BotDecision {
    pub action: Action,
    pub plan: BotPlan,
}

pub trait Bot {
    fn choose_action(&mut self, turn: &BotTurn) -> Result<Action, BotError>;
}

pub trait BotDecisionSource {
    fn choose_decision(&mut self, turn: &BotTurn) -> Result<BotDecision, BotError>;
}

#[derive(Debug, Error)]
pub enum BotError {
    #[error("{0}")]
    Engine(#[from] EngineError),
    #[error("bot received no legal actions")]
    NoLegalActions,
    #[error("{0}")]
    InvalidDecision(String),
}

impl From<BotAnalysisError> for BotError {
    fn from(value: BotAnalysisError) -> Self {
        match value {
            BotAnalysisError::Engine(error) => Self::Engine(error),
            BotAnalysisError::NoLegalActions => Self::NoLegalActions,
            BotAnalysisError::InvalidAnalysis(message) => Self::InvalidDecision(message),
        }
    }
}

pub fn turn_for_match(game: &Match, player: Player) -> Result<BotTurn, BotError> {
    let match_state = game.export_state();
    Ok(BotTurn {
        player,
        turnup: match_state
            .current_hand
            .as_ref()
            .map(|hand| hand.state.turnup.clone()),
        match_state,
        view: game.player_view(player)?,
        legal_actions: game.strategic_legal_actions(player)?,
    })
}

pub fn choose_action_for_match<B: Bot>(
    bot: &mut B,
    game: &Match,
    player: Player,
) -> Result<Action, BotError> {
    let turn = turn_for_match(game, player)?;
    bot.choose_action(&turn)
}

pub fn choose_decision_for_match<B>(
    bot: &mut B,
    game: &Match,
    player: Player,
) -> Result<BotDecision, BotError>
where
    B: BotDecisionSource,
{
    let turn = turn_for_match(game, player)?;
    bot.choose_decision(&turn)
}

pub fn resolve_weighted_action<R: Rng>(
    plan: &BotPlan,
    legal_actions: &[Action],
    rng: &mut R,
) -> Result<Action, BotError> {
    if legal_actions.is_empty() {
        return Err(BotError::NoLegalActions);
    }

    if plan.choices.is_empty() {
        return Err(BotError::InvalidDecision(
            "bot returned no weighted action choices".to_string(),
        ));
    }

    let mut filtered = Vec::new();
    for choice in &plan.choices {
        if !legal_actions.contains(&choice.action) {
            return Err(BotError::InvalidDecision(format!(
                "bot returned an illegal action: {:?}",
                choice.action
            )));
        }

        if !choice.weight.is_finite() || choice.weight <= 0.0 {
            continue;
        }

        filtered.push(choice);
    }

    if filtered.is_empty() {
        return Err(BotError::InvalidDecision(
            "bot returned no positive legal action weights".to_string(),
        ));
    }

    let total_weight: f32 = filtered.iter().map(|choice| choice.weight).sum();
    let mut draw = rng.gen_range(0.0..total_weight);
    for choice in &filtered {
        if draw <= choice.weight {
            return Ok(choice.action.clone());
        }
        draw -= choice.weight;
    }

    Ok(filtered
        .last()
        .expect("filtered choices should not be empty")
        .action
        .clone())
}
