use rand::{rngs::StdRng, Rng, SeedableRng};
use truco_engine::Action;

use super::{
    Bot, BotDecision, BotDecisionSource, BotError, BotPlan, BotTurn, WeightedActionChoice,
};

#[derive(Debug, Clone)]
pub struct UniformRandomBot<R = StdRng> {
    rng: R,
}

impl UniformRandomBot<StdRng> {
    pub fn from_seed(seed: u64) -> Self {
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        Self {
            rng: StdRng::from_seed(bytes),
        }
    }

    pub fn from_entropy() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }
}

impl<R> UniformRandomBot<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng> BotDecisionSource for UniformRandomBot<R> {
    fn choose_decision(&mut self, turn: &BotTurn) -> Result<BotDecision, BotError> {
        if turn.legal_actions.is_empty() {
            return Err(BotError::NoLegalActions);
        }

        let index = self.rng.gen_range(0..turn.legal_actions.len());
        Ok(BotDecision {
            action: turn.legal_actions[index].clone(),
            plan: BotPlan {
                choices: turn
                    .legal_actions
                    .iter()
                    .cloned()
                    .map(|action| WeightedActionChoice {
                        action,
                        weight: 1.0,
                    })
                    .collect(),
                reasoning: Some(
                    "Uniform random baseline bot. Every legal action was assigned equal weight."
                        .to_string(),
                ),
            },
        })
    }
}

impl<R: Rng> Bot for UniformRandomBot<R> {
    fn choose_action(&mut self, turn: &BotTurn) -> Result<Action, BotError> {
        Ok(self.choose_decision(turn)?.action)
    }
}
