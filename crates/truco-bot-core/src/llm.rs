use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use truco_engine::{Action, Card, EngineError, Rank, Suit};

use super::{
    resolve_weighted_action, Bot, BotDecision, BotDecisionSource, BotError, BotPlan, BotTurn,
    WeightedActionChoice,
};

const LLM_BOT_SYSTEM_PROMPT: &str = r#"You are a 2-player Truco bot.

Rules summary:
- Deck ranks from weakest to strongest are: 4, 5, 6, 7, Q, J, K, A, 2, 3.
- The turnup defines the manilha rank as the next rank in that cycle.
- Among manilhas, suit order from weakest to strongest is DIAMONDS, SPADES, HEARTS, CLUBS.
- Non-manilhas of different suits tie when ranks are equal.
- A hand is worth 1 point unless raised through the ladder 3, 6, 9, 12.
- A declined raise awards the previous stake to the raiser.
- Hiding a card on the first round is illegal.
- In Mao de onze, accepting plays for 3 and folding concedes 1.

Action notation:
- Legal actions are given in canonical JSON.
- To play a card face up: { "type": "play_face_up", "card_id": "..." }
- To play a card face down: { "type": "play_face_down", "card_id": "..." }
- To raise: { "type": "raise", "to": 3|6|9|12 }
- Other actions are accept_raise, fold, accept_eleven, fold_eleven.

Your response must be strict JSON with this shape:
{
  "reasoning": "short explanation",
  "choices": [
    {
      "action": { ... one legal action object ... },
      "weight": 0.7
    }
  ]
}

You may return a mixed strategy by assigning positive weights to multiple legal actions.
Never invent an illegal action or a card id that is not listed.
"#;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmBotRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub turn: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmBotResponse {
    #[serde(default)]
    pub reasoning: String,
    pub choices: Vec<WeightedActionChoice>,
}

pub trait LlmCompletionClient {
    fn complete_turn(&mut self, request: &LlmBotRequest) -> Result<String, LlmBotError>;
}

#[derive(Debug, Error)]
pub enum LlmBotError {
    #[error("{0}")]
    Bot(#[from] BotError),
    #[error("{0}")]
    Engine(#[from] EngineError),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Provider(String),
}

#[derive(Debug, Clone)]
pub struct LlmBot<C, R = StdRng> {
    client: C,
    rng: R,
    last_response: Option<LlmBotResponse>,
}

impl<C> LlmBot<C, StdRng> {
    pub fn from_seed(client: C, seed: u64) -> Self {
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        Self {
            client,
            rng: StdRng::from_seed(bytes),
            last_response: None,
        }
    }
}

impl<C, R> LlmBot<C, R> {
    pub fn new(client: C, rng: R) -> Self {
        Self {
            client,
            rng,
            last_response: None,
        }
    }

    pub fn last_response(&self) -> Option<&LlmBotResponse> {
        self.last_response.as_ref()
    }
}

impl<C: LlmCompletionClient, R: rand::Rng> BotDecisionSource for LlmBot<C, R> {
    fn choose_decision(&mut self, turn: &BotTurn) -> Result<BotDecision, BotError> {
        let request = build_request(turn);
        let raw = self
            .client
            .complete_turn(&request)
            .map_err(|error| BotError::InvalidDecision(error.to_string()))?;
        let response = parse_llm_response(&raw)
            .map_err(|error| BotError::InvalidDecision(error.to_string()))?;
        let plan = BotPlan {
            choices: response.choices.clone(),
            reasoning: if response.reasoning.is_empty() {
                None
            } else {
                Some(response.reasoning.clone())
            },
        };
        let action = resolve_weighted_action(&plan, &turn.legal_actions, &mut self.rng)?;
        self.last_response = Some(response);
        Ok(BotDecision { action, plan })
    }
}

impl<C: LlmCompletionClient, R: rand::Rng> Bot for LlmBot<C, R> {
    fn choose_action(&mut self, turn: &BotTurn) -> Result<Action, BotError> {
        Ok(self.choose_decision(turn)?.action)
    }
}

pub fn build_request(turn: &BotTurn) -> LlmBotRequest {
    LlmBotRequest {
        system_prompt: LLM_BOT_SYSTEM_PROMPT.to_string(),
        user_prompt:
            "Choose the next action from the legal_actions list. Use mixed weights only when you genuinely want a distribution."
                .to_string(),
        turn: build_turn_payload(turn),
    }
}

pub fn build_turn_payload(turn: &BotTurn) -> Value {
    json!({
        "player": turn.player,
        "turnup": turn.turnup,
        "score": turn.view.score,
        "winner": turn.view.winner,
        "next_dealer": turn.view.next_dealer,
        "current_player": turn.view.current_player,
        "hand_in_progress": turn.view.hand_in_progress,
        "player_view": turn.view,
        "legal_actions": turn.legal_actions,
        "card_notation": {
            "ranks": ["4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"],
            "suits": ["DIAMONDS", "SPADES", "HEARTS", "CLUBS"],
            "examples": [
                format_card_literal(&Card {
                    id: "example".into(),
                    rank: Rank::Ace,
                    suit: Suit::Spades,
                }),
                format_card_literal(&Card {
                    id: "example".into(),
                    rank: Rank::Three,
                    suit: Suit::Clubs,
                }),
            ],
        },
    })
}

pub fn parse_llm_response(raw: &str) -> Result<LlmBotResponse, LlmBotError> {
    let response: LlmBotResponse = serde_json::from_str(raw)?;
    if response.choices.is_empty() {
        return Err(LlmBotError::Provider(
            "llm bot response did not include any weighted choices".to_string(),
        ));
    }
    if response
        .choices
        .iter()
        .all(|choice| !choice.weight.is_finite() || choice.weight <= 0.0)
    {
        return Err(LlmBotError::Provider(
            "llm bot response did not include any positive weights".to_string(),
        ));
    }

    Ok(response)
}

pub fn format_card_literal(card: &Card) -> String {
    format!("{}{}", format_rank(card.rank), format_suit(card.suit))
}

fn format_rank(rank: Rank) -> &'static str {
    match rank {
        Rank::Four => "4",
        Rank::Five => "5",
        Rank::Six => "6",
        Rank::Seven => "7",
        Rank::Queen => "Q",
        Rank::Jack => "J",
        Rank::King => "K",
        Rank::Ace => "A",
        Rank::Two => "2",
        Rank::Three => "3",
    }
}

fn format_suit(suit: Suit) -> &'static str {
    match suit {
        Suit::Diamonds => "d",
        Suit::Spades => "s",
        Suit::Hearts => "h",
        Suit::Clubs => "c",
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_llm_response, LlmBot, LlmBotError, LlmCompletionClient};
    use crate::choose_action_for_match;
    use truco_engine::{Action, Card, Hands, Match, Rank, Score, Suit, Turnup};

    #[derive(Debug, Clone)]
    struct StubClient {
        raw_response: String,
    }

    impl LlmCompletionClient for StubClient {
        fn complete_turn(
            &mut self,
            _request: &super::LlmBotRequest,
        ) -> Result<String, LlmBotError> {
            Ok(self.raw_response.clone())
        }
    }

    fn sample_match() -> Match {
        let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
        game.start_hand(
            Turnup {
                rank: Rank::Ace,
                suit: Suit::Spades,
            },
            Hands {
                zero: smallvec::smallvec![
                    Card {
                        id: "p0c0".into(),
                        rank: Rank::Seven,
                        suit: Suit::Diamonds,
                    },
                    Card {
                        id: "p0c1".into(),
                        rank: Rank::Six,
                        suit: Suit::Clubs,
                    },
                    Card {
                        id: "p0c2".into(),
                        rank: Rank::Four,
                        suit: Suit::Hearts,
                    },
                ],
                one: smallvec::smallvec![
                    Card {
                        id: "p1c0".into(),
                        rank: Rank::Three,
                        suit: Suit::Clubs,
                    },
                    Card {
                        id: "p1c1".into(),
                        rank: Rank::Five,
                        suit: Suit::Spades,
                    },
                    Card {
                        id: "p1c2".into(),
                        rank: Rank::Four,
                        suit: Suit::Diamonds,
                    },
                ],
            },
        )
        .expect("hand should start");
        game
    }

    #[test]
    fn parse_llm_response_rejects_empty_choices() {
        let error = parse_llm_response(r#"{ "choices": [] }"#).expect_err("should reject");
        assert!(matches!(error, LlmBotError::Provider(_)));
    }

    #[test]
    fn llm_bot_resolves_a_legal_weighted_action() {
        let mut bot = LlmBot::from_seed(
            StubClient {
                raw_response: r#"{
                    "reasoning": "Mix between a raise and a solid lead.",
                    "choices": [
                        { "action": { "type": "raise", "to": 3 }, "weight": 0.6 },
                        { "action": { "type": "play_face_up", "card_id": "p0c0" }, "weight": 0.4 }
                    ]
                }"#
                .to_string(),
            },
            4,
        );
        let game = sample_match();

        let action = choose_action_for_match(&mut bot, &game, 0).expect("bot should act");

        assert!(
            matches!(action, Action::Raise { to: 3 })
                || matches!(
                    action,
                    Action::PlayFaceUp { ref card_id } if card_id.as_ref() == "p0c0"
                )
        );
        assert!(bot.last_response().is_some());
    }
}
