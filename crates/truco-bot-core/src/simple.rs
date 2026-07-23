use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use truco_engine::{Action, Card, Player, PublicCard, PublicPlay, Rank, Suit, Turnup};

use super::{
    Bot, BotDecision, BotDecisionSource, BotError, BotPlan, BotTurn, WeightedActionChoice,
};

#[derive(Debug, Clone)]
pub struct SimpleTrainerBot<R = StdRng> {
    rng: R,
}

impl SimpleTrainerBot<StdRng> {
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

impl<R> SimpleTrainerBot<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

#[derive(Debug, Clone)]
struct CandidateAction {
    strength: u8,
    up: Option<Action>,
    down: Option<Action>,
}

impl<R: rand::Rng> BotDecisionSource for SimpleTrainerBot<R> {
    fn choose_decision(&mut self, turn: &BotTurn) -> Result<BotDecision, BotError> {
        if turn.legal_actions.is_empty() {
            return Err(BotError::NoLegalActions);
        }

        if let Some(action) = choose_special_response(turn) {
            return Ok(single_choice_decision(
                action,
                "Simple trainer bot resolved a special response from the current hand texture.",
            ));
        }

        if let Some(action) = choose_raise(turn) {
            return Ok(single_choice_decision(
                action,
                "Simple trainer bot found an opening raise worth taking.",
            ));
        }

        if let Some(action) = choose_play(turn, &mut self.rng) {
            return Ok(single_choice_decision(
                action,
                "Simple trainer bot selected a single tactical card play.",
            ));
        }

        let index = self.rng.gen_range(0..turn.legal_actions.len());
        Ok(single_choice_decision(
            turn.legal_actions[index].clone(),
            "Simple trainer bot fell back to a random legal action.",
        ))
    }
}

impl<R: rand::Rng> Bot for SimpleTrainerBot<R> {
    fn choose_action(&mut self, turn: &BotTurn) -> Result<Action, BotError> {
        Ok(self.choose_decision(turn)?.action)
    }
}

fn single_choice_decision(action: Action, reasoning: &str) -> BotDecision {
    BotDecision {
        action: action.clone(),
        plan: BotPlan {
            choices: vec![WeightedActionChoice {
                action,
                weight: 1.0,
            }],
            reasoning: Some(reasoning.to_string()),
        },
    }
}

fn choose_special_response(turn: &BotTurn) -> Option<Action> {
    let pending_raise = turn.view.hand.as_ref()?.public_state.pending_raise.clone();
    if pending_raise.is_some() {
        let accept = turn
            .legal_actions
            .iter()
            .find(|action| matches!(action, Action::AcceptRaise))
            .cloned();
        let fold = turn
            .legal_actions
            .iter()
            .find(|action| matches!(action, Action::Fold))
            .cloned();
        if accept.is_none() || fold.is_none() {
            return accept.or(fold);
        }

        let turnup = turn.turnup.as_ref()?;
        let cards = &turn.view.hand.as_ref()?.hand;
        let top_strength = cards
            .iter()
            .map(|card| card_strength(card, turnup))
            .max()
            .unwrap_or(0);
        let manilhas = cards.iter().filter(|card| card.is_manilha(turnup)).count();
        let target = pending_raise.map(|raise| raise.to).unwrap_or(1);

        if target <= 3 || manilhas > 0 || top_strength >= 9 {
            return accept;
        }

        if target >= 6 && top_strength <= 7 {
            return fold;
        }

        return accept;
    }

    let pending_decision = turn
        .view
        .hand
        .as_ref()?
        .public_state
        .pending_decision
        .clone();
    if pending_decision.is_some() {
        let accept = turn
            .legal_actions
            .iter()
            .find(|action| matches!(action, Action::AcceptEleven))
            .cloned();
        let fold = turn
            .legal_actions
            .iter()
            .find(|action| matches!(action, Action::FoldEleven))
            .cloned();
        if accept.is_none() || fold.is_none() {
            return accept.or(fold);
        }

        let turnup = turn.turnup.as_ref()?;
        let cards = &turn.view.hand.as_ref()?.hand;
        let top_strength = cards
            .iter()
            .map(|card| card_strength(card, turnup))
            .max()
            .unwrap_or(0);
        let manilhas = cards.iter().filter(|card| card.is_manilha(turnup)).count();

        if manilhas > 0 || top_strength >= 8 {
            return accept;
        }

        return fold;
    }

    None
}

fn choose_raise(turn: &BotTurn) -> Option<Action> {
    let raise = turn.legal_actions.iter().find_map(|action| match action {
        Action::Raise { to } => Some(Action::Raise { to: *to }),
        _ => None,
    })?;
    let hand = turn.view.hand.as_ref()?;
    let turnup = turn.turnup.as_ref()?;

    if !hand.public_state.current_round.plays.is_empty() {
        return None;
    }

    let strengths: Vec<u8> = hand
        .hand
        .iter()
        .map(|card| card_strength(card, turnup))
        .collect();
    let top = strengths.iter().copied().max().unwrap_or(0);
    let strong_count = strengths.iter().filter(|strength| **strength >= 8).count();
    let manilhas = hand
        .hand
        .iter()
        .filter(|card| card.is_manilha(turnup))
        .count();

    if hand.public_state.hand_value < 6 && (manilhas > 0 || strong_count >= 2 || top >= 10) {
        return Some(raise);
    }

    None
}

fn choose_play<R: rand::Rng>(turn: &BotTurn, rng: &mut R) -> Option<Action> {
    let hand = turn.view.hand.as_ref()?;
    let turnup = turn.turnup.as_ref()?;
    let candidates = collect_candidates(turn, turnup);
    if candidates.is_empty() {
        return None;
    }

    let visible_opponent_strength =
        visible_opponent_play(&hand.public_state.current_round.plays, turn.player)
            .map(|play| card_strength(play, turnup));
    let current_round_started = !hand.public_state.current_round.plays.is_empty();

    if let Some(target_strength) = visible_opponent_strength {
        let mut winning: Vec<&CandidateAction> = candidates
            .iter()
            .filter(|candidate| candidate.strength > target_strength)
            .collect();
        winning.sort_by_key(|candidate| candidate.strength);
        if let Some(candidate) = winning.first() {
            return candidate.up.clone().or_else(|| candidate.down.clone());
        }
    }

    let mut ordered: Vec<&CandidateAction> = candidates.iter().collect();
    ordered.sort_by_key(|candidate| candidate.strength);
    let weakest_strength = ordered.first()?.strength;
    let mut weakest: Vec<&CandidateAction> = ordered
        .into_iter()
        .filter(|candidate| candidate.strength == weakest_strength)
        .collect();
    weakest.shuffle(rng);
    let chosen = weakest.first()?;

    if current_round_started && chosen.down.is_some() && weakest_strength <= 5 {
        return chosen.down.clone().or_else(|| chosen.up.clone());
    }

    chosen.up.clone().or_else(|| chosen.down.clone())
}

fn collect_candidates(turn: &BotTurn, turnup: &Turnup) -> Vec<CandidateAction> {
    let Some(hand) = &turn.view.hand else {
        return Vec::new();
    };

    hand.hand
        .iter()
        .filter_map(|card| {
            let up = turn
                .legal_actions
                .iter()
                .find(|action| matches!(action, Action::PlayFaceUp { card_id } if card_id == &card.id))
                .cloned();
            let down = turn
                .legal_actions
                .iter()
                .find(|action| matches!(action, Action::PlayFaceDown { card_id } if card_id == &card.id))
                .cloned();

            if up.is_none() && down.is_none() {
                return None;
            }

            Some(CandidateAction {
                strength: card_strength(card, turnup),
                up,
                down,
            })
        })
        .collect()
}

fn visible_opponent_play(plays: &[PublicPlay], player: Player) -> Option<&PublicCard> {
    plays
        .iter()
        .find(|play| play.player != player)
        .and_then(|play| play.card.as_ref())
}

fn card_strength(card: &impl CardLike, turnup: &Turnup) -> u8 {
    if card.rank() == turnup.rank.next_for_manilha() {
        10 + card.suit().manilha_strength() as u8
    } else {
        card.rank().index() as u8
    }
}

trait CardLike {
    fn rank(&self) -> Rank;
    fn suit(&self) -> Suit;
}

impl CardLike for Card {
    fn rank(&self) -> Rank {
        self.rank
    }

    fn suit(&self) -> Suit {
        self.suit
    }
}

impl CardLike for PublicCard {
    fn rank(&self) -> Rank {
        self.rank
    }

    fn suit(&self) -> Suit {
        self.suit
    }
}
