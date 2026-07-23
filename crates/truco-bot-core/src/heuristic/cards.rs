use truco_engine::{Action, Card, Player, PublicCard, PublicPlay, Rank, Suit, Turnup};

use super::CandidateAction;
use crate::BotTurn;

pub(crate) trait CardLike {
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

pub(crate) fn card_strength(card: &impl CardLike, turnup: &Turnup) -> u8 {
    if card.rank() == turnup.rank.next_for_manilha() {
        10 + card.suit().manilha_strength() as u8
    } else {
        card.rank().index() as u8
    }
}

pub(crate) fn card_strengths(cards: &[Card], turnup: &Turnup) -> Vec<u8> {
    cards
        .iter()
        .map(|card| card_strength(card, turnup))
        .collect()
}

pub(crate) fn is_highest_manilha_clubs(card: &Card, turnup: &Turnup) -> bool {
    card.is_manilha(turnup) && card.suit == Suit::Clubs
}

pub(crate) fn collect_candidates(turn: &BotTurn, turnup: &Turnup) -> Vec<CandidateAction> {
    let Some(hand) = &turn.view.hand else {
        return Vec::new();
    };

    hand.hand
        .iter()
        .filter_map(|card| {
            let up = turn
                .legal_actions
                .iter()
                .find(|action| {
                    matches!(action, Action::PlayFaceUp { card_id } if card_id == &card.id)
                })
                .cloned();
            let down = turn
                .legal_actions
                .iter()
                .find(|action| {
                    matches!(action, Action::PlayFaceDown { card_id } if card_id == &card.id)
                })
                .cloned();
            if up.is_none() && down.is_none() {
                return None;
            }

            Some(CandidateAction {
                card: card.clone(),
                strength: card_strength(card, turnup),
                up,
                down,
            })
        })
        .collect()
}

pub(crate) fn visible_opponent_play(plays: &[PublicPlay], player: Player) -> Option<&PublicCard> {
    plays
        .iter()
        .find(|play| play.player != player)
        .and_then(|play| play.card.as_ref())
}

pub(crate) fn visible_self_play(plays: &[PublicPlay], player: Player) -> Option<&PublicCard> {
    plays
        .iter()
        .find(|play| play.player == player)
        .and_then(|play| play.card.as_ref())
}

pub(crate) fn own_visible_current_strength(turn: &BotTurn) -> Option<u8> {
    let hand = turn.view.hand.as_ref()?;
    let turnup = turn.turnup.as_ref()?;
    visible_self_play(&hand.public_state.current_round.plays, turn.player)
        .map(|card| card_strength(card, turnup))
}
