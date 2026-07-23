use rand::Rng;
use truco_engine::Action;

use super::{
    cards::{card_strengths, own_visible_current_strength},
    BotProfile,
};
use crate::BotTurn;

pub(crate) fn choose_raise_response<R: Rng>(
    turn: &BotTurn,
    profile: BotProfile,
    rng: &mut R,
) -> Option<Action> {
    let pending_raise = turn
        .view
        .hand
        .as_ref()?
        .public_state
        .pending_raise
        .clone()?;
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

    let hand = turn.view.hand.as_ref()?;
    let turnup = turn.turnup.as_ref()?;
    let strengths = card_strengths(&hand.hand, turnup);
    let shown_strength = own_visible_current_strength(turn);
    let shown_weight = if hand.hand.is_empty() { 1.0 } else { 0.75 };
    let top_strength = strengths
        .iter()
        .copied()
        .chain(shown_strength)
        .max()
        .unwrap_or(0);
    let weighted_total_strength = strengths.iter().map(|value| *value as f32).sum::<f32>()
        + shown_strength.map_or(0.0, |value| value as f32 * shown_weight);
    let strength_samples = strengths.len() as f32 + shown_strength.map_or(0.0, |_| shown_weight);
    let average_strength = if strength_samples == 0.0 {
        0.0
    } else {
        weighted_total_strength / strength_samples
    };
    let manilhas = hand
        .hand
        .iter()
        .filter(|card| card.is_manilha(turnup))
        .count()
        + if shown_strength.is_some_and(|value| value >= 10) {
            1
        } else {
            0
        };
    let last_card = hand.hand.len() == 1;
    let shown_last_card = hand.hand.is_empty() && shown_strength.is_some();

    if last_card || shown_last_card {
        let lone_strength = if shown_last_card {
            shown_strength.unwrap_or(0) as f32
        } else {
            strengths.first().copied().unwrap_or(0) as f32
        };
        let mut accept_probability = lone_strength / 13.0;
        accept_probability += match profile {
            BotProfile::Conservative => -0.1,
            BotProfile::Balanced => 0.0,
            BotProfile::Aggressive => 0.18,
            BotProfile::Tricky => 0.08,
        };
        if shown_last_card {
            accept_probability += 0.08;
        }
        accept_probability += match pending_raise.to {
            3 => 0.18,
            6 => 0.02,
            9 => -0.1,
            12 => -0.2,
            _ => 0.0,
        };
        accept_probability = accept_probability.clamp(0.05, 0.95);
        return if rng.gen_bool(accept_probability as f64) {
            accept
        } else {
            fold
        };
    }

    if pending_raise.to <= 3 && (top_strength >= 8 || manilhas > 0 || average_strength >= 6.0) {
        return accept;
    }

    if pending_raise.to >= 9 && manilhas == 0 && top_strength <= 8 {
        return fold;
    }

    let score = average_strength
        + top_strength as f32 * 0.35
        + manilhas as f32 * 2.1
        + match profile {
            BotProfile::Conservative => -0.7,
            BotProfile::Balanced => 0.0,
            BotProfile::Aggressive => 0.8,
            BotProfile::Tricky => 0.3,
        };
    let target = match pending_raise.to {
        3 => 6.0,
        6 => 7.0,
        9 => 8.0,
        12 => 9.4,
        _ => 7.0,
    };

    if score >= target {
        accept
    } else {
        fold
    }
}

pub(crate) fn choose_mao_de_onze_response<R: Rng>(
    turn: &BotTurn,
    profile: BotProfile,
    rng: &mut R,
) -> Option<Action> {
    let pending_decision = turn
        .view
        .hand
        .as_ref()?
        .public_state
        .pending_decision
        .clone()?;
    if pending_decision.player != turn.player {
        return None;
    }

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

    let hand = turn.view.hand.as_ref()?;
    let turnup = turn.turnup.as_ref()?;
    let strengths = card_strengths(&hand.hand, turnup);
    let top_strength = strengths.iter().copied().max().unwrap_or(0);
    let manilhas = hand
        .hand
        .iter()
        .filter(|card| card.is_manilha(turnup))
        .count();

    let mut accept_probability = 0.25
        + (top_strength as f32 / 13.0) * 0.45
        + manilhas as f32 * 0.2
        + match profile {
            BotProfile::Conservative => -0.12,
            BotProfile::Balanced => 0.0,
            BotProfile::Aggressive => 0.12,
            BotProfile::Tricky => 0.05,
        };
    accept_probability = accept_probability.clamp(0.05, 0.98);

    if rng.gen_bool(accept_probability as f64) {
        accept
    } else {
        fold
    }
}
