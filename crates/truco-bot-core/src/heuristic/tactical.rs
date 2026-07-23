use rand::Rng;
use truco_engine::{bot_analysis::TacticalActionSummary, Action, Match};

use super::{
    cards::own_visible_current_strength,
    play::{choose_opening_raise, choose_play},
    raise::raise_action,
    raise_response::{choose_mao_de_onze_response, choose_raise_response},
    reasoning::tactical_raise_response_reasoning,
    BotProfile,
};
use crate::BotTurn;

pub(crate) fn should_use_tactical_oracle(turn: &BotTurn) -> bool {
    let Some(hand) = &turn.view.hand else {
        return false;
    };
    if !turn_matches_exact_state(turn) {
        return false;
    }
    if hand.public_state.pending_decision.is_some() {
        return false;
    }

    hand.hand.len() <= 2
        || hand.public_state.pending_raise.is_some()
        || !hand.public_state.completed_rounds.is_empty()
        || !hand.public_state.current_round.plays.is_empty()
}

pub(crate) fn turn_matches_exact_state(turn: &BotTurn) -> bool {
    let Ok(game) = Match::from_state(turn.match_state.clone()) else {
        return false;
    };
    let Ok(exact_turn) = crate::turn_for_match(&game, turn.player) else {
        return false;
    };

    exact_turn.turnup == turn.turnup
        && exact_turn.view == turn.view
        && exact_turn.legal_actions == turn.legal_actions
}

pub(crate) fn choose_tactical_action<R: Rng>(
    turn: &BotTurn,
    profile: BotProfile,
    best_summaries: &[&TacticalActionSummary],
    rng: &mut R,
) -> Option<Action> {
    if best_summaries.is_empty() {
        return None;
    }

    let mut restricted_turn = turn.clone();
    restricted_turn.legal_actions = best_summaries
        .iter()
        .map(|summary| summary.action.clone())
        .collect();

    if best_summaries.iter().any(|summary| summary.force_hand_win) {
        if let Some(action) = raise_action(&restricted_turn) {
            return Some(action);
        }
    }

    if let Some(action) = choose_raise_response(&restricted_turn, profile, rng) {
        return Some(action);
    }

    if let Some(action) = choose_mao_de_onze_response(&restricted_turn, profile, rng) {
        return Some(action);
    }

    if let Some(action) = choose_opening_raise(&restricted_turn, profile, rng) {
        return Some(action);
    }

    if let Some(action) = choose_play(&restricted_turn, profile, rng) {
        return Some(action);
    }

    restricted_turn.legal_actions.first().cloned()
}

pub(crate) fn choose_tactical_raise_response<R: Rng>(
    turn: &BotTurn,
    profile: BotProfile,
    summaries: &[TacticalActionSummary],
    rng: &mut R,
) -> Option<(Action, String)> {
    let hand = turn.view.hand.as_ref()?;
    let pending_raise = hand.public_state.pending_raise.as_ref()?;

    let shown_strength = own_visible_current_strength(turn)?;
    let accept_summary = summaries
        .iter()
        .find(|summary| matches!(summary.action, Action::AcceptRaise))?;
    let fold_summary = summaries
        .iter()
        .find(|summary| matches!(summary.action, Action::Fold))?;
    let accept_win_rate = accept_summary.winning_determinizations as f32
        / accept_summary.total_determinizations as f32;
    let fold_win_rate =
        fold_summary.winning_determinizations as f32 / fold_summary.total_determinizations as f32;
    let average_gap = accept_summary.average_signed_points - fold_summary.average_signed_points;
    let shown_pressure = shown_strength as f32 / 13.0;

    if average_gap >= 1.25 || (accept_win_rate >= 0.72 && shown_strength >= 9) {
        let reasoning = tactical_raise_response_reasoning(
            profile,
            accept_summary,
            fold_summary,
            shown_strength,
            1.0,
        );
        return Some((Action::AcceptRaise, reasoning));
    }

    if average_gap <= -3.0 && accept_win_rate <= 0.12 && shown_strength <= 6 {
        let reasoning = tactical_raise_response_reasoning(
            profile,
            accept_summary,
            fold_summary,
            shown_strength,
            0.0,
        );
        return Some((Action::Fold, reasoning));
    }

    let mut accept_probability = 0.1
        + shown_pressure * 0.35
        + accept_win_rate * 0.3
        + (accept_win_rate - fold_win_rate) * 0.2
        + average_gap * 0.1
        + if hand.hand.is_empty() { 0.1 } else { 0.0 }
        + if accept_summary.worst_case_signed_points >= fold_summary.worst_case_signed_points {
            0.06
        } else {
            0.0
        }
        + match profile {
            BotProfile::Conservative => -0.08,
            BotProfile::Balanced => 0.0,
            BotProfile::Aggressive => 0.08,
            BotProfile::Tricky => 0.12,
        }
        + match pending_raise.to {
            3 => 0.1,
            6 => 0.02,
            9 => -0.1,
            12 => -0.18,
            _ => 0.0,
        };
    accept_probability = accept_probability.clamp(0.08, 0.9);

    let action = if rng.gen_bool(accept_probability as f64) {
        Action::AcceptRaise
    } else {
        Action::Fold
    };
    let reasoning = tactical_raise_response_reasoning(
        profile,
        accept_summary,
        fold_summary,
        shown_strength,
        accept_probability,
    );
    Some((action, reasoning))
}

pub(crate) fn best_tactical_summaries(
    summaries: &[TacticalActionSummary],
) -> Vec<&TacticalActionSummary> {
    let mut winners = filter_tactical_ties(summaries.iter().collect(), |summary| {
        if summary.force_hand_win {
            1_i32
        } else {
            0_i32
        }
    });
    winners = filter_tactical_ties(winners, |summary| summary.worst_case_signed_points);
    filter_tactical_float_ties(winners, |summary| summary.average_signed_points)
}

pub(crate) fn forced_raise_summary<'a>(
    turn: &BotTurn,
    summaries: &'a [TacticalActionSummary],
) -> Option<&'a TacticalActionSummary> {
    let raise = raise_action(turn)?;
    summaries
        .iter()
        .find(|summary| summary.force_hand_win && summary.action == raise)
}

fn filter_tactical_ties<F>(
    summaries: Vec<&TacticalActionSummary>,
    score: F,
) -> Vec<&TacticalActionSummary>
where
    F: Fn(&TacticalActionSummary) -> i32,
{
    let Some(best_score) = summaries.iter().map(|summary| score(summary)).max() else {
        return summaries;
    };
    summaries
        .into_iter()
        .filter(|summary| score(summary) == best_score)
        .collect()
}

fn filter_tactical_float_ties<F>(
    summaries: Vec<&TacticalActionSummary>,
    score: F,
) -> Vec<&TacticalActionSummary>
where
    F: Fn(&TacticalActionSummary) -> f32,
{
    let Some(best_score) = summaries
        .iter()
        .map(|summary| score(summary))
        .max_by(|left, right| left.total_cmp(right))
    else {
        return summaries;
    };
    summaries
        .into_iter()
        .filter(|summary| score(summary).total_cmp(&best_score).is_eq())
        .collect()
}
