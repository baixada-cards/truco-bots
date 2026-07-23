use rand::Rng;
use truco_engine::Action;

use super::{
    cards::{
        card_strength, card_strengths, collect_candidates, is_highest_manilha_clubs,
        visible_opponent_play,
    },
    raise::raise_action,
    BotProfile, CandidateAction,
};
use crate::BotTurn;

pub(crate) fn choose_play<R: Rng>(
    turn: &BotTurn,
    _profile: BotProfile,
    rng: &mut R,
) -> Option<Action> {
    let hand = turn.view.hand.as_ref()?;
    let turnup = turn.turnup.as_ref()?;
    let mut candidates = collect_candidates(turn, turnup);
    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by_key(|candidate| candidate.strength);
    let visible_opponent_strength =
        visible_opponent_play(&hand.public_state.current_round.plays, turn.player)
            .map(|play| card_strength(play, turnup));

    if let Some(target_strength) = visible_opponent_strength {
        let all_win = candidates
            .iter()
            .all(|candidate| candidate.strength > target_strength);
        let none_win = candidates
            .iter()
            .all(|candidate| candidate.strength <= target_strength);

        if all_win || none_win {
            return choose_candidate_action(candidates.first()?, true, !none_win, rng);
        }

        let winning = candidates
            .iter()
            .find(|candidate| candidate.strength > target_strength)?;
        return choose_candidate_action(winning, true, true, rng);
    }

    let non_club_manilha = candidates
        .iter()
        .find(|candidate| !is_highest_manilha_clubs(&candidate.card, turnup));

    if let Some(candidate) = non_club_manilha {
        return choose_candidate_action(candidate, false, false, rng);
    }

    choose_candidate_action(candidates.first()?, false, false, rng)
}

pub(crate) fn choose_candidate_action<R: Rng>(
    candidate: &CandidateAction,
    reacting: bool,
    face_up_would_win_round: bool,
    rng: &mut R,
) -> Option<Action> {
    let mut options = Vec::new();
    if let Some(up) = candidate.up.clone() {
        options.push(up);
    }
    if reacting && !face_up_would_win_round {
        if let Some(down) = candidate.down.clone() {
            return Some(down);
        }
    }
    if options.is_empty() {
        if let Some(down) = candidate.down.clone() {
            return Some(down);
        }
    }
    if options.len() == 1 {
        return options.into_iter().next();
    }
    let index = rng.gen_range(0..options.len());
    Some(options[index].clone())
}

pub(crate) fn choose_opening_raise<R: Rng>(
    turn: &BotTurn,
    profile: BotProfile,
    rng: &mut R,
) -> Option<Action> {
    let raise = raise_action(turn)?;
    let hand = turn.view.hand.as_ref()?;
    let turnup = turn.turnup.as_ref()?;
    if !hand.public_state.current_round.plays.is_empty() {
        return None;
    }
    if hand.public_state.hand_value >= 9 {
        return None;
    }

    let strengths = card_strengths(&hand.hand, turnup);
    let top = strengths.iter().copied().max().unwrap_or(0);
    let strong_count = strengths.iter().filter(|strength| **strength >= 8).count();
    let manilhas = hand
        .hand
        .iter()
        .filter(|card| card.is_manilha(turnup))
        .count();

    let mut raise_probability: f32 = match profile {
        BotProfile::Conservative => 0.05,
        BotProfile::Balanced => 0.14,
        BotProfile::Aggressive => 0.26,
        BotProfile::Tricky => 0.18,
    };

    if manilhas > 0 {
        raise_probability += 0.26;
    }
    if strong_count >= 2 {
        raise_probability += 0.18;
    }
    if top >= 12 {
        raise_probability += 0.1;
    }
    if top <= 7 && manilhas == 0 {
        raise_probability -= 0.16;
    }

    if strong_count == 0 && manilhas == 0 && top <= 7 {
        raise_probability = raise_probability.max(match profile {
            BotProfile::Conservative => 0.01,
            BotProfile::Balanced => 0.04,
            BotProfile::Aggressive => 0.08,
            BotProfile::Tricky => 0.12,
        });
    }

    raise_probability = raise_probability.clamp(0.0, 0.9);

    if rng.gen_bool(raise_probability as f64) {
        Some(raise)
    } else {
        None
    }
}

pub(crate) fn should_raise_before_showing_killer_card(
    turn: &BotTurn,
    chosen_play: Option<&Action>,
) -> bool {
    let Some(Action::PlayFaceUp { card_id } | Action::PlayFaceDown { card_id }) = chosen_play
    else {
        return false;
    };
    let Some(hand) = &turn.view.hand else {
        return false;
    };
    let Some(turnup) = &turn.turnup else {
        return false;
    };
    let Some(card) = hand.hand.iter().find(|candidate| &candidate.id == card_id) else {
        return false;
    };
    if !is_highest_manilha_clubs(card, turnup) {
        return false;
    }

    let prior_round_win = hand
        .public_state
        .completed_rounds
        .iter()
        .any(|round| round.winner == Some(turn.player));
    prior_round_win && raise_action(turn).is_some()
}
