use truco_engine::{Action, MATCH_TARGET};

use crate::BotTurn;

pub(crate) fn raise_action(turn: &BotTurn) -> Option<Action> {
    if !raise_is_worthwhile_for_match_score(turn) {
        return None;
    }

    turn.legal_actions.iter().find_map(|action| match action {
        Action::Raise { to } => Some(Action::Raise { to: *to }),
        _ => None,
    })
}

pub(crate) fn raise_is_worthwhile_for_match_score(turn: &BotTurn) -> bool {
    let Some(hand) = turn.view.hand.as_ref() else {
        return false;
    };

    let current_stake = hand
        .public_state
        .pending_raise
        .as_ref()
        .map_or(hand.public_state.hand_value, |pending_raise| {
            pending_raise.to
        });
    let score = match turn.player {
        0 => turn.view.score.zero,
        1 => turn.view.score.one,
        _ => 0,
    };
    let points_needed = MATCH_TARGET.saturating_sub(score);

    current_stake < points_needed
}
