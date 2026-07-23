use truco_engine::{bot_analysis::TacticalActionSummary, Action};

use super::BotProfile;

pub(crate) fn single_choice_decision(action: Action, reasoning: String) -> crate::BotDecision {
    crate::BotDecision {
        action: action.clone(),
        plan: crate::BotPlan {
            choices: vec![crate::WeightedActionChoice {
                action,
                weight: 1.0,
            }],
            reasoning: Some(reasoning),
        },
    }
}

pub(crate) fn tactical_reasoning(profile: BotProfile, summary: &TacticalActionSummary) -> String {
    let action_label = describe_action(&summary.action);
    let worst_case = format_signed_points(summary.worst_case_signed_points);
    let average = format_signed_points_float(summary.average_signed_points);

    if summary.force_hand_win {
        format!(
            "{:?} heuristic profile found a forced hand win in {}/{} determinizations and chose {action_label} to preserve it (worst-case {worst_case}, average {average}).",
            profile, summary.winning_determinizations, summary.total_determinizations
        )
    } else {
        format!(
            "{:?} heuristic profile chose {action_label} from {}/{} winning determinizations (worst-case {worst_case}, average {average}).",
            profile, summary.winning_determinizations, summary.total_determinizations
        )
    }
}

pub(crate) fn tactical_raise_response_reasoning(
    profile: BotProfile,
    accept_summary: &TacticalActionSummary,
    fold_summary: &TacticalActionSummary,
    shown_strength: u8,
    accept_probability: f32,
) -> String {
    let accept_average = format_signed_points_float(accept_summary.average_signed_points);
    let fold_average = format_signed_points_float(fold_summary.average_signed_points);
    format!(
        "{:?} heuristic profile mixed the pending raise after revealing a {shown_strength}-strength card (accept avg {accept_average}, fold avg {fold_average}, accept {:.0}%).",
        profile,
        accept_probability * 100.0,
    )
}

pub(crate) fn describe_action(action: &Action) -> String {
    match action {
        Action::PlayFaceUp { card_id } => format!("face-up play {card_id}"),
        Action::PlayFaceDown { card_id } => format!("face-down play {card_id}"),
        Action::Raise { to } => format!("raise to {to}"),
        Action::AcceptRaise => "accept raise".to_string(),
        Action::Fold => "fold".to_string(),
        Action::AcceptEleven => "accept mão de onze".to_string(),
        Action::FoldEleven => "fold mão de onze".to_string(),
        Action::ConcedeHand => "concede hand".to_string(),
    }
}

pub(crate) fn format_signed_points(value: i32) -> String {
    if value >= 0 {
        format!("+{value}")
    } else {
        value.to_string()
    }
}

pub(crate) fn format_signed_points_float(value: f32) -> String {
    if value >= 0.0 {
        format!("+{value:.2}")
    } else {
        format!("{value:.2}")
    }
}
