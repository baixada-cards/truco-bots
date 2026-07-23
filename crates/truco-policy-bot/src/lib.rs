//! Live-match bot that plays the solved CFR average strategy.
//!
//! The solved region is 10x10 plus the mão-de-onze row (`EXACT_SOLVING.md`):
//! from 10x10 every continuation lands in a solved score or ends the match,
//! which is why solver matches start there. Per decision the bot rebuilds the
//! solver's `InfoSet` from the hand's exact action log (kept by the hosting
//! service — the engine's exported state drops where resolved raises sat in
//! the sequence, so the log is the only faithful source), looks the key up in
//! a mmap-ed [`bot_policy`] profile, and samples the mixed average strategy.
//! Any gap — uncovered score, missing key, unmappable action — falls back to
//! the heuristic bot, loudly.
//!
//! Seat symmetry: profiles are stored in the solve's frame. A live hand whose
//! (score, dealer) is the seat-transpose of a stored profile is served by
//! swapping `InfoSet::player` (10x10 dealer-1 via the dealer-0 solve; seat-1
//! mão-de-onze states via the seat-0 row). `is_dealer`, the turnup class, the
//! hand, and the history are all frame-independent.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use smallvec::SmallVec;

use truco_bot_core::{
    resolve_weighted_action, BotDecision, BotDecisionSource, BotError, BotPlan, BotProfile,
    BotTurn, HeuristicBot, WeightedActionChoice,
};
use truco_engine::{Action, Card, Player, Turnup};
use truco_policy_format::abstraction::{abstract_card, turnup_class, AbstractHand};
use truco_policy_format::file::{BotPolicyEntry, BotPolicyFile};
use truco_policy_format::info_set::{AbstractAction, InfoSet};
use truco_policy_format::manifest::PolicyManifest;

/// One action as observed by the hosting service, in hand order. `card` is
/// resolved at log time for play actions (the service sees the authoritative
/// state before the card leaves the hand).
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedAction {
    pub player: Player,
    pub action: Action,
    pub card: Option<Card>,
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyStoreError {
    #[error("policy manifest: {0}")]
    Manifest(String),
    #[error("policy profile {file}: {message}")]
    Profile { file: String, message: String },
}

/// (seat0 score, seat1 score, turnup class, dealer) in the solve's frame.
type ProfileKey = (u8, u8, u8, u8);

/// All bot-policy profiles from one directory, mmap-ed and shared. Selection
/// handles the seat transpose; `swapped` tells the caller to relabel
/// `InfoSet::player` into the solve's frame.
#[derive(Debug)]
pub struct PolicyStore {
    profiles: HashMap<ProfileKey, BotPolicyFile>,
    dir: PathBuf,
}

impl PolicyStore {
    pub fn load(dir: &Path) -> Result<Self, PolicyStoreError> {
        let manifest_path = dir.join("manifest.json");
        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| PolicyStoreError::Manifest(format!("{}: {e}", manifest_path.display())))?;
        let manifest =
            PolicyManifest::parse(&raw).map_err(|e| PolicyStoreError::Manifest(e.to_string()))?;
        let mut profiles = HashMap::with_capacity(manifest.profiles.len());
        for profile in manifest.profiles {
            let file = BotPolicyFile::open(&dir.join(&profile.file)).map_err(|e| {
                PolicyStoreError::Profile {
                    file: profile.file.clone(),
                    message: e.to_string(),
                }
            })?;
            let key = (
                profile.score[0],
                profile.score[1],
                profile.tc,
                profile.dealer,
            );
            profiles.insert(key, file);
        }
        Ok(Self {
            profiles,
            dir: dir.to_path_buf(),
        })
    }

    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Find the profile serving a live hand, either directly or through the
    /// seat transpose. Returns `(profile, swapped)`.
    pub(crate) fn select(
        &self,
        score: (u8, u8),
        tc: u8,
        dealer: u8,
    ) -> Option<(&BotPolicyFile, bool)> {
        if let Some(profile) = self.profiles.get(&(score.0, score.1, tc, dealer)) {
            return Some((profile, false));
        }
        self.profiles
            .get(&(score.1, score.0, tc, 1 - dealer))
            .map(|profile| (profile, true))
    }

    pub fn covers(&self, score: (u8, u8), tc: u8, dealer: u8) -> bool {
        self.select(score, tc, dealer).is_some()
    }
}

/// Why a decision could not be served from the policy artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PolicyMiss {
    NoHand,
    UncoveredSpot { score: (u8, u8), tc: u8, dealer: u8 },
    UnreconstructableLog(String),
    IncompleteHand,
    MissingKey(u64),
    CorruptProfile(String),
    NoLegalChoices,
}

impl PolicyMiss {
    fn label(&self) -> String {
        match self {
            Self::NoHand => "no hand in progress".into(),
            Self::UncoveredSpot { score, tc, dealer } => format!(
                "no profile for score {}x{} tc{tc} dealer {dealer}",
                score.0, score.1
            ),
            Self::UnreconstructableLog(reason) => format!("action log: {reason}"),
            Self::IncompleteHand => "could not reconstruct the starting hand".into(),
            Self::MissingKey(key) => format!("info set {key} missing from profile"),
            Self::CorruptProfile(reason) => format!("invalid policy profile: {reason}"),
            Self::NoLegalChoices => "no stored action is currently legal".into(),
        }
    }
}

/// The hosted solver bot: policy lookup with a heuristic fallback.
#[derive(Debug, Clone)]
pub struct SolverPolicyBot {
    store: Arc<PolicyStore>,
    rng: StdRng,
    fallback: HeuristicBot<StdRng>,
}

impl SolverPolicyBot {
    pub fn new(store: Arc<PolicyStore>, seed: Option<u64>) -> Self {
        let rng_seed = seed.unwrap_or(0x501B_EB07_0000).wrapping_add(0x501B_EB07);
        Self {
            store,
            rng: StdRng::seed_from_u64(rng_seed),
            fallback: HeuristicBot::new(
                StdRng::seed_from_u64(rng_seed.wrapping_add(1)),
                BotProfile::Balanced,
            ),
        }
    }

    pub fn store(&self) -> &Arc<PolicyStore> {
        &self.store
    }

    /// Decide from the solved policy; `hand_log` is the current hand's exact
    /// action sequence as observed by the host, oldest first.
    pub fn choose_decision(
        &mut self,
        turn: &BotTurn,
        hand_log: &[ObservedAction],
    ) -> Result<BotDecision, BotError> {
        match self.policy_decision(turn, hand_log) {
            Ok(decision) => Ok(decision),
            Err(miss) => {
                // stderr, not the `log` facade: the hosting service initializes
                // no logger, and a silent fallback would look like the solver
                // playing badly.
                eprintln!(
                    "solver bot fallback to heuristic: {} (player {}, log length {})",
                    miss.label(),
                    turn.player,
                    hand_log.len()
                );
                let mut decision = self.fallback.choose_decision(turn)?;
                decision.plan.reasoning = Some(format!(
                    "solver fallback ({}); {}",
                    miss.label(),
                    decision.plan.reasoning.as_deref().unwrap_or("heuristic")
                ));
                Ok(decision)
            }
        }
    }

    fn policy_decision(
        &mut self,
        turn: &BotTurn,
        hand_log: &[ObservedAction],
    ) -> Result<BotDecision, PolicyMiss> {
        let hand = turn
            .match_state
            .current_hand
            .as_ref()
            .ok_or(PolicyMiss::NoHand)?;
        let st = &hand.state;
        let turnup = &st.turnup;
        let tc = turnup_class(turnup);
        let dealer = st.dealer;
        let score = (st.score.zero, st.score.one);

        let (profile, swapped) = self
            .store
            .select(score, tc.blocked_plain_level, dealer)
            .ok_or(PolicyMiss::UncoveredSpot {
                score,
                tc: tc.blocked_plain_level,
                dealer,
            })?;

        let remaining: &[Card] = turn
            .view
            .hand
            .as_ref()
            .map(|hand_view| hand_view.hand.as_slice())
            .unwrap_or(&[]);
        let starting_hand = starting_abstract_hand(st, turn.player, remaining, turnup)
            .ok_or(PolicyMiss::IncompleteHand)?;

        let solve_frame_player = if swapped {
            1 - turn.player
        } else {
            turn.player
        };
        let mut info_set =
            InfoSet::new(solve_frame_player, turn.player == dealer, tc, starting_hand);
        for observed in hand_log {
            let abs =
                observed_to_abstract(observed, turnup).map_err(PolicyMiss::UnreconstructableLog)?;
            if observed.player == turn.player {
                info_set.record_own_action(abs);
            } else {
                info_set.record_opponent_action(abs);
            }
        }

        let key = info_set.key();
        let entry = profile
            .lookup_checked(key)
            .map_err(|error| PolicyMiss::CorruptProfile(error.to_string()))?
            .ok_or(PolicyMiss::MissingKey(key.0))?;

        let choices = concrete_choices(&entry, remaining, turnup, &turn.legal_actions);
        if choices.is_empty() {
            return Err(PolicyMiss::NoLegalChoices);
        }
        let plan = BotPlan {
            choices,
            reasoning: Some("solved equilibrium (average strategy)".to_string()),
        };
        let action = resolve_weighted_action(&plan, &turn.legal_actions, &mut self.rng)
            .map_err(|_| PolicyMiss::NoLegalChoices)?;
        Ok(BotDecision { action, plan })
    }
}

/// The bot's dealt hand in abstract, sorted form: remaining cards plus its own
/// plays recorded in the hand state.
fn starting_abstract_hand(
    st: &truco_engine::GameState,
    player: Player,
    remaining: &[Card],
    turnup: &Turnup,
) -> Option<AbstractHand> {
    let mut cards: SmallVec<[&Card; 3]> = remaining.iter().collect();
    for play in st
        .completed_rounds
        .iter()
        .flat_map(|round| round.plays.iter())
        .chain(st.current_round.plays.iter())
    {
        if play.player == player {
            cards.push(&play.card);
        }
    }
    if cards.len() != 3 {
        return None;
    }
    let mut hand: AbstractHand = cards
        .iter()
        .map(|card| abstract_card(card, turnup))
        .collect();
    hand.sort();
    Some(hand)
}

fn observed_to_abstract(
    observed: &ObservedAction,
    turnup: &Turnup,
) -> Result<AbstractAction, String> {
    let played_card = |label: &str| {
        observed
            .card
            .as_ref()
            .map(|card| abstract_card(card, turnup))
            .ok_or_else(|| format!("{label} entry is missing its resolved card"))
    };
    match &observed.action {
        Action::PlayFaceUp { .. } => Ok(AbstractAction::PlayFaceUp(played_card("play_face_up")?)),
        Action::PlayFaceDown { .. } => {
            Ok(AbstractAction::PlayFaceDown(played_card("play_face_down")?))
        }
        Action::Raise { to } => Ok(AbstractAction::Raise(*to)),
        Action::AcceptRaise => Ok(AbstractAction::AcceptRaise),
        Action::Fold => Ok(AbstractAction::Fold),
        Action::AcceptEleven => Ok(AbstractAction::AcceptEleven),
        Action::FoldEleven => Ok(AbstractAction::FoldEleven),
        Action::ConcedeHand => Err("concede_hand is outside the solver's action space".into()),
    }
}

/// Map a stored policy entry onto concrete, currently-legal engine actions.
/// Duplicate concrete cards of one abstract class are interchangeable; the
/// first match carries the class's whole probability.
fn concrete_choices(
    entry: &BotPolicyEntry,
    remaining: &[Card],
    turnup: &Turnup,
    legal_actions: &[Action],
) -> Vec<WeightedActionChoice> {
    let find_card = |target| {
        remaining
            .iter()
            .find(|card| abstract_card(card, turnup) == target)
            .map(|card| card.id.clone())
    };
    let mut choices = Vec::with_capacity(entry.actions.len());
    for (&abs, &weight) in entry.actions.iter().zip(entry.probabilities.iter()) {
        let action = match abs {
            AbstractAction::PlayFaceUp(card) => match find_card(card) {
                Some(card_id) => Action::PlayFaceUp { card_id },
                None => continue,
            },
            AbstractAction::PlayFaceDown(card) => match find_card(card) {
                Some(card_id) => Action::PlayFaceDown { card_id },
                None => continue,
            },
            AbstractAction::Raise(to) => Action::Raise { to },
            AbstractAction::AcceptRaise => Action::AcceptRaise,
            AbstractAction::Fold => Action::Fold,
            AbstractAction::AcceptEleven => Action::AcceptEleven,
            AbstractAction::FoldEleven => Action::FoldEleven,
            AbstractAction::OpponentPlayedHidden => continue,
        };
        if legal_actions.contains(&action) {
            choices.push(WeightedActionChoice { action, weight });
        }
    }
    choices
}

pub mod seed;

#[cfg(test)]
mod tests;
