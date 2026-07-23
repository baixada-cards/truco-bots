//! Seeded live matches: reproduce a study-lab position against the hosted bot.
//!
//! The lab knows the hero's cards, the vira, and the line played so far; the
//! villain's hand is either pinned exactly or left as a range. This module
//! turns that into a real engine `Match`:
//!
//! - **Pinned villain** → he holds exactly those cards (point-mass posterior).
//! - **Unpinned villain** → his hand is sampled from the Bayesian posterior
//!   `P(h | history) ∝ prior(h) × Π σ*(observed villain action | info set)`,
//!   where the prior is the combinatorial deal weight (NOT uniform over
//!   abstract hands: a hand with more concrete realizations is more likely)
//!   and the likelihood conditions on the villain having played the solved
//!   equilibrium along the analyzed line. When no policy artifacts are
//!   mounted, or the line is fully off-equilibrium (all likelihoods zero),
//!   sampling degrades to the prior alone and says so.
//!
//! The replay applies the history verbatim to the engine (both seats' actions,
//! no bot interference) and returns the exact `ObservedAction` log, so the
//! solver bot's info-set bridge resumes from the analyzed node with a faithful
//! history — including resolved raises that the exported state alone forgets.

use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;

use truco_engine::{Action, Card, Hands, Match, MatchState, Player, Rank, Score, Suit, Turnup};
use truco_policy_format::abstraction::{
    abstract_card, turnup_class, AbstractCard, AbstractHand, ALL_RANKS, ALL_SUITS,
};
use truco_policy_format::info_set::{AbstractAction, InfoSet};

use crate::{ObservedAction, PolicyStore};

/// One replayed action from the analyzed line. Card plays carry the abstract
/// class (0..=12); the concrete card is realized here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SeededActionKind {
    PlayFaceUp { class: u8 },
    PlayFaceDown { class: u8 },
    Raise { to: u8 },
    AcceptRaise,
    Fold,
    AcceptEleven,
    FoldEleven,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeededHistoryAction {
    pub seat: Player,
    #[serde(flatten)]
    pub kind: SeededActionKind,
}

/// A lab position to realize as a live hand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedSpec {
    pub score: Score,
    pub dealer: Player,
    pub vira_rank: Rank,
    pub human_player: Player,
    /// The human's three cards as abstract class indices (0..=12) when the
    /// lab drafted them; `None` samples the human's unspecified cards from
    /// the same line-conditioned posterior as the villain's (their committed
    /// plays stay fixed).
    #[serde(default)]
    pub hero_hand: Option<Vec<u8>>,
    /// The bot's exact hand when the lab has it pinned; `None` samples it.
    #[serde(default)]
    pub villain_hand: Option<Vec<u8>>,
    #[serde(default)]
    pub history: Vec<SeededHistoryAction>,
}

/// How the villain's hand was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VillainSampling {
    /// The lab pinned it; dealt exactly.
    Pinned,
    /// Sampled from prior × equilibrium likelihood of the observed line.
    Posterior,
    /// Sampled from the combinatorial prior alone (no artifacts, or the line
    /// has zero equilibrium mass).
    Prior,
}

/// A realized, replayed seeded hand ready to host.
#[derive(Debug)]
pub struct SeededHand {
    pub state: MatchState,
    pub log: Vec<ObservedAction>,
    pub sampling: VillainSampling,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SeedError {
    #[error("invalid seed spec: {0}")]
    Invalid(String),
    #[error("seeded history does not replay: {0}")]
    History(String),
    #[error("no deal is consistent with this position")]
    NoConsistentHand,
}

const NUM_CLASSES: usize = 13;

/// The concrete cards a class can realize under this vira, minus the turnup.
struct ConcretePool {
    /// options[class] = remaining concrete (rank, suit) pairs.
    options: [SmallVec<[(Rank, Suit); 4]>; NUM_CLASSES],
}

impl ConcretePool {
    fn new(vira: &Turnup) -> Self {
        let manilha_rank = vira.rank.next_for_manilha();
        let plain_ladder: Vec<Rank> = ALL_RANKS
            .iter()
            .copied()
            .filter(|rank| *rank != manilha_rank)
            .collect();
        let mut options: [SmallVec<[(Rank, Suit); 4]>; NUM_CLASSES] = Default::default();
        for (level, rank) in plain_ladder.iter().enumerate() {
            for suit in ALL_SUITS {
                if *rank == vira.rank && suit == vira.suit {
                    continue; // the turnup itself is out of the deck
                }
                options[level].push((*rank, suit));
            }
        }
        for suit in ALL_SUITS {
            options[9 + suit.manilha_strength()].push((manilha_rank, suit));
        }
        Self { options }
    }

    fn available(&self, class: u8) -> usize {
        self.options[class as usize].len()
    }

    /// Take one concrete card of the class; `pick` indexes the remaining
    /// options so unknown cards can be realized uniformly.
    fn take(&mut self, class: u8, pick: usize) -> Option<Card> {
        let options = &mut self.options[class as usize];
        if options.is_empty() {
            return None;
        }
        let (rank, suit) = options.swap_remove(pick % options.len());
        Some(Card {
            id: format!("{}{}", rank_char(rank), suit_char(suit)).into(),
            rank,
            suit,
        })
    }
}

fn rank_char(rank: Rank) -> char {
    match rank.index() {
        0 => '4',
        1 => '5',
        2 => '6',
        3 => '7',
        4 => 'q',
        5 => 'j',
        6 => 'k',
        7 => 'a',
        8 => '2',
        _ => '3',
    }
}

fn suit_char(suit: Suit) -> char {
    match suit {
        Suit::Diamonds => 'd',
        Suit::Spades => 's',
        Suit::Hearts => 'h',
        Suit::Clubs => 'c',
    }
}

fn class_card(class: u8) -> Result<AbstractCard, SeedError> {
    if class as usize >= NUM_CLASSES {
        return Err(SeedError::Invalid(format!(
            "card class {class} out of range"
        )));
    }
    Ok(AbstractCard::from_type_index(class as usize))
}

fn seeded_to_abstract(kind: &SeededActionKind) -> Result<AbstractAction, SeedError> {
    Ok(match kind {
        SeededActionKind::PlayFaceUp { class } => AbstractAction::PlayFaceUp(class_card(*class)?),
        SeededActionKind::PlayFaceDown { class } => {
            AbstractAction::PlayFaceDown(class_card(*class)?)
        }
        SeededActionKind::Raise { to } => AbstractAction::Raise(*to),
        SeededActionKind::AcceptRaise => AbstractAction::AcceptRaise,
        SeededActionKind::Fold => AbstractAction::Fold,
        SeededActionKind::AcceptEleven => AbstractAction::AcceptEleven,
        SeededActionKind::FoldEleven => AbstractAction::FoldEleven,
    })
}

/// The classes a seat's history plays commit it to holding, in play order.
fn played_classes(spec: &SeedSpec, seat: Player) -> Vec<u8> {
    spec.history
        .iter()
        .filter(|entry| entry.seat == seat)
        .filter_map(|entry| match entry.kind {
            SeededActionKind::PlayFaceUp { class } | SeededActionKind::PlayFaceDown { class } => {
                Some(class)
            }
            _ => None,
        })
        .collect()
}

/// `needed` must be a sub-multiset of `hand`.
fn contains_multiset(hand: &[u8], needed: &[u8]) -> bool {
    let mut counts = [0i32; NUM_CLASSES];
    for &class in hand {
        counts[class as usize] += 1;
    }
    for &class in needed {
        counts[class as usize] -= 1;
        if counts[class as usize] < 0 {
            return false;
        }
    }
    true
}

fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut result = 1.0;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Equilibrium likelihood of the villain's observed actions given a candidate
/// full hand, from the policy artifacts. `None` when a required info set is
/// missing (degrade to prior); `Some(0.0)` is a real zero (hand excluded).
fn line_likelihood(
    spec: &SeedSpec,
    store: &PolicyStore,
    villain_seat: Player,
    candidate: &AbstractHand,
) -> Option<f64> {
    let tc = turnup_class(&Turnup {
        rank: spec.vira_rank,
        suit: Suit::Hearts,
    });
    let (profile, swapped) = store.select(
        (spec.score.zero, spec.score.one),
        tc.blocked_plain_level,
        spec.dealer,
    )?;
    let solve_frame_player = if swapped {
        1 - villain_seat
    } else {
        villain_seat
    };
    let mut info_set = InfoSet::new(
        solve_frame_player,
        villain_seat == spec.dealer,
        tc,
        candidate.clone(),
    );
    let mut likelihood = 1.0f64;
    for entry in &spec.history {
        let abs = seeded_to_abstract(&entry.kind).ok()?;
        if entry.seat == villain_seat {
            let stored = profile.lookup_checked(info_set.key()).ok()??;
            let probability = stored
                .actions
                .iter()
                .zip(stored.probabilities.iter())
                .find(|(action, _)| **action == abs)
                .map(|(_, p)| *p as f64)
                // The action is absent from the solved tree at this node (e.g.
                // pruned): the equilibrium never plays it from this hand.
                .unwrap_or(0.0);
            likelihood *= probability;
            if likelihood == 0.0 {
                return Some(0.0);
            }
            info_set.record_own_action(abs);
        } else {
            info_set.record_opponent_action(abs);
        }
    }
    Some(likelihood)
}

/// Enumerate size-`unknowns` multisets over classes within `avail`, calling
/// `visit(counts, prior_weight)` for each.
fn for_each_completion(
    avail: &[usize; NUM_CLASSES],
    unknowns: usize,
    visit: &mut impl FnMut(&[u8], f64),
) {
    fn recurse(
        avail: &[usize; NUM_CLASSES],
        class: usize,
        remaining: usize,
        picked: &mut Vec<u8>,
        weight: f64,
        visit: &mut impl FnMut(&[u8], f64),
    ) {
        if remaining == 0 {
            visit(picked, weight);
            return;
        }
        if class >= NUM_CLASSES {
            return;
        }
        let max_take = avail[class].min(remaining);
        for take in 0..=max_take {
            for _ in 0..take {
                picked.push(class as u8);
            }
            recurse(
                avail,
                class + 1,
                remaining - take,
                picked,
                weight * binomial(avail[class], take),
                visit,
            );
            for _ in 0..take {
                picked.pop();
            }
        }
    }
    let mut picked = Vec::with_capacity(unknowns);
    recurse(avail, 0, unknowns, &mut picked, 1.0, visit);
}

/// Whether the policy store can serve the seeded spot — the gate for offering
/// a solver opponent at this position (checked before sampling).
pub fn store_covers_spec(store: &PolicyStore, spec: &SeedSpec) -> bool {
    let tc = turnup_class(&Turnup {
        rank: spec.vira_rank,
        suit: Suit::Hearts,
    });
    store.covers(
        (spec.score.zero, spec.score.one),
        tc.blocked_plain_level,
        spec.dealer,
    )
}

/// One enumerated way to fill a seat's unknown cards.
struct Completion {
    classes: Vec<u8>,
    counts: [u8; NUM_CLASSES],
}

/// Realize the position: deal both hands (sampling any unspecified cards from
/// the joint line-conditioned posterior), replay the history, and return the
/// resulting state plus the exact log.
pub fn build_seeded_hand(
    spec: &SeedSpec,
    store: Option<&Arc<PolicyStore>>,
    rng: &mut StdRng,
) -> Result<SeededHand, SeedError> {
    if !matches!(spec.human_player, 0 | 1) || !matches!(spec.dealer, 0 | 1) {
        return Err(SeedError::Invalid("seats must be 0 or 1".into()));
    }
    let villain_seat = 1 - spec.human_player;
    for entry in &spec.history {
        if !matches!(entry.seat, 0 | 1) {
            return Err(SeedError::Invalid("history seat must be 0 or 1".into()));
        }
        seeded_to_abstract(&entry.kind)?;
    }

    // Each seat's committed cards: a full pinned hand, or just the cards its
    // history plays force it to hold.
    let hero_played = played_classes(spec, spec.human_player);
    let villain_played = played_classes(spec, villain_seat);
    if hero_played.len() > 3 || villain_played.len() > 3 {
        return Err(SeedError::Invalid(
            "a seat cannot have played more than 3 cards".into(),
        ));
    }
    let validate_pin = |pin: &[u8], played: &[u8], who: &str| -> Result<(), SeedError> {
        if pin.len() != 3 {
            return Err(SeedError::Invalid(format!(
                "pinned {who} hand must have exactly 3 cards"
            )));
        }
        if !contains_multiset(pin, played) {
            return Err(SeedError::Invalid(format!(
                "the {who}'s played cards are not all in the pinned hand"
            )));
        }
        Ok(())
    };
    if let Some(pin) = &spec.hero_hand {
        validate_pin(pin, &hero_played, "hero")?;
    }
    if let Some(pin) = &spec.villain_hand {
        validate_pin(pin, &villain_played, "villain")?;
    }
    let hero_base = spec
        .hero_hand
        .clone()
        .unwrap_or_else(|| hero_played.clone());
    let villain_base = spec
        .villain_hand
        .clone()
        .unwrap_or_else(|| villain_played.clone());
    for &class in hero_base.iter().chain(villain_base.iter()) {
        class_card(class)?;
    }

    let turnup = Turnup {
        rank: spec.vira_rank,
        suit: Suit::Hearts,
    };
    let mut pool = ConcretePool::new(&turnup);

    // Class-level accounting: both seats' committed cards leave the pool
    // before unknowns are enumerated over what remains.
    let mut avail = [0usize; NUM_CLASSES];
    for (class, slot) in avail.iter_mut().enumerate() {
        *slot = pool.available(class as u8);
    }
    for &class in hero_base.iter().chain(villain_base.iter()) {
        let slot = &mut avail[class as usize];
        if *slot == 0 {
            return Err(SeedError::NoConsistentHand);
        }
        *slot -= 1;
    }

    let enumerate = |unknowns: usize| -> Vec<Completion> {
        let mut out = Vec::new();
        for_each_completion(&avail, unknowns, &mut |classes, _| {
            let mut counts = [0u8; NUM_CLASSES];
            for &class in classes {
                counts[class as usize] += 1;
            }
            out.push(Completion {
                classes: classes.to_vec(),
                counts,
            });
        });
        out
    };
    let hero_completions = enumerate(3 - hero_base.len());
    let villain_completions = enumerate(3 - villain_base.len());
    if hero_completions.is_empty() || villain_completions.is_empty() {
        return Err(SeedError::NoConsistentHand);
    }

    // Per-seat equilibrium likelihood of the observed line for every candidate
    // full hand. Only sampled seats are conditioned: a pinned seat's
    // likelihood is a constant factor that cancels in normalization (and a
    // literal zero there would falsely nuke the other seat's posterior).
    // `None` anywhere means the artifacts cannot answer; degrade to prior.
    let seat_acted = |seat: Player| spec.history.iter().any(|entry| entry.seat == seat);
    let likelihoods =
        |store: &Arc<PolicyStore>, seat: Player, base: &[u8], comps: &[Completion]| {
            if !seat_acted(seat) {
                return Some(vec![1.0; comps.len()]);
            }
            comps
                .iter()
                .map(|comp| {
                    let mut hand: AbstractHand = base
                        .iter()
                        .chain(comp.classes.iter())
                        .map(|&class| AbstractCard::from_type_index(class as usize))
                        .collect();
                    hand.sort();
                    line_likelihood(spec, store, seat, &hand)
                })
                .collect::<Option<Vec<f64>>>()
        };
    let uniform = |n: usize| vec![1.0f64; n];
    let mut posterior_ok = false;
    let (hero_liks, villain_liks) = match store {
        Some(store) => {
            let hero = if spec.hero_hand.is_none() {
                likelihoods(store, spec.human_player, &hero_base, &hero_completions)
            } else {
                Some(uniform(hero_completions.len()))
            };
            let villain = if spec.villain_hand.is_none() {
                likelihoods(store, villain_seat, &villain_base, &villain_completions)
            } else {
                Some(uniform(villain_completions.len()))
            };
            match (hero, villain) {
                (Some(hero), Some(villain)) => {
                    posterior_ok = true;
                    (hero, villain)
                }
                _ => (
                    uniform(hero_completions.len()),
                    uniform(villain_completions.len()),
                ),
            }
        }
        None => (
            uniform(hero_completions.len()),
            uniform(villain_completions.len()),
        ),
    };

    // Joint prior: both completions draw from the SAME remaining pool, so the
    // weight is the sequential multinomial count, zero when they collide.
    let joint_prior = |hero: &[u8; NUM_CLASSES], villain: &[u8; NUM_CLASSES]| -> f64 {
        let mut weight = 1.0;
        for class in 0..NUM_CLASSES {
            let (h, v) = (hero[class] as usize, villain[class] as usize);
            if h + v > avail[class] {
                return 0.0;
            }
            weight *= binomial(avail[class], h) * binomial(avail[class] - h, v);
        }
        weight
    };
    let fill_weights = |use_likelihood: bool| -> Vec<f64> {
        let mut weights = Vec::with_capacity(hero_completions.len() * villain_completions.len());
        for (i, hero) in hero_completions.iter().enumerate() {
            for (j, villain) in villain_completions.iter().enumerate() {
                let mut weight = joint_prior(&hero.counts, &villain.counts);
                if use_likelihood {
                    weight *= hero_liks[i] * villain_liks[j];
                }
                weights.push(weight);
            }
        }
        weights
    };

    let mut mode = VillainSampling::Prior;
    let mut weights = if posterior_ok {
        let posterior = fill_weights(true);
        if posterior.iter().sum::<f64>() > 0.0 {
            mode = VillainSampling::Posterior;
            posterior
        } else {
            // Fully off-equilibrium line: keep the prior and say so.
            fill_weights(false)
        }
    } else {
        fill_weights(false)
    };
    if spec.villain_hand.is_some() {
        mode = VillainSampling::Pinned;
    }

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Err(SeedError::NoConsistentHand);
    }
    let mut draw = rng.gen_range(0.0..total);
    let mut chosen = weights.len() - 1;
    for (index, weight) in weights.iter().enumerate() {
        if draw < *weight {
            chosen = index;
            break;
        }
        draw -= weight;
    }
    weights.clear();
    let hero_pick = &hero_completions[chosen / villain_completions.len()];
    let villain_pick = &villain_completions[chosen % villain_completions.len()];

    // Realize concrete cards: committed cards deterministically, sampled
    // unknowns as a uniform concrete copy so the deal matches the prior.
    let mut realize = |base: &[u8],
                       completion: &Completion,
                       rng: &mut StdRng|
     -> Result<SmallVec<[Card; 3]>, SeedError> {
        let mut cards: SmallVec<[Card; 3]> = SmallVec::new();
        for &class in base {
            cards.push(pool.take(class, 0).ok_or(SeedError::NoConsistentHand)?);
        }
        for &class in &completion.classes {
            let pick = rng.gen_range(0..usize::MAX);
            cards.push(pool.take(class, pick).ok_or(SeedError::NoConsistentHand)?);
        }
        Ok(cards)
    };
    let hero_cards = realize(&hero_base, hero_pick, rng)?;
    let villain_cards = realize(&villain_base, villain_pick, rng)?;

    // Deal and replay.
    let hands = if spec.human_player == 0 {
        Hands {
            zero: hero_cards.clone(),
            one: villain_cards.clone(),
        }
    } else {
        Hands {
            zero: villain_cards.clone(),
            one: hero_cards.clone(),
        }
    };
    let mut game = Match::new(spec.dealer, spec.score.clone())
        .map_err(|e| SeedError::Invalid(e.to_string()))?;
    game.start_hand(turnup, hands)
        .map_err(|e| SeedError::Invalid(e.to_string()))?;

    let mut log: Vec<ObservedAction> = Vec::with_capacity(spec.history.len());
    for entry in &spec.history {
        if game.current_player() != Some(entry.seat) {
            return Err(SeedError::History(format!(
                "seat {} acts out of turn at step {}",
                entry.seat,
                log.len()
            )));
        }
        let state = game.export_state();
        let hand_state = state
            .current_hand
            .as_ref()
            .ok_or_else(|| SeedError::History("hand ended before the line did".into()))?;
        let (action, card) = match &entry.kind {
            SeededActionKind::PlayFaceUp { class } | SeededActionKind::PlayFaceDown { class } => {
                let target = class_card(*class)?;
                let card = hand_state
                    .state
                    .hands
                    .player(entry.seat)
                    .iter()
                    .find(|card| abstract_card(card, &hand_state.state.turnup) == target)
                    .cloned()
                    .ok_or_else(|| {
                        SeedError::History(format!(
                            "seat {} does not hold a card of class {class} at step {}",
                            entry.seat,
                            log.len()
                        ))
                    })?;
                let action = if matches!(entry.kind, SeededActionKind::PlayFaceUp { .. }) {
                    Action::PlayFaceUp {
                        card_id: card.id.clone(),
                    }
                } else {
                    Action::PlayFaceDown {
                        card_id: card.id.clone(),
                    }
                };
                (action, Some(card))
            }
            SeededActionKind::Raise { to } => (Action::Raise { to: *to }, None),
            SeededActionKind::AcceptRaise => (Action::AcceptRaise, None),
            SeededActionKind::Fold => (Action::Fold, None),
            SeededActionKind::AcceptEleven => (Action::AcceptEleven, None),
            SeededActionKind::FoldEleven => (Action::FoldEleven, None),
        };
        game.apply_action(entry.seat, &action)
            .map_err(|e| SeedError::History(format!("step {}: {e}", log.len())))?;
        log.push(ObservedAction {
            player: entry.seat,
            action,
            card,
        });
    }

    let state = game.export_state();
    let still_live = state.winner.is_none()
        && state
            .current_hand
            .as_ref()
            .is_some_and(|hand| hand.hand_winner.is_none());
    if !still_live {
        return Err(SeedError::History(
            "the line already ends the hand; there is nothing left to play".into(),
        ));
    }

    Ok(SeededHand {
        state,
        log,
        sampling: mode,
    })
}

#[cfg(test)]
mod tests;
