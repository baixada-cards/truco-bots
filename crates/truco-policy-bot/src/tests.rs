//! Runtime contract tests.
//!
//! Solve-side traversal parity is tested where policies are produced and in
//! the cross-repository integration suite. This crate stays solver-independent
//! and verifies that a contract-generated key and TPB1 row drive a real match.

use std::path::PathBuf;
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use truco_bot_core::turn_for_match;
use truco_engine::{Action, Card, Hands, Match, Rank, Score, Suit, Turnup};
use truco_policy_format::abstraction::{abstract_card, turnup_class, AbstractCard, AbstractHand};
use truco_policy_format::file::write_bot_policy;
use truco_policy_format::info_set::{AbstractAction, InfoSet};

use super::{PolicyStore, SolverPolicyBot};

fn sample_turnup() -> Turnup {
    Turnup {
        rank: Rank::Ace,
        suit: Suit::Spades,
    }
}

fn card(id: &str, rank: Rank, suit: Suit) -> Card {
    Card {
        id: id.into(),
        rank,
        suit,
    }
}

fn sample_hands() -> Hands {
    Hands {
        zero: smallvec![
            card("p0-seven", Rank::Seven, Suit::Diamonds),
            card("p0-six", Rank::Six, Suit::Clubs),
            card("p0-four", Rank::Four, Suit::Hearts),
        ],
        one: smallvec![
            card("p1-three", Rank::Three, Suit::Clubs),
            card("p1-five", Rank::Five, Suit::Spades),
            card("p1-four", Rank::Four, Suit::Diamonds),
        ],
    }
}

fn abstract_hand(cards: &[Card], turnup: &Turnup) -> AbstractHand {
    let mut hand: AbstractHand = cards
        .iter()
        .map(|card| abstract_card(card, turnup))
        .collect();
    hand.sort();
    hand
}

fn write_store(
    tag: &str,
    score: (u8, u8),
    dealer: u8,
    turnup: &Turnup,
    player: u8,
    hand: &[Card],
    chosen: AbstractCard,
) -> Arc<PolicyStore> {
    let tc = turnup_class(turnup);
    let key = InfoSet::new(player, player == dealer, tc, abstract_hand(hand, turnup)).key();
    let dir = std::env::temp_dir().join(format!(
        "truco-policy-bot-runtime-{tag}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create policy directory");
    let file = format!(
        "s{}x{}-tc{}-d{dealer}.tpb",
        score.0, score.1, tc.blocked_plain_level
    );
    let actions: SmallVec<[u8; 8]> = smallvec![AbstractAction::PlayFaceUp(chosen).to_u8()];
    let probabilities: SmallVec<[f32; 8]> = smallvec![1.0];
    write_bot_policy(
        &dir.join(&file),
        std::iter::once((key.0, actions, probabilities)),
    )
    .expect("write policy");
    std::fs::write(
        dir.join("manifest.json"),
        format!(
            "{{\"format\":\"truco-policy-bot/v1\",\"profiles\":[{{\"score\":[{},{}],\"tc\":{},\"dealer\":{dealer},\"file\":\"{file}\"}}]}}",
            score.0, score.1, tc.blocked_plain_level
        ),
    )
    .expect("write manifest");
    Arc::new(PolicyStore::load(&dir).expect("load policy store"))
}

#[test]
fn contract_generated_opening_key_drives_live_action() {
    let turnup = sample_turnup();
    let hands = sample_hands();
    let chosen = abstract_card(&hands.one[0], &turnup);
    let store = write_store("direct", (10, 10), 0, &turnup, 1, &hands.one, chosen);

    let mut game = Match::new(0, Score { zero: 10, one: 10 }).expect("match");
    game.start_hand(turnup, hands).expect("start hand");
    let turn = turn_for_match(&game, 1).expect("opening turn");
    let mut bot = SolverPolicyBot::new(store, Some(7));
    let decision = bot.choose_decision(&turn, &[]).expect("policy decision");

    assert_eq!(
        decision.action,
        Action::PlayFaceUp {
            card_id: "p1-three".into()
        }
    );
    assert!(decision
        .plan
        .reasoning
        .as_deref()
        .is_some_and(|reason| reason.contains("solved equilibrium")));
}

#[test]
fn contract_profile_supports_seat_transpose() {
    let turnup = sample_turnup();
    let original = sample_hands();
    let chosen = abstract_card(&original.one[0], &turnup);
    let store = write_store("transpose", (10, 10), 0, &turnup, 1, &original.one, chosen);
    assert!(store.covers((10, 10), turnup_class(&turnup).blocked_plain_level, 1));

    let hands = Hands {
        zero: original.one,
        one: original.zero,
    };
    let mut game = Match::new(1, Score { zero: 10, one: 10 }).expect("match");
    game.start_hand(turnup, hands).expect("start hand");
    let turn = turn_for_match(&game, 0).expect("opening turn");
    let mut bot = SolverPolicyBot::new(store, Some(11));
    let decision = bot.choose_decision(&turn, &[]).expect("policy decision");

    assert_eq!(
        decision.action,
        Action::PlayFaceUp {
            card_id: "p1-three".into()
        }
    );
}

#[test]
fn uncovered_spot_falls_back_to_heuristic() {
    let turnup = sample_turnup();
    let hands = sample_hands();
    let chosen = abstract_card(&hands.one[0], &turnup);
    let store = write_store("fallback", (10, 10), 0, &turnup, 1, &hands.one, chosen);

    let mut game = Match::new(0, Score { zero: 9, one: 9 }).expect("match");
    game.start_hand(turnup, hands).expect("start hand");
    let turn = turn_for_match(&game, 1).expect("opening turn");
    let mut bot = SolverPolicyBot::new(store, Some(13));
    let decision = bot.choose_decision(&turn, &[]).expect("fallback decision");

    assert!(turn.legal_actions.contains(&decision.action));
    assert!(decision
        .plan
        .reasoning
        .as_deref()
        .is_some_and(|reason| reason.starts_with("solver fallback")));
}

#[test]
fn store_load_rejects_bad_manifest() {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "truco-policy-bot-bad-manifest-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create directory");
    std::fs::write(
        dir.join("manifest.json"),
        "{\"format\":\"other/v9\",\"profiles\":[]}",
    )
    .expect("write manifest");
    assert!(PolicyStore::load(&dir).is_err());
}
