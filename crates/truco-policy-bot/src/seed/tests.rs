//! Seeded-hand tests: exact pinned deals, prior sampling, posterior
//! conditioning against a synthetic artifact, and replay validation.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use smallvec::SmallVec;

use truco_engine::{Player, Rank, Score, Suit, Turnup};
use truco_policy_format::abstraction::{abstract_card, turnup_class, AbstractCard, AbstractHand};
use truco_policy_format::file::write_bot_policy;
use truco_policy_format::info_set::{AbstractAction, InfoSet};

use super::{
    build_seeded_hand, SeedError, SeedSpec, SeededActionKind, SeededHistoryAction, VillainSampling,
};
use crate::PolicyStore;

fn ten_ten_spec(history: Vec<SeededHistoryAction>) -> SeedSpec {
    SeedSpec {
        score: Score { zero: 10, one: 10 },
        dealer: 0,
        vira_rank: Rank::Jack,
        human_player: 0,
        hero_hand: Some(vec![2, 5, 12]),
        villain_hand: None,
        history,
    }
}

fn act(seat: Player, kind: SeededActionKind) -> SeededHistoryAction {
    SeededHistoryAction { seat, kind }
}

/// The villain's full dealt hand (remaining + played), in abstract classes.
fn villain_classes(state: &truco_engine::MatchState, villain: Player) -> Vec<u8> {
    let hand = state.current_hand.as_ref().expect("hand in progress");
    let turnup = &hand.state.turnup;
    let mut classes: Vec<u8> = hand
        .state
        .hands
        .player(villain)
        .iter()
        .map(|card| abstract_card(card, turnup).type_index() as u8)
        .collect();
    for round in hand
        .state
        .completed_rounds
        .iter()
        .map(|round| &round.plays)
        .chain(std::iter::once(&hand.state.current_round.plays))
    {
        for play in round {
            if play.player == villain {
                classes.push(abstract_card(&play.card, turnup).type_index() as u8);
            }
        }
    }
    classes.sort_unstable();
    classes
}

#[test]
fn pinned_villain_gets_exactly_that_hand() {
    let mut spec = ten_ten_spec(vec![]);
    spec.villain_hand = Some(vec![1, 5, 8]);
    let mut rng = StdRng::seed_from_u64(11);
    let seeded = build_seeded_hand(&spec, None, &mut rng).expect("seeded hand");
    assert_eq!(seeded.sampling, VillainSampling::Pinned);
    assert!(seeded.log.is_empty());
    assert_eq!(villain_classes(&seeded.state, 1), vec![1, 5, 8]);
    // Hero got exactly the requested classes too.
    let hand = seeded.state.current_hand.as_ref().unwrap();
    let mut hero: Vec<u8> = hand
        .state
        .hands
        .player(0)
        .iter()
        .map(|card| abstract_card(card, &hand.state.turnup).type_index() as u8)
        .collect();
    hero.sort_unstable();
    assert_eq!(hero, vec![2, 5, 12]);
    // The vira the lab was studying is preserved for display.
    assert_eq!(hand.state.turnup.rank, Rank::Jack);
}

#[test]
fn unpinned_villain_without_artifacts_samples_the_prior() {
    let spec = ten_ten_spec(vec![]);
    let mut rng = StdRng::seed_from_u64(7);
    let seeded = build_seeded_hand(&spec, None, &mut rng).expect("seeded hand");
    assert_eq!(seeded.sampling, VillainSampling::Prior);
    let classes = villain_classes(&seeded.state, 1);
    assert_eq!(classes.len(), 3);
    // The hero holds the only copy of manilha 12 (clubs); the villain cannot.
    assert!(!classes.contains(&12));
}

#[test]
fn draft_less_seed_samples_both_hands_around_the_committed_plays() {
    // The user's exact report: "10x10 v4 : 3 q" viewed with no drafted hand.
    // Under a 4 vira: '3' is class 8, 'q' is class 4. Mão (seat 1) led the 3,
    // pé answered the q; the human plays mão (the seat to act next).
    let spec = SeedSpec {
        score: Score { zero: 10, one: 10 },
        dealer: 0,
        vira_rank: Rank::Four,
        human_player: 1,
        hero_hand: None,
        villain_hand: None,
        history: vec![
            act(1, SeededActionKind::PlayFaceUp { class: 8 }),
            act(0, SeededActionKind::PlayFaceUp { class: 4 }),
        ],
    };
    let mut rng = StdRng::seed_from_u64(21);
    let seeded = build_seeded_hand(&spec, None, &mut rng).expect("seeded hand");
    assert_eq!(seeded.sampling, VillainSampling::Prior);
    assert_eq!(seeded.log.len(), 2);

    // Both seats hold full 3-card hands containing their committed plays.
    let hero = villain_classes(&seeded.state, 1);
    let villain = villain_classes(&seeded.state, 0);
    assert_eq!(hero.len(), 3);
    assert_eq!(villain.len(), 3);
    assert!(hero.contains(&8), "the hero must hold the 3 they played");
    assert!(
        villain.contains(&4),
        "the villain must hold the q they played"
    );

    // Trick 1 resolved (3 beats q): mão won and leads trick 2.
    let hand = seeded.state.current_hand.as_ref().unwrap();
    assert_eq!(hand.state.completed_rounds.len(), 1);
    assert_eq!(hand.state.completed_rounds[0].winner, Some(1));
    assert_eq!(
        seeded
            .state
            .current_hand
            .as_ref()
            .unwrap()
            .state
            .next_player,
        Some(1)
    );
}

#[test]
fn deck_conservation_with_duplicate_classes() {
    // Hero takes three of the four Queens (class 4 under a Jack vira);
    // villain pins the fourth plus two more. A fifth copy must refuse.
    let mut spec = ten_ten_spec(vec![]);
    spec.hero_hand = Some(vec![4, 4, 4]);
    spec.villain_hand = Some(vec![4, 1, 2]);
    let mut rng = StdRng::seed_from_u64(3);
    assert!(build_seeded_hand(&spec, None, &mut rng).is_ok());

    spec.villain_hand = Some(vec![4, 4, 2]);
    let mut rng = StdRng::seed_from_u64(3);
    assert!(matches!(
        build_seeded_hand(&spec, None, &mut rng),
        Err(SeedError::NoConsistentHand)
    ));

    // The vira's own plain level has only 3 copies (the turnup is out):
    // a fourth is impossible even split across both hands.
    spec.hero_hand = Some(vec![5, 5, 5]);
    spec.villain_hand = Some(vec![5, 1, 2]);
    let mut rng = StdRng::seed_from_u64(3);
    assert!(matches!(
        build_seeded_hand(&spec, None, &mut rng),
        Err(SeedError::NoConsistentHand)
    ));
}

/// Build an artifact whose equilibrium plays the observed opening class iff
/// the candidate hand also contains `marker`; the posterior must then keep
/// only marker hands.
fn posterior_store(spec: &SeedSpec, observed_class: u8, marker: u8, tag: &str) -> Arc<PolicyStore> {
    let tc = turnup_class(&Turnup {
        rank: spec.vira_rank,
        suit: Suit::Hearts,
    });
    let villain = 1 - spec.human_player;
    // Enumerate candidate villain hands the same way the sampler does:
    // the observed play is committed, two unknowns from what remains.
    let mut avail = [0usize; 13];
    for (class, slot) in avail.iter_mut().enumerate() {
        *slot = tc.availability()[class] as usize;
    }
    avail[tc_class_usage(spec, observed_class)] -= 1; // committed play
    for &class in spec.hero_hand.as_deref().unwrap_or(&[]) {
        avail[class as usize] -= 1;
    }

    type PolicyRow = (u64, SmallVec<[u8; 8]>, SmallVec<[f32; 8]>);
    let mut entries: Vec<PolicyRow> = Vec::new();
    super::for_each_completion(&avail, 2, &mut |completion, _| {
        let mut classes = vec![observed_class];
        classes.extend_from_slice(completion);
        let mut hand: AbstractHand = classes
            .iter()
            .map(|&class| AbstractCard::from_type_index(class as usize))
            .collect();
        hand.sort();
        let info_set = InfoSet::new(villain, villain == spec.dealer, tc, hand);
        let plays_observed = classes.contains(&marker);
        let observed =
            AbstractAction::PlayFaceUp(AbstractCard::from_type_index(observed_class as usize));
        // In-equilibrium hands open with the observed class; others open with
        // some other action, so the observed line has zero mass for them.
        let action = if plays_observed {
            observed
        } else {
            AbstractAction::Fold
        };
        entries.push((
            info_set.key().0,
            SmallVec::from_slice(&[action.to_u8()]),
            SmallVec::from_slice(&[1.0f32]),
        ));
    });

    let dir = std::env::temp_dir().join(format!("truco-seed-test-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create dir");
    write_bot_policy(&dir.join("s10x10-tc5-d0.tpb"), entries.into_iter()).expect("write");
    std::fs::write(
        dir.join("manifest.json"),
        format!(
            "{{\"format\":\"truco-policy-bot/v1\",\"profiles\":[{{\"score\":[10,10],\"tc\":{},\"dealer\":0,\"file\":\"s10x10-tc5-d0.tpb\"}}]}}",
            tc.blocked_plain_level
        ),
    )
    .expect("manifest");
    Arc::new(PolicyStore::load(&dir).expect("load"))
}

fn tc_class_usage(_spec: &SeedSpec, class: u8) -> usize {
    class as usize
}

#[test]
fn posterior_conditions_on_the_equilibrium_line() {
    // Villain (mão, seat 1) opened with class 3 face up.
    let history = vec![act(1, SeededActionKind::PlayFaceUp { class: 3 })];
    let spec = ten_ten_spec(history);
    let marker = 8u8;
    let store = posterior_store(&spec, 3, marker, "posterior");

    for seed in 0..24u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let seeded = build_seeded_hand(&spec, Some(&store), &mut rng).expect("seeded hand");
        assert_eq!(seeded.sampling, VillainSampling::Posterior);
        let classes = villain_classes(&seeded.state, 1);
        assert!(
            classes.contains(&marker),
            "posterior sampled a hand {classes:?} outside the conditioned support"
        );
        assert!(classes.contains(&3), "the committed play must be in hand");
        // The replayed log carries the villain's opening play.
        assert_eq!(seeded.log.len(), 1);
    }
}

#[test]
fn off_equilibrium_line_degrades_to_prior_and_says_so() {
    // marker class 12 is the hero's manilha: no candidate hand can contain it,
    // so every candidate's likelihood is zero and sampling falls back.
    let history = vec![act(1, SeededActionKind::PlayFaceUp { class: 3 })];
    let spec = ten_ten_spec(history);
    let store = posterior_store(&spec, 3, 12, "offeq");
    let mut rng = StdRng::seed_from_u64(5);
    let seeded = build_seeded_hand(&spec, Some(&store), &mut rng).expect("seeded hand");
    assert_eq!(seeded.sampling, VillainSampling::Prior);
}

#[test]
fn missing_artifacts_for_the_spot_degrade_to_prior() {
    // Store covers 10x10 tc5 only; an 11x10 seed finds no profile.
    let history = vec![act(1, SeededActionKind::PlayFaceUp { class: 3 })];
    let base = ten_ten_spec(history.clone());
    let store = posterior_store(&base, 3, 8, "uncovered");
    let mut spec = ten_ten_spec(history);
    spec.score = Score { zero: 11, one: 10 };
    // At 11x10 with dealer 0 the mão-de-onze decision comes first.
    spec.history
        .insert(0, act(0, SeededActionKind::AcceptEleven));
    let mut rng = StdRng::seed_from_u64(9);
    let seeded = build_seeded_hand(&spec, Some(&store), &mut rng).expect("seeded hand");
    assert_eq!(seeded.sampling, VillainSampling::Prior);
    assert_eq!(seeded.log.len(), 2);
}

#[test]
fn replays_raises_and_face_down_plays_with_a_faithful_log() {
    // mão (seat 1) opens class 3; pé (seat 0, hero) raises; mão accepts;
    // hero wins trick 1 with class 5, then leads trick 2 face down (hiding is
    // illegal in trick 1, so the hide exercises the round-2 path).
    let history = vec![
        act(1, SeededActionKind::PlayFaceUp { class: 3 }),
        act(0, SeededActionKind::Raise { to: 3 }),
        act(1, SeededActionKind::AcceptRaise),
        act(0, SeededActionKind::PlayFaceUp { class: 5 }),
        act(0, SeededActionKind::PlayFaceDown { class: 2 }),
    ];
    let spec = ten_ten_spec(history);
    let mut rng = StdRng::seed_from_u64(13);
    let seeded = build_seeded_hand(&spec, None, &mut rng).expect("seeded hand");
    assert_eq!(seeded.log.len(), 5);
    // The log records the resolved raise the exported state forgets.
    assert!(matches!(
        seeded.log[1].action,
        truco_engine::Action::Raise { to: 3 }
    ));
    assert!(seeded.log[3].card.is_some());
    let hand = seeded.state.current_hand.as_ref().unwrap();
    assert_eq!(hand.state.hand_value, 3);
}

#[test]
fn invalid_lines_refuse_cleanly() {
    // Out of turn: pé cannot lead the first trick.
    let spec = ten_ten_spec(vec![act(0, SeededActionKind::PlayFaceUp { class: 2 })]);
    let mut rng = StdRng::seed_from_u64(1);
    assert!(matches!(
        build_seeded_hand(&spec, None, &mut rng),
        Err(SeedError::History(_))
    ));

    // Hero plays a card outside the declared hero hand.
    let spec = ten_ten_spec(vec![
        act(1, SeededActionKind::PlayFaceUp { class: 3 }),
        act(0, SeededActionKind::PlayFaceUp { class: 7 }),
    ]);
    let mut rng = StdRng::seed_from_u64(1);
    assert!(matches!(
        build_seeded_hand(&spec, None, &mut rng),
        Err(SeedError::Invalid(_))
    ));

    // A hand-ending line has nothing left to play.
    let spec = ten_ten_spec(vec![
        act(1, SeededActionKind::PlayFaceUp { class: 3 }),
        act(0, SeededActionKind::Raise { to: 3 }),
        act(1, SeededActionKind::Fold),
    ]);
    let mut rng = StdRng::seed_from_u64(1);
    assert!(matches!(
        build_seeded_hand(&spec, None, &mut rng),
        Err(SeedError::History(_))
    ));
}
