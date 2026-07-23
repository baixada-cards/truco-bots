use truco_bot_core::{
    choose_action_for_match, choose_decision_for_match, turn_for_match, Bot, BotProfile, BotTurn,
    HeuristicBot, SimpleTrainerBot, UniformRandomBot,
};
use truco_engine::{
    state::Visibility, Action, Card, CompletedRound, CurrentRound, EngineState, GameState, Hands,
    Match, MatchState, PendingRaise, PlayedCard, Rank, Score, Suit, Turnup,
};

fn sample_turnup() -> Turnup {
    Turnup {
        rank: Rank::Ace,
        suit: Suit::Spades,
    }
}

fn sample_hands() -> Hands {
    Hands {
        zero: smallvec::smallvec![
            Card {
                id: "p0c0".into(),
                rank: Rank::Seven,
                suit: Suit::Diamonds,
            },
            Card {
                id: "p0c1".into(),
                rank: Rank::Six,
                suit: Suit::Clubs,
            },
            Card {
                id: "p0c2".into(),
                rank: Rank::Four,
                suit: Suit::Hearts,
            },
        ],
        one: smallvec::smallvec![
            Card {
                id: "p1c0".into(),
                rank: Rank::Three,
                suit: Suit::Clubs,
            },
            Card {
                id: "p1c1".into(),
                rank: Rank::Five,
                suit: Suit::Spades,
            },
            Card {
                id: "p1c2".into(),
                rank: Rank::Four,
                suit: Suit::Diamonds,
            },
        ],
    }
}

fn opening_turn(hand: Vec<Card>, opponent: Vec<Card>) -> BotTurn {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: hand.into(),
            one: opponent.into(),
        },
    )
    .expect("hand should start");
    turn_for_match(&game, 0).expect("bot turn should build")
}

fn third_round_raise_response_turn(lone_card: Card) -> BotTurn {
    let lone_card_copy = lone_card.clone();
    let mut turn = opening_turn(
        vec![
            Card {
                id: "unused0".into(),
                rank: Rank::Seven,
                suit: Suit::Diamonds,
            },
            Card {
                id: "unused1".into(),
                rank: Rank::Six,
                suit: Suit::Hearts,
            },
            lone_card,
        ],
        vec![
            Card {
                id: "opp0".into(),
                rank: Rank::Five,
                suit: Suit::Clubs,
            },
            Card {
                id: "opp1".into(),
                rank: Rank::King,
                suit: Suit::Hearts,
            },
            Card {
                id: "opp2".into(),
                rank: Rank::Six,
                suit: Suit::Spades,
            },
        ],
    );

    let hand_view = turn.view.hand.as_mut().expect("player hand should exist");
    hand_view.hand = vec![lone_card_copy];
    hand_view.public_state.pending_raise = Some(PendingRaise {
        raised_by: 1,
        to: 6,
        previous_value: 3,
    });
    turn.legal_actions = vec![Action::AcceptRaise, Action::Fold];
    turn
}

fn revealed_last_card_raise_match() -> Match {
    Match::from_state(MatchState {
        next_dealer: 0,
        score: Score { zero: 0, one: 0 },
        winner: None,
        current_hand: Some(EngineState {
            state: GameState {
                dealer: 0,
                next_player: Some(1),
                score: Score { zero: 0, one: 0 },
                hand_value: 3,
                turnup: sample_turnup(),
                hands: Hands {
                    zero: smallvec::smallvec![Card {
                        id: "hero-last".into(),
                        rank: Rank::King,
                        suit: Suit::Spades,
                    }],
                    one: smallvec::smallvec![],
                },
                completed_rounds: smallvec::smallvec![
                    CompletedRound {
                        leader: 1,
                        winner: Some(0),
                        plays: smallvec::smallvec![],
                    },
                    CompletedRound {
                        leader: 0,
                        winner: Some(1),
                        plays: smallvec::smallvec![],
                    },
                ],
                current_round: CurrentRound {
                    leader: 1,
                    plays: smallvec::smallvec![PlayedCard {
                        player: 1,
                        visibility: Visibility::Up,
                        card: Card {
                            id: "villain-shown".into(),
                            rank: Rank::Ace,
                            suit: Suit::Hearts,
                        },
                    }],
                },
                last_raised_by: Some(0),
                pending_raise: Some(PendingRaise {
                    raised_by: 0,
                    to: 6,
                    previous_value: 3,
                }),
                pending_decision: None,
            },
            hand_winner: None,
            match_winner: None,
        }),
    })
    .expect("revealed last-card raise state should load")
}

fn bot_mao_de_onze_match(bot_hand: Vec<Card>) -> Match {
    let mut game = Match::new(0, Score { zero: 4, one: 11 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: smallvec::smallvec![
                Card {
                    id: "opp0".into(),
                    rank: Rank::Four,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "opp1".into(),
                    rank: Rank::Five,
                    suit: Suit::Spades,
                },
                Card {
                    id: "opp2".into(),
                    rank: Rank::Six,
                    suit: Suit::Hearts,
                },
            ],
            one: bot_hand.into(),
        },
    )
    .expect("hand should start");
    game
}

#[test]
fn turn_for_match_uses_player_view_and_legal_actions() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(sample_turnup(), sample_hands())
        .expect("hand should start");

    let engine_actions = game
        .legal_actions_for_current_player()
        .expect("engine legal actions should compute");
    let strategic_actions = game
        .strategic_legal_actions_for_current_player()
        .expect("strategic legal actions should compute");
    let turn = turn_for_match(&game, 0).expect("bot turn should build");

    assert_eq!(turn.player, 0);
    assert_eq!(turn.turnup, Some(sample_turnup()));
    assert_eq!(turn.view.player, 0);
    assert_eq!(
        turn.view
            .hand
            .as_ref()
            .expect("player hand should be present")
            .hand
            .len(),
        3
    );
    assert!(engine_actions.contains(&Action::ConcedeHand));
    assert_eq!(turn.legal_actions, strategic_actions);
    assert!(!turn.legal_actions.contains(&Action::ConcedeHand));
}

#[test]
fn uniform_random_bot_always_returns_a_legal_action() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(sample_turnup(), sample_hands())
        .expect("hand should start");

    let legal_actions = game
        .legal_actions_for_current_player()
        .expect("legal actions should compute");
    let mut bot = UniformRandomBot::from_seed(7);

    for _ in 0..32 {
        let action = choose_action_for_match(&mut bot, &game, 0).expect("bot should act");
        assert!(legal_actions.contains(&action));
    }
}

#[test]
fn uniform_random_bot_can_choose_different_actions_over_time() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(sample_turnup(), sample_hands())
        .expect("hand should start");

    let mut bot = UniformRandomBot::from_seed(42);
    let mut saw_raise = false;
    let mut saw_play = false;

    for _ in 0..64 {
        let action = choose_action_for_match(&mut bot, &game, 0).expect("bot should act");
        match action {
            Action::Raise { .. } => saw_raise = true,
            Action::PlayFaceUp { .. } => saw_play = true,
            _ => {}
        }
    }

    assert!(saw_raise);
    assert!(saw_play);
}

#[test]
fn simple_trainer_bot_accepts_small_raise_with_strong_hand() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(sample_turnup(), sample_hands())
        .expect("hand should start");
    game.apply_action_for_current_player(&Action::Raise { to: 3 })
        .expect("raise should apply");

    let mut bot = SimpleTrainerBot::from_seed(5);
    let action = choose_action_for_match(&mut bot, &game, 1).expect("bot should act");

    assert_eq!(action, Action::AcceptRaise);
}

#[test]
fn simple_trainer_bot_plays_a_legal_card_action() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(sample_turnup(), sample_hands())
        .expect("hand should start");

    let mut bot = SimpleTrainerBot::from_seed(7);
    let action = choose_action_for_match(&mut bot, &game, 0).expect("bot should act");

    assert!(matches!(
        action,
        Action::Raise { .. } | Action::PlayFaceUp { .. } | Action::PlayFaceDown { .. }
    ));
    let legal_actions = game
        .legal_actions_for_current_player()
        .expect("legal actions should compute");
    assert!(legal_actions.contains(&action));
}

#[test]
fn heuristic_bot_plays_smallest_when_all_responses_win() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: smallvec::smallvec![
                Card {
                    id: "p0c0".into(),
                    rank: Rank::Four,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p0c1".into(),
                    rank: Rank::Five,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "p0c2".into(),
                    rank: Rank::Six,
                    suit: Suit::Spades,
                },
            ],
            one: smallvec::smallvec![
                Card {
                    id: "p1c0".into(),
                    rank: Rank::Six,
                    suit: Suit::Clubs,
                },
                Card {
                    id: "p1c1".into(),
                    rank: Rank::Seven,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "p1c2".into(),
                    rank: Rank::Ace,
                    suit: Suit::Hearts,
                },
            ],
        },
    )
    .expect("hand should start");
    game.apply_action_for_current_player(&Action::PlayFaceUp {
        card_id: "p0c0".into(),
    })
    .expect("opening play should apply");

    let mut bot = HeuristicBot::from_seed(9);
    let action = choose_action_for_match(&mut bot, &game, 1).expect("bot should act");

    assert_eq!(
        action,
        Action::PlayFaceUp {
            card_id: "p1c0".into(),
        }
    );
}

#[test]
fn heuristic_bot_does_not_lead_with_highest_manilha_when_other_cards_exist() {
    let mut turn = opening_turn(
        vec![
            Card {
                id: "p0c0".into(),
                rank: Rank::Two,
                suit: Suit::Clubs,
            },
            Card {
                id: "p0c1".into(),
                rank: Rank::Six,
                suit: Suit::Hearts,
            },
            Card {
                id: "p0c2".into(),
                rank: Rank::Four,
                suit: Suit::Diamonds,
            },
        ],
        vec![
            Card {
                id: "p1c0".into(),
                rank: Rank::Seven,
                suit: Suit::Spades,
            },
            Card {
                id: "p1c1".into(),
                rank: Rank::Five,
                suit: Suit::Diamonds,
            },
            Card {
                id: "p1c2".into(),
                rank: Rank::Four,
                suit: Suit::Hearts,
            },
        ],
    );
    turn.legal_actions.retain(|action| {
        matches!(
            action,
            Action::PlayFaceUp { .. } | Action::PlayFaceDown { .. }
        )
    });

    let mut bot = HeuristicBot::from_seed_with_profile(3, BotProfile::Balanced);
    let action = bot.choose_action(&turn).expect("bot should act");

    assert_eq!(
        action,
        Action::PlayFaceUp {
            card_id: "p0c2".into(),
        }
    );
}

#[test]
fn heuristic_bot_raises_before_exposing_highest_manilha_after_prior_round_win() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: smallvec::smallvec![
                Card {
                    id: "p0c0".into(),
                    rank: Rank::Seven,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "p0c1".into(),
                    rank: Rank::Four,
                    suit: Suit::Clubs,
                },
                Card {
                    id: "p0c2".into(),
                    rank: Rank::Two,
                    suit: Suit::Clubs,
                },
            ],
            one: smallvec::smallvec![
                Card {
                    id: "p1c0".into(),
                    rank: Rank::Four,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p1c1".into(),
                    rank: Rank::Seven,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p1c2".into(),
                    rank: Rank::Five,
                    suit: Suit::Spades,
                },
            ],
        },
    )
    .expect("hand should start");

    for action in [
        Action::PlayFaceUp {
            card_id: "p0c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p0c1".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c1".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c2".into(),
        },
    ] {
        game.apply_action_for_current_player(&action)
            .expect("scripted action should apply");
    }

    let mut bot = HeuristicBot::from_seed_with_profile(17, BotProfile::Balanced);
    let action = choose_action_for_match(&mut bot, &game, 0).expect("bot should act");

    assert_eq!(action, Action::Raise { to: 3 });
}

#[test]
fn heuristic_bot_does_not_reraise_when_current_stake_already_wins_the_match() {
    let mut game = Match::new(1, Score { zero: 9, one: 0 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: smallvec::smallvec![
                Card {
                    id: "p0c0".into(),
                    rank: Rank::Two,
                    suit: Suit::Clubs,
                },
                Card {
                    id: "p0c1".into(),
                    rank: Rank::Two,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p0c2".into(),
                    rank: Rank::Three,
                    suit: Suit::Clubs,
                },
            ],
            one: smallvec::smallvec![
                Card {
                    id: "p1c0".into(),
                    rank: Rank::Four,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "p1c1".into(),
                    rank: Rank::Five,
                    suit: Suit::Spades,
                },
                Card {
                    id: "p1c2".into(),
                    rank: Rank::Four,
                    suit: Suit::Hearts,
                },
            ],
        },
    )
    .expect("hand should start");

    for action in [Action::Raise { to: 3 }, Action::AcceptRaise] {
        game.apply_action_for_current_player(&action)
            .expect("scripted action should apply");
    }

    for seed in 0..64 {
        let mut bot = HeuristicBot::from_seed_with_profile(seed, BotProfile::Aggressive);
        let action = choose_action_for_match(&mut bot, &game, 0).expect("bot should act");

        assert!(matches!(action, Action::PlayFaceUp { .. }));
    }
}

#[test]
fn heuristic_bot_accepts_more_often_with_a_stronger_last_card() {
    let strong_turn = third_round_raise_response_turn(Card {
        id: "hero".into(),
        rank: Rank::Two,
        suit: Suit::Clubs,
    });
    let weak_turn = third_round_raise_response_turn(Card {
        id: "hero".into(),
        rank: Rank::Four,
        suit: Suit::Diamonds,
    });

    let mut strong_accepts = 0;
    let mut weak_accepts = 0;

    for seed in 0..64 {
        let mut strong_bot = HeuristicBot::from_seed_with_profile(seed, BotProfile::Balanced);
        let mut weak_bot = HeuristicBot::from_seed_with_profile(seed, BotProfile::Balanced);
        let strong_action = strong_bot
            .choose_action(&strong_turn)
            .expect("strong bot should act");
        let weak_action = weak_bot
            .choose_action(&weak_turn)
            .expect("weak bot should act");

        if matches!(strong_action, Action::AcceptRaise) {
            strong_accepts += 1;
        }
        if matches!(weak_action, Action::AcceptRaise) {
            weak_accepts += 1;
        }
    }

    assert!(strong_accepts > weak_accepts);
    assert!(strong_accepts >= 50);
    assert!(weak_accepts <= 12);
}

#[test]
fn heuristic_bot_balanced_profile_bluffs_sometimes_with_a_weak_opening_hand() {
    let turn = opening_turn(
        vec![
            Card {
                id: "hero0".into(),
                rank: Rank::Seven,
                suit: Suit::Diamonds,
            },
            Card {
                id: "hero1".into(),
                rank: Rank::Six,
                suit: Suit::Hearts,
            },
            Card {
                id: "hero2".into(),
                rank: Rank::Four,
                suit: Suit::Diamonds,
            },
        ],
        vec![
            Card {
                id: "opp0".into(),
                rank: Rank::King,
                suit: Suit::Clubs,
            },
            Card {
                id: "opp1".into(),
                rank: Rank::Five,
                suit: Suit::Spades,
            },
            Card {
                id: "opp2".into(),
                rank: Rank::Four,
                suit: Suit::Hearts,
            },
        ],
    );

    let mut raises = 0;
    for seed in 0..256 {
        let mut bot = HeuristicBot::from_seed_with_profile(seed, BotProfile::Balanced);
        let action = bot.choose_action(&turn).expect("bot should act");
        if matches!(action, Action::Raise { to: 3 }) {
            raises += 1;
        }
    }

    assert!(raises > 0);
}

#[test]
fn heuristic_bot_balanced_profile_accepts_strong_mao_de_onze_more_often_than_not() {
    let game = bot_mao_de_onze_match(vec![
        Card {
            id: "bot0".into(),
            rank: Rank::Two,
            suit: Suit::Diamonds,
        },
        Card {
            id: "bot1".into(),
            rank: Rank::Three,
            suit: Suit::Hearts,
        },
        Card {
            id: "bot2".into(),
            rank: Rank::Seven,
            suit: Suit::Clubs,
        },
    ]);
    let legal_actions = game
        .legal_actions(1)
        .expect("legal actions should compute for the eleven response");
    assert!(legal_actions.contains(&Action::AcceptEleven));
    assert!(legal_actions.contains(&Action::FoldEleven));

    let mut accepts = 0;
    let mut folds = 0;
    for seed in 0..128 {
        let mut bot = HeuristicBot::from_seed_with_profile(seed, BotProfile::Balanced);
        let action = choose_action_for_match(&mut bot, &game, 1).expect("bot should act");
        match action {
            Action::AcceptEleven => accepts += 1,
            Action::FoldEleven => folds += 1,
            _ => panic!("mão de onze should only resolve with accept or fold"),
        }
    }

    assert!(accepts > folds);
}

#[test]
fn heuristic_bot_does_not_auto_fold_revealed_last_card_raise_pressure() {
    let game = revealed_last_card_raise_match();
    let legal_actions = game
        .legal_actions(1)
        .expect("legal actions should compute for the raise response");
    assert!(legal_actions.contains(&Action::AcceptRaise));
    assert!(legal_actions.contains(&Action::Fold));

    let mut accepts = 0;
    for seed in 0..64 {
        let mut bot = HeuristicBot::from_seed_with_profile(seed, BotProfile::Balanced);
        let action = choose_action_for_match(&mut bot, &game, 1).expect("bot should act");
        if matches!(action, Action::AcceptRaise) {
            accepts += 1;
        }
    }

    assert!(accepts > 0);
}

#[test]
fn heuristic_bot_reraises_when_reraise_preserves_a_forced_hand_win() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: smallvec::smallvec![
                Card {
                    id: "p0c0".into(),
                    rank: Rank::Seven,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "p0c1".into(),
                    rank: Rank::Four,
                    suit: Suit::Clubs,
                },
                Card {
                    id: "p0c2".into(),
                    rank: Rank::Two,
                    suit: Suit::Clubs,
                },
            ],
            one: smallvec::smallvec![
                Card {
                    id: "p1c0".into(),
                    rank: Rank::Four,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p1c1".into(),
                    rank: Rank::Seven,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p1c2".into(),
                    rank: Rank::Five,
                    suit: Suit::Spades,
                },
            ],
        },
    )
    .expect("hand should start");

    for action in [
        Action::Raise { to: 3 },
        Action::AcceptRaise,
        Action::PlayFaceUp {
            card_id: "p0c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p0c1".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c1".into(),
        },
        Action::Raise { to: 6 },
    ] {
        game.apply_action_for_current_player(&action)
            .expect("scripted action should apply");
    }

    let mut bot = HeuristicBot::from_seed_with_profile(19, BotProfile::Balanced);
    let decision = choose_decision_for_match(&mut bot, &game, 0).expect("bot should act");

    assert_eq!(decision.action, Action::Raise { to: 9 });
    assert!(decision
        .plan
        .reasoning
        .as_deref()
        .expect("heuristic reasoning should be present")
        .contains("forced hand win"));
}

#[test]
fn heuristic_bot_does_not_reraise_past_match_point_when_accepting_is_already_enough() {
    let mut game = Match::new(1, Score { zero: 6, one: 0 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: smallvec::smallvec![
                Card {
                    id: "p0c0".into(),
                    rank: Rank::Seven,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "p0c1".into(),
                    rank: Rank::Four,
                    suit: Suit::Clubs,
                },
                Card {
                    id: "p0c2".into(),
                    rank: Rank::Two,
                    suit: Suit::Clubs,
                },
            ],
            one: smallvec::smallvec![
                Card {
                    id: "p1c0".into(),
                    rank: Rank::Four,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p1c1".into(),
                    rank: Rank::Seven,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p1c2".into(),
                    rank: Rank::Five,
                    suit: Suit::Spades,
                },
            ],
        },
    )
    .expect("hand should start");

    for action in [
        Action::Raise { to: 3 },
        Action::AcceptRaise,
        Action::PlayFaceUp {
            card_id: "p0c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p0c1".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c1".into(),
        },
        Action::Raise { to: 6 },
    ] {
        game.apply_action_for_current_player(&action)
            .expect("scripted action should apply");
    }

    for seed in 0..32 {
        let mut bot = HeuristicBot::from_seed_with_profile(seed, BotProfile::Balanced);
        let decision = choose_decision_for_match(&mut bot, &game, 0).expect("bot should act");

        assert!(matches!(
            decision.action,
            Action::AcceptRaise | Action::Fold
        ));
    }
}

#[test]
fn heuristic_bot_hides_only_when_face_up_cannot_win_the_current_round() {
    let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
    game.start_hand(
        sample_turnup(),
        Hands {
            zero: smallvec::smallvec![
                Card {
                    id: "p0c0".into(),
                    rank: Rank::Four,
                    suit: Suit::Hearts,
                },
                Card {
                    id: "p0c1".into(),
                    rank: Rank::Three,
                    suit: Suit::Clubs,
                },
                Card {
                    id: "p0c2".into(),
                    rank: Rank::Seven,
                    suit: Suit::Diamonds,
                },
            ],
            one: smallvec::smallvec![
                Card {
                    id: "p1c0".into(),
                    rank: Rank::Four,
                    suit: Suit::Diamonds,
                },
                Card {
                    id: "p1c1".into(),
                    rank: Rank::Five,
                    suit: Suit::Spades,
                },
                Card {
                    id: "p1c2".into(),
                    rank: Rank::Six,
                    suit: Suit::Hearts,
                },
            ],
        },
    )
    .expect("hand should start");

    for action in [
        Action::PlayFaceUp {
            card_id: "p0c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p1c0".into(),
        },
        Action::PlayFaceUp {
            card_id: "p0c1".into(),
        },
    ] {
        game.apply_action_for_current_player(&action)
            .expect("scripted action should apply");
    }

    let mut bot = HeuristicBot::from_seed_with_profile(21, BotProfile::Balanced);
    let action = choose_action_for_match(&mut bot, &game, 1).expect("bot should act");

    assert_eq!(
        action,
        Action::PlayFaceDown {
            card_id: "p1c1".into(),
        }
    );
}
