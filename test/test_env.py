from src.skyjo_env import SkyjoEnv
import pytest


def test_env_init():
    for num_players in [2, 8]:
        env = SkyjoEnv(num_players=num_players)
        assert env.num_players == num_players
        assert env.current_player == 0
        assert env.deck == []
        assert env.discard_pile == []
        assert env.players == []
        assert not env.done
        assert env.final_player is None
        assert env.remaining_final_turns is None
        assert env.pending_card is None

    with pytest.raises(ValueError):
        env = SkyjoEnv(num_players=1)
    with pytest.raises(ValueError):
        env = SkyjoEnv(num_players=9)


def test_env_reset():
    player_number = 2
    env = SkyjoEnv(num_players=2)
    env.reset()

    assert len(env.deck) == 150 - 12 * player_number - 1

    assert len(env.players) == player_number
    for player in env.players:
        assert len(player["grid"]) == 12
        assert len(player["visible"]) == 12
        assert player["return_count"] == 2
        visible_cards = [card for card, visible in zip(player["grid"], player["visible"]) if visible]
        assert player["score"] == sum(visible_cards)
        assert sum(player["visible"]) == 2

    assert len(env.discard_pile) == 1

    assert env.current_player == 0
    assert not env.done
    assert env.final_player is None
    assert env.remaining_final_turns is None
    assert env.pending_card is None


def test_env_obs():
    env = SkyjoEnv(num_players=2)
    env.reset()

    obs = env._get_obs()
    assert "players" in obs
    assert "discard" in obs
    assert "deck_count" in obs
    assert "current_player" in obs
    assert "pending_card" in obs

def test_env_remove_columns():
    # No columns to remove
    env = SkyjoEnv(num_players=2)
    env.reset()
    player = {
        "grid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "visible": [True, True, True, True, True, True, True, True, True, True, True, True]
    }
    initial_grid = player["grid"].copy()
    initial_visible = player["visible"].copy()
    env._check_and_remove_columns(player)
    assert player["grid"] == initial_grid
    assert player["visible"] == initial_visible
    assert len(env.discard_pile) == 1

    # One column to remove (all 7s)
    env = SkyjoEnv(num_players=2)
    env.reset()
    player = {
        "grid": [1, 2, 3, 7, 5, 6, 8, 7, 9, 12, 11, 7],
        "visible": [True, True, True, True, True, True, True, True, True, True, True, True]
    }
    env._check_and_remove_columns(player)
    expected_grid = [1, 2, 3, None, 5, 6, 8, None, 9, 12, 11, None]
    expected_visible = [True, True, True, False, True, True, True, False, True, True, True, False]
    assert player["grid"] == expected_grid
    assert player["visible"] == expected_visible
    assert len(env.discard_pile) == 4

    # Multiple columns to remove
    env = SkyjoEnv(num_players=2)
    env.reset()
    player = {
        "grid": [1, 1, 2, 3, 1, 1, 3, 2, 1, 1, 2, 3],
        "visible": [True, True, True, True, True, True, True, True, True, True, True, True]
    }
    env._check_and_remove_columns(player)
    expected_grid = [None, None, 2, 3, None, None, 3, 2, None, None, 2, 3]
    expected_visible = [False, False, True, True, False, False, True, True, False, False, True, True]
    assert player["grid"] == expected_grid
    assert player["visible"] == expected_visible
    assert len(env.discard_pile) == 7

    # Column with same numbers but not all visible
    env = SkyjoEnv(num_players=2)
    env.reset()
    player = {
        "grid": [1, 2, 3, 4, 1, 5, 6, 7, 1, 8, 9, 10],
        "visible": [True, True, True, True, True, True, True, True, False, True, True, True]
    }
    initial_grid = player["grid"].copy()
    initial_visible = player["visible"].copy()
    env._check_and_remove_columns(player)
    assert player["grid"] == initial_grid
    assert player["visible"] == initial_visible
    assert len(env.discard_pile) == 1

    # All columns to remove
    env = SkyjoEnv(num_players=2)
    env.reset()
    player = {
        "grid": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "visible": [True, True, True, True, True, True, True, True, True, True, True, True]
    }
    env._check_and_remove_columns(player)
    assert player["grid"] == [None, None, None, None, None, None, None, None, None, None, None, None]
    assert player["visible"] == [False, False, False, False, False, False, False, False, False, False, False, False]
    assert len(env.discard_pile) == 13


def test_draw_from_deck_and_discard():
    env = SkyjoEnv(num_players=2)
    env.reset()

    top_discard = env.discard_pile[-1]
    deck_top = env.deck[-1]

    with pytest.raises(ValueError):
        env.step({"flip": 0})

    # Draw from deck
    _, reward, done, _ = env.step({"draw": "deck"})
    assert env.pending_card == deck_top
    assert reward == 0.0
    assert not done

    # Cannot draw again without using pending card
    with pytest.raises(ValueError):
        env.step({"draw": "deck"})

    # Use the card, then draw from discard
    env.step({"flip": next(i for i, v in enumerate(env.players[0]["visible"]) if not v)})
    assert env.pending_card is None
    
    top_discard = env.discard_pile[-1]
    _, reward, done, _ = env.step({"draw": "discard"})
    assert env.pending_card == top_discard


def test_invalid_draw_source():
    env = SkyjoEnv(num_players=2)
    env.reset()
    with pytest.raises(ValueError):
        env.step({"draw": "garbage"})


def test_draw_with_empty_deck():
    env = SkyjoEnv(num_players=2)
    env.reset()

    # Empty deck
    env.deck = []
    env.discard_pile = [5, 6, 7]

    # First draw recycle the discard pile
    _, _, _, _ = env.step({"draw": "deck"})
    assert len(env.deck) == 1 
    assert env.pending_card in [5, 6, 7]


def test_replace_card():
    env = SkyjoEnv(num_players=2)
    env.reset()
    env.step({"draw": "deck"})

    replace_index = next(i for i, v in enumerate(env.players[0]["grid"]) if v is not None)

    env.step({"replace": replace_index})
    assert env.players[0]["visible"][replace_index]
    assert env.pending_card is None
    assert env.current_player == 1


def test_flip_card():
    env = SkyjoEnv(num_players=2)
    env.reset()
    env.step({"draw": "deck"})

    flip_index = next(i for i, v in enumerate(env.players[0]["visible"]) if not v)
    env.step({"flip": flip_index})
    assert env.players[0]["visible"][flip_index]
    assert env.pending_card is None


def test_invalid_replace_index():
    env = SkyjoEnv(num_players=2)
    env.reset()
    env.step({"draw": "deck"})

    # Simulate a card already removed
    env.players[0]["grid"][5] = None
    with pytest.raises(ValueError):
        env.step({"replace": 5})


def test_invalid_flip_index():
    env = SkyjoEnv(num_players=2)
    env.reset()
    env.step({"draw": "deck"})

    # Flip a card already visible
    visible_index = next(i for i, v in enumerate(env.players[0]["visible"]) if v)
    with pytest.raises(ValueError):
        env.step({"flip": visible_index})

    # Flip a card already removed
    env.players[0]["grid"][3] = None
    env.players[0]["visible"][3] = False
    with pytest.raises(ValueError):
        env.step({"flip": 3})


def test_missing_and_conflicting_action():
    env = SkyjoEnv(num_players=2)
    env.reset()
    env.step({"draw": "deck"})

    # Neither flip nor replace
    with pytest.raises(ValueError):
        env.step({})

    # Both at the same time
    with pytest.raises(ValueError):
        env.step({"flip": 1, "replace": 2})



def test_player_transition():
    env = SkyjoEnv(num_players=2)
    env.reset()
    env.step({"draw": "deck"})
    env.step({"flip": next(i for i, v in enumerate(env.players[0]["visible"]) if not v)})
    assert env.current_player == 1

    env.step({"draw": "deck"})
    env.step({"flip": next(i for i, v in enumerate(env.players[1]["visible"]) if not v)})
    assert env.current_player == 0


def test_final_round_trigger_and_game_end():
    env = SkyjoEnv(num_players=2)
    env.reset()

    # Force the final round
    player = env.players[0]
    player["visible"] = [False] + [True] * 11
    player["return_count"] = 11
    env.pending_card = 5
    player["grid"][0] = 2

    env.step({"replace": 0})
    assert env.final_player == 0
    assert env.remaining_final_turns == 1
    assert not env.done

    # Joueur 1 joue
    env.step({"draw": "deck"})
    flip_idx = next(i for i, v in enumerate(env.players[1]["visible"]) if not v)
    env.step({"flip": flip_idx})

    assert env.done

def test_score_calculation_in_step():
    env = SkyjoEnv(num_players=2)
    env.reset()
    
    # Test score when replacing a card
    initial_score = env.players[0]["score"]
    env.step({"draw": "deck"})
    replace_index = next(i for i, v in enumerate(env.players[0]["grid"]) if v is not None)
    old_card_value = env.players[0]["grid"][replace_index]
    new_card_value = env.pending_card
    
    env.step({"replace": replace_index})
    expected_score = initial_score - old_card_value + new_card_value
    assert env.players[0]["score"] == expected_score
    
    # Test score when flipping a card
    env.reset()
    env.step({"draw": "deck"})
    flip_index = next(i for i, v in enumerate(env.players[0]["visible"]) if not v)
    flipped_card_value = env.players[0]["grid"][flip_index]
    initial_score = env.players[0]["score"]
    
    env.step({"flip": flip_index})
    expected_score = initial_score + flipped_card_value
    assert env.players[0]["score"] == expected_score

def test_previous_action_logic():
    env = SkyjoEnv(num_players=2)
    env.reset()
    
    # Test draw from deck
    env.step({"draw": "deck"})
    assert env.previous_action == "draw_deck"
    
    # Test draw from discard
    env.step({"flip": next(i for i, v in enumerate(env.players[0]["visible"]) if not v)})
    env.step({"draw": "discard"})
    assert env.previous_action == "draw_discard"
    
    # Test error when trying to flip after drawing from discard
    with pytest.raises(ValueError, match="You cannot flip a card after drawing from the discard pile"):
        env.step({"flip": next(i for i, v in enumerate(env.players[0]["visible"]) if not v)})
    
    # Test that previous_action is reset after completing a turn
    env.step({"replace": next(i for i, v in enumerate(env.players[0]["grid"]) if v is not None)})
    assert env.previous_action == ""
    
    # Test that previous_action is properly set and reset in a complete turn
    env.step({"draw": "deck"})
    assert env.previous_action == "draw_deck"
    env.step({"flip": next(i for i, v in enumerate(env.players[0]["visible"]) if not v)})
    assert env.previous_action == ""

def test_get_final_scores():
    env = SkyjoEnv(num_players=2)
    env.reset()
    
    # Test that we can't get final scores before game is done
    with pytest.raises(ValueError, match="Cannot calculate final scores before the game is done"):
        env.get_final_scores()
    
    # Test normal scoring (no penalty)
    # Set up a game state where player 0 ends with lowest score
    env.players[0]["grid"] = [1, 2, 3, None, None, None, None, None, None, None, None, None]  # Score: 6
    env.players[1]["grid"] = [4, 5, 6, None, None, None, None, None, None, None, None, None]  # Score: 15
    env.final_player = 0
    env.done = True
    
    final_scores = env.get_final_scores()
    assert final_scores[0] == 6  # No penalty as player 0 has lowest score
    assert final_scores[1] == 15
    
    # Test penalty scoring (positive score)
    # Set up a game state where player 0 ends with higher score
    env.players[0]["grid"] = [4, 5, 6, None, None, None, None, None, None, None, None, None]  # Score: 15
    env.players[1]["grid"] = [1, 2, 3, None, None, None, None, None, None, None, None, None]  # Score: 6
    env.final_player = 0
    env.done = True
    
    final_scores = env.get_final_scores()
    assert final_scores[0] == 30  # Score doubled as player 0 has higher score
    assert final_scores[1] == 6
    
    # Test penalty scoring (negative score)
    # Set up a game state where player 0 ends with higher negative score
    env.players[0]["grid"] = [-4, -5, -6, None, None, None, None, None, None, None, None, None]  # Score: -15
    env.players[1]["grid"] = [-1, -2, -3, None, None, None, None, None, None, None, None, None]  # Score: -6
    env.final_player = 0
    env.done = True
    
    final_scores = env.get_final_scores()
    assert final_scores[0] == -15  # Score not doubled as it's negative
    assert final_scores[1] == -6

    # Test with multiple players having the same lowest score
    # Set up a game state where players 0 and 1 have the same lowest score
    env.players[0]["grid"] = [1, 2, 3, None, None, None, None, None, None, None, None, None]  # Score: 6
    env.players[1]["grid"] = [1, 2, 3, None, None, None, None, None, None, None, None, None]  # Score: 6
    env.final_player = 0
    env.done = True
    
    final_scores = env.get_final_scores()
    assert final_scores[0] == 12  # penalty as player 0 and 1 have the same score
    assert final_scores[1] == 6
