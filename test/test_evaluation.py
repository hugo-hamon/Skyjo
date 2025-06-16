from src.evaluation.win_based import WinBasedEvaluator
from src.evaluation.elo_based import EloBasedEvaluator
from src.models.base_model import SkyjoModel
from typing import Any
import random
import pytest


class RandomModel(SkyjoModel):
    """
    A simple model that plays randomly.
    """

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Get a random action based on the current game state.

        Args:
            observation (dict[str, Any]): The current game state

        Returns:
            dict[str, Any]: A random valid action
        """
        # If there's a pending card, we use it
        if observation["pending_card"] is not None:
            # Get all valid positions (not None in grid)
            valid_positions = [
                i
                for i, card in enumerate(
                    observation["players"][observation["current_player"]]["grid"]
                )
                if card is not None
            ]

            # Randomly choose between replace and flip
            if random.random() < 0.5 or observation["previous_action"] == "draw_discard":
                # Replace a random card
                return {"replace": random.choice(valid_positions)}
            else:
                # Flip a random hidden card
                hidden_positions = [
                    i
                    for i, visible in enumerate(
                        observation["players"][observation["current_player"]]["visible"]
                    )
                    if not visible
                    and observation["players"][observation["current_player"]]["grid"][i]
                    is not None
                ]
                if hidden_positions:
                    return {"flip": random.choice(hidden_positions)}
                else:
                    return {"replace": random.choice(valid_positions)}
        else:
            # Draw a card
            return {"draw": random.choice(["deck", "discard"])}

    def reset(self) -> None:
        """Reset the model's internal state (nothing to reset for random model)"""
        pass


def test_win_based_evaluator_1v1():
    # Create test models with fixed actions
    model1 = RandomModel()
    model2 = RandomModel()
    
    evaluator = WinBasedEvaluator(num_games=10)
    results = evaluator.evaluate_1v1(model1, model2)
    
    # Check that results contain expected keys
    assert "model1_win_rate" in results
    assert "model2_win_rate" in results
    
    # Check that win rates are between 0 and 1
    assert 0 <= results["model1_win_rate"] <= 1
    assert 0 <= results["model2_win_rate"] <= 1
    
    # Check that win rates sum to 1 (including ties)
    assert abs(results["model1_win_rate"] + results["model2_win_rate"] - 1) < 1e-10


def test_win_based_evaluator_2v2():
    # Create test models with fixed actions
    models_1 = [RandomModel(), RandomModel()]
    models_2 = [RandomModel(), RandomModel()]
    
    evaluator = WinBasedEvaluator(num_games=10)
    results = evaluator.evaluate_2v2(models_1, models_2)
    
    # Check that results contain expected keys
    assert "team1_win_rate" in results
    assert "team2_win_rate" in results
    
    # Check that win rates are between 0 and 1
    assert 0 <= results["team1_win_rate"] <= 1
    assert 0 <= results["team2_win_rate"] <= 1
    
    # Check that win rates sum to 1 (including ties)
    assert abs(results["team1_win_rate"] + results["team2_win_rate"] - 1) < 1e-10


def test_win_based_evaluator_invalid_2v2():
    evaluator = WinBasedEvaluator()
    
    # Test with wrong number of models
    with pytest.raises(ValueError, match="Must provide exactly 2 models for each team"):
        evaluator.evaluate_2v2(
            [RandomModel()],
            [RandomModel(), RandomModel()]
        )


def test_elo_based_evaluator_1v1():
    # Create test models with fixed actions
    model1 = RandomModel()
    model2 = RandomModel()
    
    evaluator = EloBasedEvaluator(num_games=10)
    results = evaluator.evaluate_1v1(model1, model2)
    
    # Check that results contain expected keys
    assert "model1_rating" in results
    assert "model2_rating" in results
    
    # Check that ratings are close to initial rating
    # (with small number of games, ratings shouldn't change much)
    assert abs(results["model1_rating"] - evaluator.initial_rating) < 100
    assert abs(results["model2_rating"] - evaluator.initial_rating) < 100


def test_elo_based_evaluator_2v2():
    # Create test models with fixed actions
    models_1 = [RandomModel(), RandomModel()]
    models_2 = [RandomModel(), RandomModel()]
    
    evaluator = EloBasedEvaluator(num_games=10)
    results = evaluator.evaluate_2v2(models_1, models_2)
    
    # Check that results contain expected keys
    for i in range(4):
        assert f"model{i+1}_rating" in results
    
    # Check that ratings are close to initial rating
    for i in range(4):
        assert abs(results[f"model{i+1}_rating"] - evaluator.initial_rating) < 100


def test_elo_based_evaluator_invalid_2v2():
    evaluator = EloBasedEvaluator()
    
    # Test with wrong number of models
    with pytest.raises(ValueError, match="Must provide exactly 2 models for each team"):
        evaluator.evaluate_2v2(
            [RandomModel()],
            [RandomModel(), RandomModel()]
        )


def test_elo_rating_update():
    evaluator = EloBasedEvaluator(k_factor=32)
    
    # Test rating update for a win
    model1_id = "model1"
    model2_id = "model2"
    
    # Initial ratings
    evaluator.ratings[model1_id] = 1500
    evaluator.ratings[model2_id] = 1500
    
    # Update ratings for a win
    evaluator._update_ratings(model1_id, model2_id, 1, 0)
    
    # Check that ratings changed appropriately
    assert evaluator.ratings[model1_id] > 1500  # Winner's rating increased
    assert evaluator.ratings[model2_id] < 1500  # Loser's rating decreased
    assert abs(evaluator.ratings[model1_id] - 1500) == abs(evaluator.ratings[model2_id] - 1500)  # Equal change


def test_elo_expected_score():
    evaluator = EloBasedEvaluator()
    
    # Test expected score calculation
    rating1 = 1500
    rating2 = 1500
    
    # Equal ratings should give 0.5 expected score
    expected = evaluator._get_expected_score(rating1, rating2)
    assert abs(expected - 0.5) < 1e-10
    
    # Higher rating should give higher expected score
    rating1 = 1600
    rating2 = 1400
    expected = evaluator._get_expected_score(rating1, rating2)
    assert expected > 0.5 