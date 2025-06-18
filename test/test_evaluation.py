from src.evaluation.evaluator import Evaluator
from src.models.base_model import SkyjoModel
from unittest.mock import Mock, patch
import pytest


class MockModel(SkyjoModel):
    """Mock model for testing purposes."""

    def __init__(self, name: str = "MockModel"):
        self.name = name
        self.reset_called = False

    def get_action(self, observation):
        """Return a simple action for testing."""
        if observation["pending_card"] is not None:
            # Use the pending card
            for i, visible in enumerate(
                observation["players"][observation["current_player"]]["visible"]
            ):
                if (
                    not visible
                    and observation["players"][observation["current_player"]]["grid"][i]
                    is not None
                ):
                    return {"flip": i}
            # If no hidden cards, replace a visible one
            for i, visible in enumerate(
                observation["players"][observation["current_player"]]["visible"]
            ):
                if (
                    visible
                    and observation["players"][observation["current_player"]]["grid"][i]
                    is not None
                ):
                    return {"replace": i}
        else:
            # Draw a card
            return {"draw": "deck"}

    def reset(self):
        """Reset the model's state."""
        self.reset_called = True


class TestEvaluator:
    """Test suite for the Evaluator class."""

    def test_init_default_values(self):
        """Test evaluator initialization with default values."""
        evaluator = Evaluator()

        assert evaluator.num_games == 1000
        assert evaluator.k_factor == 32
        assert evaluator.initial_rating == 1500
        assert evaluator.verbose is True
        assert evaluator.ratings == {}
        assert evaluator.win_based_ratings == {}
        assert evaluator.total_games == 0

    def test_init_custom_values(self):
        """Test evaluator initialization with custom values."""
        evaluator = Evaluator(
            num_games=100, k_factor=16, initial_rating=1200, verbose=False
        )

        assert evaluator.num_games == 100
        assert evaluator.k_factor == 16
        assert evaluator.initial_rating == 1200
        assert evaluator.verbose is False

    def test_reset(self):
        """Test evaluator reset functionality."""
        evaluator = Evaluator()

        # Add some data
        evaluator.ratings = {"model1": 1600, "model2": 1400}
        evaluator.win_based_ratings = {"model1": 5.5, "model2": 4.5}
        evaluator.total_games = 10

        evaluator.reset()

        assert evaluator.ratings == {}
        assert evaluator.win_based_ratings == {}
        assert evaluator.total_games == 0

    def test_evaluate_too_few_models(self):
        """Test that evaluate raises error with too few models."""
        evaluator = Evaluator(num_games=1)
        models = [(MockModel("model1"), "model1")]

        with pytest.raises(ValueError, match="Must provide at least 2 models"):
            evaluator.evaluate(models)

    def test_evaluate_too_many_models(self):
        """Test that evaluate raises error with too many models."""
        evaluator = Evaluator(num_games=1)
        models = [(MockModel(f"model{i}"), f"model{i}") for i in range(9)]

        with pytest.raises(
            ValueError, match="Too many models. Maximum number of models is 8."
        ):
            evaluator.evaluate(models)

    def test_get_expected_score(self):
        """Test expected score calculation."""
        evaluator = Evaluator()

        # Test with equal ratings
        ratings = [1500, 1500, 1500]
        expected = evaluator._get_expected_score(1500, ratings)
        assert abs(expected - 1 / 3) < 1e-6

        # Test with different ratings
        ratings = [1600, 1400, 1500]
        expected = evaluator._get_expected_score(1600, ratings)
        # Higher rating should have higher expected score
        assert expected > 1 / 3

        expected = evaluator._get_expected_score(1400, ratings)
        # Lower rating should have lower expected score
        assert expected < 1 / 3

    def test_update_ratings_new_models(self):
        """Test rating updates for new models."""
        evaluator = Evaluator(initial_rating=1500, k_factor=32)

        model_names = ["model1", "model2"]
        scores = [1.0, 0.0]  # model1 wins, model2 loses

        evaluator._update_ratings(model_names, scores)

        # Check win-based ratings
        assert evaluator.win_based_ratings["model1"] == 1.0
        assert evaluator.win_based_ratings["model2"] == 0.0

        # Check Elo ratings (should be initialized and updated)
        assert "model1" in evaluator.ratings
        assert "model2" in evaluator.ratings
        assert evaluator.ratings["model1"] > 1500  # Winner should gain rating
        assert evaluator.ratings["model2"] < 1500  # Loser should lose rating

    def test_update_ratings_existing_models(self):
        """Test rating updates for existing models."""
        evaluator = Evaluator(initial_rating=1500, k_factor=32)

        # Set initial ratings
        evaluator.ratings = {"model1": 1600, "model2": 1400}
        evaluator.win_based_ratings = {"model1": 5.0, "model2": 3.0}

        model_names = ["model1", "model2"]
        scores = [0.0, 1.0]  # model2 wins, model1 loses

        evaluator._update_ratings(model_names, scores)

        # Check win-based ratings are accumulated
        assert evaluator.win_based_ratings["model1"] == 5.0
        assert evaluator.win_based_ratings["model2"] == 4.0

        # Check Elo ratings are updated
        assert evaluator.ratings["model1"] < 1600  # Loser should lose rating
        assert evaluator.ratings["model2"] > 1400  # Winner should gain rating

    def test_update_ratings_draw(self):
        """Test rating updates for a draw scenario."""
        evaluator = Evaluator(initial_rating=1500, k_factor=32)

        model_names = ["model1", "model2", "model3"]
        scores = [1 / 3, 1 / 3, 1 / 3]  # All tie

        evaluator._update_ratings(model_names, scores)

        # Check win-based ratings
        assert evaluator.win_based_ratings["model1"] == 1 / 3
        assert evaluator.win_based_ratings["model2"] == 1 / 3
        assert evaluator.win_based_ratings["model3"] == 1 / 3

        # Check Elo ratings (should be close to initial for equal players)
        for model_name in model_names:
            assert abs(evaluator.ratings[model_name] - 1500) < 10

    @patch("src.evaluation.evaluator.alive_bar")
    def test_evaluate_single_game(self, mock_alive_bar):
        """Test evaluation with a single game."""
        # Mock the progress bar
        mock_bar = Mock()
        mock_bar.__enter__ = Mock(return_value=mock_bar)
        mock_bar.__exit__ = Mock(return_value=None)
        mock_alive_bar.return_value = mock_bar

        evaluator = Evaluator(num_games=1, verbose=True)
        models = [(MockModel("model1"), "model1"), (MockModel("model2"), "model2")]

        evaluator.evaluate(models)

        # Check that models were reset
        assert models[0][0].reset_called
        assert models[1][0].reset_called

        # Check that total games was incremented
        assert evaluator.total_games == 1

        # Check that ratings were updated
        assert len(evaluator.ratings) == 2
        assert len(evaluator.win_based_ratings) == 2

    @patch("src.evaluation.evaluator.alive_bar")
    def test_evaluate_verbose_false(self, mock_alive_bar):
        """Test evaluation with verbose=False."""
        evaluator = Evaluator(num_games=1, verbose=False)
        models = [(MockModel("model1"), "model1"), (MockModel("model2"), "model2")]

        evaluator.evaluate(models)

        # Should not call alive_bar when verbose=False
        mock_alive_bar.assert_not_called()

        # Check that evaluation still works
        assert evaluator.total_games == 1
        assert len(evaluator.ratings) == 2

    def test_evaluate_multiple_players(self):
        """Test evaluation with more than 2 players."""
        evaluator = Evaluator(num_games=1, verbose=False)
        models = [
            (MockModel("model1"), "model1"),
            (MockModel("model2"), "model2"),
            (MockModel("model3"), "model3"),
            (MockModel("model4"), "model4"),
        ]

        evaluator.evaluate(models)

        # Check that all models were processed
        assert evaluator.total_games == 1
        assert len(evaluator.ratings) == 4
        assert len(evaluator.win_based_ratings) == 4

        # Check that all models were reset
        for model, _ in models:
            assert model.reset_called

    def test_elo_calculation_accuracy(self):
        """Test that Elo calculations are mathematically correct."""
        evaluator = Evaluator(k_factor=32, initial_rating=1500)

        # Test with two players of equal rating
        ratings = [1500, 1500]
        expected_scores = [
            evaluator._get_expected_score(1500, ratings),
            evaluator._get_expected_score(1500, ratings),
        ]

        # Equal ratings should have equal expected scores
        assert abs(expected_scores[0] - expected_scores[1]) < 1e-10
        assert abs(expected_scores[0] - 0.5) < 1e-10

        # Test rating update formula
        model_names = ["player1", "player2"]
        scores = [1.0, 0.0]  # player1 wins

        evaluator._update_ratings(model_names, scores)

        # Calculate expected rating change
        expected_change = evaluator.k_factor * (1.0 - 0.5)  # 16 points
        assert abs(evaluator.ratings["player1"] - (1500 + expected_change)) < 1e-10
        assert abs(evaluator.ratings["player2"] - (1500 - expected_change)) < 1e-10

    def test_win_based_rating_accumulation(self):
        """Test that win-based ratings accumulate correctly."""
        evaluator = Evaluator(num_games=1, verbose=False)
        models = [(MockModel("model1"), "model1"), (MockModel("model2"), "model2")]

        # First evaluation
        evaluator.evaluate(models)

        # Second evaluation
        evaluator.evaluate(models)

        assert evaluator.total_games == 2

    def test_model_shuffling(self):
        """Test that models are randomly assigned to players."""
        evaluator = Evaluator(num_games=10, verbose=False)
        models = [
            (MockModel("model1"), "model1"),
            (MockModel("model2"), "model2"),
            (MockModel("model3"), "model3"),
        ]

        # Run multiple games to ensure shuffling occurs
        evaluator.evaluate(models)

        # All models should have been evaluated
        assert len(evaluator.ratings) == 3
        assert len(evaluator.win_based_ratings) == 3
        assert evaluator.total_games == 10

    def test_edge_case_zero_games(self):
        """Test edge case with zero games."""
        evaluator = Evaluator(num_games=0, verbose=False)
        models = [(MockModel("model1"), "model1"), (MockModel("model2"), "model2")]

        evaluator.evaluate(models)

        # Should complete without error
        assert evaluator.total_games == 0
        assert len(evaluator.ratings) == 0
        assert len(evaluator.win_based_ratings) == 0

    def test_k_factor_impact(self):
        """Test that different k-factors affect rating changes appropriately."""
        # High k-factor should cause larger rating changes
        evaluator_high = Evaluator(k_factor=64, initial_rating=1500)
        evaluator_low = Evaluator(k_factor=16, initial_rating=1500)

        model_names = ["model1", "model2"]
        scores = [1.0, 0.0]

        evaluator_high._update_ratings(model_names, scores)
        evaluator_low._update_ratings(model_names, scores)

        # High k-factor should result in larger rating changes
        high_change = abs(evaluator_high.ratings["model1"] - 1500)
        low_change = abs(evaluator_low.ratings["model1"] - 1500)

        assert high_change > low_change
