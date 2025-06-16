import sys
import os

# Add src/ to the path to import the skyjo_env.py file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.elo_based import EloBasedEvaluator
from src.evaluation.win_based import WinBasedEvaluator
from src.models.base_model import SkyjoModel
from typing import Any
import random


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


def main():
    num_games = 10000

    # Create some random models
    model1 = RandomModel()
    model2 = RandomModel()
    model3 = RandomModel()
    model4 = RandomModel()

    # Evaluate using win-based system
    print("=== Win-based Evaluation ===")
    win_evaluator = WinBasedEvaluator(num_games=num_games)

    # 1v1 evaluation
    print("\n1v1 Evaluation:")
    results_1v1 = win_evaluator.evaluate_1v1(model1, model2)
    print(f"Model 1 win rate: {results_1v1['model1_win_rate']:.2%}")
    print(f"Model 2 win rate: {results_1v1['model2_win_rate']:.2%}")

    # 2v2 evaluation
    print("\n2v2 Evaluation:")
    results_2v2 = win_evaluator.evaluate_2v2(
        [model1, model2],  # Team 1
        [model3, model4],  # Team 2
    )
    print(f"Team 1 win rate: {results_2v2['team1_win_rate']:.2%}")
    print(f"Team 2 win rate: {results_2v2['team2_win_rate']:.2%}")

    # Evaluate using Elo-based system
    print("\n=== Elo-based Evaluation ===")
    elo_evaluator = EloBasedEvaluator(num_games=num_games)

    # 1v1 evaluation
    print("\n1v1 Evaluation:")
    results_1v1_elo = elo_evaluator.evaluate_1v1(model1, model2)
    print(f"Model 1 Elo rating: {results_1v1_elo['model1_rating']:.1f}")
    print(f"Model 2 Elo rating: {results_1v1_elo['model2_rating']:.1f}")

    # 2v2 evaluation
    print("\n2v2 Evaluation:")
    results_2v2_elo = elo_evaluator.evaluate_2v2(
        [model1, model2],  # Team 1
        [model3, model4],  # Team 2
    )
    print("Team 1:")
    print(f"  Model 1 Elo rating: {results_2v2_elo['model1_rating']:.1f}")
    print(f"  Model 2 Elo rating: {results_2v2_elo['model2_rating']:.1f}")
    print("Team 2:")
    print(f"  Model 3 Elo rating: {results_2v2_elo['model3_rating']:.1f}")
    print(f"  Model 4 Elo rating: {results_2v2_elo['model4_rating']:.1f}")


if __name__ == "__main__":
    main()
