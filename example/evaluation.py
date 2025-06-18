import sys
import os

# Add src/ to the path to import the skyjo_env.py file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluator import Evaluator
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
            if (
                random.random() < 0.5
                or observation["previous_action"] == "draw_discard"
            ):
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
    num_games = 10_000
    model_number = 5

    # Create some random models
    models = [RandomModel() for _ in range(model_number)]
    model_names = [f"Model {i + 1}" for i in range(model_number)]

    # Evaluate
    print("=== Evaluation ===")
    evaluator = Evaluator(num_games=num_games, k_factor=16, initial_rating=1500, verbose=True)

    evaluator.evaluate(
        [(model, model_name) for model, model_name in zip(models, model_names)]
    )

    for model_name, win_rate in evaluator.win_based_ratings.items():
        print(f"{model_name} win rate: {win_rate / evaluator.total_games:.2%}")

    for model_name, rating in evaluator.ratings.items():
        print(f"{model_name} Elo rating: {rating:.1f}")


if __name__ == "__main__":
    main()
