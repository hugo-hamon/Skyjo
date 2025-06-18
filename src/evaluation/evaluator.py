from src.models.base_model import SkyjoModel
from alive_progress import alive_bar
from src.skyjo_env import SkyjoEnv
import random
import math


class Evaluator:
    """
    Evaluates models using the Elo rating system and a win-based system.
    """

    def __init__(
        self,
        num_games: int = 1000,
        k_factor: float = 32,
        initial_rating: float = 1500,
        verbose: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            num_games (int): Number of games to play for evaluation
            k_factor (float): K-factor for Elo rating updates (the higher the more weight is given to the outcome of the game)
            initial_rating (float): Initial Elo rating for new models
            verbose (bool): Whether to show a progress bar
        """
        self.num_games = num_games
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.verbose = verbose

        self.ratings = {}  # model_name -> rating
        self.win_based_ratings = {}  # model_name -> win-based rating
        self.total_games = 0

    def evaluate(self, models: list[tuple[SkyjoModel, str]]) -> None:
        """Evaluate a list of models in a given scenario."""
        if len(models) < 2:
            raise ValueError("Must provide at least 2 models")
        if len(models) > 8:
            raise ValueError("Too many models. Maximum number of models is 8.")

        num_players = len(models)

        with self._get_progress_bar() as bar:
            for _ in range(self.num_games):
                env = SkyjoEnv(num_players=num_players)
                obs = env.reset()

                # Randomly assign models to players
                shuffled_models = random.sample(models, num_players)
                done = False

                while not done:
                    current_model = shuffled_models[env.current_player][0]
                    action = current_model.get_action(obs)
                    obs, _, done, _ = env.step(action)

                # Get final scores
                final_scores = env.get_final_scores()

                # Calculate scores (1 for win, 1 / num_players for draw, 0 for loss)
                min_score = min(final_scores.values())
                num_min_score = sum(
                    1 for score in final_scores.values() if score == min_score
                )

                # Determine the winner
                elo_scores = []
                for idx in range(num_players):
                    if final_scores[idx] == min_score:
                        elo_scores.append(1 / num_min_score)
                    else:
                        elo_scores.append(0)

                # Update ratings
                self._update_ratings(
                    [model_name for _, model_name in shuffled_models], elo_scores
                )

                # Reset models
                for model in shuffled_models:
                    model[0].reset()

                self.total_games += 1
                bar()

    def reset(self) -> None:
        """Reset the evaluator's internal state."""
        self.ratings = {}
        self.win_based_ratings = {}
        self.total_games = 0
        
    def _get_expected_score(self, player_rating: float, ratings: list[float]) -> float:
        """Calculate expected score for each player"""
        player_q = math.pow(10, player_rating / 400)
        all_players_q = [math.pow(10, rating / 400) for rating in ratings]
        return player_q / sum(all_players_q)

    def _update_ratings(self, model_names: list[str], scores: list[float]) -> None:
        """Update Elo ratings and win-based ratings based on game outcome"""
        # Update win-based ratings
        for model_name, score in zip(model_names, scores):
            self.win_based_ratings[model_name] = (
                self.win_based_ratings.get(model_name, 0) + score
            )

        # Get current ratings or initialize if new
        ratings = [
            self.ratings.get(model_name, self.initial_rating)
            for model_name in model_names
        ]

        # Calculate expected scores
        expected_scores = [
            self._get_expected_score(rating, ratings) for rating in ratings
        ]
        for expected_score, actual_score, actual_rating, model_name in zip(
            expected_scores, scores, ratings, model_names
        ):
            self.ratings[model_name] = actual_rating + self.k_factor * (
                actual_score - expected_score
            )

    def _get_progress_bar(self):
        """Get the appropriate progress bar based on verbose setting."""
        if self.verbose:
            return alive_bar(self.num_games, title="Evaluating models", force_tty=True)
        else:
            # Dummy context manager that does nothing
            class DummyBar:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def __call__(self):
                    pass

            return DummyBar()
