from src.models.base_model import SkyjoModel
from src.skyjo_env import SkyjoEnv
import random
import math


class EloBasedEvaluator:
    """
    Evaluates models using the Elo rating system.
    """

    def __init__(
        self, num_games: int = 1000, k_factor: float = 32, initial_rating: float = 1500
    ):
        """
        Initialize the evaluator.

        Args:
            num_games (int): Number of games to play for evaluation
            k_factor (float): K-factor for Elo rating updates (the higher the more weight is given to the outcome of the game)
            initial_rating (float): Initial Elo rating for new models
        """
        self.num_games = num_games
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}  # model_id -> rating

    def _get_expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score for player 1"""
        return 1 / (1 + math.pow(10, (rating2 - rating1) / 400))

    def _update_ratings(
        self, model1_id: str, model2_id: str, score1: float, score2: float
    ):
        """Update Elo ratings based on game outcome"""
        # Get current ratings or initialize if new
        rating1 = self.ratings.get(model1_id, self.initial_rating)
        rating2 = self.ratings.get(model2_id, self.initial_rating)

        # Calculate expected scores
        expected1 = self._get_expected_score(rating1, rating2)
        expected2 = 1 - expected1

        # Update ratings
        self.ratings[model1_id] = rating1 + self.k_factor * (score1 - expected1)
        self.ratings[model2_id] = rating2 + self.k_factor * (score2 - expected2)

    def evaluate_1v1(self, model1: SkyjoModel, model2: SkyjoModel) -> dict[str, float]:
        """
        Evaluate two models in a 1v1 scenario using Elo ratings.

        Args:
            model1 (SkyjoModel): First model
            model2 (SkyjoModel): Second model

        Returns:
            dict[str, float]: Current Elo ratings for each model
        """
        model1_id = id(model1)
        model2_id = id(model2)

        for _ in range(self.num_games):
            env = SkyjoEnv(num_players=2)
            obs = env.reset()
            models = [(model1, "model1"), (model2, "model2")]

            # Randomly assign models to players
            if random.random() < 0.5:
                models = [(model2, "model2"), (model1, "model1")]
                model1_id, model2_id = model2_id, model1_id

            done = False
            while not done:
                current_model = models[env.current_player][0]
                action = current_model.get_action(obs)
                obs, _, done, _ = env.step(action)

            # Get final scores
            final_scores = env.get_final_scores()

            # Calculate scores (1 for win, 0.5 for draw, 0 for loss)
            if final_scores[0] < final_scores[1]:
                score1, score2 = 1, 0
            elif final_scores[0] > final_scores[1]:
                score1, score2 = 0, 1
            else:
                score1, score2 = 0.5, 0.5

            self._update_ratings(model1_id, model2_id, score1, score2)

            # Reset models
            models[0][0].reset()
            models[1][0].reset()

        return {
            "model1_rating": self.ratings.get(model1_id, self.initial_rating),
            "model2_rating": self.ratings.get(model2_id, self.initial_rating),
        }

    def evaluate_2v2(
        self, models_1: list[SkyjoModel], models_2: list[SkyjoModel]
    ) -> dict[str, float]:
        """
        Evaluate four models in a 2v2 scenario using Elo ratings.

        Args:
            models (list[SkyjoModel]): list of four models

        Returns:
            dict[str, float]: Current Elo ratings for each model
        """
        if len(models_1) != 2 or len(models_2) != 2:
            raise ValueError("Must provide exactly 2 models for each team")

        models = [
            (models_1[0], id(models_1[0]), "team1"),
            (models_1[1], id(models_1[1]), "team1"),
            (models_2[0], id(models_2[0]), "team2"),
            (models_2[1], id(models_2[1]), "team2"),
        ]

        for _ in range(self.num_games):
            env = SkyjoEnv(num_players=4)
            obs = env.reset()

            # Randomly assign models to players
            shuffled_indices = random.sample(range(4), 4)
            shuffled_models = [models[i] for i in shuffled_indices]

            done = False
            while not done:
                current_model = shuffled_models[env.current_player][0]
                action = current_model.get_action(obs)
                obs, _, done, _ = env.step(action)

            # Get final scores
            final_scores = env.get_final_scores()

            # Calculate team scores
            team1_score = 0
            team2_score = 0
            for idx, model in enumerate(models):
                if model[2] == "team1":
                    team1_score += final_scores[idx]
                else:
                    team2_score += final_scores[idx]

            # Calculate scores for each model
            if team1_score < team2_score:
                scores = [1, 1, 0, 0]  # Team 1 wins
            elif team1_score > team2_score:
                scores = [0, 0, 1, 1]  # Team 2 wins
            else:
                scores = [0.5, 0.5, 0.5, 0.5]  # Draw

            # Update ratings for each model
            for i in range(4):
                for j in range(i + 1, 4):
                    self._update_ratings(
                        models[i][1], models[j][1], scores[i], scores[j]
                    )

            # Reset models
            for model in models:
                model[0].reset()

        return {
            f"model{i + 1}_rating": self.ratings.get(models[i][1], self.initial_rating)
            for i in range(4)
        }
