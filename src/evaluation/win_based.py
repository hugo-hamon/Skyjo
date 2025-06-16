from src.models.base_model import SkyjoModel
from src.skyjo_env import SkyjoEnv
import random

class WinBasedEvaluator:
    """
    Evaluates models based on their win rates in different scenarios.
    """
    
    def __init__(self, num_games: int = 1000):
        """
        Initialize the evaluator.
        
        Args:
            num_games (int): Number of games to play for evaluation
        """
        self.num_games = num_games
        
    def evaluate_1v1(self, model1: SkyjoModel, model2: SkyjoModel) -> dict[str, float]:
        """
        Evaluate two models in a 1v1 scenario.
        
        Args:
            model1 (SkyjoModel): First model
            model2 (SkyjoModel): Second model
            
        Returns:
            dict[str, float]: Win rates for each model
        """
        wins = {"model1": 0, "model2": 0}
        
        for _ in range(self.num_games):
            env = SkyjoEnv(num_players=2)
            obs = env.reset()
            models = [(model1, "model1"), (model2, "model2")]

            # Randomly assign models to players
            if random.random() < 0.5:
                models = [(model2, "model2"), (model1, "model1")]
            
            done = False
            while not done:
                current_model = models[env.current_player][0]
                action = current_model.get_action(obs)
                obs, _, done, _ = env.step(action)
            
            # Get final scores
            final_scores = env.get_final_scores()

            # Two players with the same score are considered a tie
            if final_scores[0] == final_scores[1]:
                wins["model1"] += 0.5
                wins["model2"] += 0.5
            else:
                winner = min(final_scores.items(), key=lambda x: x[1])[0]
                wins[models[winner][1]] += 1
            
            # Reset models
            models[0][0].reset()
            models[1][0].reset()
            
        return {
            "model1_win_rate": wins["model1"] / self.num_games,
            "model2_win_rate": wins["model2"] / self.num_games
        }
    
    def evaluate_2v2(self, models_1: list[SkyjoModel], models_2: list[SkyjoModel]) -> dict[str, float]:
        """
        Evaluate four models in a 2v2 scenario.
        
        Args:
            models_1 (list[SkyjoModel]): list of two models
            models_2 (list[SkyjoModel]): list of two models
            
        Returns:
            dict[str, float]: Win rates for each team
        """
        if len(models_1) != 2 or len(models_2) != 2:
            raise ValueError("Must provide exactly 2 models for each team")
            
        team_wins = {"team1": 0, "team2": 0}
        models = [(models_1[0], "team1"), (models_1[1], "team1"), (models_2[0], "team2"), (models_2[1], "team2")]
        
        for _ in range(self.num_games):
            env = SkyjoEnv(num_players=4)
            obs = env.reset()
            
            # Randomly assign models to players
            shuffled_models = random.sample(models, 4)
            
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
                if model[1] == "team1":
                    team1_score += final_scores[idx]
                else:
                    team2_score += final_scores[idx]
            
            # Determine winner
            if team1_score < team2_score:
                team_wins["team1"] += 1
            elif team1_score > team2_score:
                team_wins["team2"] += 1
            else:
                team_wins["team1"] += 0.5
                team_wins["team2"] += 0.5
                
            # Reset models
            for model in models:
                model[0].reset()
                
        return {
            "team1_win_rate": team_wins["team1"] / self.num_games,
            "team2_win_rate": team_wins["team2"] / self.num_games
        } 