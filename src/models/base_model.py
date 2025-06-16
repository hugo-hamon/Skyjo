from abc import ABC, abstractmethod
from typing import Any


class SkyjoModel(ABC):
    """
    Abstract base class for Skyjo playing models.
    All models must implement these methods to be evaluated.
    """

    @abstractmethod
    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Get the next action to play based on the current observation.

        Args:
            observation (dict[str, Any]): The current game state observation

        Returns:
            dict[str, Any]: The action to take, either:
                - {"draw": "deck" or "discard"} to draw a card
                - {"replace": int (0-11) or "flip": int (0-11)} to use the drawn card
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the model's internal state if needed.
        Called at the start of each new game.
        """
        pass
