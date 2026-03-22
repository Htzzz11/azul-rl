"""
Base abstract class for all Azul agents.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class BaseAgent(ABC):
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def choose_action(self, observation: dict, player_index: int, valid_actions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Choose an action given the current game state.
        """
        pass

    def reset(self):
        """
        Reset agent state between games (if needed).
        """
        pass
