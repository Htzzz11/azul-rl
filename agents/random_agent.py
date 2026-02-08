"""
Random agent implementation for Azul.
"""

import random
from typing import List, Tuple
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, name: str = None, seed: int = None):
        super().__init__(name)
        self.seed = seed
        self.rng = random.Random(seed)

    def choose_action(self, observation: dict, valid_actions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        if not valid_actions:
            raise ValueError("Cannot choose action: valid_actions list is empty")

        return self.rng.choice(valid_actions)

    def reset(self):
        """
        Reset the random number generator to initial seed (if provided).
        This ensures reproducibility across multiple games when a seed is set.
        """
        if self.seed is not None:
            self.rng = random.Random(self.seed)
