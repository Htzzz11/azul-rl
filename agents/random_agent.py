"""
Random agent implementation for Azul.
"""

import random
from typing import List, Tuple
from collections import defaultdict
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Flat random agent - uniformly samples from all valid actions.
    This can create bias toward sources with more colors (typically center in late game).
    """
    def __init__(self, name: str = None, seed: int = None):
        super().__init__(name)
        self.seed = seed
        self.rng = random.Random(seed)

    def choose_action(self, observation: dict, player_index: int, valid_actions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
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


class RandomAgentHierarchical(BaseAgent):
    """
    Hierarchical random agent - first randomly selects a factory/center,
    then randomly selects an action from that source.
    This creates uniform probability over sources regardless of how many colors each has.
    """
    def __init__(self, name: str = None, seed: int = None):
        super().__init__(name)
        self.seed = seed
        self.rng = random.Random(seed)

        # Statistics tracking
        self.center_selections = 0
        self.factory_selections = 0
        self.total_actions = 0

    def choose_action(self, observation: dict, player_index: int, valid_actions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        if not valid_actions:
            raise ValueError("Cannot choose action: valid_actions list is empty")

        # Group actions by factory index (0 = center, 1+ = factories)
        actions_by_factory = defaultdict(list)
        for action in valid_actions:
            factory_idx = action[0]
            actions_by_factory[factory_idx].append(action)

        # Randomly choose a factory/center (uniform over available sources)
        available_sources = list(actions_by_factory.keys())
        chosen_factory = self.rng.choice(available_sources)

        # Track statistics
        self.total_actions += 1
        if chosen_factory == 0:
            self.center_selections += 1
        else:
            self.factory_selections += 1

        # Randomly choose an action from that factory/center
        return self.rng.choice(actions_by_factory[chosen_factory])

    def reset(self):
        """
        Reset the random number generator to initial seed (if provided).
        This ensures reproducibility across multiple games when a seed is set.
        """
        if self.seed is not None:
            self.rng = random.Random(self.seed)

        # Reset statistics
        self.center_selections = 0
        self.factory_selections = 0
        self.total_actions = 0

    def get_selection_stats(self):
        """Return statistics about center vs factory selection rates."""
        if self.total_actions == 0:
            return {"center_rate": 0.0, "factory_rate": 0.0}

        return {
            "center_rate": self.center_selections / self.total_actions,
            "factory_rate": self.factory_selections / self.total_actions,
            "total_actions": self.total_actions
        }
