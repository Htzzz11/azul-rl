"""
Minimax agent implementation with action heuristics.
"""

import numpy as np
from typing import List, Tuple
from agents.base_agent import BaseAgent


class MinimaxAgent(BaseAgent):
    def __init__(self, depth: int = 1, name: str = None):
        super().__init__(name)
        self.depth = depth

    def choose_action(self, observation: dict, player_index: int,
                     valid_actions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Choose the action with the highest heuristic value.
        """
        best_action = None
        best_value = float('-inf')

        for action in valid_actions:
            value = self.evaluate_action(observation, action, player_index)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def evaluate_action(self, observation: dict, action: Tuple[int, int, int, int],
                       player_index: int) -> float:
        factory_idx, tile_color, tiles_to_floor, pattern_line_idx = action
        my_state = observation['players'][player_index]

        value = 0.0

        #Avoid Floor Penalties
        value -= tiles_to_floor * 20

        #Prefer Taking from Factories
        if factory_idx == 0:  # Taking from center
            value -= 3
        else:  # Taking from factory
            value += 5

        #Bonus for Completing Pattern Lines
        pattern_line = my_state['pattern_lines'][pattern_line_idx]
        num_tiles_in_line = np.sum(pattern_line != 5)  # 5 = empty
        max_capacity = pattern_line_idx + 1  # Row 0 holds 1, row 1 holds 2, etc.

        # Calculate how many tiles we're placing (total taken - floor)
        num_tiles_from_factory = self._count_tiles_in_source(
            observation, factory_idx, tile_color
        )
        tiles_to_pattern_line = num_tiles_from_factory - tiles_to_floor

        # Will this complete the line?
        if num_tiles_in_line + tiles_to_pattern_line == max_capacity:
            # Big bonus for completing a line!
            value += 30
            # Bigger lines give more points
            value += pattern_line_idx * 5
        elif num_tiles_in_line + tiles_to_pattern_line > max_capacity:
            # Overflow - some tiles go to floor (already penalized above)
            pass
        else:
            # Partial fill - still good progress
            value += tiles_to_pattern_line * 2

        # Prefer Higher Pattern Lines
        value += pattern_line_idx * 1

        # Avoid Taking Tiles for Colors Already on Wall
        wall_row = my_state['wall'][pattern_line_idx]
        if tile_color in wall_row[wall_row != 5]:  # 5 = empty
            # This action is illegal or useless - penalize heavily
            value -= 100

        # Prefer Taking More Tiles
        if tiles_to_floor == 0:  # Only bonus if not wasting them
            value += num_tiles_from_factory * 3

        return value

    def _count_tiles_in_source(self, observation: dict, factory_idx: int,
                               tile_color: int) -> int:
        """
        Count how many tiles of a given color are in the source.
        """
        if factory_idx == 0:  # Center
            return int(observation['center'][tile_color])
        else:  # Factory (1-indexed, so subtract 1 for array access)
            factory_array_idx = factory_idx - 1
            return int(observation['factories'][factory_array_idx][tile_color])

