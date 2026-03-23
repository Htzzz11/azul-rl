"""
Minimax agent with alpha-beta pruning for Azul.

At depth 1 (default), uses a fast heuristic evaluation on the current
observation without cloning the environment.  At depth >= 2, requires
access to the live environment via ``set_env(env)`` so it can clone and
simulate future moves with alpha-beta search.
"""

import copy
import random
from typing import List, Tuple, Optional

import numpy as np

from agents.base_agent import BaseAgent


class MinimaxAgent(BaseAgent):
    def __init__(self, depth: int = 1, name: str = None):
        super().__init__(name)
        self.depth = depth
        self.env = None

    # ── environment access (only needed for depth >= 2) ──────────────

    def set_env(self, env):
        """Store a reference to the live environment for deep search."""
        self.env = env

    def reset(self):
        self.env = None

    # ── public interface ─────────────────────────────────────────────

    def choose_action(self, observation: dict, player_index: int,
                      valid_actions: List[Tuple[int, int, int, int]]
                      ) -> Tuple[int, int, int, int]:
        if not valid_actions:
            raise ValueError("valid_actions is empty")

        if len(valid_actions) == 1:
            return valid_actions[0]

        # Fast path: depth-1 uses only the heuristic (no env clone needed)
        if self.depth <= 1:
            return self._choose_greedy(observation, player_index, valid_actions)

        # Deep search requires environment access
        if self.env is None:
            raise RuntimeError(
                "MinimaxAgent(depth>=2) requires the environment. "
                "Call agent.set_env(env) before choose_action()."
            )

        return self._choose_alphabeta(observation, player_index, valid_actions)

    # ── depth-1 greedy (original logic, no env needed) ───────────────

    def _choose_greedy(self, observation: dict, player_index: int,
                       valid_actions: list) -> Tuple[int, int, int, int]:
        best_action = None
        best_value = float('-inf')
        for action in valid_actions:
            value = self._evaluate_action(observation, action, player_index)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    # ── alpha-beta search (depth >= 2) ───────────────────────────────

    def _choose_alphabeta(self, observation: dict, player_index: int,
                          valid_actions: list) -> Tuple[int, int, int, int]:
        # Preserve global RNG (game engine uses random module)
        rng_state = random.getstate()

        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in valid_actions:
            env_copy = self._clone_env()
            env_copy.step(action)

            value = self._alphabeta(
                env_copy, self.depth - 1, alpha, beta,
                maximising=False, root_player=player_index,
            )

            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)

        random.setstate(rng_state)
        return best_action

    def _alphabeta(self, env_copy, depth: int, alpha: float, beta: float,
                   maximising: bool, root_player: int) -> float:
        obs, _, term, trunc, info = env_copy.last()

        if term or trunc:
            return self._evaluate_terminal(env_copy.state, root_player)

        valid_moves = info.get('valid_moves', [])
        if not valid_moves:
            return self._evaluate_terminal(env_copy.state, root_player)

        # Leaf: evaluate with heuristic
        if depth <= 0:
            return self._evaluate_state(obs, root_player)

        if maximising:
            value = float('-inf')
            for action in valid_moves:
                child = copy.deepcopy(env_copy)
                child.step(action)
                value = max(value, self._alphabeta(
                    child, depth - 1, alpha, beta,
                    maximising=False, root_player=root_player,
                ))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for action in valid_moves:
                child = copy.deepcopy(env_copy)
                child.step(action)
                value = min(value, self._alphabeta(
                    child, depth - 1, alpha, beta,
                    maximising=True, root_player=root_player,
                ))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # ── evaluation functions ─────────────────────────────────────────

    def _evaluate_terminal(self, state: dict, root_player: int) -> float:
        """Score differential at a terminal (or near-terminal) state."""
        my_score = state['players'][root_player]['score']
        opp_score = state['players'][1 - root_player]['score']
        return float(my_score - opp_score)

    def _evaluate_state(self, observation: dict, root_player: int) -> float:
        """
        Heuristic board evaluation from root_player's perspective.
        Combines score differential with pattern-line progress and wall coverage.
        """
        me = observation['players'][root_player]
        opp = observation['players'][1 - root_player]

        score_diff = float(me['score'] - opp['score'])

        # Pattern-line progress: reward lines close to completion
        my_progress = self._pattern_line_progress(me)
        opp_progress = self._pattern_line_progress(opp)

        # Wall coverage: tiles on wall contribute to adjacency scoring
        my_wall = float(np.sum(np.asarray(me['wall']) != 5))
        opp_wall = float(np.sum(np.asarray(opp['wall']) != 5))

        return (score_diff
                + 2.0 * (my_progress - opp_progress)
                + 0.5 * (my_wall - opp_wall))

    @staticmethod
    def _pattern_line_progress(player_state: dict) -> float:
        """Sum of fill-ratio for each pattern line (higher = closer to scoring)."""
        total = 0.0
        for row_idx in range(5):
            line = player_state['pattern_lines'][row_idx]
            filled = int(np.sum(np.asarray(line) != 5))
            capacity = row_idx + 1
            total += filled / capacity
        return total

    # ── single-action heuristic (used by depth-1 greedy) ────────────

    def _evaluate_action(self, observation: dict,
                         action: Tuple[int, int, int, int],
                         player_index: int) -> float:
        factory_idx, tile_color, tiles_to_floor, pattern_line_idx = action
        my_state = observation['players'][player_index]

        value = 0.0

        # Avoid floor penalties
        value -= tiles_to_floor * 20

        # Prefer taking from factories over center
        if factory_idx == 0:
            value -= 3
        else:
            value += 5

        # Bonus for completing pattern lines
        pattern_line = my_state['pattern_lines'][pattern_line_idx]
        num_tiles_in_line = np.sum(pattern_line != 5)
        max_capacity = pattern_line_idx + 1

        num_tiles_from_factory = self._count_tiles_in_source(
            observation, factory_idx, tile_color
        )
        tiles_to_pattern_line = num_tiles_from_factory - tiles_to_floor

        if num_tiles_in_line + tiles_to_pattern_line == max_capacity:
            value += 30
            value += pattern_line_idx * 5
        elif num_tiles_in_line + tiles_to_pattern_line > max_capacity:
            pass
        else:
            value += tiles_to_pattern_line * 2

        # Prefer higher pattern lines
        value += pattern_line_idx * 1

        # Avoid tiles for colors already on wall
        wall_row = my_state['wall'][pattern_line_idx]
        if tile_color in wall_row[wall_row != 5]:
            value -= 100

        # Prefer taking more tiles (if not wasting)
        if tiles_to_floor == 0:
            value += num_tiles_from_factory * 3

        return value

    @staticmethod
    def _count_tiles_in_source(observation: dict, factory_idx: int,
                               tile_color: int) -> int:
        if factory_idx == 0:
            return int(observation['center'][tile_color])
        else:
            return int(observation['factories'][factory_idx - 1][tile_color])

    # ── env cloning ──────────────────────────────────────────────────

    def _clone_env(self):
        """Deep-copy the environment, stripping the renderer."""
        renderer = self.env._renderer
        self.env._renderer = None
        env_copy = copy.deepcopy(self.env)
        self.env._renderer = renderer
        return env_copy
