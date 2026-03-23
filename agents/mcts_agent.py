"""
MCTS (Monte Carlo Tree Search) agent implementation for Azul.

Uses UCB1 for tree policy and random rollouts for evaluation.
Designed for 2-player games. For 3-4 player games, the opponent-aware
UCB1 selection would need to be reworked.
"""

import copy
import math
import random
from typing import List, Tuple, Optional

from agents.base_agent import BaseAgent


class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, valid_actions: List[Tuple[int, int, int, int]],
                 player_index: int, parent: Optional['MCTSNode'] = None,
                 action_taken: Optional[Tuple[int, int, int, int]] = None):
        self.valid_actions = valid_actions
        self.player_index = player_index
        self.parent = parent
        self.action_taken = action_taken

        self.children = {}  # action_tuple -> MCTSNode
        self.untried_actions = list(valid_actions)
        self.visits = 0
        self.total_value = 0.0  # cumulative value from root player's perspective

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return len(self.valid_actions) == 0

    def ucb_score(self, c: float, root_player: int) -> float:
        """
        Calculate UCB1 score for this node.

        At opponent nodes, the exploitation term is negated so that
        selection favours moves that are best for the opponent
        (worst for the root player).
        """
        if self.visits == 0:
            return float('inf')

        exploitation = self.total_value / self.visits

        # Negate exploitation at opponent nodes
        if self.player_index != root_player:
            exploitation = -exploitation

        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration

    def best_child(self, c: float, root_player: int) -> Tuple[Tuple[int, int, int, int], 'MCTSNode']:
        """Select the child with the highest UCB1 score."""
        best_action = None
        best_node = None
        best_score = float('-inf')

        for action, child in self.children.items():
            score = child.ucb_score(c, root_player)
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child

        return best_action, best_node


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search agent for Azul.

    Uses UCB1 selection, random rollouts, and score differential
    as the value function. Requires access to the environment via
    set_env() for state cloning during search.

    Designed for 2-player games only.
    """

    def __init__(self, n_simulations: int = 100, rollout_depth: int = 50,
                 c: float = 1.41, name: str = None, seed: int = None):
        super().__init__(name)
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.c = c
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = None

    def set_env(self, env):
        """
        Store a reference to the live environment.
        Must be called before choose_action so MCTS can clone the env for simulations.
        """
        self.env = env

    def choose_action(self, observation: dict, player_index: int,
                      valid_actions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        if not valid_actions:
            raise ValueError("Cannot choose action: valid_actions list is empty")

        if self.env is None:
            raise RuntimeError(
                "MCTSAgent requires access to the environment. "
                "Call agent.set_env(env) before choose_action()."
            )

        # Single valid action - no need to search
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Filter out obviously bad actions: floor-dump actions where a
        # placement action exists for the same (factory, color) pair.
        # Floor-dump actions always waste tiles; we only keep them as
        # fallback when no placement exists for that source+color.
        placement_actions, fallback_actions = self._filter_actions(valid_actions)
        search_actions = placement_actions if placement_actions else fallback_actions

        root = MCTSNode(search_actions, player_index)

        # Save global random state before simulations.
        # The game engine (Bag.take_tiles) uses Python's global random module,
        # so MCTS rollouts would corrupt the real game's RNG if not preserved.
        global_rng_state = random.getstate()

        for _ in range(self.n_simulations):
            env_copy = self._clone_env()

            # Selection: walk down tree via UCB1, stepping env_copy
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                action, node = node.best_child(self.c, player_index)
                env_copy.step(action)

            # Expansion: add one new child for an untried action
            if not node.is_terminal():
                node = self._expand(node, env_copy)

            # Simulation: random rollout from current state
            value = self._rollout(env_copy, player_index)

            # Backpropagation: update values up to root
            self._backpropagate(node, value)

        # Restore global random state so the real game is unaffected
        random.setstate(global_rng_state)

        # Return the action with the most visits, breaking ties by avg value
        best_action = max(
            root.children,
            key=lambda a: (
                root.children[a].visits,
                root.children[a].total_value / max(root.children[a].visits, 1)
            )
        )
        return best_action

    @staticmethod
    def _filter_actions(valid_actions):
        """
        Separate placement actions from floor-dump actions.

        Floor-dump actions (tiles_to_floor > 0) are only kept when no
        placement action exists for that (factory, color) pair - i.e.
        the player is forced to dump to floor.
        """
        placement_actions = []
        floor_dump_actions = []

        # Track which (factory, color) pairs have a placement action
        has_placement = set()

        for action in valid_actions:
            factory_idx, tile_color, tiles_to_floor, pattern_line = action
            if tiles_to_floor == 0:
                placement_actions.append(action)
                has_placement.add((factory_idx, tile_color))
            else:
                floor_dump_actions.append(action)

        # Keep floor-dump actions only when forced (no placement exists)
        forced_floor = [
            a for a in floor_dump_actions
            if (a[0], a[1]) not in has_placement
        ]
        placement_actions.extend(forced_floor)

        return placement_actions, floor_dump_actions

    def _clone_env(self):
        """
        Deep copy the environment for simulation.
        Strips the matplotlib renderer to avoid copy issues.
        """
        renderer = self.env._renderer
        self.env._renderer = None
        env_copy = copy.deepcopy(self.env)
        self.env._renderer = renderer
        return env_copy

    def _expand(self, node: MCTSNode, env_copy) -> MCTSNode:
        """Pick an untried action, step the env, create a child node."""
        action = node.untried_actions.pop()
        env_copy.step(action)

        obs, _, term, trunc, info = env_copy.last()
        valid = info.get('valid_moves', []) if not (term or trunc) else []
        next_player = int(env_copy.agent_selection.split('_')[1])

        child = MCTSNode(
            valid_actions=valid,
            player_index=next_player,
            parent=node,
            action_taken=action
        )
        node.children[action] = child
        return child

    def _rollout(self, env_copy, root_player: int) -> float:
        """
        Random rollout from the current env state for rollout_depth steps.
        Returns score differential from root player's perspective.
        """
        for _ in range(self.rollout_depth):
            obs, _, term, trunc, info = env_copy.last()
            if term or trunc:
                break

            valid = info.get('valid_moves', [])
            if not valid:
                break

            action = self.rng.choice(valid)
            env_copy.step(action)

        # Value = score differential from root player's perspective
        state = env_copy.state
        my_score = state['players'][root_player]['score']
        opp_scores = [
            state['players'][i]['score']
            for i in range(len(state['players']))
            if i != root_player
        ]
        return my_score - max(opp_scores)

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagate the rollout value up to the root."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def reset(self):
        """Reset agent state between games."""
        if self.seed is not None:
            self.rng = random.Random(self.seed)
        self.env = None
