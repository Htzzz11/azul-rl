from typing import Dict

import numpy as np
from azul_game_engine.board import Board
from azul_game_engine.center import Center
from azul_game_engine.floor import Floor
from azul_game_engine.game import Game
from azul_game_engine.lid import Lid
from azul_game_engine.player import Player
from azul_game_engine.tile import Tile
from azul_game_engine.wall import Wall
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector


class AzulEnv(AECEnv):
    metadata = {
        "name": "azul_env_v1"
    }
    def __init__(self, player_count=2, max_moves=None):
        super(AzulEnv, self).__init__()
        self.player_count = player_count
        self.agents = [f"player_{i}" for i in range(player_count)]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.factories = 1 + 2 * player_count
        self.game = None
        self.state = None
        self.current_move = 0
        self.max_moves = player_count * 150 if max_moves is None else max_moves
        
        # Renderer is created lazily on first render() call
        self._renderer = None

        self.observation_spaces: Dict[str, spaces.Space] = {
            agent: spaces.Dict({
                "factories": spaces.Box(low=0, high=4, shape=(self.factories, 5), dtype=np.int32),
                "center": spaces.Box(low=0, high=3 * self.factories, shape=(5,), dtype=np.int32),
                "players": spaces.Tuple([
                    spaces.Dict({
                        "pattern_lines": spaces.Box(low=0, high=5, shape=(5, 5), dtype=np.int32),
                        "wall": spaces.Box(low=0, high=5, shape=(5, 5), dtype=np.int32),
                        "floor": spaces.Box(low=0, high=5, shape=(7,), dtype=np.int32),
                        # high is 5 instead of 4 because of first player marker.
                        "is_starting": spaces.Discrete(2),
                        "score": spaces.Discrete(241)
                    }) for _ in range(player_count)
                ]),
                "bag": spaces.Box(low=0, high=100, shape=(5,), dtype=np.int32),
                "lid": spaces.Box(low=0, high=100, shape=(5,), dtype=np.int32)
            }) for agent in self.agents
        }

        self.action_spaces: Dict[str, spaces.Space] = {
            agent: spaces.MultiDiscrete([
                self.factories + 1,  # factory index (0 is center)
                5,  # tile color (0-4)
                20,  # tiles to place on the floor
                5  # pattern line index (0-4)
            ]) for agent in self.agents
        }

        self.reset()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.game = self._create_game()
        self._set_state()
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self._agent_selector.reset()
        
        valid_moves = self._get_all_valid_moves()
        self.infos[self.agent_selection] = {"valid_moves": valid_moves}
        
        return self.state, {"valid_moves": valid_moves}
    
    def step(self, action):
        factory_index, tile_to_take_number, tiles_to_place_floor, pattern_line_index = action
        tile_to_take = Tile(self.__number_to_tile__(tile_to_take_number))

        # Get valid moves for current player
        valid_moves = self._get_all_valid_moves()
        
        # Check if the action is valid
        if not self._is_valid_move(action):
            reward = -2
            self._cumulative_rewards[self.agent_selection] += reward
            self.infos[self.agent_selection] = {"valid_moves": valid_moves}
            return self.state, reward, False, False, {"valid_moves": valid_moves}

        if factory_index == 0:
            self.game.execute_factory_offer_phase_with_center(
                tile_to_take, tiles_to_place_floor, pattern_line_index
            )
        else:
            self.game.execute_factory_offer_phase_with_factory(
                factory_index - 1, tile_to_take, tiles_to_place_floor, pattern_line_index
            )

        self._set_state()
        self.agent_selection = f"player_{self.game.current_player}"
        
        reward = -1
        self._cumulative_rewards[self.agent_selection] += reward

        terminated = not self.game.json_object().get("isRunning")
        if terminated:
            self.terminations = {agent: True for agent in self.agents}
            self._add_score()

        self.current_move += 1
        truncated = self.current_move >= self.max_moves

        if truncated:
            self.truncations = {agent: True for agent in self.agents}
            self._add_score()

        # Get valid moves for the next player and store in infos
        next_valid_moves = self._get_all_valid_moves()
        self.infos[self.agent_selection] = {"valid_moves": next_valid_moves}
        
        return self.state, reward, terminated, truncated, {"valid_moves": next_valid_moves}
    
    def observe(self, agent):
        return self.state

    def render(self): # pragma: no cover
        if self.state is None:
            return
            
        # Get tile counts from the state
        bag_counts = self.state["bag"]
        lid_counts = self.state["lid"]
        center_counts = self.state["center"]
        factories = self.state["factories"]
        
        # Lazily create renderer on first use (deferred import avoids
        # loading matplotlib when rendering is not needed, e.g. training)
        if self._renderer is None:
            from azul_marl_env.render.azul_renderer import AzulRenderer
            self._renderer = AzulRenderer()
        self._renderer.render(self.state, bag_counts, lid_counts, center_counts, factories)

    def close(self): # pragma: no cover
        if self._renderer is not None:
            self._renderer.close()

    @staticmethod
    def __convert_tile_dict_to_array__(tile_dict):
        tile_array = np.zeros(5, dtype=np.int32)
        for tile, count in tile_dict.items():
            tile_index = AzulEnv.__tile_to_number__(tile)
            if tile_index is not None:
                tile_array[tile_index] = count
        return tile_array

    @staticmethod
    def __number_to_tile__(number):
        return {
            0: "B",
            1: "Y",
            2: "R",
            3: "K",
            4: "W"
        }.get(number)

    @staticmethod
    def __tile_to_number__(tile):
        return {
            "B": 0,
            "Y": 1,
            "R": 2,
            "K": 3,
            "W": 4
        }.get(tile)

    def _create_game(self):
        lid = Lid()
        players = []
        for i in range(self.player_count):
            players.append(Player(Board(wall=Wall(), floor=Floor(lid)), f"Player {i + 1}"))
        return Game(players, Center(), 0, lid)

    def _set_state(self):
        game_state = self.game.json_object()
        self.state = {
            "factories": np.array(
                [AzulEnv.__convert_tile_dict_to_array__(factory) for factory in game_state.get("Factory displays")]),
            "center": AzulEnv.__convert_tile_dict_to_array__(game_state.get("Center")),
            "players": [
                {
                    "pattern_lines": np.array(
                        [
                            [AzulEnv.__tile_to_number__(tile) for tile in row] + [5] * (5 - len(row))
                            # Fill empty spots with 5
                            for row in player_state.get("Board").get("Pattern lines")
                        ], dtype=np.int32),
                    "wall": np.array(
                        [[AzulEnv.__tile_to_number__(tile) if tile.isupper() else 5 for tile in row] for row in
                         player_state.get("Board").get("Wall")], dtype=np.int32),
                    "floor": [AzulEnv.__tile_to_number__(tile) if tile != "M" else 5 for tile in
                              player_state.get("Board").get("Floor")],
                    "score": player_state.get("Score")
                }
                for player_state in game_state.get("Players")
            ],
            "bag": AzulEnv.__convert_tile_dict_to_array__(game_state.get("Bag")),
            "lid": AzulEnv.__convert_tile_dict_to_array__(game_state.get("Lid"))
        }

    def _get_current_player_index(self):
        """Get the index of the current player."""
        return self.agents.index(self.agent_selection)

    def _is_pattern_line_full(self, player_state, pattern_line_index):
        """Check if a pattern line is full."""
        pattern_line = player_state["pattern_lines"][pattern_line_index]
        # Count non-empty slots (tiles that are not 5, which represents empty)
        non_empty_count = np.sum(pattern_line != 5)
        return non_empty_count == pattern_line_index + 1

    def _is_pattern_line_empty(self, player_state, pattern_line_index):
        """Check if a pattern line is empty."""
        pattern_line = player_state["pattern_lines"][pattern_line_index]
        return np.all(pattern_line == 5)

    def _get_pattern_line_tile_color(self, player_state, pattern_line_index):
        """Get the tile color in a pattern line (if any)."""
        pattern_line = player_state["pattern_lines"][pattern_line_index]
        non_empty_tiles = pattern_line[pattern_line != 5]
        if len(non_empty_tiles) > 0:
            return non_empty_tiles[0]  # All tiles in a pattern line are the same color
        return None

    def _is_tile_already_on_wall(self, player_state, pattern_line_index, tile_color):
        """Check if a tile color is already placed on the wall at the given row."""
        # Calculate the correct position for this tile color in this row
        # Following the Azul wall pattern: position = (tile_color + row) % 5
        wall_position = (tile_color + pattern_line_index) % 5
        wall_row = player_state["wall"][pattern_line_index]
        # Check if the specific position for this color is already filled (not 5, which means empty)
        return wall_row[wall_position] != 5

    def _get_valid_moves_from_center(self):
        """Get all valid moves when taking from center."""
        valid_moves = []
        current_player_index = self._get_current_player_index()
        player_state = self.state["players"][current_player_index]
        center = self.state["center"]
        
        # For each tile color in center
        for tile_color in range(5):
            if center[tile_color] == 0:
                continue
            # For each pattern line
            for pattern_line_index in range(5):
                valid_moves.append((0, tile_color, center[tile_color], pattern_line_index))
                # Check if wall already has this tile in this row
                if self._is_tile_already_on_wall(player_state, pattern_line_index, tile_color):
                    continue
                
                # Check if pattern line is full
                if self._is_pattern_line_full(player_state, pattern_line_index):
                    continue
                
                # Check if pattern line is empty
                if self._is_pattern_line_empty(player_state, pattern_line_index):
                    # Valid move: take from center (factory_index=0), tile_color, 0 tiles to floor, pattern_line_index
                    valid_moves.append((0, tile_color, 0, pattern_line_index))
                    continue
                
                # Check if color on pattern line matches current tile
                existing_color = self._get_pattern_line_tile_color(player_state, pattern_line_index)
                if existing_color != tile_color:
                    continue
                
                # Valid move
                valid_moves.append((0, tile_color, 0, pattern_line_index))
        
        return valid_moves

    def _get_valid_moves_from_factory(self, factory_index):
        """Get all valid moves when taking from a specific factory."""
        valid_moves = []
        current_player_index = self._get_current_player_index()
        player_state = self.state["players"][current_player_index]
        factory = self.state["factories"][factory_index]
        
        # For each tile color in factory
        for tile_color in range(5):
            if factory[tile_color] == 0:
                continue
            # For each pattern line
            for pattern_line_index in range(5):
                valid_moves.append((factory_index + 1, tile_color, factory[tile_color], pattern_line_index))
                # Check if wall already has this tile in this row
                if self._is_tile_already_on_wall(player_state, pattern_line_index, tile_color):
                    continue
                
                # Check if pattern line is full
                if self._is_pattern_line_full(player_state, pattern_line_index):
                    continue
                
                # Check if pattern line is empty
                if self._is_pattern_line_empty(player_state, pattern_line_index):
                    # Valid move: take from factory (factory_index+1), tile_color, 0 tiles to floor, pattern_line_index
                    valid_moves.append((factory_index + 1, tile_color, 0, pattern_line_index))
                    continue
                
                # Check if color on pattern line matches current tile
                existing_color = self._get_pattern_line_tile_color(player_state, pattern_line_index)
                if existing_color != tile_color:
                    continue
                
                # Valid move
                valid_moves.append((factory_index + 1, tile_color, 0, pattern_line_index))
        
        return valid_moves

    def _get_all_valid_moves(self):
        """Get all valid moves for the current player."""
        valid_moves = self._get_valid_moves_from_center()
        
        # Get valid moves from each factory
        for factory_index in range(len(self.state["factories"])):
            valid_moves.extend(self._get_valid_moves_from_factory(factory_index))
        
        return valid_moves

    def _is_valid_move(self, action):
        """Check if an action is valid."""
        # Convert action to tuple if it's a numpy array
        if isinstance(action, np.ndarray):
            action_tuple = tuple(action)
        else:
            action_tuple = action
        return action_tuple in self._get_all_valid_moves()
    
    def _add_score(self):
        for i, a in enumerate(self.agents):
            self._cumulative_rewards[a] += self.state["players"][i]["score"]
