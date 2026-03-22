"""
SimulationManager: orchestrates batch games between agents and records results to CSV.

Handles player order rotation to control for first-player bias,
automatic set_env for MCTS agents, and structured CSV output.
"""

import csv
import os
import time
from azul_marl_env import azul_v1_2players


class SimulationManager:
    """
    Runs batch experiments between two agents.

    For each game, records: scores, winner, starting player, move count,
    and agent descriptions. Alternates player order across games to
    control for first-player bias.
    """

    CSV_FIELDS = [
        'game_id', 'seed', 'agent1_type', 'agent2_type',
        'agent1_score', 'agent2_score', 'winner',
        'starting_player', 'move_count', 'game_completed',
        'time_seconds'
    ]

    def __init__(self, agent1, agent2, num_games, seed=42, rotate_order=True):
        """
        Args:
            agent1: First agent instance.
            agent2: Second agent instance.
            num_games: Number of games to play.
            seed: Base seed for reproducibility.
            rotate_order: If True, alternate which agent goes first each game.
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.num_games = num_games
        self.seed = seed
        self.rotate_order = rotate_order
        self.results = []

    def _agent_description(self, agent):
        """Generate a description string for an agent including its parameters."""
        name = agent.__class__.__name__
        params = []

        if hasattr(agent, 'n_simulations'):
            params.append(f"sims={agent.n_simulations}")
        if hasattr(agent, 'rollout_depth'):
            params.append(f"depth={agent.rollout_depth}")
        if hasattr(agent, 'depth') and agent.__class__.__name__ == 'MinimaxAgent':
            params.append(f"depth={agent.depth}")
        if hasattr(agent, 'seed') and agent.seed is not None:
            params.append(f"seed={agent.seed}")

        if params:
            return f"{name}({', '.join(params)})"
        return name

    def _play_single_game(self, game_id, game_seed, swap_order):
        """
        Play a single game between two agents.

        Args:
            game_id: Sequential game number.
            game_seed: Seed for this game.
            swap_order: If True, agent2 plays as player_0 and agent1 as player_1.

        Returns:
            dict with game results.
        """
        env = azul_v1_2players()

        if swap_order:
            agents = [self.agent2, self.agent1]
            starting_player = 'agent2'
        else:
            agents = [self.agent1, self.agent2]
            starting_player = 'agent1'

        # Give MCTS agents access to the environment
        for agent in agents:
            if hasattr(agent, 'set_env'):
                agent.set_env(env)

        observation, info = env.reset(seed=game_seed)

        move_count = 0
        game_completed = False

        start_time = time.time()

        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                game_completed = termination
                break

            agent_idx = int(agent_name.split('_')[1])
            current_agent = agents[agent_idx]

            valid_moves = info.get('valid_moves', [])
            if not valid_moves:
                break

            action = current_agent.choose_action(observation, agent_idx, valid_moves)
            env.step(action)
            move_count += 1

        elapsed = time.time() - start_time

        # Extract final scores
        final_state = env.state
        player_scores = [player['score'] for player in final_state['players']]

        env.close()

        # Map player_0/player_1 scores back to agent1/agent2
        if swap_order:
            agent1_score = player_scores[1]
            agent2_score = player_scores[0]
        else:
            agent1_score = player_scores[0]
            agent2_score = player_scores[1]

        # Determine winner
        if agent1_score > agent2_score:
            winner = 'agent1'
        elif agent2_score > agent1_score:
            winner = 'agent2'
        else:
            winner = 'tie'

        return {
            'game_id': game_id,
            'seed': game_seed,
            'agent1_type': self._agent_description(self.agent1),
            'agent2_type': self._agent_description(self.agent2),
            'agent1_score': agent1_score,
            'agent2_score': agent2_score,
            'winner': winner,
            'starting_player': starting_player,
            'move_count': move_count,
            'game_completed': game_completed,
            'time_seconds': round(elapsed, 2)
        }

    def run_batch(self, verbose=True):
        """
        Run all games and collect results.

        Args:
            verbose: If True, print progress updates.

        Returns:
            List of result dicts.
        """
        self.results = []

        if verbose:
            print(f"Running {self.num_games} games: "
                  f"{self._agent_description(self.agent1)} vs "
                  f"{self._agent_description(self.agent2)}")
            if self.rotate_order:
                print(f"  Player order: alternating (first-player bias control)")
            print()

        for i in range(self.num_games):
            game_seed = self.seed + i if self.seed is not None else None
            swap_order = self.rotate_order and (i % 2 == 1)

            result = self._play_single_game(
                game_id=i + 1,
                game_seed=game_seed,
                swap_order=swap_order
            )
            self.results.append(result)

            if verbose and ((i + 1) % 10 == 0 or i == 0):
                wins1 = sum(1 for r in self.results if r['winner'] == 'agent1')
                wins2 = sum(1 for r in self.results if r['winner'] == 'agent2')
                ties = sum(1 for r in self.results if r['winner'] == 'tie')
                avg_time = sum(r['time_seconds'] for r in self.results) / len(self.results)
                print(f"  Game {i+1}/{self.num_games} | "
                      f"agent1 wins: {wins1}, agent2 wins: {wins2}, ties: {ties} | "
                      f"avg {avg_time:.1f}s/game")

        if verbose:
            self._print_summary()

        return self.results

    def _print_summary(self):
        """Print summary statistics after a batch."""
        if not self.results:
            return

        n = len(self.results)
        wins1 = sum(1 for r in self.results if r['winner'] == 'agent1')
        wins2 = sum(1 for r in self.results if r['winner'] == 'agent2')
        ties = sum(1 for r in self.results if r['winner'] == 'tie')
        avg_score1 = sum(r['agent1_score'] for r in self.results) / n
        avg_score2 = sum(r['agent2_score'] for r in self.results) / n
        total_time = sum(r['time_seconds'] for r in self.results)

        print()
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"  {self._agent_description(self.agent1)}:")
        print(f"    Wins: {wins1}/{n} ({wins1/n*100:.1f}%) | Avg score: {avg_score1:.1f}")
        print(f"  {self._agent_description(self.agent2)}:")
        print(f"    Wins: {wins2}/{n} ({wins2/n*100:.1f}%) | Avg score: {avg_score2:.1f}")
        print(f"  Ties: {ties}/{n} ({ties/n*100:.1f}%)")
        print(f"  Total time: {total_time:.1f}s ({total_time/n:.1f}s per game)")
        print("=" * 60)

    def export_results(self, filepath):
        """
        Export results to CSV.

        Args:
            filepath: Path to output CSV file.
        """
        if not self.results:
            print("No results to export. Run run_batch() first.")
            return

        # Create parent directories if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results exported to {filepath} ({len(self.results)} games)")
