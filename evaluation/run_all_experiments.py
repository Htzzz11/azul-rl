"""
Experiment runner: defines all agent matchups and runs them via SimulationManager.

Usage:
    python -m evaluation.run_all_experiments              # run all experiments
    python -m evaluation.run_all_experiments --only 0 2   # run experiments 0 and 2 only
    python -m evaluation.run_all_experiments --list        # list all experiments
"""

import argparse
import os

from agents.random_agent import RandomAgent, RandomAgentHierarchical
from agents.minimax_agent import MinimaxAgent
from agents.mcts_agent import MCTSAgent
from evaluation.simulation_manager import SimulationManager

# Resolve paths relative to project root, not the working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# EXPERIMENT CONFIGURATION
#
# Each experiment defines:
#   - agent1, agent2: agent instances
#   - games: number of games to play
#   - output: CSV output path
# ============================================================

EXPERIMENTS = [
    # Control: Random vs Random
    {
        "name": "Random vs Random",
        "agent1": RandomAgent(name="Random_1", seed=42),
        "agent2": RandomAgent(name="Random_2", seed=99),
        "games": 100,
        "output": "results/random_vs_random.csv",
    },

    # Control: Hierarchical Random vs Flat Random
    {
        "name": "Flat Random vs Hierarchical Random",
        "agent1": RandomAgent(name="Flat", seed=42),
        "agent2": RandomAgentHierarchical(name="Hierarchical", seed=99),
        "games": 100,
        "output": "results/flat_vs_hierarchical.csv",
    },

    # Minimax vs Random
    {
        "name": "Minimax vs Random",
        "agent1": MinimaxAgent(name="Minimax"),
        "agent2": RandomAgent(name="Random", seed=42),
        "games": 100,
        "output": "results/minimax_vs_random.csv",
    },

    # MCTS(200) vs Random
    {
        "name": "MCTS(200) vs Random",
        "agent1": MCTSAgent(n_simulations=200, rollout_depth=60, name="MCTS_200", seed=42),
        "agent2": RandomAgent(name="Random", seed=99),
        "games": 50,
        "output": "results/mcts200_vs_random.csv",
    },

    # MCTS(200) vs Minimax
    {
        "name": "MCTS(200) vs Minimax",
        "agent1": MCTSAgent(n_simulations=200, rollout_depth=60, name="MCTS_200", seed=42),
        "agent2": MinimaxAgent(name="Minimax"),
        "games": 50,
        "output": "results/mcts200_vs_minimax.csv",
    },

    # Minimax vs Minimax (control - test first-player advantage)
    {
        "name": "Minimax vs Minimax",
        "agent1": MinimaxAgent(name="Minimax_1"),
        "agent2": MinimaxAgent(name="Minimax_2"),
        "games": 100,
        "output": "results/minimax_vs_minimax.csv",
    },
]

SEED = 42


def list_experiments():
    print("Available experiments:")
    print("-" * 60)
    for i, exp in enumerate(EXPERIMENTS):
        print(f"  [{i}] {exp['name']} ({exp['games']} games) -> {exp['output']}")
    print()


def run_experiment(idx):
    exp = EXPERIMENTS[idx]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {idx}: {exp['name']}")
    print(f"{'='*60}\n")

    manager = SimulationManager(
        agent1=exp['agent1'],
        agent2=exp['agent2'],
        num_games=exp['games'],
        seed=SEED,
        rotate_order=True
    )

    manager.run_batch(verbose=True)
    output_path = os.path.join(PROJECT_ROOT, exp['output'])
    manager.export_results(output_path)


def main():
    parser = argparse.ArgumentParser(description="Run agent comparison experiments")
    parser.add_argument('--only', type=int, nargs='+',
                        help="Run only these experiment indices")
    parser.add_argument('--list', action='store_true',
                        help="List all experiments and exit")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if args.only:
        indices = args.only
    else:
        indices = range(len(EXPERIMENTS))

    list_experiments()

    for idx in indices:
        if idx < 0 or idx >= len(EXPERIMENTS):
            print(f"Skipping invalid experiment index: {idx}")
            continue
        run_experiment(idx)

    print(f"\nAll experiments complete.")


if __name__ == "__main__":
    main()
