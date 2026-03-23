"""
Smoke test for MCTS agent.
Supports running a single game or a small batch via command line args.

Usage:
    python scripts/test_mcts.py                      # single game, 10 sims
    python scripts/test_mcts.py --sims 50            # single game, 50 sims
    python scripts/test_mcts.py --games 5 --sims 20  # 5 games, 20 sims
    python scripts/test_mcts.py --depth 20           # custom rollout depth
"""

import argparse
import time

from azul_marl_env import azul_v1_2players
from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent


def play_single_game(agent1, agent2, seed=None):
    """Play a single game between two agents."""
    env = azul_v1_2players()

    agents = [agent1, agent2]

    for agent in agents:
        if hasattr(agent, 'set_env'):
            agent.set_env(env)

    observation, info = env.reset(seed=seed)

    move_count = 0

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            break

        agent_idx = int(agent_name.split('_')[1])
        current_agent = agents[agent_idx]

        valid_moves = info.get('valid_moves', [])
        if not valid_moves:
            break

        action = current_agent.choose_action(observation, agent_idx, valid_moves)
        env.step(action)
        move_count += 1

    final_state = env.state
    player_scores = [player['score'] for player in final_state['players']]

    env.close()

    return {
        'scores': player_scores,
        'winner': player_scores.index(max(player_scores)),
        'moves': move_count
    }


def main():
    parser = argparse.ArgumentParser(description="MCTS agent smoke test")
    parser.add_argument('--games', type=int, default=1, help="Number of games to play (default: 1)")
    parser.add_argument('--sims', type=int, default=10, help="MCTS simulations per move (default: 10)")
    parser.add_argument('--depth', type=int, default=30, help="Rollout depth (default: 30)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    num_games = args.games
    n_sims = args.sims
    depth = args.depth
    seed = args.seed

    print(f"MCTS vs Random | {num_games} game(s) | {n_sims} simulations | depth {depth}")
    print("="*60)

    mcts = MCTSAgent(n_simulations=n_sims, rollout_depth=depth, name="MCTS", seed=seed)
    rand = RandomAgent(name="Random", seed=seed + 1)

    wins = [0, 0]
    total_scores = [0, 0]
    total_time = 0

    for i in range(num_games):
        mcts.reset()
        rand.reset()

        start = time.time()
        result = play_single_game(mcts, rand, seed=seed + i)
        elapsed = time.time() - start
        total_time += elapsed

        if result['scores'][0] > result['scores'][1]:
            wins[0] += 1
        elif result['scores'][1] > result['scores'][0]:
            wins[1] += 1

        total_scores[0] += result['scores'][0]
        total_scores[1] += result['scores'][1]

        print(f"  Game {i+1}: MCTS {result['scores'][0]} - {result['scores'][1]} Random "
              f"({'MCTS wins' if result['scores'][0] > result['scores'][1] else 'Random wins'}) "
              f"[{elapsed:.1f}s, {result['moves']} moves]")

    if num_games > 1:
        print()
        print(f"  MCTS wins: {wins[0]}/{num_games}")
        print(f"  Random wins: {wins[1]}/{num_games}")
        print(f"  MCTS avg score: {total_scores[0]/num_games:.1f}")
        print(f"  Random avg score: {total_scores[1]/num_games:.1f}")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/num_games:.1f}s per game)")


if __name__ == "__main__":
    main()
