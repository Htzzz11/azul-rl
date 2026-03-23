"""
Training script for the DQN Azul agent.

Supports curriculum learning: Random -> Self-play -> Minimax opponents.
Logs episode rewards, win-rate benchmarks, and saves checkpoints.

Usage:
    python -m scripts.train_dqn --episodes 10000 --lr 1e-4
    python -m scripts.train_dqn --episodes 10000 --curriculum   # curriculum training
"""

import argparse
import gc
import os
import random
import time

import numpy as np

from azul_marl_env import azul_v1_2players
from agents.dqn_agent import (
    DQNAgent, encode_state, action_to_index, ACTION_SPACE_SIZE,
)
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent


# ── reward shaping ──────────────────────────────────────────────────────────
def shaped_reward(obs_before: dict, obs_after: dict, player_index: int,
                  action: tuple, env_reward: float) -> float:
    """
    Augment sparse environment reward with small shaping bonuses:
      +0.1  per tile placed on a pattern line (tiles_to_floor == 0)
      -0.1  per tile sent to floor  (tiles_to_floor > 0)
      +env_reward  (score delta at end of round)
    """
    factory_idx, tile_color, tiles_to_floor, pattern_line = action
    shaped = env_reward

    if tiles_to_floor == 0:
        shaped += 0.1
    else:
        shaped -= 0.1 * int(tiles_to_floor)

    return shaped


# ── single game ─────────────────────────────────────────────────────────────
def play_training_game(dqn_agent: DQNAgent, opponent,
                       dqn_player_idx: int, seed: int = None,
                       use_shaping: bool = True,
                       store_transitions: bool = True):
    """
    Play one full game, optionally storing transitions for the DQN agent.

    Returns:
        (dqn_score, opp_score, total_reward)  where total_reward is the sum
        of (shaped) rewards collected by the DQN agent during the game.
    """
    env = azul_v1_2players()

    # Give env access to agents that need it (Minimax depth>=2)
    if hasattr(opponent, 'set_env'):
        opponent.set_env(env)

    obs, info = env.reset(seed=seed)

    agents = {dqn_player_idx: dqn_agent,
              1 - dqn_player_idx: opponent}

    # Track state for DQN transitions
    prev_state = None
    prev_action_idx = None
    prev_action = None
    total_reward = 0.0

    for agent_name in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            # Terminal transition: use raw reward only (no shaping bonus)
            if prev_state is not None:
                state_vec = encode_state(obs, dqn_player_idx)
                r = reward
                if store_transitions:
                    dqn_agent.replay_buffer.push(
                        prev_state, prev_action_idx, r, state_vec, True)
                total_reward += r
            break

        agent_idx = int(agent_name.split('_')[1])
        valid_moves = info.get('valid_moves', [])
        if not valid_moves:
            break

        current_agent = agents[agent_idx]
        action = current_agent.choose_action(obs, agent_idx, valid_moves)

        if agent_idx == dqn_player_idx:
            state_vec = encode_state(obs, dqn_player_idx)
            action_idx = action_to_index(action)

            # Store previous transition (reward comes one step later)
            # Use prev_action for shaping — it's the action that generated this reward
            if prev_state is not None:
                r = shaped_reward({}, obs, dqn_player_idx, prev_action,
                                  reward) if use_shaping else reward
                if store_transitions:
                    dqn_agent.replay_buffer.push(
                        prev_state, prev_action_idx, r, state_vec, False)
                total_reward += r

            prev_state = state_vec
            prev_action_idx = action_idx
            prev_action = action

        env.step(action)

    # Extract final scores
    final_state = env.state
    scores = [p['score'] for p in final_state['players']]
    env.close()

    return scores[dqn_player_idx], scores[1 - dqn_player_idx], total_reward


# ── evaluation ──────────────────────────────────────────────────────────────
def evaluate_vs(dqn_agent: DQNAgent, opponent, n_games: int = 20,
                seed: int = 9999) -> float:
    """
    Win-rate benchmark (epsilon=0 for DQN). Returns win rate in [0, 1].
    """
    old_eps = dqn_agent.epsilon
    dqn_agent.epsilon = 0.0

    wins = 0
    for i in range(n_games):
        dqn_idx = i % 2
        s_dqn, s_opp, _ = play_training_game(
            dqn_agent, opponent, dqn_player_idx=dqn_idx,
            seed=seed + i, use_shaping=False, store_transitions=False)
        if s_dqn > s_opp:
            wins += 1

    dqn_agent.epsilon = old_eps
    return wins / n_games


# ── curriculum opponent selection ────────────────────────────────────────────
def get_opponent(ep: int, total_episodes: int, curriculum: bool,
                 dqn_agent: DQNAgent,
                 random_opp: RandomAgent, minimax_opp: MinimaxAgent):
    """
    Return the opponent for this episode.

    Without curriculum: always RandomAgent.
    With curriculum:
      Phase 1 (first 40%):  RandomAgent
      Phase 2 (40%-70%):    self-play (copy of DQN with current epsilon)
      Phase 3 (70%-100%):   MinimaxAgent(depth=1)
    """
    if not curriculum:
        return random_opp, "Random"

    progress = ep / total_episodes
    if progress <= 0.4:
        return random_opp, "Random"
    elif progress <= 0.7:
        # Self-play: use a snapshot of the agent itself
        return dqn_agent, "Self"
    else:
        return minimax_opp, "Minimax"


# ── main training loop ──────────────────────────────────────────────────────
def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    dqn = DQNAgent(
        name='DQNTrainee',
        epsilon=args.epsilon_start,
        lr=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
    )

    # Resume from checkpoint if provided
    if args.resume:
        dqn.load(args.resume)
        print(f"Resumed from {args.resume} (ep={dqn.episodes_done}, "
              f"steps={dqn.train_steps})")

    random_opp = RandomAgent(name='TrainRandom')
    minimax_opp = MinimaxAgent(name='TrainMinimax', depth=1)

    eps_decay = (args.epsilon_start - args.epsilon_end) / max(args.episodes, 1)

    print(f"Training DQN for {args.episodes} episodes")
    print(f"  lr={args.lr}  gamma={args.gamma}  batch={args.batch_size}")
    print(f"  epsilon: {args.epsilon_start} -> {args.epsilon_end}")
    print(f"  curriculum: {args.curriculum}")
    print(f"  device: {dqn.device}")
    print()

    rewards_log = []
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        # Select opponent
        opponent, opp_label = get_opponent(
            ep, args.episodes, args.curriculum,
            dqn, random_opp, minimax_opp,
        )

        # Alternate sides each episode
        dqn_idx = ep % 2
        game_seed = args.seed + ep if args.seed is not None else None

        dqn_score, opp_score, ep_reward = play_training_game(
            dqn, opponent, dqn_player_idx=dqn_idx,
            seed=game_seed, use_shaping=True)

        rewards_log.append(ep_reward)

        # Gradient updates (multiple per episode to use data efficiently)
        n_updates = max(1, min(4, len(dqn.replay_buffer) // args.batch_size))
        losses = []
        for _ in range(n_updates):
            loss = dqn.update()
            if loss is not None:
                losses.append(loss)

        # Epsilon decay
        dqn.epsilon = max(args.epsilon_end,
                          dqn.epsilon - eps_decay)
        dqn.episodes_done = ep

        # Logging
        if ep % args.log_every == 0 or ep == 1:
            avg_r = np.mean(rewards_log[-args.log_every:])
            avg_loss = np.mean(losses) if losses else 0.0
            elapsed = time.time() - start_time
            print(f"Ep {ep:>5}/{args.episodes} | "
                  f"vs={opp_label:<7} | "
                  f"eps={dqn.epsilon:.3f} | "
                  f"avg_r={avg_r:+.2f} | "
                  f"loss={avg_loss:.4f} | "
                  f"buf={len(dqn.replay_buffer)} | "
                  f"score={dqn_score}-{opp_score} | "
                  f"{elapsed:.0f}s")

        # Evaluation benchmark
        if ep % args.eval_every == 0:
            wr_rand = evaluate_vs(dqn, RandomAgent(name='EvalRandom'),
                                  n_games=20, seed=7777 + ep)
            wr_mini = evaluate_vs(dqn, MinimaxAgent(name='EvalMinimax'),
                                  n_games=20, seed=8888 + ep)
            print(f"  >>> Win rate: vs Random {wr_rand:.0%} | "
                  f"vs Minimax {wr_mini:.0%}")

        # Periodic garbage collection
        if ep % 100 == 0:
            gc.collect()

        # Checkpoint
        if ep % args.save_every == 0:
            path = os.path.join(args.save_dir, f"dqn_ep{ep}.pt")
            dqn.save(path)
            print(f"  Saved checkpoint: {path}")

    # Final save
    final_path = os.path.join(args.save_dir, "dqn_final.pt")
    dqn.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")

    # Final evaluation
    wr_rand = evaluate_vs(dqn, RandomAgent(name='EvalRandom'),
                          n_games=20, seed=12345)
    wr_mini = evaluate_vs(dqn, MinimaxAgent(name='EvalMinimax'),
                          n_games=20, seed=12345)
    print(f"Final win rate: vs Random {wr_rand:.0%} | vs Minimax {wr_mini:.0%}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Azul")
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer-capacity', type=int, default=50000)
    parser.add_argument('--target-update', type=int, default=50,
                        help='Sync target network every N training steps')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum: Random -> Self-play -> Minimax')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a checkpoint file')
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
