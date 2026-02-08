from azul_marl_env import azul_v1_2players
from agents.random_agent import RandomAgent


def play_single_game(render=False, seed=None):
    """
    Play a single game with two random agents.
    """
    # Create environment
    env = azul_v1_2players()

    # Create two random agents
    agent1 = RandomAgent(name="Random_1", seed=seed)
    agent2 = RandomAgent(name="Random_2", seed=seed + 1 if seed else None)
    agents = [agent1, agent2]

    # Reset environment
    observation, info = env.reset()

    move_count = 0
    game_over = False

    # Play the game
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            game_over = True
            break

        # Get the agent index from agent name (e.g., "player_0" -> 0)
        agent_idx = int(agent_name.split('_')[1])
        current_agent = agents[agent_idx]

        # Get valid moves from info
        valid_moves = info.get('valid_moves', [])

        if not valid_moves:
            print(f"Warning: No valid moves for {agent_name}")
            break

        # Agent chooses an action
        action = current_agent.choose_action(observation, valid_moves)

        # Execute the action
        env.step(action)
        move_count += 1

        # Optional: render the game state
        if render:
            env.render()

    # Extract final results
    final_state = env.state
    player_scores = [player['score'] for player in final_state['players']]
    winner_idx = player_scores.index(max(player_scores))
    winner_name = agents[winner_idx].name

    results = {
        'player_0_score': player_scores[0],
        'player_1_score': player_scores[1],
        'winner_index': winner_idx,
        'winner_name': winner_name,
        'move_count': move_count,
        'game_completed': game_over
    }

    env.close()

    return results


def play_multiple_games(num_games=10, seed=None):
    """
    Play multiple games and collect statistics.

    Args:
        num_games: Number of games to play
        seed: Base random seed for reproducibility

    Returns:
        dict: Aggregate statistics
    """
    print(f"Playing {num_games} games with RandomAgent vs RandomAgent...\n")

    wins = [0, 0]
    total_scores = [0, 0]
    all_scores = [[], []]

    for i in range(num_games):
        game_seed = seed + i if seed else None
        results = play_single_game(render=False, seed=game_seed)

        # Update statistics
        wins[results['winner_index']] += 1
        total_scores[0] += results['player_0_score']
        total_scores[1] += results['player_1_score']
        all_scores[0].append(results['player_0_score'])
        all_scores[1].append(results['player_1_score'])

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_games} games")

    # Calculate statistics
    win_rates = [w / num_games for w in wins]
    avg_scores = [s / num_games for s in total_scores]

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Total games played: {num_games}")
    print(f"\nPlayer 0 (Random_1):")
    print(f"  Wins: {wins[0]} ({win_rates[0]*100:.1f}%)")
    print(f"  Average score: {avg_scores[0]:.1f}")
    print(f"\nPlayer 1 (Random_2):")
    print(f"  Wins: {wins[1]} ({win_rates[1]*100:.1f}%)")
    print(f"  Average score: {avg_scores[1]:.1f}")
    print("\n" + "="*50)

    return {
        'wins': wins,
        'win_rates': win_rates,
        'avg_scores': avg_scores,
        'all_scores': all_scores
    }


if __name__ == "__main__":
    # Test 1: Play a single game with rendering
    print("Test 1: Playing a single game...")
    print("-" * 50)
    result = play_single_game(render=False, seed=42)
    print(f"Player 0 score: {result['player_0_score']}")
    print(f"Player 1 score: {result['player_1_score']}")
    print(f"Winner: {result['winner_name']}")
    print(f"Total moves: {result['move_count']}")
    print(f"Game completed: {result['game_completed']}")
    print()

    # Test 2: Play multiple games and analyze
    print("\nTest 2: Playing multiple games for statistical analysis...")
    print("-" * 50)
    stats = play_multiple_games(num_games=100, seed=42)

    # With two identical random agents, win rates should be approximately 50-50
    print("\nExpected behavior: With two identical random agents,")
    print("win rates should be approximately 50-50 (within statistical variance).")
