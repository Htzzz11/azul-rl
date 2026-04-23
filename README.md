# AzulMARL: Agent Comparison and Balance Analysis

Source code for the final-year project *Strategy Behaviour of Different Agents and Game Balance Analysis in Azul* (Tianzhi Hou, BSc Computer Science, King's College London).

The project implements four AI agent paradigms for the two-player variant of the board game [Azul](https://boardgamegeek.com/boardgame/230802/azul), pits them against each other in batch tournaments, and uses the resulting data to analyse the game's competitive balance along four dimensions (win rate, score distribution, first-player advantage, luck versus skill). Full methodology and results are in the accompanying report.

![AzulRendering](images/azul_rendering.gif)

## Layout

```
azul-rl-submission/
├── agents/               # BaseAgent, Random, Minimax, MCTS, DQN
├── azul_marl_env/        # PettingZoo environment (forked from Visockas 2025)
├── evaluation/           # SimulationManager, Analyzer, run_all_experiments
├── scripts/              # train_dqn.py, generate_report.py
├── tests/                # pytest suite
├── models/               # dqn_final.pt (trained DQN used in the report)
├── results/              # per-game CSVs + generated figures
├── requirements.txt      # pip-style dependency list
├── environment.yaml      # conda environment specification
└── LICENSE               # GNU Affero General Public License v3
```

## Installation

Python 3.12 or later. Two install paths, both tested against the pinned versions in the report's experimental setup.

**Option 1 — conda (recommended):**

```bash
conda env create -f environment.yaml
conda activate azul
```

**Option 2 — pip in a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Both routes install the same set of direct dependencies: PyTorch, NumPy, pandas, SciPy, Matplotlib, PettingZoo, Gymnasium, `azul-game-engine`, and pytest.

## Verifying the Install

```bash
pytest
```

The test suite covers the environment, the DQN agent, and the Analyzer. A clean run completes in under a minute on a laptop.

## Reproducing the Report's Results

The `results/` directory contains the raw per-game CSVs used to produce every figure and table in Chapter 5 of the report. To regenerate the figures and the plain-text summary from those CSVs:

```bash
python -m scripts.generate_report
```

Output (PNG figures plus `analysis_report.txt`) lands in `results/figures/`. Use `--results-dir` and `--figures-dir` to override the defaults.

## Re-running the Experiments from Scratch

To discard the supplied CSVs and rerun every matchup (roughly three to four hours end-to-end on a laptop, dominated by the MCTS matchups):

```bash
python -m evaluation.run_all_experiments               # all experiments
python -m evaluation.run_all_experiments --list        # list matchups
python -m evaluation.run_all_experiments --only 0 2    # only these indices
```

Each experiment writes one CSV with one row per game. Re-running an experiment overwrites its CSV; regenerate the figures afterwards with `python -m scripts.generate_report`.

## Retraining the DQN Agent

The `models/dqn_final.pt` checkpoint is the one used for the DQN experiments in the report. To retrain from scratch with the same three-phase curriculum (Random → self-play → Minimax depth 1):

```bash
python -m scripts.train_dqn --episodes 10000 --curriculum
```

Intermediate checkpoints save to `models/` every `--save-every` episodes (default 500); the final model is `models/dqn_final.pt`. Short smoke run:

```bash
python -m scripts.train_dqn --episodes 5
```

Resume from a checkpoint:

```bash
python -m scripts.train_dqn --episodes 10000 --resume models/dqn_ep5000.pt
```

Training is CPU-only as shipped; a full 10,000-episode curriculum run takes on the order of several hours on a modern laptop.

## Programmatic Use

Minimal self-play loop between the depth-1 Minimax agent and the flat random baseline:

```python
from azul_marl_env import azul_v1_2players
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent

env = azul_v1_2players()
observation, info = env.reset(seed=42)

agents_by_idx = {
    0: MinimaxAgent(depth=1, name="Minimax"),
    1: RandomAgent(name="Random", seed=0),
}

for agent_name in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        break
    player_index = env.agent_name_mapping[agent_name]
    action = agents_by_idx[player_index].choose_action(
        observation, player_index, info["valid_moves"]
    )
    env.step(action)

env.close()
```

Search-based agents (MCTS, Minimax at depth ≥ 2) also need a reference to the live environment so they can deep-copy it during search:

```python
from agents.mcts_agent import MCTSAgent

mcts = MCTSAgent(simulations=200, name="MCTS")
mcts.set_env(env)   # must be called before choose_action
```

Loading the trained DQN checkpoint for evaluation (forces a deterministic greedy policy):

```python
from agents.dqn_agent import DQNAgent

dqn = DQNAgent.from_pretrained("models/dqn_final.pt", name="DQN")
```

## Attribution

The PettingZoo environment wrapper is forked from [AzulMARL](https://github.com/AzulImplementation/AzulMARL) by Evaldas Visockas, which itself wraps [AzulGameEngine](https://github.com/AzulImplementation/AzulGameEngine) by the same author. Both are redistributed here under the terms of their upstream licences. The agent implementations, the evaluation framework, and the Analyzer are this project's own work.

## License

GNU Affero General Public License v3. See `LICENSE`.
