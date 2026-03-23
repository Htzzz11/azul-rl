"""
Deep Q-Network (DQN) agent for Azul.

Uses a neural network to approximate Q-values for each possible action,
with invalid action masking to ensure only legal moves are selected.
"""

import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------
# All possible action tuples: (factory, color, tiles_to_floor, pattern_line)
#   factory:        0-4   (0 = center, 1-4 = display factories for 2-player)
#   color:          0-4
#   tiles_to_floor: 0-4   (0 = place on line, 1-4 = dump count)
#   pattern_line:   0-4
#
# Total: 5 * 5 * 5 * 5 = 625 possible indices.
# Many will never be valid; that's fine – we just mask them.

NUM_FACTORIES = 6       # 0 = center, 1-5 = display (2-player: 5 display factories)
NUM_COLORS = 5
MAX_FLOOR_TILES = 21    # 0..20 (20 tiles per color, all could land in center)
NUM_PATTERN_LINES = 5
ACTION_SPACE_SIZE = NUM_FACTORIES * NUM_COLORS * MAX_FLOOR_TILES * NUM_PATTERN_LINES  # 3150


def action_to_index(action: Tuple[int, int, int, int]) -> int:
    """Convert an action tuple to a flat index."""
    f, c, t, p = int(action[0]), int(action[1]), int(action[2]), int(action[3])
    return ((f * NUM_COLORS + c) * MAX_FLOOR_TILES + t) * NUM_PATTERN_LINES + p


def index_to_action(idx: int) -> Tuple[int, int, int, int]:
    """Convert a flat index back to an action tuple."""
    p = idx % NUM_PATTERN_LINES
    idx //= NUM_PATTERN_LINES
    t = idx % MAX_FLOOR_TILES
    idx //= MAX_FLOOR_TILES
    c = idx % NUM_COLORS
    f = idx // NUM_COLORS
    return (f, c, t, p)


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------
STATE_SIZE = 156  # see breakdown below


def encode_state(observation: dict, player_index: int) -> np.ndarray:
    """
    Flatten the observation dict into a fixed-size feature vector.

    Layout (156 features):
        factories       5×5 = 25   (raw tile counts, each factory row)
        center              5
        my pattern_lines   25
        my wall             25
        my floor             7
        my score             1   (normalised /240)
        opp pattern_lines  25
        opp wall            25
        opp floor            7
        opp score            1
        bag                  5
        lid                  5
        ---------------------------
        Total              156
    """
    parts: list[np.ndarray] = []

    # Factories: shape (num_display_factories, 5) – flatten, normalise by max 4
    fac = np.asarray(observation['factories'], dtype=np.float32).flatten()
    fac /= np.float32(4.0)
    # Pad/truncate to 25 (5 factories * 5 colors) to match STATE_SIZE layout
    if len(fac) < 25:
        padded = np.zeros(25, dtype=np.float32)
        padded[:len(fac)] = fac
        fac = padded
    else:
        fac = fac[:25]
    parts.append(fac)

    # Center: shape (5,), normalise by 20 (reasonable upper bound)
    center = np.asarray(observation['center'], dtype=np.float32)[:5]
    center /= np.float32(20.0)
    parts.append(center)

    # Current player
    me = observation['players'][player_index]
    opp_index = 1 - player_index
    opp = observation['players'][opp_index]

    for player_obs in (me, opp):
        # pattern_lines (5,5) – shift so empty(5)=0, Blue(0)=1, ..., White(4)=5
        # then normalise to [0, 1]
        pl = np.asarray(player_obs['pattern_lines'], dtype=np.float32).flatten()
        mask = pl != np.float32(5.0)
        pl[mask] += np.float32(1.0)
        pl[~mask] = np.float32(0.0)
        pl /= np.float32(5.0)
        parts.append(pl)

        # wall (5,5) – binary: 1.0 if tile placed, 0.0 if empty
        # (wall position determines color, so binary is sufficient)
        wall = np.asarray(player_obs['wall'], dtype=np.float32).flatten()
        wall = (wall != np.float32(5.0)).astype(np.float32)
        parts.append(wall)

        # floor (7,) – binary: 1.0 if occupied, 0.0 if not
        floor_raw = player_obs['floor']
        floor = np.zeros(7, dtype=np.float32)
        n_floor = min(len(floor_raw), 7)
        if n_floor > 0:
            floor[:n_floor] = np.float32(1.0)
        parts.append(floor)

        # score (normalised)
        parts.append(np.array([np.float32(player_obs['score']) / np.float32(240.0)],
                              dtype=np.float32))

    # bag (5,), normalise by 20
    bag = np.asarray(observation['bag'], dtype=np.float32)[:5]
    bag /= np.float32(20.0)
    parts.append(bag)
    # lid (5,), normalise by 20
    lid = np.asarray(observation['lid'], dtype=np.float32)[:5]
    lid /= np.float32(20.0)
    parts.append(lid)

    state = np.concatenate(parts)
    # Ensure exactly STATE_SIZE
    if len(state) < STATE_SIZE:
        padded = np.zeros(STATE_SIZE, dtype=np.float32)
        padded[:len(state)] = state
        state = padded
    elif len(state) > STATE_SIZE:
        state = state[:STATE_SIZE]
    return state


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Simple MLP Q-network: state -> Q-value for each action index."""

    def __init__(self, state_size: int = STATE_SIZE,
                 action_size: int = ACTION_SPACE_SIZE,
                 hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size ring buffer for experience tuples."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action_idx: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for Azul.

    During evaluation (epsilon=0), performs a forward pass and picks the
    highest-Q valid action.  During training, uses epsilon-greedy exploration
    and maintains a replay buffer + target network.
    """

    def __init__(self, name: str = None, model_path: Optional[str] = None,
                 epsilon: float = 0.0, lr: float = 1e-4,
                 gamma: float = 0.99, buffer_capacity: int = 50_000,
                 batch_size: int = 64, target_update_freq: int = 50,
                 device: str = None):
        super().__init__(name)

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimiser
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Bookkeeping
        self.train_steps = 0
        self.episodes_done = 0

        # Load pretrained weights if provided
        if model_path is not None:
            self.load(model_path)

    # ----- BaseAgent interface -----

    def choose_action(self, observation: dict, player_index: int,
                      valid_actions: List[Tuple[int, int, int, int]]
                      ) -> Tuple[int, int, int, int]:
        if not valid_actions:
            raise ValueError("valid_actions is empty")

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Encode state and run through Q-network
        state = encode_state(observation, player_index)
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_t).squeeze(0)  # (ACTION_SPACE_SIZE,)

        # Mask: set all invalid actions to -inf
        valid_indices = [action_to_index(a) for a in valid_actions]
        mask = torch.full_like(q_values, float('-inf'))
        for idx in valid_indices:
            mask[idx] = 0.0
        masked_q = q_values + mask

        best_idx = int(masked_q.argmax().item())
        return index_to_action(best_idx)

    def reset(self):
        """Reset per-game state (nothing needed for DQN)."""
        pass

    # ----- Training helpers -----

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.

        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Current Q-values for chosen actions
        q_values = self.q_network(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values (no grad)
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1

        # Periodically sync target network
        if self.train_steps % self.target_update_freq == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ----- Persistence -----

    def save(self, path: str):
        """Save model weights and training metadata."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'episodes_done': self.episodes_done,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str):
        """Load model weights and training metadata."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'train_steps' in checkpoint:
            self.train_steps = checkpoint['train_steps']
        if 'episodes_done' in checkpoint:
            self.episodes_done = checkpoint['episodes_done']
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']

    @classmethod
    def from_pretrained(cls, model_path: str, name: str = None) -> 'DQNAgent':
        """Convenience constructor for evaluation (epsilon=0)."""
        agent = cls(name=name, model_path=model_path, epsilon=0.0)
        agent.epsilon = 0.0  # override epsilon restored from checkpoint
        return agent
