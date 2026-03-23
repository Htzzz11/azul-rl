"""
Tests for the DQN agent.
"""

import os
import tempfile

import numpy as np
import pytest

from agents.dqn_agent import (
    DQNAgent,
    QNetwork,
    ReplayBuffer,
    action_to_index,
    encode_state,
    index_to_action,
    STATE_SIZE,
    ACTION_SPACE_SIZE,
    NUM_FACTORIES,
    NUM_COLORS,
    MAX_FLOOR_TILES,
    NUM_PATTERN_LINES,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_dummy_observation():
    """Create a minimal observation dict matching the real env structure."""
    return {
        'factories': np.zeros((4, 5), dtype=np.int32),
        'center': np.zeros(5, dtype=np.int32),
        'players': (
            {
                'pattern_lines': np.full((5, 5), 5, dtype=np.int32),
                'wall': np.full((5, 5), 5, dtype=np.int32),
                'floor': [],  # real env returns list of present tiles only
                'is_starting': 0,
                'score': 0,
            },
            {
                'pattern_lines': np.full((5, 5), 5, dtype=np.int32),
                'wall': np.full((5, 5), 5, dtype=np.int32),
                'floor': [],
                'is_starting': 0,
                'score': 0,
            },
        ),
        'bag': np.array([20, 20, 20, 20, 20], dtype=np.int32),
        'lid': np.zeros(5, dtype=np.int32),
    }


DUMMY_VALID_ACTIONS = [
    (1, 0, 0, 0),
    (1, 0, 0, 1),
    (1, 2, 0, 2),
    (2, 3, 0, 3),
    (0, 4, 0, 4),
]


# ── State encoding ──────────────────────────────────────────────────────────

class TestStateEncoding:
    def test_output_size(self):
        obs = _make_dummy_observation()
        state = encode_state(obs, player_index=0)
        assert state.shape == (STATE_SIZE,), f"Expected {STATE_SIZE}, got {state.shape}"

    def test_dtype(self):
        obs = _make_dummy_observation()
        state = encode_state(obs, player_index=0)
        assert state.dtype == np.float32

    def test_score_normalisation(self):
        obs = _make_dummy_observation()
        obs['players'][0]['score'] = 120
        state = encode_state(obs, player_index=0)
        # score is at position: 25 (fac) + 5 (center) + 25 (pl) + 25 (wall) + 7 (floor) = 87
        assert abs(state[87] - 0.5) < 1e-5, f"Expected 0.5, got {state[87]}"

    def test_different_player_index(self):
        obs = _make_dummy_observation()
        obs['players'][0]['score'] = 100
        obs['players'][1]['score'] = 50
        s0 = encode_state(obs, player_index=0)
        s1 = encode_state(obs, player_index=1)
        # The two encodings should differ (me/opp swapped)
        assert not np.allclose(s0, s1)


# ── Action encoding ─────────────────────────────────────────────────────────

class TestActionEncoding:
    def test_roundtrip(self):
        for action in DUMMY_VALID_ACTIONS:
            idx = action_to_index(action)
            recovered = index_to_action(idx)
            assert recovered == action, f"{action} -> {idx} -> {recovered}"

    def test_index_range(self):
        for f in range(NUM_FACTORIES):
            for c in range(NUM_COLORS):
                for t in range(MAX_FLOOR_TILES):
                    for p in range(NUM_PATTERN_LINES):
                        idx = action_to_index((f, c, t, p))
                        assert 0 <= idx < ACTION_SPACE_SIZE

    def test_unique_indices(self):
        indices = set()
        for f in range(NUM_FACTORIES):
            for c in range(NUM_COLORS):
                for t in range(MAX_FLOOR_TILES):
                    for p in range(NUM_PATTERN_LINES):
                        indices.add(action_to_index((f, c, t, p)))
        assert len(indices) == ACTION_SPACE_SIZE


# ── Action masking ──────────────────────────────────────────────────────────

class TestActionMasking:
    def test_choose_action_returns_valid(self):
        agent = DQNAgent(epsilon=0.0)
        obs = _make_dummy_observation()
        for _ in range(20):
            action = agent.choose_action(obs, 0, DUMMY_VALID_ACTIONS)
            assert action in DUMMY_VALID_ACTIONS, (
                f"Agent chose {action} which is not in valid_actions")

    def test_choose_action_with_exploration(self):
        agent = DQNAgent(epsilon=1.0)  # always explore
        obs = _make_dummy_observation()
        for _ in range(20):
            action = agent.choose_action(obs, 0, DUMMY_VALID_ACTIONS)
            assert action in DUMMY_VALID_ACTIONS

    def test_single_valid_action(self):
        agent = DQNAgent(epsilon=0.0)
        obs = _make_dummy_observation()
        single = [(2, 1, 0, 3)]
        action = agent.choose_action(obs, 0, single)
        assert action == (2, 1, 0, 3)


# ── Save / load roundtrip ──────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_load_roundtrip(self):
        agent = DQNAgent(epsilon=0.3)
        agent.train_steps = 42
        agent.episodes_done = 7

        obs = _make_dummy_observation()
        action_before = agent.choose_action(obs, 0, DUMMY_VALID_ACTIONS)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            agent.save(path)

            loaded = DQNAgent.from_pretrained(path)
            # from_pretrained sets epsilon=0, so just check weights work
            action_after = loaded.choose_action(obs, 0, DUMMY_VALID_ACTIONS)
            # Actions should be valid (may differ due to epsilon difference)
            assert action_after in DUMMY_VALID_ACTIONS

            # Check metadata was restored
            assert loaded.train_steps == 42
            assert loaded.episodes_done == 7
        finally:
            os.unlink(path)

    def test_from_pretrained(self):
        agent = DQNAgent(epsilon=0.5)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            loaded = DQNAgent.from_pretrained(path, name='TestAgent')
            assert loaded.epsilon == 0.0  # from_pretrained forces eval mode
            assert loaded.name == 'TestAgent'
        finally:
            os.unlink(path)


# ── Replay buffer ───────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(np.zeros(STATE_SIZE), i % ACTION_SPACE_SIZE, 1.0,
                     np.zeros(STATE_SIZE), False)
        assert len(buf) == 10
        states, actions, rewards, next_states, dones = buf.sample(5)
        assert states.shape == (5, STATE_SIZE)
        assert actions.shape == (5,)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(np.zeros(STATE_SIZE), 0, 0.0, np.zeros(STATE_SIZE), False)
        assert len(buf) == 5


# ── Q-Network ───────────────────────────────────────────────────────────────

class TestQNetwork:
    def test_output_shape(self):
        import torch
        net = QNetwork()
        x = torch.randn(1, STATE_SIZE)
        out = net(x)
        assert out.shape == (1, ACTION_SPACE_SIZE)

    def test_batch(self):
        import torch
        net = QNetwork()
        x = torch.randn(8, STATE_SIZE)
        out = net(x)
        assert out.shape == (8, ACTION_SPACE_SIZE)


# ── Training update ─────────────────────────────────────────────────────────

class TestTrainingUpdate:
    def test_update_returns_none_when_buffer_small(self):
        agent = DQNAgent(batch_size=64)
        # Buffer has 0 items
        assert agent.update() is None

    def test_update_returns_loss(self):
        agent = DQNAgent(batch_size=4)
        # Fill buffer with dummy transitions
        for i in range(10):
            agent.replay_buffer.push(
                np.random.randn(STATE_SIZE).astype(np.float32),
                i % ACTION_SPACE_SIZE,
                1.0,
                np.random.randn(STATE_SIZE).astype(np.float32),
                False,
            )
        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0
