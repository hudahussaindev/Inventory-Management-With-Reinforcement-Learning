import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class SimpleDQNNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super(SimpleDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.BoolTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DoubleDQNAgent:
    """Simplified Double DQN agent with minimal configuration."""

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 9,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update_freq: int = 100,
        hidden_size: int = 128
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.main_network = SimpleDQNNetwork(
            state_dim, action_dim, hidden_size).to(self.device)
        self.target_network = SimpleDQNNetwork(
            state_dim, action_dim, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        # Optimizer and buffer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=lr)
        self.replay_buffer = SimpleReplayBuffer(buffer_size)
        self.step_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.main_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.main_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_value = rewards + (self.gamma * next_q_value * ~dones)

        # Loss and update
        loss = F.mse_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Exponential epsilon decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, state: np.ndarray) -> int:
        """Predict best action (greedy, no exploration)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)
            return q_values.argmax().item()

    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['main_network'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.step_count = checkpoint.get('step_count', 0)
