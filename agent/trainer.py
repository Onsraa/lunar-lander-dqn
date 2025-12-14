import os
import json
import time
import random
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .dqn import DQN
from .replay_buffer import ReplayBuffer


class DQNTrainer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 5000,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 100,
        hidden_dims: Tuple[int, ...] = (128, 128),
        save_dir: str = "models/base",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Réseaux
        self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_size)

        self.steps_done = 0
        self.episodes_done = 0

        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_successes': [],
            'losses': [],
            'epsilons': [],
            'training_time': 0
        }

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def get_epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).argmax(dim=1).item()

    def optimize_model(self) -> float:

        if not self.memory.is_ready(self.batch_size):
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones).float()
        
        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_episode(self, env) -> Dict:
        state = env.reset()
        total_reward = 0
        episode_length = 0
        losses = []

        done = truncated = False

        while not (done or truncated):
            action = self.select_action(state)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            self.memory.push(state, action, reward, next_state, done or truncated)
            
            loss = self.optimize_model()
            if loss > 0:
                losses.append(loss)
            
            self.steps_done += 1
            if self.steps_done % self.target_update == 0:
                self.update_target_network()

            state = next_state
            total_reward += reward
            episode_length += 1

        self.episodes_done += 1

        return {
            'reward': total_reward,
            'length': episode_length,
            'success': info.get('success', False),
            'loss': np.mean(losses) if losses else 0,
            'epsilon': self.get_epsilon()
        }

    def _prefill_buffer(self, env, num_steps: int = 1000):
        state = env.reset()
        for _ in range(num_steps):
            action = random.randrange(self.action_dim)
            next_state, reward, done, truncated, _ = env.step(action)
            self.memory.push(state, action, reward, next_state, done or truncated)
            if done or truncated:
                state = env.reset()
            else:
                state = next_state

    def train(self, env, num_episodes: int = 2000,
              save_every: int = 100, verbose: bool = True) -> Dict:
        start_time = time.time()

        if len(self.memory) < self.batch_size * 10:
            if verbose:
                print("Pré-remplissage du buffer...")
            self._prefill_buffer(env, num_steps=self.batch_size * 10)

        progress = tqdm(range(num_episodes), desc="Training", disable=not verbose)

        for episode in progress:
            stats = self.train_episode(env)

            self.training_stats['episode_rewards'].append(stats['reward'])
            self.training_stats['episode_lengths'].append(stats['length'])
            self.training_stats['episode_successes'].append(stats['success'])
            self.training_stats['losses'].append(stats['loss'])
            self.training_stats['epsilons'].append(stats['epsilon'])

            if verbose:
                recent_rewards = self.training_stats['episode_rewards'][-100:]
                recent_successes = self.training_stats['episode_successes'][-100:]

                progress.set_postfix({
                    'R': f"{np.mean(recent_rewards):.0f}",
                    'S': f"{100*np.mean(recent_successes):.0f}%",
                    'ε': f"{stats['epsilon']:.2f}",
                    'L': f"{stats['loss']:.3f}" if stats['loss'] > 0 else "N/A"
                })

            if (episode + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_ep{episode+1}.pt")

        self.training_stats['training_time'] = time.time() - start_time
        self.save_checkpoint("final_model.pt")
        self.save_statistics()

        return self.training_stats

    def save_checkpoint(self, filename: str):
        torch.save({
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']

    def save_statistics(self, filename: str = "training_stats.json"):
        stats = {
            'episode_rewards': self.training_stats['episode_rewards'],
            'episode_lengths': self.training_stats['episode_lengths'],
            'episode_successes': [int(s) for s in self.training_stats['episode_successes']],
            'losses': self.training_stats['losses'],
            'epsilons': self.training_stats['epsilons'],
            'training_time': self.training_stats['training_time'],
            'total_episodes': self.episodes_done,
            'total_steps': self.steps_done,
            'timestamp': datetime.now().isoformat()
        }

        os.makedirs("statistics", exist_ok=True)
        with open(os.path.join("statistics", filename), 'w') as f:
            json.dump(stats, f, indent=2)

    def get_action_greedy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).argmax(dim=1).item()
