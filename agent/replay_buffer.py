"""
Replay Buffer (Simplifié)
=========================
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, NamedTuple
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Buffer d'expérience simple."""
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append(Transition(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=reward,
            next_state=np.array(next_state, dtype=np.float32),
            done=done
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        
        return (
            torch.FloatTensor(np.array(batch.state)),
            torch.LongTensor(batch.action),
            torch.FloatTensor(batch.reward),
            torch.FloatTensor(np.array(batch.next_state)),
            torch.BoolTensor(batch.done)
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
