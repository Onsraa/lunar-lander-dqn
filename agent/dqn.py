"""
Deep Q-Network (Simplifié)
==========================
"""

import torch
import torch.nn as nn
from typing import Tuple


class DQN(nn.Module):
    """Réseau DQN simple avec 2 couches cachées."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def get_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
