"""
Lunar Lander - Environnement (avec contraintes)
===============================================

Contraintes disponibles:
- fuel_constraint: Carburant limité
- max_time: Temps maximum configurable
"""

import math
import random
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .lander import Lander, LanderConfig


@dataclass
class EnvironmentConfig:
    """Configuration de l'environnement."""
    width: int = 800
    height: int = 600

    gravity: float = 15.0
    max_time: float = 30.0
    fps: int = 60
    
    platform_width: int = 120
    platform_height: int = 10
    
    max_landing_speed: float = 40.0
    max_landing_angle: float = 0.3
    
    spawn_y: float = 100.0
    spawn_x_range: Tuple[float, float] = (150, 650)
    
    # Récompenses
    reward_landing: float = 100.0
    reward_crash: float = -100.0
    reward_out_of_fuel: float = -50.0  # Pénalité si plus de carburant
    
    # Contraintes
    fuel_constraint: bool = False      # Activer la contrainte de carburant
    max_fuel: float = 100.0            # Quantité de carburant max


# 4 actions simples
ACTIONS = [
    (0.0, 0.0, 0.0),  # 0: NOP
    (1.0, 0.0, 0.0),  # 1: MAIN
    (0.0, 1.0, 0.0),  # 2: LEFT
    (0.0, 0.0, 1.0),  # 3: RIGHT
]
ACTIONS_NAMES = ["NOP", "MAIN", "LEFT", "RIGHT"]
NUM_ACTIONS = 4


class LunarLanderEnv:
    """Environnement Lunar Lander avec contraintes optionnelles."""
    
    def __init__(self, config: EnvironmentConfig = None, render_mode: str = None):
        self.config = config or EnvironmentConfig()
        self.render_mode = render_mode
        
        self.platform_x = self.config.width // 2
        self.platform_y = self.config.height - self.config.platform_height
        
        # Config du lander avec carburant
        self.lander_config = LanderConfig(max_fuel=self.config.max_fuel)
        self.lander: Optional[Lander] = None
        
        self.time_elapsed = 0.0
        self.episode_done = False
        self.success = False
        self.out_of_fuel = False  # Flag pour savoir si on a manqué de carburant
        
        # 6 dimensions de base + 1 si contrainte carburant
        self.state_dim = 7 if self.config.fuel_constraint else 6
        self.action_dim = NUM_ACTIONS

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        margin = 150
        self.platform_x = random.randint(
            self.config.width // 2 - margin,
            self.config.width // 2 + margin
        )

        spawn_x = random.uniform(*self.config.spawn_x_range)
        spawn_y = self.config.spawn_y

        if self.lander is None:
            self.lander = Lander(spawn_x, spawn_y, self.lander_config)
        else:
            self.lander.reset(spawn_x, spawn_y)

        self.lander.vx = random.uniform(-15, 15)
        self.lander.vy = random.uniform(0, 10)
        self.lander.angle = random.uniform(-0.1, 0.1)

        self.time_elapsed = 0.0
        self.episode_done = False
        self.success = False
        self.out_of_fuel = False

        return self._get_state()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.episode_done:
            raise RuntimeError("Episode terminé. Appelez reset().")
        
        # Appliquer l'action
        main, left, right = ACTIONS[action_idx]
        self.lander.set_thrusters(main, left, right)

        # Physique (avec gestion du carburant si activé)
        dt = 1.0 / self.config.fps
        self.lander.update(dt, self.config.gravity, fuel_enabled=self.config.fuel_constraint)
        self.time_elapsed += dt
        
        # Détecter si on vient de tomber à court de carburant
        if self.config.fuel_constraint and not self.lander.has_fuel and not self.out_of_fuel:
            self.out_of_fuel = True

        # Récompense et terminaison
        reward = self._calculate_reward()
        terminated, truncated = self._check_termination()

        info = {
            'time_elapsed': self.time_elapsed,
            'success': self.success,
            'speed': self.lander.get_speed(),
            'angle': abs(self.lander.angle),
            'fuel': self.lander.fuel if self.config.fuel_constraint else None,
            'fuel_ratio': self.lander.fuel_ratio if self.config.fuel_constraint else None,
            'out_of_fuel': self.out_of_fuel,
        }

        self.episode_done = terminated or truncated

        return self._get_state(), reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """
        État normalisé.
        6 dims de base + 1 dim carburant si contrainte activée.
        """
        rel_x = (self.lander.x - self.platform_x) / (self.config.width / 2)
        rel_y = (self.lander.y - self.platform_y) / self.config.height
        
        vx_norm = self.lander.vx / 100.0
        vy_norm = self.lander.vy / 100.0
        
        angle_norm = self.lander.angle / math.pi
        ang_vel_norm = self.lander.angular_velocity / 3.0
        
        state = [rel_x, rel_y, vx_norm, vy_norm, angle_norm, ang_vel_norm]
        
        # Ajouter le carburant si contrainte activée
        if self.config.fuel_constraint:
            fuel_norm = self.lander.fuel_ratio  # Déjà entre 0 et 1
            state.append(fuel_norm)
        
        return np.clip(np.array(state, dtype=np.float32), -1.5, 1.5)

    def _calculate_reward(self) -> float:
        """Reward shaping avec pénalité carburant."""
        reward = 0.0
        
        # === 1. PÉNALITÉ VITESSE ===
        height = self.platform_y - self.lander.y
        height_ratio = max(0.0, min(1.0, height / self.config.height))
        speed = self.lander.get_speed()
        max_safe_speed = 30 + 70 * height_ratio
        
        if speed > max_safe_speed:
            excess = speed - max_safe_speed
            reward -= excess * 0.005
        
        # === 2. BONUS DESCENTE CONTRÔLÉE ===
        if 0 < self.lander.vy < 25:
            reward += 0.02
        
        # === 3. BONUS ALIGNEMENT ===
        dist_to_platform = abs(self.lander.x - self.platform_x)
        if dist_to_platform < self.config.platform_width / 2:
            reward += 0.02
        
        # === 4. BONUS ANGLE STABLE ===
        if abs(self.lander.angle) < 0.15:
            reward += 0.01
        
        # === 5. PÉNALITÉ CARBURANT (si contrainte activée) ===
        if self.config.fuel_constraint:
            # Pénalité progressive quand le carburant est bas
            if self.lander.fuel_ratio < 0.2:
                reward -= 0.02 * (0.2 - self.lander.fuel_ratio)
            
            # Grosse pénalité si on tombe à court
            if self.out_of_fuel:
                reward += self.config.reward_out_of_fuel * 0.01  # Pénalité par step
        
        # === 6. TERMINAISON ===
        if self._check_landing():
            if self._is_successful_landing():
                reward += self.config.reward_landing
                # Bonus si atterri avec du carburant restant
                if self.config.fuel_constraint:
                    reward += 20 * self.lander.fuel_ratio
                self.success = True
            else:
                reward += self.config.reward_crash
        elif self._check_out_of_bounds():
            reward += self.config.reward_crash
        
        return reward
    
    def _check_landing(self) -> bool:
        bottom_y = self.lander.y + self.lander_config.size / 2
        return bottom_y >= self.platform_y
    
    def _is_successful_landing(self) -> bool:
        half_platform = self.config.platform_width / 2
        on_platform = abs(self.lander.x - self.platform_x) < half_platform
        speed_ok = self.lander.get_speed() < self.config.max_landing_speed
        angle_ok = abs(self.lander.angle) < self.config.max_landing_angle
        return on_platform and speed_ok and angle_ok
    
    def _check_out_of_bounds(self) -> bool:
        margin = 50
        return (self.lander.x < -margin or 
                self.lander.x > self.config.width + margin or
                self.lander.y < -margin)
    
    def _check_termination(self) -> Tuple[bool, bool]:
        terminated = self._check_landing() or self._check_out_of_bounds()
        truncated = self.time_elapsed >= self.config.max_time
        return terminated, truncated
    
    def get_platform_rect(self) -> Tuple[int, int, int, int]:
        return (
            self.platform_x - self.config.platform_width // 2,
            self.platform_y,
            self.config.platform_width,
            self.config.platform_height
        )
    
    @property
    def observation_space_dim(self) -> int:
        return self.state_dim
    
    @property
    def action_space_dim(self) -> int:
        return self.action_dim
