"""
Lunar Lander - Environnement (Corrigé)
======================================

Corrections principales:
1. Reward shaping qui pénalise la vitesse excessive près du sol
2. Actions réduites à 4 (plus simple)
3. FPS réduit à 60 (moins de steps inutiles)
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


# 4 actions simples
ACTIONS = [
    (0.0, 0.0, 0.0),  # 0: NOP - rien
    (1.0, 0.0, 0.0),  # 1: MAIN - moteur principal
    (0.0, 1.0, 0.0),  # 2: LEFT - moteur gauche (tourne à droite)
    (0.0, 0.0, 1.0),  # 3: RIGHT - moteur droit (tourne à gauche)
]
ACTIONS_NAMES = ["NOP", "MAIN", "LEFT", "RIGHT"]
NUM_ACTIONS = 4


class LunarLanderEnv:
    """Environnement Lunar Lander simplifié."""
    
    def __init__(self, config: EnvironmentConfig = None, render_mode: str = None):
        self.config = config or EnvironmentConfig()
        self.render_mode = render_mode
        
        self.platform_x = self.config.width // 2
        self.platform_y = self.config.height - self.config.platform_height
        
        self.lander_config = LanderConfig()
        self.lander: Optional[Lander] = None
        
        self.time_elapsed = 0.0
        self.episode_done = False
        self.success = False
        
        # 6 dimensions d'état
        self.state_dim = 6
        self.action_dim = NUM_ACTIONS

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Plateforme avec position variable
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

        # Vitesse initiale aléatoire légère
        self.lander.vx = random.uniform(-15, 15)
        self.lander.vy = random.uniform(0, 10)
        self.lander.angle = random.uniform(-0.1, 0.1)

        self.time_elapsed = 0.0
        self.episode_done = False
        self.success = False

        return self._get_state()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.episode_done:
            raise RuntimeError("Episode terminé. Appelez reset().")
        
        # Appliquer l'action
        main, left, right = ACTIONS[action_idx]
        self.lander.set_thrusters(main, left, right)

        # Physique
        dt = 1.0 / self.config.fps
        self.lander.update(dt, self.config.gravity)
        self.time_elapsed += dt

        # Récompense et terminaison
        reward = self._calculate_reward()
        terminated, truncated = self._check_termination()

        info = {
            'time_elapsed': self.time_elapsed,
            'success': self.success,
            'speed': self.lander.get_speed(),
            'angle': abs(self.lander.angle),
        }

        self.episode_done = terminated or truncated

        return self._get_state(), reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """
        État normalisé en 6 dimensions:
        - Position relative à la plateforme (x, y)
        - Vitesses (vx, vy)
        - Angle et vitesse angulaire
        """
        # Position relative à la plateforme
        rel_x = (self.lander.x - self.platform_x) / (self.config.width / 2)
        rel_y = (self.lander.y - self.platform_y) / self.config.height
        
        # Vitesses (normalisées sur ~100 px/s max)
        vx_norm = self.lander.vx / 100.0
        vy_norm = self.lander.vy / 100.0
        
        # Angle (normalisé sur π) et vitesse angulaire
        angle_norm = self.lander.angle / math.pi
        ang_vel_norm = self.lander.angular_velocity / 3.0
        
        state = np.array([
            rel_x, rel_y, vx_norm, vy_norm, angle_norm, ang_vel_norm
        ], dtype=np.float32)
        
        return np.clip(state, -1.5, 1.5)

    def _calculate_reward(self) -> float:
        """
        Reward shaping corrigé.
        
        CLEF: On pénalise la vitesse excessive, surtout près du sol.
        Cela empêche l'agent de simplement tomber sans freiner.
        """
        reward = 0.0
        
        # === 1. PÉNALITÉ VITESSE (clé du fix!) ===
        # Plus on est proche du sol, plus la vitesse max autorisée diminue
        height = self.platform_y - self.lander.y
        height_ratio = max(0.0, min(1.0, height / self.config.height))
        
        speed = self.lander.get_speed()
        
        # Vitesse "safe" décroît avec l'altitude:
        # En haut (height_ratio=1): 100 px/s OK
        # En bas (height_ratio=0): 30 px/s max
        max_safe_speed = 30 + 70 * height_ratio
        
        if speed > max_safe_speed:
            excess = speed - max_safe_speed
            reward -= excess * 0.005  # Pénalité progressive
        
        # === 2. BONUS DESCENTE CONTRÔLÉE ===
        # Récompense si on descend doucement (vy positif = descend)
        if 0 < self.lander.vy < 25:
            reward += 0.02
        
        # === 3. BONUS ALIGNEMENT HORIZONTAL ===
        # Petit bonus si on est au-dessus de la plateforme
        dist_to_platform = abs(self.lander.x - self.platform_x)
        if dist_to_platform < self.config.platform_width / 2:
            reward += 0.02
        
        # === 4. BONUS ANGLE STABLE ===
        if abs(self.lander.angle) < 0.15:
            reward += 0.01
        
        # === 5. TERMINAISON ===
        if self._check_landing():
            if self._is_successful_landing():
                reward += self.config.reward_landing
                self.success = True
            else:
                reward += self.config.reward_crash
        elif self._check_out_of_bounds():
            reward += self.config.reward_crash
        
        return reward
    
    def _check_landing(self) -> bool:
        """Vérifie si le vaisseau touche le sol."""
        bottom_y = self.lander.y + self.lander_config.size / 2
        return bottom_y >= self.platform_y
    
    def _is_successful_landing(self) -> bool:
        """Vérifie si l'atterrissage est réussi."""
        half_platform = self.config.platform_width / 2
        on_platform = abs(self.lander.x - self.platform_x) < half_platform
        speed_ok = self.lander.get_speed() < self.config.max_landing_speed
        angle_ok = abs(self.lander.angle) < self.config.max_landing_angle
        return on_platform and speed_ok and angle_ok
    
    def _check_out_of_bounds(self) -> bool:
        """Vérifie si le vaisseau est sorti des limites."""
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
