import math
import random
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .lander import Lander, LanderConfig


@dataclass
class EnvironmentConfig:
    width: int = 800
    height: int = 600

    gravity: float = 15.0
    max_time: float = 20.0
    fps: int = 60

    platform_width: int = 150
    platform_height: int = 10

    max_landing_speed: float = 50.0
    max_landing_angle: float = 0.4

    spawn_y: float = 100.0
    spawn_x_range: Tuple[float, float] = (200, 600)

    fuel_constraint: bool = False
    max_fuel: float = 100.0


ACTIONS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]
ACTIONS_NAMES = ["NOP", "MAIN", "LEFT", "RIGHT"]
NUM_ACTIONS = 4


class LunarLanderEnv:

    def __init__(self, config: EnvironmentConfig = None, render_mode: str = None):
        self.config = config or EnvironmentConfig()
        self.render_mode = render_mode
        
        self.platform_x = self.config.width // 2
        self.platform_y = self.config.height - self.config.platform_height
        
        self.lander_config = LanderConfig(max_fuel=self.config.max_fuel)
        self.lander: Optional[Lander] = None
        
        self.time_elapsed = 0.0
        self.episode_done = False
        self.success = False
        self.out_of_fuel = False
        
        self.previous_shaping = None
        
        self.state_dim = 7 if self.config.fuel_constraint else 6
        self.action_dim = NUM_ACTIONS

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        margin = 100
        self.platform_x = random.randint(margin, self.config.width - margin)

        spawn_x = random.uniform(*self.config.spawn_x_range)
        spawn_y = self.config.spawn_y

        if self.lander is None:
            self.lander = Lander(spawn_x, spawn_y, self.lander_config)
        else:
            self.lander.reset(spawn_x, spawn_y)

        self.lander.vx = random.uniform(-10, 10)
        self.lander.vy = random.uniform(0, 5)
        self.lander.angle = random.uniform(-0.05, 0.05)

        self.time_elapsed = 0.0
        self.episode_done = False
        self.success = False
        self.out_of_fuel = False
        
        self.previous_shaping = self._calculate_shaping()

        return self._get_state()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.episode_done:
            raise RuntimeError("Episode terminé. Appelez reset().")
        
        main, left, right = ACTIONS[action_idx]
        self.lander.set_thrusters(main, left, right)

        dt = 1.0 / self.config.fps
        self.lander.update(dt, self.config.gravity, fuel_enabled=self.config.fuel_constraint)
        self.time_elapsed += dt
        
        if self.config.fuel_constraint and not self.lander.has_fuel and not self.out_of_fuel:
            self.out_of_fuel = True

        terminated, truncated = self._check_termination()
        
        reward = self._calculate_reward(terminated, truncated)

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
        rel_x = (self.lander.x - self.platform_x) / (self.config.width / 2)
        rel_y = (self.lander.y - self.platform_y) / self.config.height
        
        vx_norm = self.lander.vx / 100.0
        vy_norm = self.lander.vy / 100.0
        
        angle_norm = self.lander.angle / math.pi
        ang_vel_norm = self.lander.angular_velocity / 3.0
        
        state = [rel_x, rel_y, vx_norm, vy_norm, angle_norm, ang_vel_norm]
        
        if self.config.fuel_constraint:
            state.append(self.lander.fuel_ratio)
        
        return np.clip(np.array(state, dtype=np.float32), -1.5, 1.5)

    def _calculate_shaping(self) -> float:

        dist_x = abs(self.lander.x - self.platform_x) / self.config.width

        height = (self.platform_y - self.lander.y) / self.config.height
        height = max(0, min(1, height))

        speed = self.lander.get_speed() / self.config.max_landing_speed

        shaping = -dist_x * 100 - height * 50

        if height < 0.3:
            speed_penalty = max(0, speed - 0.8) * 50 * (1 - height / 0.3)
            shaping -= speed_penalty

        return shaping

    def _calculate_reward(self, terminated: bool, truncated: bool) -> float:

        reward = 0.0
        current_shaping = self._calculate_shaping()

        if self.previous_shaping is not None:
            shaping_delta = current_shaping - self.previous_shaping
            reward += shaping_delta

        self.previous_shaping = current_shaping
        reward -= 0.1

        if truncated:
            reward -= 100.0

        elif self._check_out_of_bounds():
            reward -= 100.0

        elif self._check_landing():
            half_platform = self.config.platform_width / 2
            dist_to_center = abs(self.lander.x - self.platform_x)
            on_platform = dist_to_center < half_platform
            speed = self.lander.get_speed()
            angle = abs(self.lander.angle)

            speed_ok = speed < self.config.max_landing_speed
            angle_ok = angle < self.config.max_landing_angle

            if on_platform and speed_ok and angle_ok:
                self.success = True
                reward += 100.0

                if speed < self.config.max_landing_speed * 0.5:
                    reward += 50.0

                if angle < self.config.max_landing_angle * 0.5:
                    reward += 50.0

                if self.config.fuel_constraint:
                    reward += self.lander.fuel_ratio * 30

            elif on_platform:
                reward -= 80.0

            else:
                reward -= 100.0

        return reward
    
    def _check_landing(self) -> bool:
        bottom_y = self.lander.y + self.lander_config.size / 2
        return bottom_y >= self.platform_y
    
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
