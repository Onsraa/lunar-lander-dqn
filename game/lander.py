"""
Lunar Lander - Physique du vaisseau (Simplifié)
===============================================
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LanderConfig:
    """Configuration du vaisseau."""
    mass: float = 10.0
    size: float = 40.0
    
    main_thrust: float = 600.0      # Force moteur principal
    side_thrust: float = 400.0      # Force moteurs latéraux
    
    moment_of_inertia: float = 1000.0
    angular_damping: float = 0.8   # Amortissement rotation


class Lander:
    """Vaisseau spatial avec 3 moteurs."""
    
    def __init__(self, x: float, y: float, config: LanderConfig = None):
        self.config = config or LanderConfig()
        
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        
        # Moteurs (0 = off, 1 = on)
        self.main_thruster = 0.0
        self.left_thruster = 0.0
        self.right_thruster = 0.0
    
    def set_thrusters(self, main: float, left: float, right: float):
        """Active les moteurs (valeurs 0 ou 1)."""
        self.main_thruster = max(0.0, min(1.0, main))
        self.left_thruster = max(0.0, min(1.0, left))
        self.right_thruster = max(0.0, min(1.0, right))

    def update(self, dt: float, gravity: float):
        """Met à jour la physique."""
        sin_a = math.sin(self.angle)
        cos_a = math.cos(self.angle)
        
        # Force moteur principal (pousse vers le haut du vaisseau)
        thrust_x = -sin_a * self.main_thruster * self.config.main_thrust
        thrust_y = -cos_a * self.main_thruster * self.config.main_thrust
        
        # Forces latérales (pour la rotation principalement)
        side_force = (self.left_thruster - self.right_thruster) * self.config.side_thrust
        thrust_x += cos_a * side_force
        thrust_y += sin_a * side_force
        
        # Gravité
        gravity_force = self.config.mass * gravity
        
        # Accélérations
        ax = thrust_x / self.config.mass
        ay = (thrust_y + gravity_force) / self.config.mass
        
        # Couple (torque) des moteurs latéraux
        lever_arm = self.config.size / 4
        torque = (self.right_thruster - self.left_thruster) * self.config.side_thrust * lever_arm
        angular_acc = torque / self.config.moment_of_inertia

        # Intégration
        self.vx += ax * dt
        self.vy += ay * dt
        self.angular_velocity += angular_acc * dt
        
        # Amortissement angulaire
        self.angular_velocity *= self.config.angular_damping

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.angular_velocity * dt

        # Normaliser angle entre -π et π
        while self.angle > math.pi:
            self.angle -= 2 * math.pi
        while self.angle < -math.pi:
            self.angle += 2 * math.pi

    def get_state(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.vx, self.vy, self.angle, self.angular_velocity)
    
    def get_speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2)
    
    def get_corners(self) -> list:
        """Coins du vaisseau pour le rendu."""
        half = self.config.size / 2
        corners_local = [(-half, -half), (half, -half), (half, half), (-half, half)]
        
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        corners = []
        for lx, ly in corners_local:
            gx = self.x + lx * cos_a - ly * sin_a
            gy = self.y + lx * sin_a + ly * cos_a
            corners.append((gx, gy))
        return corners

    def get_thruster_positions(self) -> Tuple[Tuple[float, float], ...]:
        """Positions des 3 moteurs pour le rendu."""
        half = self.config.size / 2
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        
        # Main: bas centre, Left: flanc gauche, Right: flanc droit
        local_pos = [(0, half), (-half, 0), (half, 0)]
        
        positions = []
        for lx, ly in local_pos:
            gx = self.x + lx * cos_a - ly * sin_a
            gy = self.y + lx * sin_a + ly * cos_a
            positions.append((gx, gy))
        return tuple(positions)
    
    def reset(self, x: float, y: float):
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.angle = self.angular_velocity = 0.0
        self.main_thruster = self.left_thruster = self.right_thruster = 0.0
