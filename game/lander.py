import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LanderConfig:
    size: float = 30.0
    mass: float = 1.0

    main_thrust: float = 40.0
    side_thrust: float = 20.0
    rotation_speed: float = 1.5

    # Carburant
    max_fuel: float = 100.0
    main_fuel_cost: float = 10.0
    side_fuel_cost: float = 5.0


class Lander:

    def __init__(self, x: float, y: float, config: LanderConfig = None):
        self.config = config or LanderConfig()

        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

        self.angle = 0.0
        self.angular_velocity = 0.0

        self.main_thruster = 0.0
        self.left_thruster = 0.0
        self.right_thruster = 0.0

        self.fuel = self.config.max_fuel

    def reset(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.main_thruster = 0.0
        self.left_thruster = 0.0
        self.right_thruster = 0.0
        self.fuel = self.config.max_fuel

    def set_thrusters(self, main: float, left: float, right: float):
        self.main_thruster = max(0.0, min(1.0, main))
        self.left_thruster = max(0.0, min(1.0, left))
        self.right_thruster = max(0.0, min(1.0, right))

    def update(self, dt: float, gravity: float, fuel_enabled: bool = False):

        if fuel_enabled:
            fuel_used = (
                self.main_thruster * self.config.main_fuel_cost * dt +
                self.left_thruster * self.config.side_fuel_cost * dt +
                self.right_thruster * self.config.side_fuel_cost * dt
            )
            self.fuel = max(0.0, self.fuel - fuel_used)

            if self.fuel <= 0:
                self.main_thruster = 0.0
                self.left_thruster = 0.0
                self.right_thruster = 0.0

        main_thrust_x = -math.sin(self.angle) * self.main_thruster * self.config.main_thrust
        main_thrust_y = -math.cos(self.angle) * self.main_thruster * self.config.main_thrust

        perp_x = math.cos(self.angle)
        perp_y = -math.sin(self.angle)

        side_net = self.left_thruster - self.right_thruster
        side_thrust_x = perp_x * side_net * self.config.side_thrust
        side_thrust_y = perp_y * side_net * self.config.side_thrust

        total_thrust_x = main_thrust_x + side_thrust_x
        total_thrust_y = main_thrust_y + side_thrust_y

        ax = total_thrust_x / self.config.mass
        ay = gravity + total_thrust_y / self.config.mass

        rotation_torque = (self.left_thruster - self.right_thruster) * self.config.rotation_speed

        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        self.angular_velocity += rotation_torque * dt
        angular_damping = 0.85
        self.angular_velocity *= pow(angular_damping, dt * 60)
        self.angle += self.angular_velocity * dt

    def get_speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2)

    def get_corners(self) -> List[Tuple[float, float]]:
        s = self.config.size / 2
        corners_local = [
            (-s * 0.6, -s),
            (s * 0.6, -s),
            (s, s * 0.5),
            (0, s),
            (-s, s * 0.5),
        ]

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        corners = []
        for lx, ly in corners_local:
            rx = lx * cos_a - ly * sin_a + self.x
            ry = lx * sin_a + ly * cos_a + self.y
            corners.append((rx, ry))

        return corners

    def get_thruster_positions(self) -> List[Tuple[float, float]]:
        s = self.config.size / 2
        positions_local = [
            (0, s),
            (-s, 0),
            (s, 0),
        ]

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        positions = []
        for lx, ly in positions_local:
            rx = lx * cos_a - ly * sin_a + self.x
            ry = lx * sin_a + ly * cos_a + self.y
            positions.append((rx, ry))

        return positions

    @property
    def fuel_ratio(self) -> float:
        return self.fuel / self.config.max_fuel

    @property
    def has_fuel(self) -> bool:
        return self.fuel > 0