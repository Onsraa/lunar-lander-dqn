"""
Lunar Lander - Rendu Pygame (Simplifié)
=======================================
"""

import pygame
import math
import random
from typing import List, Tuple
from dataclasses import dataclass

from .lander import Lander
from .environment import LunarLanderEnv


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float = 1.0
    size: float = 4.0
    color: Tuple[int, int, int] = (255, 200, 50)


class ParticleSystem:
    def __init__(self, max_particles: int = 200):
        self.particles: List[Particle] = []
        self.max_particles = max_particles

    def emit(self, x: float, y: float, angle: float, power: float, is_main: bool = True):
        if power < 0.1 or len(self.particles) >= self.max_particles:
            return

        num = int(power * 8) if is_main else int(power * 4)
        
        for _ in range(num):
            spread = random.uniform(-0.2, 0.2)
            speed = random.uniform(150, 250) * power
            
            vx = -math.sin(angle + spread) * speed
            vy = math.cos(angle + spread) * speed
            
            color = (255, random.randint(100, 200), random.randint(0, 50))
            
            self.particles.append(Particle(
                x=x + random.uniform(-2, 2),
                y=y + random.uniform(-2, 2),
                vx=vx, vy=vy,
                life=random.uniform(0.3, 0.6),
                size=random.uniform(3, 6) * power,
                color=color
            ))

    def update(self, dt: float):
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt * 3
            p.size *= 0.95
        self.particles = [p for p in self.particles if p.life > 0 and p.size > 0.5]

    def draw(self, screen: pygame.Surface):
        for p in self.particles:
            alpha = int(255 * max(0, p.life))
            size = max(1, int(p.size))
            
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p.color, alpha), (size, size), size)
            screen.blit(surf, (int(p.x - size), int(p.y - size)))

    def clear(self):
        self.particles.clear()


class Renderer:
    # Couleurs
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    DARK_GRAY = (50, 50, 50)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 150, 0)

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None
        self.initialized = False
        self.particles = ParticleSystem()
        self.font = None

    def init(self):
        if self.initialized:
            return
        pygame.init()
        pygame.display.set_caption("Lunar Lander DQN")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Font optionnel (peut échouer sur Python 3.14)
        try:
            self.font = pygame.font.Font(None, 24)
        except Exception:
            self.font = None
            print("Note: Font non disponible, HUD simplifié")

        self.initialized = True

    def close(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False

    def render(self, env: LunarLanderEnv, info: dict = None, fps: int = 60) -> bool:
        if not self.initialized:
            self.init()

        # Événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        # Dessin
        self.screen.fill(self.BLACK)
        self._draw_stars()
        self._draw_ground()
        self._draw_platform(env)

        if env.lander:
            self._update_particles(env.lander, 1.0 / fps)
            self.particles.draw(self.screen)
            self._draw_lander(env.lander)

        self._draw_hud(env, info)

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def _draw_stars(self):
        """Dessine quelques étoiles fixes."""
        random.seed(42)
        for _ in range(50):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height - 100)
            pygame.draw.circle(self.screen, self.WHITE, (x, y), 1)
        random.seed()

    def _draw_ground(self):
        pygame.draw.rect(self.screen, self.GRAY,
                        (0, self.height - 5, self.width, 5))

    def _draw_platform(self, env: LunarLanderEnv):
        rect = env.get_platform_rect()

        if env.lander:
            dist = abs(env.lander.x - env.platform_x)
            if dist < env.config.platform_width / 2:
                color = self.GREEN
            elif dist < env.config.platform_width:
                color = self.YELLOW
            else:
                color = self.RED
        else:
            color = self.GREEN

        pygame.draw.rect(self.screen, color, rect)

        for i in range(1, 4):
            x = rect[0] + rect[2] * i // 4
            pygame.draw.line(self.screen, self.WHITE,
                           (x, rect[1]), (x, rect[1] + rect[3]), 2)

    def _draw_lander(self, lander: Lander):
        corners = lander.get_corners()
        pygame.draw.polygon(self.screen, self.WHITE, corners)
        pygame.draw.polygon(self.screen, self.GRAY, corners, 2)

        positions = lander.get_thruster_positions()

        if lander.main_thruster > 0:
            pygame.draw.circle(self.screen, self.ORANGE,
                             (int(positions[0][0]), int(positions[0][1])), 8)

        if lander.left_thruster > 0:
            pygame.draw.circle(self.screen, self.YELLOW,
                             (int(positions[1][0]), int(positions[1][1])), 5)

        if lander.right_thruster > 0:
            pygame.draw.circle(self.screen, self.YELLOW,
                             (int(positions[2][0]), int(positions[2][1])), 5)

    def _update_particles(self, lander: Lander, dt: float):
        positions = lander.get_thruster_positions()

        if lander.main_thruster > 0:
            self.particles.emit(positions[0][0], positions[0][1],
                              lander.angle, lander.main_thruster, is_main=True)

        if lander.left_thruster > 0:
            self.particles.emit(positions[1][0], positions[1][1],
                              lander.angle - math.pi/2, lander.left_thruster, is_main=False)

        if lander.right_thruster > 0:
            self.particles.emit(positions[2][0], positions[2][1],
                              lander.angle + math.pi/2, lander.right_thruster, is_main=False)

        self.particles.update(dt)

    def _draw_hud(self, env: LunarLanderEnv, info: dict = None):
        """HUD avec ou sans texte selon disponibilité du font."""
        if not env.lander:
            return

        speed = env.lander.get_speed()
        angle = abs(env.lander.angle)

        # Indicateur de vitesse (barre)
        speed_pct = min(1.0, speed / env.config.max_landing_speed)
        speed_color = self.GREEN if speed_pct < 1.0 else self.RED
        pygame.draw.rect(self.screen, self.DARK_GRAY, (10, 10, 100, 15))
        pygame.draw.rect(self.screen, speed_color, (10, 10, int(100 * speed_pct), 15))
        pygame.draw.rect(self.screen, self.WHITE, (10, 10, 100, 15), 1)

        # Indicateur d'angle (barre)
        angle_pct = min(1.0, angle / env.config.max_landing_angle)
        angle_color = self.GREEN if angle_pct < 1.0 else self.RED
        pygame.draw.rect(self.screen, self.DARK_GRAY, (10, 30, 100, 15))
        pygame.draw.rect(self.screen, angle_color, (10, 30, int(100 * angle_pct), 15))
        pygame.draw.rect(self.screen, self.WHITE, (10, 30, 100, 15), 1)

        # Indicateur de temps
        time_pct = env.time_elapsed / env.config.max_time
        pygame.draw.rect(self.screen, self.DARK_GRAY, (10, 50, 100, 15))
        pygame.draw.rect(self.screen, self.YELLOW, (10, 50, int(100 * time_pct), 15))
        pygame.draw.rect(self.screen, self.WHITE, (10, 50, 100, 15), 1)

        # Texte si font disponible
        if self.font:
            text = self.font.render(f"Speed: {speed:.0f}", True, speed_color)
            self.screen.blit(text, (115, 10))

            angle_deg = math.degrees(env.lander.angle)
            text = self.font.render(f"Angle: {angle_deg:.1f}°", True, angle_color)
            self.screen.blit(text, (115, 30))

            text = self.font.render(f"Time: {env.time_elapsed:.1f}s", True, self.WHITE)
            self.screen.blit(text, (115, 50))

            if info and 'reward' in info:
                text = self.font.render(f"Reward: {info['reward']:.0f}", True, self.WHITE)
                self.screen.blit(text, (self.width - 120, 10))

    def reset_particles(self):
        self.particles.clear()

    def wait_for_key(self) -> bool:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    return True
            self.clock.tick(30)