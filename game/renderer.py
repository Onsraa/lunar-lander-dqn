import pygame
import math
import random
from typing import List, Tuple
from dataclasses import dataclass

from .lander import Lander
from .environment import LunarLanderEnv

FREETYPE_AVAILABLE = False
try:
    import pygame.freetype
    FREETYPE_AVAILABLE = True
except ImportError:
    pass


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
                x=x + random.uniform(-2, 2), y=y + random.uniform(-2, 2),
                vx=vx, vy=vy, life=random.uniform(0.3, 0.6),
                size=random.uniform(3, 6) * power, color=color
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
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    DARK_GRAY = (40, 40, 40)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 150, 0)
    CYAN = (0, 200, 255)
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None
        self.initialized = False
        self.particles = ParticleSystem()
        self.font = None
        self.font_small = None
    
    def init(self):
        if self.initialized:
            return
        pygame.init()
        pygame.display.set_caption("Lunar Lander DQN")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        if FREETYPE_AVAILABLE:
            try:
                pygame.freetype.init()
                for font_name in ['arial', 'helvetica', 'sans', None]:
                    try:
                        self.font = pygame.freetype.SysFont(font_name, 18)
                        self.font_small = pygame.freetype.SysFont(font_name, 13)
                        break
                    except Exception:
                        continue
            except Exception:
                pass
        
        self.initialized = True
    
    def _render_text(self, text: str, color: Tuple[int, int, int], 
                     pos: Tuple[int, int], font=None, center: bool = False):
        if font is None:
            font = self.font
        if font is None:
            return
        try:
            surf, rect = font.render(text, color)
            if center:
                rect.center = pos
            else:
                rect.topleft = pos
            self.screen.blit(surf, rect)
        except Exception:
            pass
    
    def close(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False

    def render(self, env: LunarLanderEnv, info: dict = None, fps: int = 60) -> bool:
        if not self.initialized:
            self.init()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

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
        random.seed(42)
        for _ in range(50):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height - 100)
            pygame.draw.circle(self.screen, self.WHITE, (x, y), 1)
        random.seed()

    def _draw_ground(self):
        pygame.draw.rect(self.screen, self.GRAY, (0, self.height - 5, self.width, 5))

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
            pygame.draw.line(self.screen, self.WHITE, (x, rect[1]), (x, rect[1] + rect[3]), 2)

    def _draw_lander(self, lander: Lander):
        corners = lander.get_corners()
        pygame.draw.polygon(self.screen, self.WHITE, corners)
        pygame.draw.polygon(self.screen, self.GRAY, corners, 2)
        positions = lander.get_thruster_positions()
        if lander.main_thruster > 0:
            pygame.draw.circle(self.screen, self.ORANGE, (int(positions[0][0]), int(positions[0][1])), 8)
        if lander.left_thruster > 0:
            pygame.draw.circle(self.screen, self.YELLOW, (int(positions[1][0]), int(positions[1][1])), 5)
        if lander.right_thruster > 0:
            pygame.draw.circle(self.screen, self.YELLOW, (int(positions[2][0]), int(positions[2][1])), 5)

    def _update_particles(self, lander: Lander, dt: float):
        positions = lander.get_thruster_positions()
        if lander.main_thruster > 0:
            self.particles.emit(positions[0][0], positions[0][1], lander.angle, lander.main_thruster, is_main=True)
        if lander.left_thruster > 0:
            self.particles.emit(positions[1][0], positions[1][1], lander.angle - math.pi/2, lander.left_thruster, is_main=False)
        if lander.right_thruster > 0:
            self.particles.emit(positions[2][0], positions[2][1], lander.angle + math.pi/2, lander.right_thruster, is_main=False)
        self.particles.update(dt)

    def _draw_hud(self, env: LunarLanderEnv, info: dict = None):
        if not env.lander:
            return
        if env.config.fuel_constraint:
            self._draw_fuel_gauge(env)
        self._draw_thruster_bars(env.lander)
        self._draw_info(env, info)
        self._draw_landing_indicator(env)

    def _draw_fuel_gauge(self, env: LunarLanderEnv):
        x, y, width, height = 20, 50, 30, 150
        fuel_pct = env.lander.fuel_ratio
        if fuel_pct > 0.3:
            color = self.CYAN
        elif fuel_pct > 0.1:
            color = self.ORANGE
        else:
            color = self.RED
        pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, width, height))
        fill_height = int(height * fuel_pct)
        if fill_height > 0:
            pygame.draw.rect(self.screen, color, (x, y + height - fill_height, width, fill_height))
        pygame.draw.rect(self.screen, self.WHITE, (x, y, width, height), 2)
        self._render_text("FUEL", self.WHITE, (x + width // 2, y - 10), center=True)
        self._render_text(f"{env.lander.fuel:.0f}", color, (x + width // 2, y + height + 8), self.font_small, center=True)

    def _draw_thruster_bars(self, lander: Lander):
        base_x, base_y = self.width - 90, 50
        side_width, side_height = 18, 100
        main_width, main_height = 28, 130
        spacing = 8
        left_x = base_x
        main_x = left_x + side_width + spacing
        right_x = main_x + main_width + spacing
        main_y = base_y
        side_y = base_y + (main_height - side_height) // 2
        self._draw_single_thruster_bar(left_x, side_y, side_width, side_height, lander.left_thruster, self.YELLOW, "L")
        self._draw_single_thruster_bar(main_x, main_y, main_width, main_height, lander.main_thruster, self.ORANGE, "M")
        self._draw_single_thruster_bar(right_x, side_y, side_width, side_height, lander.right_thruster, self.YELLOW, "R")
        total_width = right_x + side_width - left_x
        self._render_text("THRUST", self.WHITE, (left_x + total_width // 2, base_y - 10), center=True)

    def _draw_single_thruster_bar(self, x: int, y: int, width: int, height: int, power: float, color: Tuple[int, int, int], label: str):
        pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, width, height))
        fill_height = int(height * power)
        if fill_height > 0:
            pygame.draw.rect(self.screen, color, (x, y + height - fill_height, width, fill_height))
        border_color = color if power > 0 else self.GRAY
        pygame.draw.rect(self.screen, border_color, (x, y, width, height), 2)
        self._render_text(label, self.WHITE, (x + width // 2, y + height + 5), self.font_small, center=True)

    def _draw_info(self, env: LunarLanderEnv, info: dict = None):
        time_left = env.config.max_time - env.time_elapsed
        time_color = self.WHITE if time_left > 5 else self.RED
        self._render_text(f"Time: {time_left:.1f}s", time_color, (self.width // 2, 12), center=True)

    def _draw_landing_indicator(self, env: LunarLanderEnv):
        speed = env.lander.get_speed()
        angle_deg = math.degrees(env.lander.angle)
        dist_to_platform = abs(env.lander.x - env.platform_x)

        speed_ok = speed < env.config.max_landing_speed
        angle_ok = abs(env.lander.angle) < env.config.max_landing_angle
        position_ok = dist_to_platform < env.config.platform_width / 2

        x_start = 15
        y_start = self.height - 70
        line_height = 18
        indicator_radius = 5

        indicators = [
            (f"Speed: {speed:.0f}", speed_ok),
            (f"Angle: {angle_deg:+.1f}°", angle_ok),
            (f"Position", position_ok),
        ]

        for i, (text, ok) in enumerate(indicators):
            y = y_start + i * line_height

            text_color = self.WHITE
            self._render_text(text, text_color, (x_start, y), self.font_small)

            indicator_x = x_start + 90
            indicator_y = y + 6
            color = self.GREEN if ok else self.RED
            pygame.draw.circle(self.screen, color, (indicator_x, indicator_y), indicator_radius)
            pygame.draw.circle(self.screen, self.GRAY, (indicator_x, indicator_y), indicator_radius, 1)

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
