#!/usr/bin/env python3
"""
Lunar Lander - Visualisation
============================

Usage:
    python play.py                          # Agent IA
    python play.py --manual                 # Contrôle manuel
    python play.py --model models/final.pt  # Modèle spécifique
"""

import argparse
import os
import sys
import time

import torch
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import LunarLanderEnv, EnvironmentConfig, Renderer, ACTIONS_NAMES
from agent import DQN


def parse_args():
    parser = argparse.ArgumentParser(description='Jouer à Lunar Lander')
    parser.add_argument('--model', type=str, default='models/final_model.pt',
                        help='Chemin vers le modèle')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Nombre d\'épisodes')
    parser.add_argument('--manual', action='store_true',
                        help='Mode manuel')
    parser.add_argument('--random', action='store_true',
                        help='Agent aléatoire')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS (défaut: 60)')
    return parser.parse_args()


def get_manual_action() -> int:
    """Contrôles clavier."""
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_UP]:
        return 1  # MAIN
    elif keys[pygame.K_LEFT]:
        return 2  # LEFT (tourne à droite pour aller à gauche)
    elif keys[pygame.K_RIGHT]:
        return 3  # RIGHT (tourne à gauche pour aller à droite)
    else:
        return 0  # NOP


def play_episode(env, renderer, policy_net, mode='ai', fps=60):
    import random
    
    state = env.reset()
    total_reward = 0
    renderer.reset_particles()

    done = truncated = False

    while not (done or truncated):
        # Action
        if mode == 'manual':
            action = get_manual_action()
        elif mode == 'random':
            action = random.randrange(4)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_t).argmax(dim=1).item()

        next_state, reward, done, truncated, info = env.step(action)

        if not renderer.render(env, {'reward': total_reward}, fps):
            return {'aborted': True}

        state = next_state
        total_reward += reward

    return {
        'reward': total_reward,
        'success': info.get('success', False),
        'time': info.get('time_elapsed', 0)
    }


def main():
    args = parse_args()

    print("=" * 50)
    print("       LUNAR LANDER - VISUALISATION")
    print("=" * 50)
    print()

    env_config = EnvironmentConfig()
    env = LunarLanderEnv(env_config)

    # Charger le modèle
    policy_net = None
    if not args.manual and not args.random:
        if not os.path.exists(args.model):
            print(f"Modèle non trouvé: {args.model}")
            print("Lancez d'abord: python train.py")
            return

        print(f"Chargement: {args.model}")
        policy_net = DQN(env.observation_space_dim, env.action_space_dim)
        checkpoint = torch.load(args.model, map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy_net_state'])
        policy_net.eval()

    if args.manual:
        print("\nContrôles:")
        print("  ↑ : Moteur principal")
        print("  ← : Tourner à droite")
        print("  → : Tourner à gauche")
        print("  ESC : Quitter")

    print()

    renderer = Renderer(env_config.width, env_config.height)
    renderer.init()

    mode = 'manual' if args.manual else ('random' if args.random else 'ai')
    successes = 0

    try:
        for i in range(args.episodes):
            print(f"Episode {i+1}/{args.episodes}...", end=" ")
            
            result = play_episode(env, renderer, policy_net, mode, args.fps)
            
            if result.get('aborted'):
                break
            
            status = "✓ Succès" if result['success'] else "✗ Échec"
            print(f"{status} | Reward: {result['reward']:.0f}")
            
            if result['success']:
                successes += 1
            
            time.sleep(0.5)

        print()
        print(f"Résultat: {successes}/{args.episodes} succès ({100*successes/args.episodes:.0f}%)")

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
