#!/usr/bin/env python3
"""
Lunar Lander - Visualisation (avec contraintes)
===============================================

Usage:
    python play.py                                    # Base
    python play.py --model models/fuel_100/final_model.pt --fuel
    python play.py --manual --fuel --max-fuel 50
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
    
    # Modèle
    parser.add_argument('--model', type=str, default='models/base/final_model.pt',
                        help='Chemin vers le modèle')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Nombre d\'épisodes')
    
    # Mode
    parser.add_argument('--manual', action='store_true',
                        help='Mode manuel')
    parser.add_argument('--random', action='store_true',
                        help='Agent aléatoire')
    
    # Contraintes (doivent correspondre à l'entraînement!)
    parser.add_argument('--fuel', action='store_true',
                        help='Activer la contrainte de carburant')
    parser.add_argument('--max-fuel', type=float, default=100.0,
                        help='Quantité de carburant max (défaut: 100)')
    parser.add_argument('--time', type=float, default=30.0,
                        help='Temps maximum en secondes (défaut: 30)')
    
    # Affichage
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS (défaut: 60)')
    
    return parser.parse_args()


def get_manual_action() -> int:
    """Contrôles clavier."""
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_UP]:
        return 1  # MAIN
    elif keys[pygame.K_LEFT]:
        return 2  # LEFT
    elif keys[pygame.K_RIGHT]:
        return 3  # RIGHT
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
        'time': info.get('time_elapsed', 0),
        'fuel': info.get('fuel', None),
        'out_of_fuel': info.get('out_of_fuel', False),
    }


def main():
    args = parse_args()

    print("=" * 55)
    print("       LUNAR LANDER - VISUALISATION")
    print("=" * 55)
    print()
    
    # Afficher les contraintes
    print("Contraintes:")
    if args.fuel:
        print(f"  ✓ Carburant: {args.max_fuel}")
    else:
        print("  ✗ Carburant: illimité")
    print(f"  ✓ Temps max: {args.time}s")
    print()

    # Environnement avec mêmes contraintes que l'entraînement
    env_config = EnvironmentConfig(
        max_time=args.time,
        fuel_constraint=args.fuel,
        max_fuel=args.max_fuel,
    )
    env = LunarLanderEnv(env_config)

    # Charger le modèle
    policy_net = None
    if not args.manual and not args.random:
        if not os.path.exists(args.model):
            print(f"❌ Modèle non trouvé: {args.model}")
            print("\nModèles disponibles:")
            
            # Lister les modèles disponibles
            if os.path.exists("models"):
                for folder in os.listdir("models"):
                    model_path = os.path.join("models", folder, "final_model.pt")
                    if os.path.exists(model_path):
                        print(f"  - {model_path}")
            
            print("\nLancez d'abord l'entraînement avec les mêmes contraintes.")
            return

        print(f"Chargement: {args.model}")
        
        # Créer le réseau avec la bonne dimension d'état
        policy_net = DQN(env.observation_space_dim, env.action_space_dim)
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net_state'])
        policy_net.eval()
        print(f"  - État dim: {env.observation_space_dim}")

    if args.manual:
        print("\nContrôles:")
        print("  ↑ : Moteur principal")
        print("  ← : Tourner à droite")
        print("  → : Tourner à gauche")
        print("  ESC : Quitter")
        if args.fuel:
            print(f"\n⚠️  Attention: carburant limité à {args.max_fuel}!")

    print()

    renderer = Renderer(env_config.width, env_config.height)
    renderer.init()

    mode = 'manual' if args.manual else ('random' if args.random else 'ai')
    successes = 0
    out_of_fuel_count = 0

    try:
        for i in range(args.episodes):
            print(f"Episode {i+1}/{args.episodes}...", end=" ")
            
            result = play_episode(env, renderer, policy_net, mode, args.fps)
            
            if result.get('aborted'):
                break
            
            # Status
            if result['success']:
                status = "✓ Succès"
                successes += 1
            else:
                status = "✗ Échec"
            
            # Info carburant
            fuel_info = ""
            if args.fuel:
                if result.get('out_of_fuel'):
                    fuel_info = " | ⛽ VIDE!"
                    out_of_fuel_count += 1
                elif result.get('fuel') is not None:
                    fuel_info = f" | Fuel: {result['fuel']:.0f}"
            
            print(f"{status} | Reward: {result['reward']:.0f}{fuel_info}")
            
            time.sleep(0.5)

        print()
        print("-" * 40)
        print(f"Résultat: {successes}/{args.episodes} succès ({100*successes/args.episodes:.0f}%)")
        if args.fuel and out_of_fuel_count > 0:
            print(f"À court de carburant: {out_of_fuel_count} fois")

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
