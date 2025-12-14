import argparse
import os
import sys

import pygame
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import LunarLanderEnv, EnvironmentConfig, Renderer, ACTIONS_NAMES
from agent import DQN


def parse_args():
    parser = argparse.ArgumentParser(description='Visualiser Lunar Lander')
    
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--fps', type=int, default=60)
    
    # Contraintes (doivent correspondre au modèle)
    parser.add_argument('--fuel', action='store_true')
    parser.add_argument('--max-fuel', type=float, default=100.0)
    parser.add_argument('--time', type=float, default=30.0)
    
    return parser.parse_args()


def load_model(model_path: str, state_dim: int, action_dim: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    first_layer_weight = checkpoint['policy_net_state']['network.0.weight']
    hidden_dim = first_layer_weight.shape[0]

    layer_count = sum(1 for k in checkpoint['policy_net_state'].keys()
                      if k.startswith('network.') and k.endswith('.weight'))
    num_hidden = layer_count - 1

    hidden_dims = tuple([hidden_dim] * num_hidden)
    print(f"Architecture détectée: {hidden_dims}")

    model = DQN(state_dim, action_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(checkpoint['policy_net_state'])
    model.eval()

    return model, device


def get_action_from_model(model, state, device):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        return model(state_tensor).argmax(dim=1).item()


def get_manual_action():
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 1
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 2
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 3
    else:
        return 0


def run_episode(env, renderer, model=None, device=None, manual=False, fps=60):
    state = env.reset()
    renderer.reset_particles()
    
    total_reward = 0
    done = truncated = False
    
    while not (done or truncated):
        if manual:
            action = get_manual_action()
        elif model is not None:
            action = get_action_from_model(model, state, device)
        else:
            action = 0
        
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if not renderer.render(env, {'reward': total_reward, 'action': ACTIONS_NAMES[action]}, fps=fps):
            return None, None
    
    return total_reward, info.get('success', False)


def main():
    args = parse_args()
    
    print("=" * 55)
    print("      LUNAR LANDER - VISUALIZATION")
    print("=" * 55)
    print()
    
    if args.manual:
        print("Mode: MANUEL")
        print("Contrôles:")
        print("  ↑ ou W : Réacteur principal")
        print("  ← ou A : Réacteur gauche (tourne à droite)")
        print("  → ou D : Réacteur droit (tourne à gauche)")
        print("  ESC    : Quitter")
    elif args.model:
        print(f"Mode: AGENT")
        print(f"Modèle: {args.model}")
    else:
        print("Erreur: Spécifiez --model ou --manual")
        return
    
    print()
    
    env_config = EnvironmentConfig(
        width=800,
        height=600,
        gravity=15.0,
        max_time=args.time,
        fps=args.fps,
        fuel_constraint=args.fuel,
        max_fuel=args.max_fuel,
    )
    env = LunarLanderEnv(env_config)
    
    model, device = None, None
    if args.model and os.path.exists(args.model):
        model, device = load_model(args.model, env.observation_space_dim, env.action_space_dim)
        print(f"Modèle chargé sur {device}")
    elif args.model:
        print(f"Erreur: Modèle non trouvé: {args.model}")
        return
    
    renderer = Renderer(env_config.width, env_config.height)
    renderer.init()
    
    successes = 0
    rewards = []
    
    try:
        for ep in range(args.episodes):
            print(f"\nÉpisode {ep + 1}/{args.episodes}")
            
            reward, success = run_episode(
                env, renderer, model, device, 
                manual=args.manual, fps=args.fps
            )
            
            if reward is None:
                break
            
            rewards.append(reward)
            if success:
                successes += 1
                print(f"  ✓ Succès! Récompense: {reward:.1f}")
            else:
                print(f"  ✗ Échec. Récompense: {reward:.1f}")
            
            print("  Appuyez sur une touche...")
            if not renderer.wait_for_key():
                break
    
    finally:
        renderer.close()
    
    if rewards:
        print()
        print("=" * 55)
        print("                 RÉSULTATS")
        print("=" * 55)
        print(f"Épisodes: {len(rewards)}")
        print(f"Succès: {successes}/{len(rewards)} ({100*successes/len(rewards):.1f}%)")
        print(f"Récompense moyenne: {np.mean(rewards):.1f}")


if __name__ == "__main__":
    main()
