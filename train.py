#!/usr/bin/env python3
"""
Lunar Lander DQN - Entraînement (Version Corrigée)
==================================================

Corrections principales:
1. epsilon_decay réduit (5000 au lieu de 100000)
2. FPS réduit (60 au lieu de 240)
3. Reward shaping corrigé dans environment.py

Usage:
    python train.py                     # Entraînement standard
    python train.py --episodes 3000     # Plus d'épisodes
    python train.py --render            # Avec visualisation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import LunarLanderEnv, EnvironmentConfig, Renderer
from agent import DQNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Entraîner Lunar Lander DQN')
    
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Nombre d\'épisodes (défaut: 2000)')
    parser.add_argument('--render', action='store_true',
                        help='Afficher le jeu')
    parser.add_argument('--render-every', type=int, default=50,
                        help='Afficher tous les N épisodes (défaut: 50)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (défaut: 5e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (défaut: 0.99)')
    parser.add_argument('--epsilon-decay', type=float, default=5000,
                        help='Epsilon decay (défaut: 5000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (défaut: 64)')
    parser.add_argument('--load', type=str, default=None,
                        help='Charger un checkpoint')
    
    return parser.parse_args()


def train_with_render(trainer: DQNTrainer, env: LunarLanderEnv,
                      renderer: Renderer, num_episodes: int, render_every: int):
    """Entraîne avec affichage périodique."""
    from tqdm import tqdm
    import numpy as np
    import time

    start_time = time.time()
    progress = tqdm(range(num_episodes), desc="Training")

    for episode in progress:
        state = env.reset()
        total_reward = 0
        done = truncated = False
        
        should_render = (episode % render_every == 0)
        if should_render:
            renderer.reset_particles()

        while not (done or truncated):
            action = trainer.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            trainer.memory.push(state, action, reward, next_state, done or truncated)
            trainer.optimize_model()

            trainer.steps_done += 1
            if trainer.steps_done % trainer.target_update == 0:
                trainer.update_target_network()

            state = next_state
            total_reward += reward

            if should_render:
                if not renderer.render(env, {'reward': total_reward}, fps=60):
                    trainer.training_stats['training_time'] = time.time() - start_time
                    return trainer.training_stats

        trainer.episodes_done += 1
        trainer.training_stats['episode_rewards'].append(total_reward)
        trainer.training_stats['episode_successes'].append(info.get('success', False))
        trainer.training_stats['epsilons'].append(trainer.get_epsilon())
        trainer.training_stats['losses'].append(0)
        trainer.training_stats['episode_lengths'].append(0)

        # Affichage
        recent_r = trainer.training_stats['episode_rewards'][-100:]
        recent_s = trainer.training_stats['episode_successes'][-100:]
        progress.set_postfix({
            'R': f"{np.mean(recent_r):.0f}",
            'S': f"{100*np.mean(recent_s):.0f}%",
            'ε': f"{trainer.get_epsilon():.2f}"
        })

    trainer.training_stats['training_time'] = time.time() - start_time
    return trainer.training_stats


def main():
    args = parse_args()
    
    print("=" * 50)
    print("    LUNAR LANDER - DQN TRAINING (Corrigé)")
    print("=" * 50)
    print()
    
    # Environnement
    env_config = EnvironmentConfig(
        width=800,
        height=600,
        gravity=15.0,
        max_time=30.0,
        fps=60,
    )
    env = LunarLanderEnv(env_config)
    
    print(f"Environnement:")
    print(f"  - États: {env.observation_space_dim}")
    print(f"  - Actions: {env.action_space_dim}")
    print(f"  - FPS: {env_config.fps}")
    print(f"  - Temps max: {env_config.max_time}s")
    print()
    
    # Agent
    trainer = DQNTrainer(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        hidden_dims=(128, 128),
    )
    
    print(f"Agent DQN:")
    print(f"  - LR: {args.lr}")
    print(f"  - Gamma: {args.gamma}")
    print(f"  - Epsilon decay: {args.epsilon_decay}")
    print(f"  - Batch size: {args.batch_size}")
    print()
    
    if args.load:
        print(f"Chargement: {args.load}")
        trainer.load_checkpoint(args.load)
    
    print(f"Entraînement: {args.episodes} épisodes")
    print()

    # Entraînement
    if args.render:
        renderer = Renderer(env_config.width, env_config.height)
        renderer.init()
        try:
            stats = train_with_render(trainer, env, renderer, 
                                     args.episodes, args.render_every)
        finally:
            renderer.close()
    else:
        stats = trainer.train(env, num_episodes=args.episodes, 
                             save_every=100, verbose=True)
    
    # Résumé
    print()
    print("=" * 50)
    print("                 RÉSUMÉ")
    print("=" * 50)
    
    import numpy as np
    final_r = stats['episode_rewards'][-100:]
    final_s = stats['episode_successes'][-100:]
    
    print(f"Épisodes: {trainer.episodes_done}")
    print(f"Steps: {trainer.steps_done:,}")
    print(f"Temps: {stats['training_time']:.1f}s")
    print()
    print(f"Performances (100 derniers):")
    print(f"  - Récompense: {np.mean(final_r):.1f}")
    print(f"  - Succès: {100*np.mean(final_s):.1f}%")
    print()
    print("Pour visualiser: python play.py")


if __name__ == "__main__":
    main()
