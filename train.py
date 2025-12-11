#!/usr/bin/env python3
"""
Lunar Lander DQN - Entraînement (avec contraintes)
==================================================

Contraintes disponibles:
    --fuel           Active la contrainte de carburant
    --max-fuel N     Quantité de carburant (défaut: 100)
    --time N         Temps maximum en secondes (défaut: 30)

Les modèles sont sauvegardés dans des dossiers spécifiques:
    models/base/              Sans contrainte
    models/fuel_100/          Avec carburant 100
    models/time_20/           Avec temps 20s
    models/fuel_50_time_15/   Combiné

Usage:
    python train.py                          # Base
    python train.py --fuel                   # Avec carburant
    python train.py --fuel --max-fuel 50     # Carburant limité
    python train.py --time 20                # Temps réduit
    python train.py --fuel --time 15         # Combiné
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import LunarLanderEnv, EnvironmentConfig, Renderer
from agent import DQNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Entraîner Lunar Lander DQN')
    
    # Entraînement
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Nombre d\'épisodes (défaut: 2000)')
    parser.add_argument('--render', action='store_true',
                        help='Afficher le jeu')
    parser.add_argument('--render-every', type=int, default=50,
                        help='Afficher tous les N épisodes (défaut: 50)')
    
    # Hyperparamètres
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (défaut: 5e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (défaut: 0.99)')
    parser.add_argument('--epsilon-decay', type=float, default=5000,
                        help='Epsilon decay (défaut: 5000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (défaut: 64)')
    
    # Contraintes
    parser.add_argument('--fuel', action='store_true',
                        help='Activer la contrainte de carburant')
    parser.add_argument('--max-fuel', type=float, default=100.0,
                        help='Quantité de carburant max (défaut: 100)')
    parser.add_argument('--time', type=float, default=30.0,
                        help='Temps maximum en secondes (défaut: 30)')
    
    # Chargement
    parser.add_argument('--load', type=str, default=None,
                        help='Charger un checkpoint')
    
    return parser.parse_args()


def get_model_dir(args) -> str:
    """
    Génère le nom du dossier selon les contraintes actives.
    """
    parts = []
    
    if args.fuel:
        parts.append(f"fuel_{int(args.max_fuel)}")
    
    if args.time != 30.0:
        parts.append(f"time_{int(args.time)}")
    
    if not parts:
        return "models/base"
    
    return "models/" + "_".join(parts)


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
    
    # Déterminer le dossier de sauvegarde
    model_dir = get_model_dir(args)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 55)
    print("    LUNAR LANDER - DQN TRAINING")
    print("=" * 55)
    print()
    
    # Afficher les contraintes actives
    print("Contraintes:")
    if args.fuel:
        print(f"  ✓ Carburant: {args.max_fuel}")
    else:
        print("  ✗ Carburant: illimité")
    
    print(f"  ✓ Temps max: {args.time}s")
    print(f"\nModèles sauvegardés dans: {model_dir}/")
    print()
    
    # Environnement avec contraintes
    env_config = EnvironmentConfig(
        width=800,
        height=600,
        gravity=15.0,
        max_time=args.time,
        fps=60,
        fuel_constraint=args.fuel,
        max_fuel=args.max_fuel,
    )
    env = LunarLanderEnv(env_config)
    
    print(f"Environnement:")
    print(f"  - États: {env.observation_space_dim}")
    print(f"  - Actions: {env.action_space_dim}")
    print(f"  - FPS: {env_config.fps}")
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
        save_dir=model_dir,
    )
    
    print(f"Agent DQN:")
    print(f"  - LR: {args.lr}")
    print(f"  - Gamma: {args.gamma}")
    print(f"  - Epsilon decay: {args.epsilon_decay}")
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
    print("=" * 55)
    print("                 RÉSUMÉ")
    print("=" * 55)
    
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
    print(f"Modèle sauvegardé: {model_dir}/final_model.pt")
    print()
    
    # Commande pour visualiser
    cmd_parts = ["python play.py"]
    cmd_parts.append(f"--model {model_dir}/final_model.pt")
    if args.fuel:
        cmd_parts.append("--fuel")
        if args.max_fuel != 100.0:
            cmd_parts.append(f"--max-fuel {args.max_fuel}")
    if args.time != 30.0:
        cmd_parts.append(f"--time {args.time}")
    
    print("Pour visualiser:")
    print(f"  {' '.join(cmd_parts)}")


if __name__ == "__main__":
    main()
