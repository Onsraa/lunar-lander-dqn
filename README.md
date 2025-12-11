# 🚀 Lunar Lander DQN

### 1. Présentation du projet

(plus tard à l'aide)

### 2. Hyperparamètres

| Paramètre | Avant | Après | Pourquoi |
|-----------|-------|-------|----------|
| `epsilon_decay` | 100000 | **5000** | Exploration trop longue |
| `fps` | 240 | **60** | Trop de steps inutiles |
| `max_time` | 60s | **30s** | Force l'agent à agir |
| Actions | 6 | **4** | Plus simple |

### 3. Simplifications

- État: 8 → **6 dimensions** (suppression de dx/dy redondants)
- Réseau: 3 couches → **2 couches** (suffisant)
- Code: suppression de Dueling DQN et PER (non nécessaires)

## 📁 Structure

```
lunar_lander_fix/
├── game/
│   ├── lander.py       # Physique
│   ├── environment.py  # Env + Rewards (CORRIGÉ)
│   └── renderer.py     # Affichage
├── agent/
│   ├── dqn.py          # Réseau simple
│   ├── replay_buffer.py
│   └── trainer.py      # Hyperparamètres (CORRIGÉS)
├── train.py            # Entraînement
├── play.py             # Visualisation
└── requirements.txt
```

## 🚀 Utilisation

```bash
# Installation
pip install -r requirements.txt

# Entraînement (~5-10 min)
python train.py

# Avec visualisation
python train.py --render

# Visualiser l'agent entraîné
python play.py

# Mode manuel
python play.py --manual
```

## 🎮 Contrôles (mode manuel)

- `↑` : Moteur principal (freiner)
- `←` : Tourner vers la droite
- `→` : Tourner vers la gauche
- `ESC` : Quitter
