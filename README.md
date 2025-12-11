# 🚀 Lunar Lander DQN

### 1. Présentation du projet

(plus tard à l'aide)

## 📁 Structure

```
lunar_lander_fix/
├── game/
│   ├── lander.py       # Physique
│   ├── environment.py  # Env + Rewards 
│   └── renderer.py     # Affichage
├── agent/
│   ├── dqn.py          # Réseau de neurones simple
│   ├── replay_buffer.py
│   └── trainer.py      # Hyperparamètres 
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
