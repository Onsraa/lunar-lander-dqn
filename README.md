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
# Base (sans contrainte)
python train.py

# Avec carburant limité
python train.py --fuel

# Avec carburant réduit (plus difficile)
python train.py --fuel --max-fuel 50

# Avec temps réduit
python train.py --time 20

# Combiné (très difficile!)
python train.py --fuel --max-fuel 50 --time 15
``` 

## 🎮 Contrôles (mode manuel)

- `↑` : Moteur principal (freiner)
- `←` : Tourner vers la droite
- `→` : Tourner vers la gauche
- `ESC` : Quitter
