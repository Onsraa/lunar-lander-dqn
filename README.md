# 🚀 Lunar Lander DQN

### 1. Présentation du projet

(plus tard à l'aide)

## 📁 Structure

```
lunar-lander/
├── game/
│   ├── lander.py      
│   ├── environment.py  
│   └── renderer.py     
├── agent/
│   ├── dqn.py           
│   ├── replay_buffer.py
│   └── trainer.py       
├── train.py             
├── play.py              
└── requirements.txt
```

## 🚀 Utilisation

```bash
# Base (sans contrainte)
python train.py

# Avec carburant limité
python train.py --fuel

# Avec carburant réduit 
python train.py --fuel --max-fuel 50

# Avec temps réduit
python train.py --time 20

# Combiné 
python train.py --fuel --max-fuel 50 --time 15
``` 

## 🎮 Contrôles (mode manuel)

```bash
# Lance une partie en tant que joueur
python play.py --manual
```

- `↑` : Moteur principal
- `←` : Moteur gauche
- `→` : Moteur droit
- `ESC` : Quitter
