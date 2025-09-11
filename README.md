# Multi-Agent Bird Localisation with PPO

This project is a multi-agent reinforcement learning (MARL) experiment where agents collaborate to identify the locations of hidden "birds" in a 2D environment.

Agents can move around, take noisy measurements of overlapping bird calls, and exchange information with nearby neighbors.
The group forms a consensus heatmap and proposes top-K guesses of bird locations.

A shared PPO policy (Stable-Baselines3) trains the agents to maximize group reward by improving the accuracy of these guesses.

Currently, this project only implements a single agent identifying the total number of birds in the environment.

---

##  Overview

The project aims to demonstrate the usage of reinforcement learning and consensus algorithms to generate a unified prediction of the location of birds in the environment.

It features a custom Gymnasium environment, Stable-Baselines3’s PPO implementation, and basic visualisation.

---

## Project Structure

```
bird-localisation-via-reinforcement-learning
├── bird_env.py      # Gymnasium environment and PPO training
├── view_birds.py    # View a trained model
├── requirements.txt # Python dependencies
└── README.md        # This documentation
```

---

##  Getting Started

### Prerequisites

- Requires Python 3.9+ and pip

`git clone https://github.com/yourusername/multiagent-bird-localisation.git
cd multiagent-bird-localisation

pip install -r requirements.txt`

### Usage
1. Train agents
`python bird_env.py`

Trains PPO model.

Saves the trained model as trained_bird_chaser.zip

2. Watch a trained run
python view_birds.py

Loads the saved PPO model and runs visualisation over 10 runs.

## Roadmap
The project is being developed iteratively, with each step adding complexity to the environment and agents’ capabilities.

- [x] **1. Simple environment with random birds and agent randomly moving**
- [x] **2. Add reward printout**
- [x] **3. Estimate total bird count**
  - No neighbour communication included in observations.    

Next steps:  

- [ ] **4. Multiple agents estimating birds**
- [ ] **5. Heat map of bird locations**  
- [ ] **6. Top-K bird locations**  
- [ ] **7. Gossip**
  – Incorporate neighbour observations to allow consensus to emerge *without “magic averaging” over the whole group*.  
