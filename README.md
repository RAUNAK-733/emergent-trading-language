# Emergent Trading Language

[![tests](https://github.com/RAUNAK-733/emergent-trading-language/actions/workflows/ci.yml/badge.svg)](https://github.com/RAUNAK-733/emergent-trading-language/actions/workflows/ci.yml)

This is my major project on emergent communication between reinforcement-learning agents.

The basic idea is to place two agents in a small trading environment. Each agent has a private inventory and its own preferences for different resources. The agents can send a short discrete message before proposing a trade. I am studying whether they learn to use those messages in a meaningful way.

## Current progress

The following parts are working:

- trading environment with private inventories and preferences;
- random-agent baseline;
- PyTorch speaker and actor networks;
- policy-gradient training loop;
- verification using normal, zero, and random messages;
- heatmaps for inspecting symbol use.

An earlier experiment produced these results:

| Test condition | Efficiency |
|---|---:|
| Normal messages | 0.367 |
| Zero messages | 0.361 |
| Random messages | 0.365 |
| Random baseline | 0.092 |

The trained agents were better than random, but normal messages performed almost the same as zero and random messages. This means the agents learned a trading strategy, but it did not prove that they had learned a useful language.

I changed the environment after this result. Resources given away now have a cost, trades must benefit both agents, and the actor has limited information so that messages have a real purpose. New experiments are still needed for this version.

## How the model works

Each agent has two small neural networks:

- `speak_net` sees the agent's private state and sends a discrete symbol;
- `act_net` uses the agent's inventory and the received symbol to choose an offer.

The environment checks whether both agents can afford the trade and whether both benefit from it. Training starts with small offers and a simple partner-value signal, then gradually introduces larger offers and the strict mutual-benefit reward used during evaluation.

## Project structure

```text
agents/agent.py          agent speaker and actor networks
env/trading_env.py       trading environment
env/baseline.py          random-agent baseline
training/train.py        training loop
analysis/verify.py       communication checks and plots
tests/test_trading_env.py
```

The remaining files in `analysis/`, `training/`, and `utils/` are planned work for later stages of the project.

## Running the project

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the random baseline:

```bash
python main.py baseline
```

Train a new pair of agents:

```bash
python main.py train
```

Verify whether the messages help:

```bash
python main.py verify
```

Run the environment tests:

```bash
python -m unittest discover -s tests -v
```

## What I am working on next

- retrain the agents using the updated environment;
- run experiments with several random seeds;
- compare normal messages with shuffled and removed messages;
- add topographic similarity and entropy analysis;
- test larger resource spaces and longer messages.

The main goal is not only to achieve successful trades. The goal is to show whether communication itself improves cooperation.
