# Emergent Trading Language

[![tests](https://github.com/RAUNAK-733/emergent-trading-language/actions/workflows/ci.yml/badge.svg)](https://github.com/RAUNAK-733/emergent-trading-language/actions/workflows/ci.yml)
[![CodeQL](https://github.com/RAUNAK-733/emergent-trading-language/actions/workflows/codeql.yml/badge.svg)](https://github.com/RAUNAK-733/emergent-trading-language/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This research project studies emergent communication between
reinforcement-learning agents.

The basic idea is to place two agents in a small trading environment. Each agent has a private inventory and its own preferences for different resources. The agents can send a short discrete message before proposing a trade. I am studying whether they learn to use those messages in a meaningful way.

## Current progress

The following parts are working:

- trading environment with private inventories and preferences;
- random-agent baseline;
- PyTorch speaker and actor networks;
- policy-gradient training loop;
- verification using normal, zero, and random messages;
- heatmaps for inspecting symbol use;
- persistent JSON training metrics and training-curve plots;
- reproducible seeded runs and interruption-safe checkpoints;
- linear probing and UMAP representation analysis;
- isolated run directories and multi-seed comparison tooling.

An earlier experiment showed that agents could trade but did not depend on
language:

| Test condition | Efficiency |
|---|---:|
| Normal messages | 0.367 |
| Zero messages | 0.361 |
| Random messages | 0.365 |
| Random baseline | 0.092 |

The trained agents were better than random, but normal messages performed almost the same as zero and random messages. This means the agents learned a trading strategy, but it did not prove that they had learned a useful language.

I changed the environment after this result. Resources given away now have a
cost, trades must benefit both agents, and the actor has limited information so
that messages have a real purpose.

The latest paused team-reward run provides preliminary evidence that
communication helps:

| Test condition | Efficiency |
|---|---:|
| Normal messages | 0.197 |
| Zero messages | 0.000 |
| Random messages | 0.031 |

The minimum language advantage is `+0.166`, although useful trades remain at
about `6.4%`. This means the learned protocol is functionally important but
still has room to become more efficient and structured. See
`docs/CURRENT_RESULTS.md` for the full interpretation.

## How the model works

Each agent has two small neural networks:

- `speak_net` sees the agent's private state and sends a discrete symbol;
- `act_net` uses the agent's inventory and the received symbol to choose an offer.

The environment checks whether both agents can afford the trade and whether both benefit from it. Because an agent's give action benefits its partner, both agents train from the same cooperative team reward. Training starts with small offers, then gradually introduces larger offers while evaluation keeps the strict mutual-benefit rule.

The speaker is also trained with a small information regularizer. An early auxiliary task rewards messages that help the receiving actor identify which resource the sender values most. This guidance fades out as the strict trading objective takes over.

## Project structure

```text
agents/agent.py             agent speaker and actor networks
env/trading_env.py          trading environment
env/baseline.py             random-agent baseline
training/train.py           training loop
training/curriculum.py      curriculum scheduler
analysis/verify.py          communication checks and plots
analysis/topsim.py          topographic-similarity analysis
analysis/entropy.py         positional-entropy analysis
analysis/plot_training.py   training-curve plot
analysis/probing.py         linear utility probe
analysis/umap_viz.py        speaker-representation UMAP
analysis/compare_runs.py    multi-seed comparison
docs/EXPERIMENT_PROTOCOL.md standard evaluation protocol
docs/ARCHITECTURE.md        information flow and evidence design
docs/DEMO.md                presentation and experiment commands
tests/                      automated regression tests
```

## Running the project

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows with a compatible NVIDIA GPU, install the CUDA-enabled PyTorch
build after the other dependencies:

```bash
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Training automatically uses CUDA when it is available and prints the selected
device at startup.

Run the random baseline:

```bash
python main.py baseline
```

Train a new pair of agents:

```bash
python main.py train --fresh
```

Use an isolated directory when running experiments that must not overwrite one
another:

```bash
python main.py train --fresh --seed 7 --run-dir runs/seed-7
```

Training runs for 25,000 updates by default. Override it when running a larger
experiment:

```bash
python main.py train --fresh --updates 50000
```

Use a different seed for a repeat experiment:

```bash
python main.py train --fresh --seed 42
```

Training saves team-reward progress and JSON metrics every 500 updates, and
saves progress when interrupted with `Ctrl + C`.
Run `python main.py train` to resume the saved run.

Inspect the current checkpoint:

```bash
python main.py status
```

Verify whether the messages help:

```bash
python main.py verify
```

Analyze the learned communication system:

```bash
python analysis/topsim.py
python analysis/entropy.py
python analysis/plot_training.py
```

Topographic similarity checks whether similar private states produce similar
messages. Positional entropy measures how actively each message position uses
the available vocabulary. The training-curve plot compares normal-message
efficiency with the no-message control over time when JSON logs are available.

Run the full verification and analysis workflow:

```bash
python main.py analyze
```

This generates strict controls, heatmaps, topographic similarity, positional
entropy, a linear utility probe, UMAP representation plots, and a training
curve.

Compare several completed seed runs:

```bash
python main.py compare --runs runs/seed-7 runs/seed-17 runs/seed-42
```

See `docs/EXPERIMENT_PROTOCOL.md` for the standard evidence and reporting rules.
See `docs/ARCHITECTURE.md` for the information-flow design and `docs/DEMO.md`
for a concise presentation workflow.

On Windows, run all reliability checks with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_project.ps1
```

Run the standard three-seed experiment with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_seed_sweep.ps1
```

Run the environment tests:

```bash
python -m unittest discover -s tests -v
```

## Project Policies

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Support](SUPPORT.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Changelog](CHANGELOG.md)
- [Citation](CITATION.cff)

## What I am working on next

- finish the current team-reward training run;
- run experiments with several random seeds;
- compare normal messages with shuffled and removed messages;
- test larger resource spaces and longer messages.

The main goal is not only to achieve successful trades. The goal is to show whether communication itself improves cooperation.
