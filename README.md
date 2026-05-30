# Multi-Agent Trading System with Reinforcement Learning

A sophisticated multi-agent system where neural network agents learn to negotiate and execute trades through learned communication protocols.

## 🎯 Project Overview

This project builds autonomous agents that communicate through coded signals and learn to make beneficial trades through reinforcement learning. The baseline for two random agents achieving a valid trade is **18.6%** — our trained agents aim to achieve **3–4x better performance**.

## 📊 Project Roadmap & Progress

### Level 1 — The Trading Room ✅ **DONE**
**Status:** Complete

The foundational environment where agents will interact.

**What was built:**
- `TradingEnv`: Full trading environment with transaction validation
- Random baseline agents: 18.6% success rate on valid trades
- Environment metrics and reward calculation

**Success criteria met:** ✓ Environment stable and ready for agent training

---

### Level 2 — The Agent Brains 🧠 **IN PROGRESS**
**Status:** Current Focus

The neural network brain that powers each agent with communication and decision-making.

**What you will build:**
- **Neural Network Architecture:**
  - `speak()`: Encodes trade intent into 4-signal communication protocol
  - `act()`: Decodes opponent signals and determines trade response
  
- **Agent Logic:**
  - Learns which signal to broadcast to other agents
  - Learns how to interpret received signals
  - Develops adaptive trading strategies through self-play

Think of it like a person who can only communicate using 4 hand signals — the brain learns which signal to show and how to interpret signals it receives.

**Success criteria:**
- [ ] Run `python agents/agent.py` with no errors
- [ ] Shapes printed correctly (network architecture validation)
- [ ] Network wires correctly without training

---

### Level 3 — Training 🔒 **LOCKED**
**Status:** Waiting for Level 2 Completion

End-to-end training pipeline with the environment and agent brains.

---

### Level 4 — Verify Language 🔒 **LOCKED**
**Status:** Waiting for Level 3 Completion

Analyze learned communication protocols to understand agent strategies.

---

### Level 5 — Research Analysis 🔒 **LOCKED**
**Status:** Waiting for Level 4 Completion

Extract insights and publish findings.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Major\ project

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: Deep learning framework for agent neural networks
- **NumPy & SciPy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **UMAP**: Dimensionality reduction for analysis
- **Matplotlib & Seaborn**: Visualization
- **Weights & Biases (wandb)**: Experiment tracking
- **editdistance**: String similarity for protocol analysis

### Current Development

```bash
# Test agent brain architecture (Level 2)
python agents/agent.py
```

## 📁 Project Structure

```
Major project/
├── agents/              # Agent implementations
│   ├── agent.py        # Agent neural network brain
│   └── ...
├── training/           # Training pipeline (Level 3)
├── analysis/           # Analysis tools (Level 4+)
├── checkpoints/        # Model checkpoints
├── figures/            # Generated visualizations
├── utils/              # Utility functions
├── main.py             # Entry point
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🎓 How It Works

### Agent Communication Protocol

Agents communicate using exactly **4 signals**:
- Each signal encodes information about: offer amount, willingness to trade, etc.
- The `speak()` function generates these signals based on agent state
- The `act()` function interprets opponent signals and decides action

### Learning Process

1. **Environment**: TradingEnv simulates market conditions and validates trades
2. **Agent Brain**: Neural network learns to map game state → signals (speak) and signals → actions (act)
3. **Reward Signal**: Valid trade execution → positive reward; failed trade → negative reward
4. **Self-Play**: Agents compete against each other to improve strategies

### Success Metric

- **Baseline**: 18.6% (random agents)
- **Target**: 55–74% (3–4x improvement)

## 📝 Status & Updates

**Current Focus:** Implementing agent neural network architecture (Level 2)

This project is **actively maintained** and updated regularly. Check back for progress on each level.

## 🤝 Contributing

Contributions welcome! Please coordinate on levels:
- Completed levels are stable
- Current level (2) may have rapid changes
- Locked levels are not ready yet

## 📄 License

MIT License

---

**Last Updated:** 2026-05-30  
**Maintained By:** [@your-username]
