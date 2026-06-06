# Contributing

Thank you for helping improve Emergent Trading Language.

## Development Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the basic checks before opening a pull request:

```bash
python -m py_compile agents/agent.py env/trading_env.py env/baseline.py training/train.py analysis/verify.py
python env/baseline.py
```

## Experiment Standards

Changes affecting training or evaluation should report:

- configuration and random seed;
- normal-message performance;
- zero-message performance;
- random-message performance;
- blind random baseline;
- whether checkpoints were trained before or after the change.

Do not claim emergent communication from trade success alone. A communication claim requires a clear advantage over message-ablation controls.

## Pull Requests

- Keep changes focused.
- Explain the research hypothesis behind behavioral changes.
- Include verification commands and relevant metrics.
- Do not commit checkpoints, generated figures, virtual environments, or unrelated projects.

