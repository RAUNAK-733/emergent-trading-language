$ErrorActionPreference = "Stop"

python -m py_compile `
    agents/agent.py `
    env/trading_env.py `
    env/baseline.py `
    training/train.py `
    training/curriculum.py `
    analysis/verify.py `
    analysis/topsim.py `
    analysis/entropy.py `
    analysis/plot_training.py `
    analysis/probing.py `
    analysis/umap_viz.py `
    analysis/compare_runs.py `
    utils/checkpoints.py `
    utils/logger.py `
    utils/status.py `
    main.py

python -m unittest discover -s tests -v
python main.py status
