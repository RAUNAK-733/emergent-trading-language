# Demo Guide

## Show Current Status

```bash
python main.py status
```

## Resume Training

```bash
python main.py train
```

## Run Strict Verification

```bash
python main.py verify
```

## Generate Every Analysis

```bash
python main.py analyze
```

The strongest demo is the communication-control comparison. Explain that normal
messages retain trading performance while zero and random messages damage it.

## Run Independent Seed Experiments

```bash
python main.py train --fresh --seed 7 --run-dir runs/seed-7
python main.py train --fresh --seed 17 --run-dir runs/seed-17
python main.py train --fresh --seed 42 --run-dir runs/seed-42
```

Analyze one run:

```bash
python main.py analyze --run-dir runs/seed-7 --figure-dir figures/seed-7
```

Compare runs:

```bash
python main.py compare --runs runs/seed-7 runs/seed-17 runs/seed-42
```
