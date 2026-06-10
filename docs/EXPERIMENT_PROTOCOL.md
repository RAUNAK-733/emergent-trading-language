# Experiment Protocol

This project tests whether discrete messages improve cooperation between two
resource-trading agents.

## Primary Claim

Communication is useful only when normal-message performance is reliably better
than both zero-message and random-message controls.

## Standard Run

```bash
python main.py train --fresh --seed 7 --run-dir runs/seed-7
python main.py analyze --run-dir runs/seed-7 --figure-dir figures/seed-7
```

Training uses 25,000 updates by default. Each update contains 64 independently
sampled trading episodes.

## Metrics

- **Valid trade rate:** both agents can afford the exchange and both benefit.
- **Useful trade rate:** efficiency reaches at least 60% of the estimated optimum.
- **Efficiency:** achieved joint reward divided by estimated optimal joint reward.
- **Language advantage:** normal-message efficiency minus zero-message efficiency.
- **Average give:** total number of resource units given by both agents.

## Communication Controls

Final evidence must report:

1. normal learned messages;
2. messages replaced with zeros;
3. messages replaced with random symbols;
4. advantage over both controls.

Topographic similarity and entropy describe message structure, but they do not
replace the control experiments. A structured language is useful only if it
improves behavior.

## Reliable Reporting

Use several random seeds before making a final claim:

```bash
python main.py train --fresh --seed 7 --run-dir runs/seed-7
python main.py train --fresh --seed 17 --run-dir runs/seed-17
python main.py train --fresh --seed 42 --run-dir runs/seed-42
```

Compare the runs with:

```bash
python main.py compare --runs runs/seed-7 runs/seed-17 runs/seed-42
```

Report means and standard deviations rather than only the best run.
