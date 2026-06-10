# Experiment Protocol

This project tests whether discrete messages improve cooperation between two
resource-trading agents.

## Primary Claim

Communication is useful only when normal-message performance is reliably better
than both zero-message and random-message controls.

## Standard Run

```bash
python main.py train --fresh --seed 7
python main.py analyze
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
python main.py train --fresh --seed 7
python main.py train --fresh --seed 17
python main.py train --fresh --seed 42
```

Keep each run's checkpoints and `training_log.json` separately when comparing
seeds. Report means and standard deviations rather than only the best run.
