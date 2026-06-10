# Current Results

These are preliminary results from the latest paused checkpoint at update
13,339, evaluated on June 10, 2026.

## Strict Communication Controls

| Condition | Valid trades | Useful trades | Efficiency | Average give |
|---|---:|---:|---:|---:|
| Normal learned messages | 49.2% | 6.1% | 0.191 | 2.000 |
| Zero messages | 0.0% | 0.0% | 0.000 | 2.644 |
| Random messages | 7.2% | 1.0% | 0.029 | 2.475 |

The normal-message policy exceeds the zero-message control by `0.191` and the
random-message control by `0.162`. Removing or corrupting messages causes a
large behavioral failure, which is strong evidence that the agents use their
communication channel.

## Language Structure

| Analysis | Result | Interpretation |
|---|---:|---|
| Topographic similarity | 0.114 | Weak structural relationship |
| Positional entropy | 0.997 bits | About 50% of maximum vocabulary entropy |

The language is functionally useful, but it is not yet strongly compositional.
The agents mainly use two of the four available symbols. This is a useful and
honest result: communication emerged before a rich symbolic structure did.

## Current Claim

> The agents learned a single-round discrete communication protocol that is
> necessary for coordinating mutually beneficial trades in the current
> environment.

The project does not yet claim human-like negotiation, multi-round bargaining,
or a fully compositional grammar.

## Next Evidence Needed

- finish the current 25,000-update run;
- repeat the experiment across several random seeds;
- report mean and standard deviation for all controls;
- improve the useful-trade rate without weakening communication pressure;
- compare vocabulary sizes and message lengths.
