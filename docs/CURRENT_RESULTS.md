# Current Results

These are preliminary results from the latest paused checkpoint at update
13,339, evaluated on June 10, 2026.

## Strict Communication Controls

| Condition | Valid trades | Useful trades | Efficiency | Average give |
|---|---:|---:|---:|---:|
| Normal learned messages | 50.4% | 6.4% | 0.197 | 2.000 |
| Zero messages | 0.0% | 0.0% | 0.000 | 2.627 |
| Random messages | 8.1% | 1.1% | 0.031 | 2.448 |

The normal-message policy exceeds the zero-message control by `0.197` and the
random-message control by `0.166`. Removing or corrupting messages causes a
large behavioral failure, which is strong evidence that the agents use their
communication channel.

## Language Structure

| Analysis | Result | Interpretation |
|---|---:|---|
| Topographic similarity | 0.114 | Weak structural relationship |
| Positional entropy | 0.999 bits | About 50% of maximum vocabulary entropy |
| Utility-preference probe | 99.6% accuracy | Messages almost perfectly encode preferred resource |

The language is functionally useful, but it is not yet strongly compositional.
The agents mainly use two of the four available symbols. This is a useful and
honest result: communication emerged before a rich symbolic structure did.

The probe result explains the strong behavioral dependence: the learned symbol
nearly determines which resource the sender privately prefers.

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
