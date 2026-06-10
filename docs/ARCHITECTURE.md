# Architecture

## Information Flow

```text
private inventory + private utility
              |
          speak_net
              |
       discrete message
              |
partner inventory + received message
              |
           act_net
              |
        resources given
```

The speaker sees the full private observation. The actor cannot directly access
utility preferences; it sees only its own inventory and the partner's message.
This information bottleneck creates pressure for messages to communicate useful
private information.

## Trading Semantics

- `give_a_to_b`: resources Agent A gives to Agent B;
- `give_b_to_a`: resources Agent B gives to Agent A;
- Agent A benefits from what B gives;
- Agent B benefits from what A gives;
- a trade succeeds only when both agents benefit.

Both policies train from the same team reward because each give action mainly
affects the partner's outcome.

## Evidence Pipeline

1. Train with a fixed seed and isolated run directory.
2. Evaluate normal, zero, and random messages.
3. Measure language advantage over both controls.
4. Inspect symbol entropy and topographic similarity.
5. Probe messages for private utility information.
6. Visualize speaker representations with UMAP.
7. Repeat across seeds and compare means and standard deviations.

The strict control experiments establish whether communication is useful.
Structural analyses explain what the learned communication may encode.
