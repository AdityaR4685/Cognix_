# COGNIX Architecture Overview

The COGNIX architecture is built around a unidirectional data flow from raw agent predictions to a system decision with uncertainty and escalation metadata. This architecture does not establish physical driving safety.

The evaluated graph mechanism in closed Experiment 2B compares NoGraph, StandardGAT, and an EpistemicGAT sender prior `w_j = 1 / (1 + E_j)` (attention-logit adjustment `-log(1 + E_j)`). StandardGAT improved held-out TEST BCE across all five frozen seeds while reducing frozen-threshold corruption F1 across all five. The sender prior showed no descriptive evidence of additional benefit. COGNIX's contribution is the uncertainty-aware architecture and evaluation of its metric-dependent behavior; experimental superiority of this prior is not established. `q_normal` represents P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task and must not be interpreted as a probability of physical driving safety. See the [final report](../research/COGNIX_FINAL_STATUS.md#6-experiment-2b-closed-held-out-graph-evaluation).

## Pipeline Architecture

```ascii
[Agent 1: Camera]       [Agent 2: Lidar]      [Agent N...]
       │                       │                   │
       ▼                       ▼                   ▼
┌──────────────┐        ┌──────────────┐     ┌──────────────┐
│ Uncertainty  │        │ Uncertainty  │     │ Uncertainty  │
│ Estimator    │        │ Estimator    │     │ Estimator    │
└──────┬───────┘        └──────┬───────┘     └──────┬───────┘
       │ (Predict, Epistemic,  │                    │
       │  Aleatoric)           │                    │
       ▼                       ▼                    ▼
       └───────────────┬───────┘────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   Belief Fusion     │ ◄── Epistemic-Weighted Strategy
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │    Calibration      │ ◄── Ensure P(y|x) matches accuracy
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Decision Engine    │ ◄── Outputs (ACT, WAIT, ABSTAIN)
            └─────────────────────┘
```

## Core Principles
1. **Uncertainty as a First-Class Citizen**: No prediction is passed without its uncertainty profile.
2. **Epistemic vs. Aleatoric Isolation**: We explicitly isolate model ignorance (epistemic) from data noise (aleatoric).
3. **Fail-Safe Defaults**: If total uncertainty exceeds the safety threshold, the system defaults to ABSTAIN or ESCALATE.
4. **Transparent Attribution**: The `explainability` module can always trace a bad decision back to the specific agent and uncertainty type that caused it.
