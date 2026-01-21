# AITimeStepper Training Refactor

## What This Is

A refactored training routine for AITimeStepper that uses a two-phase approach: iterative trajectory collection with energy-based quality gates, followed by generalization training on validated data. This replaces the existing training loop to ensure the model only learns from physically valid transitions.

## Core Value

**Every training sample must satisfy energy conservation before being accepted.** The model learns from validated physics, not noisy approximations.

## Requirements

### Validated

- ✓ Differentiable N-body physics engine (ParticleTorch) — existing
- ✓ Feature extraction with history support (HistoryBuffer, nbody_features) — existing
- ✓ Neural network for dt prediction (FullyConnectedNN) — existing
- ✓ Energy/momentum conservation loss functions — existing
- ✓ Checkpoint save/load with config persistence — existing
- ✓ CLI interface via runner.py — existing
- ✓ Simulation mode with ML-predicted dt — existing
- ✓ W&B experiment tracking (optional) — existing

### Active

- [ ] Part 1: Iterative trajectory collection
  - Predict dt, advance particles, check energy threshold
  - If fails: retrain on same state, loop until threshold met
  - Record accepted state to trajectory
  - Parametrizable steps per epoch (default: 1)

- [ ] Part 2: Generalization training
  - Random minibatch sampling from collected trajectory
  - Train until ALL samples pass energy threshold
  - Single timestep predictions

- [ ] Epoch structure: Part 1 + Part 2, repeated for fixed N epochs

- [ ] History bootstrapping: Pad with zeros initially, discard early steps once real trajectory exists

- [ ] Replace existing training routine entirely (remove old loop)

### Out of Scope

- Variable particle count generalization — training uses fixed N
- Re-verification after Part 2 — once data accepted, it stays
- Max retry limit in Part 1 — iterate until converged (no cap)
- Separate models for Part 1/Part 2 — single shared model

## Context

**Existing codebase:**
- Training loop in `run/runner.py` (run_training function)
- Loss functions in `src/losses.py` and `src/losses_history.py`
- Feature extraction via `ModelAdapter` which wraps HistoryBuffer
- Current approach: multi-step integration per loss evaluation, standard gradient descent

**New approach differs fundamentally:**
- Single-step integration with accept/reject loop
- Data quality gate before recording
- Two-phase structure per epoch
- Convergence defined by all-samples-pass, not loss threshold

**History handling:**
- HistoryBuffer already pads with oldest state when history incomplete
- New requirement: pad with zeros instead, discard warmup steps

## Constraints

- **Tech stack**: Python/PyTorch — maintain existing stack
- **Fixed N**: Model trained for specific particle count
- **Energy threshold**: Central parameter — must be configurable
- **Backward compatibility**: Simulation mode should still work with new checkpoints

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace existing training | New approach is fundamentally different, not a mode variation | — Pending |
| No retry cap in Part 1 | User preference: iterate until physics is satisfied | — Pending |
| Trust accepted data (no re-verify) | Part 2 generalizes, doesn't invalidate Part 1 quality | — Pending |
| Zero-padding for history bootstrap | Cleaner than repeating first state, discard warmup anyway | — Pending |

---
*Last updated: 2026-01-20 after initialization*
