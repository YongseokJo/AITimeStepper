# ROADMAP: AITimeStepper Training Refactor

**Created:** 2026-01-20
**Milestone:** v1.0 - Two-Phase Training System
**Mode:** standard (5-8 phases, 3-5 plans each)

---

## Overview

Replace the existing multi-step integration training loop with a two-phase system:
- **Part 1**: Iterative trajectory collection with energy-based quality gates
- **Part 2**: Generalization training on validated data

Each epoch combines both parts. Training converges when all samples pass energy threshold.

---

## Phase 1: Configuration Parameters

**Goal:** Add energy threshold and steps-per-epoch parameters to Config system

**Requirements:** CONF-01, CONF-02, CONF-03

**Success Criteria:**
1. `Config` dataclass has `energy_threshold` field (float, default 0.01)
2. `Config` dataclass has `steps_per_epoch` field (int, default 1)
3. `Config` dataclass has `epochs` field (already exists, validated > 0)
4. CLI args auto-generated via `add_cli_args()` for new fields
5. Parameters serialized in checkpoint contract (to_dict/from_dict)

**Estimated Plans:** 2-3
- Add config fields with validation
- Add CLI argument generation
- Test serialization roundtrip

---

## Phase 2: History Buffer Zero-Padding

**Goal:** Replace oldest-state padding with zero-padding for history bootstrap

**Requirements:** HIST-01, HIST-02

**Success Criteria:**
1. `HistoryBuffer.features_for()` pads with zeros when history incomplete
2. Zero-padding applies to all feature types (basic, rich, delta_mag)
3. `features_for_histories()` (batch version) uses zero-padding
4. Warmup steps discarded once `history_len` steps collected
5. Tests confirm zero vectors for initial incomplete history

**Estimated Plans:** 2-3
- Modify padding logic in HistoryBuffer
- Handle warmup discard logic
- Add unit tests for padding behavior

---

## Phase 3: Part 1 - Trajectory Collection Loop

**Goal:** Implement iterative single-step collection with accept/reject based on energy threshold

**Requirements:** TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04

**Success Criteria:**
1. Function `collect_trajectory_step()` predicts dt, integrates 1 step, checks energy
2. If energy exceeds threshold: reject step, retrain on same state (loop)
3. If energy within threshold: accept step, record state to trajectory buffer
4. Collects N steps per epoch (configurable via `steps_per_epoch`)
5. No retry limit - iterates until energy threshold satisfied
6. Returns list of accepted (state, dt) tuples

**Estimated Plans:** 4-5
- Design trajectory collection function signature
- Implement single-step prediction + integration
- Add energy threshold check and accept/reject logic
- Add retrain loop (optimizer step on same state)
- Test with mock model and particles

---

## Phase 4: Part 2 - Generalization Training Loop

**Goal:** Train on random minibatches from collected trajectory until all samples pass threshold

**Requirements:** TRAIN-05, TRAIN-06, TRAIN-07

**Success Criteria:**
1. Function `generalize_on_trajectory()` samples minibatches from trajectory
2. Each sample: single-step prediction + integration
3. Loss computed on single timestep (not multi-step)
4. Training continues until ALL samples in minibatch pass energy threshold
5. Convergence criterion: energy residual < threshold for entire batch
6. Returns convergence status and iteration count

**Estimated Plans:** 3-4
- Design minibatch sampling from trajectory buffer
- Implement single-step loss evaluation
- Add convergence check (all-pass criterion)
- Test convergence on synthetic trajectory

---

## Phase 5: Unified Epoch Structure

**Goal:** Combine Part 1 and Part 2 into single epoch, repeat for N epochs

**Requirements:** TRAIN-08, TRAIN-09

**Success Criteria:**
1. Function `train_epoch_two_phase()` orchestrates Part 1 + Part 2
2. Part 1 output (trajectory) fed directly to Part 2
3. Epoch completes only after Part 2 converges
4. Outer loop runs for `config.epochs` iterations
5. Checkpoint saved after each epoch (or every 10 epochs)
6. W&B logging tracks Part 1 acceptance rate and Part 2 iterations

**Estimated Plans:** 3-4
- Design epoch orchestration function
- Connect Part 1 output to Part 2 input
- Add logging for both phases
- Test full epoch cycle end-to-end

---

## Phase 6: Integration into runner.py

**Goal:** Replace existing `run_training()` with new two-phase system

**Requirements:** INTG-01, INTG-03

**Success Criteria:**
1. `run_training()` calls new two-phase training functions
2. Existing Config fields (history_len, feature_type, num_orbits) still supported
3. Checkpoint contract preserved - simulation mode loads checkpoints correctly
4. CLI interface unchanged for end users
5. W&B logging maintains compatibility with existing dashboards
6. Multi-orbit support preserved (batched training)

**Estimated Plans:** 4-5
- Refactor run_training() to call new epoch function
- Preserve ModelAdapter and device/dtype logic
- Test with history-enabled and analytic modes
- Test multi-orbit batching
- Verify checkpoint backward compatibility

---

## Phase 7: Cleanup Legacy Code

**Goal:** Remove old training loop code and mark deprecated files

**Requirements:** INTG-02

**Success Criteria:**
1. Old multi-step loss evaluation removed from run_training()
2. Legacy scripts in `run/legacy/` marked for deletion (if not already removed)
3. Unused loss function calls removed (if replaced)
4. Code comments updated to reflect new training approach
5. No orphaned imports or dead code in runner.py

**Estimated Plans:** 2
- Remove old training loop logic
- Clean up imports and comments
- Verify no functionality broken

---

## Requirements Coverage

| Requirement | Phase | Description |
|-------------|-------|-------------|
| **TRAIN-01** | 3 | Predict dt, advance particles one step, check energy threshold |
| **TRAIN-02** | 3 | If energy exceeds threshold, retrain on same state |
| **TRAIN-03** | 3 | Loop until energy within threshold, then accept and record |
| **TRAIN-04** | 3 | Support parametrizable N steps per epoch |
| **TRAIN-05** | 4 | Random minibatch sampling from collected trajectory |
| **TRAIN-06** | 4 | Train until ALL samples pass energy threshold |
| **TRAIN-07** | 4 | Single timestep predictions per sample |
| **TRAIN-08** | 5 | One epoch = Part 1 (collection) + Part 2 (generalization) |
| **TRAIN-09** | 5 | Run for fixed N epochs (configurable) |
| **HIST-01** | 2 | Pad history with zeros for initial steps |
| **HIST-02** | 2 | Discard warmup steps once real trajectory exists |
| **CONF-01** | 1 | Energy threshold as configurable parameter |
| **CONF-02** | 1 | Steps per epoch as configurable parameter |
| **CONF-03** | 1 | Total epochs as configurable parameter |
| **INTG-01** | 6 | Replace existing training routine in runner.py |
| **INTG-02** | 7 | Remove old training loop code |
| **INTG-03** | 6 | Maintain checkpoint compatibility with simulation mode |

**Total:** 17/17 requirements mapped (100% coverage)

---

## Phase Dependencies

```
Phase 1 (Config) → Phase 2 (History), Phase 3 (Collection)
Phase 2 (History) → Phase 3 (Collection)
Phase 3 (Collection) → Phase 4 (Generalization)
Phase 4 (Generalization) → Phase 5 (Epoch)
Phase 5 (Epoch) → Phase 6 (Integration)
Phase 6 (Integration) → Phase 7 (Cleanup)
```

**Critical Path:** 1 → 3 → 4 → 5 → 6 → 7 (Phase 2 can proceed in parallel with Phase 3 after Phase 1)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Part 1 never converges (infinite loop) | Log acceptance rate; add diagnostic mode to detect stuck states |
| Part 2 convergence too slow | Implement early stopping if no improvement after N iterations |
| History buffer OOM for long trajectories | Limit trajectory buffer size or implement circular buffer |
| Backward compatibility break | Test checkpoint loading in simulation mode before merging |
| Multi-orbit batching complex | Start with single-orbit, generalize after validation |

---

## Milestones

- **M1 (Phase 1-2):** Configuration and history infrastructure ready
- **M2 (Phase 3-4):** Both training parts implemented and tested independently
- **M3 (Phase 5):** Unified epoch structure working end-to-end
- **M4 (Phase 6-7):** Fully integrated and old code removed

---

## Notes

- Energy threshold is the central parameter - must be exposed in Config and CLI
- No retry cap in Part 1 by design (user preference)
- Single shared model for both parts (no separate models)
- Simulation mode must continue to work with new checkpoints
- Multi-orbit training supported but start testing with single-orbit

---

*Roadmap created: 2026-01-20*
*Last updated: 2026-01-20 (initial creation)*
