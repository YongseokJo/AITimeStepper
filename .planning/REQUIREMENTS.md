# Requirements: AITimeStepper Training Refactor

**Defined:** 2026-01-20
**Core Value:** Every training sample must satisfy energy conservation before being accepted.

## v1 Requirements

Requirements for the two-phase training routine refactor.

### Training Loop

- [ ] **TRAIN-01**: Predict dt, advance particles one step, check energy threshold
- [ ] **TRAIN-02**: If energy exceeds threshold, retrain on same state (reject step)
- [ ] **TRAIN-03**: Loop until energy within threshold, then accept and record state
- [ ] **TRAIN-04**: Support parametrizable N steps per epoch (default: 1)
- [ ] **TRAIN-05**: Random minibatch sampling from collected trajectory
- [ ] **TRAIN-06**: Train until ALL samples pass energy threshold
- [ ] **TRAIN-07**: Single timestep predictions per sample
- [ ] **TRAIN-08**: One epoch = Part 1 (collection) + Part 2 (generalization)
- [ ] **TRAIN-09**: Run for fixed N epochs (configurable)

### History Handling

- [ ] **HIST-01**: Pad history with zeros for initial steps (not repeat first state)
- [ ] **HIST-02**: Discard warmup steps once real trajectory exists

### Configuration

- [ ] **CONF-01**: Energy threshold as configurable parameter
- [ ] **CONF-02**: Steps per epoch as configurable parameter
- [ ] **CONF-03**: Total epochs as configurable parameter

### Integration

- [ ] **INTG-01**: Replace existing training routine in runner.py
- [ ] **INTG-02**: Remove old training loop code
- [ ] **INTG-03**: Maintain checkpoint compatibility with simulation mode

## v2 Requirements

Deferred to future work.

### Enhancements

- **ENH-01**: Variable particle count generalization
- **ENH-02**: Max retry limit with fallback strategy in Part 1
- **ENH-03**: Re-verification of historical data after Part 2 training
- **ENH-04**: Progress visualization during training

## Out of Scope

| Feature | Reason |
|---------|--------|
| Separate models for Part 1/Part 2 | Single shared model by design |
| Momentum threshold in addition to energy | Energy threshold sufficient for v1 |
| Parallel trajectory collection | Sequential collection maintains physics consistency |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRAIN-01 | TBD | Pending |
| TRAIN-02 | TBD | Pending |
| TRAIN-03 | TBD | Pending |
| TRAIN-04 | TBD | Pending |
| TRAIN-05 | TBD | Pending |
| TRAIN-06 | TBD | Pending |
| TRAIN-07 | TBD | Pending |
| TRAIN-08 | TBD | Pending |
| TRAIN-09 | TBD | Pending |
| HIST-01 | TBD | Pending |
| HIST-02 | TBD | Pending |
| CONF-01 | TBD | Pending |
| CONF-02 | TBD | Pending |
| CONF-03 | TBD | Pending |
| INTG-01 | TBD | Pending |
| INTG-02 | TBD | Pending |
| INTG-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 0
- Unmapped: 17 (roadmap pending)

---
*Requirements defined: 2026-01-20*
*Last updated: 2026-01-20 after initial definition*
