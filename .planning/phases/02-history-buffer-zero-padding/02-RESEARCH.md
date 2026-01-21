# Phase 2: History Buffer Zero-Padding - Research

**Researched:** 2026-01-20
**Domain:** Python tensor manipulation and temporal feature padding for time-series ML
**Confidence:** HIGH

## Summary

Phase 2 replaces the current oldest-state padding strategy in `HistoryBuffer` with zero-padding for incomplete history during the bootstrap phase. The current implementation pads incomplete histories by repeating the oldest available state (or current state if empty), creating non-zero features even when no true historical data exists. This research examined the history buffer architecture, identified the exact padding locations, and documented the feature structures that must be zero-padded.

The codebase uses PyTorch tensors exclusively for this operation—no external libraries required beyond torch.zeros(). The padding logic exists in three methods (`features_for`, `features_for_batch`, `features_for_histories`), all calling a shared `_features_from_sequence` static method that processes state sequences into features. Padding happens before this call by manipulating the state list.

**Primary recommendation:** Replace list-based state padding with zero-tensor injection at the sequence assembly stage, then discard warmup steps in training loops once history_len states collected.

## Standard Stack

The history buffer is pure PyTorch with no additional dependencies.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | ≥1.x | Tensor operations, autograd | Already project dependency, no additional imports needed |
| collections.deque | stdlib | Fixed-size FIFO buffer | Standard Python, maxlen parameter handles overflow |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| typing | stdlib | Type hints for _HistoryState | Already used throughout codebase |
| dataclasses | stdlib | _HistoryState frozen dataclass | Already used for immutable state storage |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torch.zeros() | Repeat oldest state | Current behavior—creates non-zero features that misrepresent missing data |
| torch.zeros() | NaN padding | Would require special handling in feature computation and loss functions |
| deque | list with manual bounds | deque maxlen is cleaner and automatic |

**Installation:**
No additional packages required—all functionality exists in current dependencies.

## Architecture Patterns

### Current Padding Architecture

The `HistoryBuffer` stores detached particle states in a deque and provides three public methods for feature extraction:

```python
# Current structure (src/history_buffer.py lines 26-46)
class HistoryBuffer:
    def __init__(self, history_len: int = 3, feature_type: FeatureType = "delta_mag"):
        self.history_len: int = int(history_len)
        self.feature_type: FeatureType = feature_type
        self._buf: Deque[_HistoryState] = deque(maxlen=self.history_len)
```

**State representation** (_HistoryState dataclass, lines 17-24):
- `position`: torch.Tensor (N, D) or (B, N, D)
- `velocity`: torch.Tensor (N, D) or (B, N, D)
- `mass`: torch.Tensor (N,) or (B, N)
- `dt`: torch.Tensor (scalar or batch)
- `softening`: float

### Pattern 1: Current Padding Logic (Oldest-State Repeat)

**Where:** Lines 260-267 in `features_for()`

```python
# Source: /u/gkerex/projects/AITimeStepper/src/history_buffer.py:260-267
past_list: List[_HistoryState] = list(self._buf)
if len(past_list) < self.history_len:
    pad_count = self.history_len - len(past_list)
    if past_list:
        past_list = [past_list[0]] * pad_count + past_list
    else:
        past_list = [self._state_from_particle(current, detach=False)] * pad_count
```

**Problem:** Repeating oldest state creates features that suggest the system was static at the beginning, but this is not true—we simply lack data.

### Pattern 2: Feature Sequence Processing

**Where:** Lines 183-229 in `_features_from_sequence()` static method

**What it does:**
- Takes stacked tensors: `(K+1, N, D)` for positions/velocities
- Computes per-step features based on `feature_type`
- Concatenates along time axis to produce `(F_total,)` or `(B, F_total)` output

**Feature types and sizes:**

1. **basic** (11 features per timestep):
   - n_val, d_val, total_mass, r_mean, r_max, v_mean, v_max, a_mean, a_max, pair_min, pair_mean
   - Total for history_len=3: 11 × 4 = 44 features

2. **rich** (23 features per timestep):
   - n_val, d_val, total_mass, m_mean, m_min, m_max, m_rms, r_mean, r_min, r_max, r_rms, v_mean, v_min, v_max, v_rms, a_mean, a_min, a_max, a_rms, pair_min, pair_mean, pair_max, soft_val
   - Total for history_len=3: 23 × 4 = 92 features

3. **delta_mag** (10 features per transition):
   - Computes differences between consecutive states: Δpos, Δvel, Δacc
   - Statistics: mean, max, rms for each delta magnitude
   - Plus dt for each transition
   - Total for history_len=3: 10 × 3 = 30 features (3 transitions from 4 states)

### Pattern 3: Zero-Padding Strategy

**Recommended approach:**

```python
# Zero-state factory for padding
@staticmethod
def _zero_state(reference: _HistoryState) -> _HistoryState:
    """Create a zero-valued state matching reference shapes."""
    pos = reference.position
    device, dtype = pos.device, pos.dtype

    zero_pos = torch.zeros_like(pos)
    zero_vel = torch.zeros_like(pos)
    zero_mass = torch.zeros_like(reference.mass)
    zero_dt = torch.zeros_like(reference.dt)

    return _HistoryState(
        position=zero_pos,
        velocity=zero_vel,
        mass=zero_mass,
        dt=zero_dt,
        softening=reference.softening,  # Keep softening constant
    )
```

**Usage in features_for():**

```python
past_list: List[_HistoryState] = list(self._buf)
if len(past_list) < self.history_len:
    pad_count = self.history_len - len(past_list)
    if past_list:
        zero_state = self._zero_state(past_list[0])
    else:
        zero_state = self._zero_state(self._state_from_particle(current, detach=False))

    past_list = [zero_state] * pad_count + past_list
```

### Pattern 4: Warmup Step Handling

**Concept:** "Warmup" refers to the initial training steps where history buffer is filling up. Once `history_len` states are pushed, the buffer is "warm" and contains real trajectory data.

**Current behavior:** Training includes warmup steps in loss computation, but they use repeated-state padding.

**New behavior:** Warmup steps should be discarded once real history exists. This means:
- Training loop tracks steps and only computes loss after ≥ history_len steps
- OR loss function ignores/masks first history_len steps
- In practice, this is a training loop concern, not a HistoryBuffer concern

**Where warmup matters:** `run/runner.py` training loops and `src/losses_history.py` loss functions would need to skip first `history_len` steps.

### Anti-Patterns to Avoid

- **Don't use NaN padding:** Feature computation will propagate NaNs through statistics
- **Don't pad after feature computation:** Must pad state sequences, not feature tensors (different sizes for different feature_types)
- **Don't modify softening during padding:** Keep consistent across all states
- **Don't forget batched methods:** `features_for_batch` and `features_for_histories` also need zero-padding

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Zero tensor creation | Manual loops to fill arrays | `torch.zeros_like(reference)` | Preserves device, dtype, shape automatically |
| State sequence stacking | Manual concatenation | `torch.stack([s.position for s in seq], dim=0)` | Already used in codebase (line 275) |
| Deque size checking | Manual counter | `len(self._buf) < self.history_len` | Already implemented correctly |

**Key insight:** PyTorch's `zeros_like` handles all the device/dtype/shape complexity that makes padding error-prone. Don't reimplement it.

## Common Pitfalls

### Pitfall 1: Padding After Stacking
**What goes wrong:** Attempting to pad the stacked tensor `(K+1, N, D)` instead of the state list.
**Why it happens:** It seems simpler to pad tensors than state objects.
**How to avoid:** Pad the `past_list` before the stacking operation (lines 275-278). Feature computation needs full state objects (mass, dt), not just pos/vel.
**Warning signs:** Missing mass/dt in padded states; shape mismatches in `_features_from_sequence`.

### Pitfall 2: Inconsistent Softening Across States
**What goes wrong:** Using 0.0 softening for zero-padded states when other states have non-zero softening.
**Why it happens:** Forgetting softening is a physics parameter, not a data field.
**How to avoid:** Copy softening from reference state (line 273 checks this: "All history states must share the same softening value").
**Warning signs:** ValueError raised at line 273.

### Pitfall 3: Delta_Mag Edge Case
**What goes wrong:** Computing delta_mag features from zero-padded states produces misleading zero deltas.
**Why it happens:** Delta_mag computes differences between consecutive states; zero→zero has zero delta.
**How to avoid:** This is expected behavior—zero padding will produce zero delta features, signaling "no information". Alternative: mask/weight these features in loss function.
**Warning signs:** Initial training steps show suspiciously low loss due to zero features.

### Pitfall 4: Forgetting Batched Methods
**What goes wrong:** Implementing zero-padding in `features_for()` but not `features_for_batch()` or `features_for_histories()`.
**Why it happens:** The three methods have similar but not identical padding logic (lines 309-316, 401-406).
**How to avoid:** Create a shared helper method `_pad_past_list()` that all three methods call.
**Warning signs:** Tests pass for single-orbit but fail for multi-orbit training.

### Pitfall 5: Warmup Discard Logic in Wrong Layer
**What goes wrong:** Implementing warmup step filtering inside `HistoryBuffer` instead of training loop.
**Why it happens:** It seems like history-buffer responsibility.
**How to avoid:** HistoryBuffer provides features; training loop decides when to compute loss. Keep separation of concerns.
**Warning signs:** HistoryBuffer knows about epochs/steps (it shouldn't).

## Code Examples

### Zero-State Factory (Add to HistoryBuffer)

```python
# Source: Design pattern, location: ~line 150 in src/history_buffer.py
@staticmethod
def _zero_state(reference: _HistoryState) -> _HistoryState:
    """
    Create a zero-valued state matching the shape, device, and dtype
    of the reference state. Used for zero-padding incomplete history.
    """
    pos = reference.position
    device = pos.device
    dtype = pos.dtype

    return _HistoryState(
        position=torch.zeros_like(pos),
        velocity=torch.zeros_like(pos),
        mass=torch.zeros_like(reference.mass),
        dt=torch.zeros_like(reference.dt),
        softening=reference.softening,
    )
```

### Modified features_for() Padding

```python
# Source: Modification of src/history_buffer.py:260-267
def features_for(self, current: "ParticleTorch") -> torch.Tensor:
    past_list: List[_HistoryState] = list(self._buf)
    if len(past_list) < self.history_len:
        pad_count = self.history_len - len(past_list)
        if past_list:
            # Use oldest state as reference for shape/device/dtype
            zero_state = self._zero_state(past_list[0])
        else:
            # Use current state as reference
            current_state = self._state_from_particle(current, detach=False)
            zero_state = self._zero_state(current_state)

        # Pad with zeros
        past_list = [zero_state] * pad_count + past_list

    seq = past_list + [self._state_from_particle(current, detach=False)]
    # ... rest of method unchanged ...
```

### Modified features_for_batch() Padding

```python
# Source: Modification of src/history_buffer.py:309-316
def features_for_batch(self, batch_state: "ParticleTorch") -> torch.Tensor:
    # ... initial setup ...

    past_list: List[_HistoryState] = list(self._buf)
    if len(past_list) < self.history_len:
        pad_count = self.history_len - len(past_list)
        if past_list:
            zero_state = self._zero_state(past_list[0])
            # Expand to batch size
            zero_state = self._expand_state_to_batch(zero_state, B)
        else:
            # No past states; pad with zeros matching current batch
            zero_state = self._zero_state(current_state)

        past_list = [zero_state] * pad_count + past_list

    seq = [self._expand_state_to_batch(s, B) for s in past_list] + [current_state]
    # ... rest unchanged ...
```

### Modified features_for_histories() Padding

```python
# Source: Modification of src/history_buffer.py:401-406
# Inside the per-history loop at ~line 384
for i, history in enumerate(histories):
    past_list = list(history._buf)

    # ... current_state assembly ...

    if len(past_list) < history_len:
        pad_count = history_len - len(past_list)
        if past_list:
            zero_state = HistoryBuffer._zero_state(past_list[0])
        else:
            zero_state = HistoryBuffer._zero_state(current_state)

        past_list = [zero_state] * pad_count + past_list

    seq = past_list + [current_state]
    # ... rest of loop unchanged ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Repeat oldest state | Zero-padding | Phase 2 (planned) | Cleaner signal—no false history information |
| Include warmup in loss | Discard warmup steps | Phase 2 (planned) | Training focuses on true temporal patterns |

**Deprecated/outdated:**
Nothing deprecated in this phase—this is a refinement of current behavior.

## Open Questions

### 1. Should warmup discard be in loss function or training loop?

**What we know:** Requirements HIST-02 states "discard warmup steps once real trajectory exists". This could mean:
- Option A: Training loop skips first `history_len` steps before computing loss
- Option B: Loss function masks/weights first `history_len` samples differently
- Option C: HistoryBuffer provides a `is_warm()` method, caller decides what to do

**What's unclear:** Which layer owns this responsibility?

**Recommendation:** Option A (training loop). Loss function should remain stateless and not know about step counts. The training loop in `run/runner.py` already tracks epochs and steps—it should skip loss computation until `len(history._buf) == history_len`.

### 2. Should zero-padding behavior be configurable?

**What we know:** User decision in PROJECT.md is "zero-padding for history bootstrap".

**What's unclear:** Should we keep the old repeat-state behavior as an option, or hard-replace it?

**Recommendation:** Hard-replace. The decision document explicitly chose zero-padding; keeping both options adds complexity with no identified use case.

### 3. How to handle delta_mag zero-padding semantics?

**What we know:** Delta_mag computes `Δx = x[t] - x[t-1]`. Zero states will produce zero deltas.

**What's unclear:** Is this the desired signal, or should we use a special marker (e.g., large negative value) to indicate "no data"?

**Recommendation:** Zero deltas are correct semantics. They indicate "no change information available". The model will learn to interpret this during training. Warm/non-warm masking in the training loop will ensure the model doesn't overtrain on these zero features.

## Sources

### Primary (HIGH confidence)
- `/u/gkerex/projects/AITimeStepper/src/history_buffer.py` - Complete implementation source (lines 1-435)
- `/u/gkerex/projects/AITimeStepper/src/nbody_features.py` - Feature type definitions (lines 106-204)
- `/u/gkerex/projects/AITimeStepper/src/config.py` - history_len and feature_type configuration (lines 34-35)
- `.planning/REQUIREMENTS.md` - HIST-01 and HIST-02 requirements
- `.planning/PROJECT.md` - User decision on zero-padding

### Secondary (MEDIUM confidence)
- `/u/gkerex/projects/AITimeStepper/src/model_adapter.py` - Usage patterns for HistoryBuffer (lines 45-52)
- `/u/gkerex/projects/AITimeStepper/src/losses_history.py` - Loss functions using history features (lines 38, 279)

### Tertiary (LOW confidence)
None—all information verified from codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch is already project dependency, verified in imports
- Architecture: HIGH - Complete code inspection of all padding locations
- Pitfalls: HIGH - Identified from actual code structure and PyTorch tensor semantics

**Research date:** 2026-01-20
**Valid until:** 2026-02-20 (30 days, stable domain—PyTorch tensor APIs are stable)
