# Plan 02-01 Summary: Zero-State Factory and Zero-Padding Implementation

**Status:** COMPLETED
**Commits:** 286a5cc, f168818, d7f744d

## Changes Made

### Commit 286a5cc: Add _zero_state() factory method
Added static method `_zero_state()` to HistoryBuffer class (lines 254-268):
- Creates zero-valued _HistoryState from reference state
- Preserves device, dtype, and softening from reference
- Uses `torch.zeros_like()` for position, velocity, mass, dt tensors

### Commit f168818: Implement zero-padding for incomplete history
Updated three feature extraction methods to use zero-padding instead of oldest-state repetition:

1. **features_for()** (lines 276-288):
   - When buffer has states: use oldest state as reference for zero creation
   - When buffer is empty: use current state as reference for zero creation

2. **features_for_batch()** (lines 341-350):
   - Same zero-padding logic as features_for()
   - Pads before batch expansion

3. **features_for_histories()** (lines 427-436):
   - Per-history zero-padding with same logic
   - Each history gets its own zero-state based on available data

### Commit d7f744d: Add unit test for zero-padding
Added `_test_zero_padding()` function (lines 467-513) with four test cases:
- Test 1: Empty buffer uses zero-padding
- Test 2: Partially filled buffer pads remaining slots
- Test 3: Full buffer works without padding
- Test 4: _zero_state() preserves softening and creates zeros

## Verification

All verification steps passed:
- _zero_state() creates correct zero tensors with preserved softening
- features_for() returns correct feature shapes with zero-padding
- Unit test passes all assertions

Quick test available via:
```bash
python -c "from src.history_buffer import _test_zero_padding; _test_zero_padding()"
```

## Requirements Satisfied

All "must_haves" from plan satisfied:
- Zero-state factory exists and creates zero-valued _HistoryState
- features_for() returns zero features for missing history slots
- Zero-padding preserves device, dtype, and softening from reference
- _zero_state() static method present in src/history_buffer.py
- Zero-padding implemented in all three feature extraction methods
- Key link established: _zero_state() called when len(past_list) < history_len

## Impact

Zero-padding provides cleaner signal to model during bootstrap:
- Zeros explicitly indicate "no data available"
- Avoids false repetition that could mislead the model
- Consistent behavior across all feature extraction paths
