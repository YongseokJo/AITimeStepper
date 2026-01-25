# Phase 1 Research: Configuration Parameters

**Phase:** 1 - Configuration Parameters
**Date:** 2026-01-20
**Researcher:** Claude Code

---

## Executive Summary

Phase 1 requires adding two new configuration parameters (`energy_threshold` and `steps_per_epoch`) to the existing Config system. Research reveals that **`energy_threshold` already exists** in the codebase (line 21 of config.py, default 2e-4), so only `steps_per_epoch` needs to be added. The existing Config infrastructure is well-structured and provides clear patterns for adding new parameters with automatic CLI generation and checkpoint serialization.

---

## Research Questions

### 1. How does the existing Config dataclass work? What fields exist?

**Location:** `/u/gkerex/projects/AITimeStepper/src/config.py`

**Structure:**
- `Config` is a `@dataclass` (lines 10-88) with 50+ fields organized by category
- Uses Python 3.10+ type hints with `|` syntax for unions
- Defaults provided for all fields
- Sections clearly marked with comments:
  - Training / optimization (lines 12-25)
  - Loss bounds (lines 26-30)
  - Feature / history (lines 32-34)
  - Simulation / integrator (lines 36-49)
  - External field (lines 51-53)
  - Multi-orbit sampling (lines 55-60)
  - Device / dtype (lines 62-67)
  - Optimizer phases (lines 69-75)
  - Logging / misc (lines 77-82)
  - Checkpoint / model loading (lines 84-85)
  - Extra / unknown fields (line 88)

**Key Finding: `energy_threshold` already exists!**
- Line 21: `energy_threshold: float = 2e-4`
- Located in "Training / optimization" section
- Default value is 2e-4 (NOT 0.01 as specified in roadmap)
- Already has CLI arg generation (line 112)

**Field Types Present:**
- Simple types: `int`, `float`, `str`, `bool`
- Optional types: `Optional[float]`, `Optional[str]`, `Optional[Tuple[float, float, float]]`
- Complex types: `Dict[str, Any]` with `field(default_factory=dict)`
- Union types: `float | None` (modern syntax)

**Validation Patterns:**
```python
def validate(self) -> None:  # Line 238
    if self.history_len and self.history_len > 0 and not self.feature_type:
        raise ValueError("history_len > 0 requires feature_type")
    if self.num_particles is not None and self.num_particles < 2:
        raise ValueError("num_particles must be >= 2")
    if self.dim is not None and self.dim < 1:
        raise ValueError("dim must be >= 1")
    if self.duration is not None and self.duration < 0:
        raise ValueError("duration must be >= 0")
```

**Pattern:** Validates logical constraints, raises `ValueError` with descriptive messages

---

### 2. How does add_cli_args() generate CLI arguments?

**Location:** Lines 91-183

**Mechanism:**
```python
@classmethod
def add_cli_args(cls, parser: argparse.ArgumentParser,
                 include: Optional[Iterable[str]] = None) -> None:
```

**Key Features:**
1. **Category filtering:** Uses `include` parameter to filter which arg groups to add
2. **Duplicate detection:** Helper function `add_arg()` (lines 97-101) checks if option already exists before adding
3. **Default values:** Uses `default=cls.field_name` to reference class defaults
4. **Categories:** `train`, `bounds`, `history`, `orbit`, `multi`, `device`, `logging`, `sim`, `external`

**Example Pattern (line 104):**
```python
if want("train"):
    add_arg("--epochs", "-n", type=int, default=cls.epochs, help="number of training epochs")
```

**Example for energy_threshold (line 112):**
```python
add_arg("--energy-threshold", type=float, default=cls.energy_threshold,
        help="accept/reject energy threshold")
```

**Naming Convention:**
- Field name: `energy_threshold` (snake_case)
- CLI flag: `--energy-threshold` (kebab-case)
- Auto-converted by argparse

**Category Grouping in runner.py (line 402):**
```python
Config.add_cli_args(train, include=["train", "bounds", "history", "device", "logging", "sim", "multi"])
```

---

### 3. How does to_dict/from_dict handle serialization for checkpoints?

**Serialization (`to_dict()`):** Lines 216-218
```python
def to_dict(self) -> Dict[str, Any]:
    data = {f.name: getattr(self, f.name) for f in fields(self)}
    return data
```

**Simple approach:** Iterates over all dataclass fields, extracts values via `getattr()`

**Deserialization (`from_dict()`):** Lines 201-214
```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "Config":
    payload = dict(data) if data is not None else {}
    field_names = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in field_names and key != "extra":
            kwargs[key] = value
        else:
            extra[key] = value
    # Special handling for external_field_position (list -> tuple)
    if "external_field_position" in kwargs and isinstance(kwargs["external_field_position"], list):
        kwargs["external_field_position"] = tuple(kwargs["external_field_position"])
    kwargs["extra"] = extra
    return cls(**kwargs)
```

**Key Features:**
1. **Automatic field mapping:** Matches dict keys to dataclass field names
2. **Unknown fields:** Stored in `extra` dict (graceful degradation)
3. **Type coercion:** Special handling for tuple fields (JSON can't serialize tuples)
4. **Backward compatibility:** Extra fields don't break loading

**Checkpoint Integration:** Lines 64-70 in `checkpoint.py`
```python
if config is not None:
    payload["config"] = _config_payload(config)  # Uses config.as_wandb_dict()
    payload["config_summary"] = config.summary()
    payload["history_len"] = config.history_len
    payload["feature_type"] = config.feature_type
    payload["dtype"] = config.dtype
```

**Key Insight:**
- Full config saved as dict via `as_wandb_dict()` (line 220-229)
- Critical fields (`history_len`, `feature_type`, `dtype`) duplicated at top level for backward compat

---

### 4. What validation patterns are used?

**Validation Method:** Lines 238-246

**Patterns:**
1. **Cross-field validation:** Check consistency between related fields
   ```python
   if self.history_len and self.history_len > 0 and not self.feature_type:
       raise ValueError("history_len > 0 requires feature_type")
   ```

2. **Boundary checks:** Ensure numeric fields meet constraints
   ```python
   if self.num_particles is not None and self.num_particles < 2:
       raise ValueError("num_particles must be >= 2")
   ```

3. **Non-negative checks:** For duration, dim, etc.
   ```python
   if self.duration is not None and self.duration < 0:
       raise ValueError("duration must be >= 0")
   ```

**Call Sites:**
- `run_training()` in runner.py (line 228): `config.validate()`
- `run_simulation()` in runner.py (line 84): `config.validate()`

**Pattern for Phase 1:**
- `epochs` already exists but has no validation
- Should add validation: `epochs > 0`
- `steps_per_epoch` should validate: `steps_per_epoch >= 1`
- `energy_threshold` could validate: `energy_threshold > 0` (but currently has no validation)

---

### 5. Are there similar numeric threshold fields to use as patterns?

**Threshold/Bound Fields:**
1. **`energy_threshold`** (line 21): `float = 2e-4` - ALREADY EXISTS
2. **`dt_bound`** (line 19): `float = 1e-8` - Time step bound for loss heuristics
3. **`rel_loss_bound`** (line 20): `float = 1e-5` - Relative loss bound
4. **`E_lower`** (line 27): `float = 1e-6` - Lower energy bound for loss
5. **`E_upper`** (line 28): `float = 1e-4` - Upper energy bound for loss
6. **`L_lower`** (line 29): `float = 1e-4` - Lower angular momentum bound
7. **`L_upper`** (line 30): `float = 1e-2` - Upper angular momentum bound

**Step/Count Fields:**
1. **`epochs`** (line 14): `int = 1000` - ALREADY EXISTS
2. **`n_steps`** (line 18): `int = 10` - Integration steps per loss eval
3. **`replay_steps`** (line 22): `int = 1000` - Max replay optimization steps
4. **`replay_batch_size`** (line 23): `int = 512` - Replay batch size
5. **`min_replay_size`** (line 24): `int = 2` - Min replay buffer size

**CLI Patterns for Numeric Fields:**
- Float thresholds: `type=float, default=cls.field_name`
- Integer counts: `type=int, default=cls.field_name`
- Help text describes purpose and units

---

## Current Usage of energy_threshold

**Search Results:** Used in legacy code (being deprecated)
- `run/legacy/ML_history_multi_wandb.py` (line 188, 228, 230)
- `run/legacy/ML_history_wandb.py` (line 144, 205, 208, 343)

**Usage Pattern:**
```python
energy_threshold = config.energy_threshold
# ...
accepted = rel_dE_val <= energy_threshold
if accepted:
    # Record trajectory step
else:
    # Reject and retrain
```

**Note:** These are in `legacy/` directory, suggesting old training loop being replaced by Phase 3-6 implementation

---

## Implementation Plan for Phase 1

### Required Changes

**1. Add `steps_per_epoch` field to Config**
- Location: Line ~22 (after `energy_threshold`)
- Type: `int = 1`
- Section: Training / optimization

**2. Adjust `energy_threshold` default (OPTIONAL)**
- Current: `2e-4`
- Roadmap specifies: `0.01`
- Decision: Keep existing default or update?

**3. Add CLI arg for `steps_per_epoch`**
- Location: Line ~113 (in "train" section of `add_cli_args`)
- Pattern: `add_arg("--steps-per-epoch", type=int, default=cls.steps_per_epoch, help="number of steps to collect per epoch")`

**4. Add validation for `epochs` and `steps_per_epoch`**
- Location: `validate()` method (line 238)
- Add checks:
  ```python
  if self.epochs is not None and self.epochs < 1:
      raise ValueError("epochs must be >= 1")
  if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
      raise ValueError("steps_per_epoch must be >= 1")
  ```

**5. No serialization changes needed**
- `to_dict()` and `from_dict()` automatically handle new fields via dataclass introspection
- Backward compatibility: Old checkpoints without `steps_per_epoch` will use default value

---

## Key Findings Summary

### ✅ What Already Works
1. **`energy_threshold` exists** - No need to add, just verify default value
2. **`epochs` exists** - Just needs validation added
3. **Serialization is automatic** - Dataclass introspection handles new fields
4. **CLI generation is straightforward** - Clear pattern to follow
5. **Checkpoint contract is robust** - Extra fields stored in `extra` dict for forward compatibility

### ⚠️ Considerations
1. **Default value mismatch:** Roadmap says `energy_threshold` default should be 0.01, but code has 2e-4
2. **Category placement:** `steps_per_epoch` belongs in "train" category for CLI
3. **Validation gaps:** `epochs` has no validation currently, should add
4. **Legacy usage:** `energy_threshold` used in deprecated code, but new implementation will use it differently

### 📋 Minimal Changes Required
1. Add single field: `steps_per_epoch: int = 1`
2. Add single CLI arg: `--steps-per-epoch`
3. Add validation for `epochs >= 1` and `steps_per_epoch >= 1`
4. Optionally adjust `energy_threshold` default (needs decision)

---

## Testing Strategy

### Unit Tests
1. **Field defaults:**
   ```python
   config = Config()
   assert config.energy_threshold == 2e-4  # or 0.01 if changed
   assert config.steps_per_epoch == 1
   assert config.epochs == 1000
   ```

2. **CLI parsing:**
   ```python
   parser = argparse.ArgumentParser()
   Config.add_cli_args(parser, include=["train"])
   args = parser.parse_args(["--steps-per-epoch", "5", "--energy-threshold", "0.02"])
   config = Config.from_dict(vars(args))
   assert config.steps_per_epoch == 5
   assert config.energy_threshold == 0.02
   ```

3. **Validation:**
   ```python
   config = Config(epochs=0)
   with pytest.raises(ValueError, match="epochs must be >= 1"):
       config.validate()

   config = Config(steps_per_epoch=-1)
   with pytest.raises(ValueError, match="steps_per_epoch must be >= 1"):
       config.validate()
   ```

4. **Serialization roundtrip:**
   ```python
   config = Config(steps_per_epoch=10, energy_threshold=0.05)
   data = config.to_dict()
   config2 = Config.from_dict(data)
   assert config2.steps_per_epoch == 10
   assert config2.energy_threshold == 0.05
   ```

5. **Backward compatibility:**
   ```python
   # Old checkpoint without steps_per_epoch
   old_dict = {"epochs": 500, "lr": 1e-3}
   config = Config.from_dict(old_dict)
   assert config.steps_per_epoch == 1  # Uses default
   ```

### Integration Tests
1. Verify checkpoint saved with new fields
2. Verify old checkpoints load without error
3. Verify CLI args parsed correctly in runner.py

---

## Dependencies

**External:**
- Python 3.10+ (for `|` union syntax, already used)
- `dataclasses` module (stdlib)
- `argparse` module (stdlib)

**Internal:**
- No dependencies on other phases
- Phase 3 will consume `energy_threshold` and `steps_per_epoch`
- Phase 5 will consume `epochs`

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Default value confusion (0.01 vs 2e-4) | Medium | Clarify with stakeholder; document choice |
| Backward compat break | Low | `from_dict()` handles missing fields gracefully |
| Validation too strict | Low | Use `>= 1` for counts, `> 0` for thresholds |
| CLI arg name collision | Low | `add_arg()` checks for duplicates |

---

## Open Questions for Planning

1. **Should `energy_threshold` default be changed from 2e-4 to 0.01?**
   - Roadmap specifies 0.01
   - Code currently has 2e-4
   - Need stakeholder decision

2. **Should existing uses of `energy_threshold` in legacy code be updated?**
   - Likely not, since legacy code is being replaced
   - But verify nothing in active codebase depends on old default

3. **Should validation be added for `energy_threshold > 0`?**
   - Logically makes sense (negative threshold meaningless)
   - Not currently validated
   - Suggest: add validation

4. **Should `steps_per_epoch` allow value of 0?**
   - Roadmap default is 1
   - Logically, 0 steps per epoch makes no sense
   - Suggest: validate `>= 1`

---

## References

**Files Analyzed:**
- `/u/gkerex/projects/AITimeStepper/src/config.py` (274 lines)
- `/u/gkerex/projects/AITimeStepper/src/checkpoint.py` (100 lines)
- `/u/gkerex/projects/AITimeStepper/run/runner.py` (430 lines)
- `/u/gkerex/projects/AITimeStepper/.planning/codebase/CONVENTIONS.md`
- `/u/gkerex/projects/AITimeStepper/.planning/codebase/ARCHITECTURE.md`
- `/u/gkerex/projects/AITimeStepper/.planning/ROADMAP.md`

**Key Patterns:**
- Dataclass field definition: `field_name: type = default`
- CLI arg naming: snake_case → kebab-case
- Validation: Early checks with descriptive ValueErrors
- Serialization: Automatic via dataclass fields()
- Checkpoint contract: Full config + critical fields at top level

---

## RESEARCH COMPLETE

**Summary:** Phase 1 is straightforward. Only one new field (`steps_per_epoch`) needs to be added. `energy_threshold` and `epochs` already exist. Implementation requires ~15 lines of code: 1 field, 1 CLI arg, 2-3 validation checks. Main decision point is whether to adjust `energy_threshold` default value.

**Estimated Implementation Time:** 1-2 hours including tests

**Ready for Planning:** Yes
