---
wave: 1
depends_on: []
files_modified:
  - src/config.py
autonomous: true
---

# Plan 01: Add steps_per_epoch Field and CLI Argument

## Objective
Add the `steps_per_epoch` configuration parameter to the Config dataclass with corresponding CLI argument support.

## Tasks

<task id="1">
Add `steps_per_epoch` field to Config dataclass after `energy_threshold` (line 21).

The field should be:
```python
steps_per_epoch: int = 1
```

Insert this as a new line after line 21 (`energy_threshold: float = 2e-4`).
</task>

<task id="2">
Add CLI argument for `--steps-per-epoch` in the "train" section of `add_cli_args()`.

Add this line after the `--energy-threshold` argument (around line 112):
```python
add_arg("--steps-per-epoch", type=int, default=cls.steps_per_epoch, help="number of training steps per epoch")
```
</task>

## Verification

Run the following checks:

1. **Syntax check**: `python -m py_compile src/config.py`
2. **Field exists**: `python -c "from src.config import Config; c = Config(); print(f'steps_per_epoch={c.steps_per_epoch}')"` - should print `steps_per_epoch=1`
3. **CLI help**: `python run/runner.py train --help | grep -i steps-per-epoch` - should show the argument
4. **Serialization round-trip**:
   ```python
   python -c "
   from src.config import Config
   c = Config(steps_per_epoch=5)
   d = c.to_dict()
   c2 = Config.from_dict(d)
   assert c2.steps_per_epoch == 5, 'round-trip failed'
   print('Serialization OK')
   "
   ```

## must_haves
- [ ] `steps_per_epoch` field exists in Config dataclass with default value 1
- [ ] `--steps-per-epoch` CLI argument is available in train subcommand
- [ ] Field serializes correctly via to_dict/from_dict (automatic via dataclass)
