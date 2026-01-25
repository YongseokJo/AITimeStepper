---
wave: 1
depends_on: []
files_modified:
  - src/config.py
autonomous: true
---

# Plan 02: Add Validation for Training Parameters

## Objective
Add validation rules to the Config.validate() method to ensure training parameters have valid values.

## Tasks

<task id="1">
Add validation for `epochs >= 1` in the `validate()` method.

Add this check in the validate() method (after line 246, before the final blank line):
```python
if self.epochs < 1:
    raise ValueError("epochs must be >= 1")
```
</task>

<task id="2">
Add validation for `steps_per_epoch >= 1` in the `validate()` method.

Add this check immediately after the epochs validation:
```python
if self.steps_per_epoch < 1:
    raise ValueError("steps_per_epoch must be >= 1")
```
</task>

<task id="3">
Add validation for `energy_threshold > 0` in the `validate()` method.

Add this check after the steps_per_epoch validation:
```python
if self.energy_threshold <= 0:
    raise ValueError("energy_threshold must be > 0")
```
</task>

## Verification

Run the following checks:

1. **Syntax check**: `python -m py_compile src/config.py`

2. **epochs validation**:
   ```python
   python -c "
   from src.config import Config
   try:
       c = Config(epochs=0)
       c.validate()
       print('FAIL: should have raised ValueError')
       exit(1)
   except ValueError as e:
       assert 'epochs' in str(e), f'Wrong error: {e}'
       print('epochs validation OK')
   "
   ```

3. **steps_per_epoch validation**:
   ```python
   python -c "
   from src.config import Config
   try:
       c = Config(steps_per_epoch=0)
       c.validate()
       print('FAIL: should have raised ValueError')
       exit(1)
   except ValueError as e:
       assert 'steps_per_epoch' in str(e), f'Wrong error: {e}'
       print('steps_per_epoch validation OK')
   "
   ```

4. **energy_threshold validation**:
   ```python
   python -c "
   from src.config import Config
   try:
       c = Config(energy_threshold=-0.01)
       c.validate()
       print('FAIL: should have raised ValueError')
       exit(1)
   except ValueError as e:
       assert 'energy_threshold' in str(e), f'Wrong error: {e}'
       print('energy_threshold validation OK')
   "
   ```

5. **Valid config passes**:
   ```python
   python -c "
   from src.config import Config
   c = Config(epochs=100, steps_per_epoch=5, energy_threshold=0.01)
   c.validate()
   print('Valid config passes validation OK')
   "
   ```

## must_haves
- [ ] `epochs < 1` raises ValueError with message containing "epochs"
- [ ] `steps_per_epoch < 1` raises ValueError with message containing "steps_per_epoch"
- [ ] `energy_threshold <= 0` raises ValueError with message containing "energy_threshold"
- [ ] Valid configurations pass validate() without error
