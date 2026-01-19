Integration Plan for ML + Simulation
===================================

Phase 1: Shared Config & Checkpoint Plumbing
---------------------------------------------
- [x] **Shared config module (`src/config.py`)**  
  - Define a `Config` dataclass that collects every argument required by ML runners and simulators: optimizer type/phases, `epochs`, `lr`, `weight_decay`, `n_steps`, replay bounds, `history_len`, `feature_type`, `E/L` bands, `device`/`dtype`, `tf32`/`compile` toggles, `integrator_mode` (`analytic`, `ml`, `history`), `external_field` params, `save_name`, random seeds, etc.  
  - Provide constructors (`Config.from_cli`, `Config.from_dict`) so each CLI only wires its `argparse` parser into the shared config, eliminating duplicated argument definitions.  
  - Add helpers to resolve device/dtype, convert to W&B config dict, and validate interdependent fields (e.g., `history_len > 0` must pair with `feature_type`).
- [x] **Unified checkpoint helper (`src/checkpoint.py`)**  
  - Save both `model_state_dict` (expected by simulator) and the legacy `model_state` blob (used by `trainer.save_checkpoint`) along with metadata (config summary, dtype, history settings).  
  - Load checkpoints by checking either key and optionally loading optimizer state; expose `load_model_state(model, path)` plus `load_config_from_checkpoint(path)` so scripts automatically read the right history/dtype info.  
  - Ensure saving uses the shared `Config` so future scripts can compare what config was used without manual bookkeeping.

Phase 2: Model/Feature Adapter Consolidation
--------------------------------------------
- [x] **Model adapter (`src/model_adapter.py`)**  
  - Given a `Config`, build the `HistoryBuffer` (if `history_len>0`) or plain features via `system_features`.  
  - Instantiate `FullyConnectedNN` with consistent `hidden_dims`, `dropout`, activation, and dtype/device determined by the config.  
  - Provide helpers such as `build_feature_tensor(particle_or_batch)`, `predict_dt(model, state, history=None)`, and `update_history(history, particle)` so both training loops and simulators reuse the same pipeline.  
  - Return metadata (e.g., input `feature_dim`, history enabled flag) for logging and checkpointing.
- [x] **Refactor ML runners to consume adapter**  
  - Update `run/ML_history_wandb.py`, `run/ML_history_multi_wandb.py`, and `run/ML_test_wandb.py` to instantiate the shared config + adapter up front, reduce duplicated HistoryBuffer logic, and log a single `config.as_wandb_dict()`.  
  - Replace manual feature dimension inference with `adapter.input_dim`.  
  - Save checkpoints via the new helper so training and simulator share a canonical artifact.
- [x] **Mirror adapter usage in simulators**  
  - Refactor `simulators/particle.py` prediction helpers to call `adapter.predict_dt` for history-aware and history-free cases, ensuring the same config (history len/feature type/dtype) is honored.  
  - When the simulator steps using history-aware ML, use the adapter’s `update_history` helper to push accepted states, mimicking training replay semantics.  
  - Ensure dtype/device resolution falls back to CPU when necessary but prefers the training config (e.g., `torch.double` + `cuda` if available).

Phase 3: Simulation Integration with N-body & External Field
------------------------------------------------------------
- [ ] **Extend simulators to accept shared config**  
  - Update `simulators/nbody_simulator.py` (and `two_body_simulator.py` as needed) to take `Config` and optional `external_acceleration` callbacks (from `src/external_potentials`).  
  - Use `config.integrator_mode` to select between analytic dt, ML dt, or history-aware ML dt, and ensure ML modes call the shared adapter + checkpoint loader.  
  - Pass through tide settings (mass, position) to the simulator and to the `ParticleTorch` states so the external field is applied consistently in both training and simulation.
- [ ] **Single CLI for analytic + ML simulations**  
  - Replace `run/simulator_test.py` (or add `run/simulator_runner.py`) with a script that loads `Config.from_cli`, allows selecting `--integrator-mode {analytic,ml,history}`, toggles the external field, and loads the shared checkpoint automatically.  
  - Document the drop-in options for `--history-len`, `--feature-type`, `--external-field-mass/position`, etc., so you can switch between modes without editing the code.  
  - Log the effective config at start (e.g., `print(config.summary())`) to ensure you never manually mismatch history vs simulation settings again.
- [ ] **History bookkeeping parity**  
  - When the simulator accepts a state under history-aware ML, push the state into the shared `HistoryBuffer` just like the training replay loop does; keep a local structure (`accepted_states`) to support replay training if desired.  
  - Provide helper utilities for sampling multi-orbit ICs (`e` range, `a` range) so future N-body sims reuse the same dataset creation logic as `ML_history_multi_wandb.py`.

Phase 4: Documentation & Integration Sanity
------------------------------------------
- [ ] **Docs refresh**  
  - Update `README.md` (and/or `Notes.md`) with the following sections:  
    1. “Shared Config”: describe `Config.from_cli`, how to extend it, and list the default parameter set.  
    2. “Checkpoint contract”: explain why we save both `model_state_dict` and `model_state`, how to load them, and where files live (e.g., `data/<save_name>/model/`).  
    3. “Simulation matrix”: show how to run analytic, history-free ML, history-aware ML, and how to enable the external tidal field using the new shared CLI.  
- [ ] **Integration sanity script (`run/integration_sanity.py`)**  
  - Use the shared config + adapter to train for a few epochs with simplified settings (small replay batch, short `epochs`) and save via the new checkpoint helper.  
  - Immediately load that checkpoint and run a short N-body simulation (with optional tidal field) using the shared CLI, logging acceptance rate, energy drift, and final state.  
  - Output a short summary (could be printed or saved to `data/<save_name>/sanity.log`) to confirm the entire pipeline works end-to-end.

Next Steps
----------
1. [ ] Prototype the shared `Config` dataclass + CLI wiring; update `run/ML_history_wandb.py` to rely on it and verify argument coverage.  
2. [ ] Implement the checkpoint helper, refactor training scripts to use it, and ensure `run/simulator_test.py` loads the resulting artifact without format issues.  
3. [ ] Build the model adapter, update both training and simulator code to share features/history handling, and extend `simulators/nbody_simulator.py` with the new config + external field path.  
4. [ ] Refresh the docs and add `run/integration_sanity.py` so every future change can be validated against the shared workflow.
