# Training routine summary (run/runner.py)

## Entry point
- `run/runner.py` uses `argparse` with a `train` subcommand.
- CLI arguments are provided via `Config.add_cli_args(... include=["train","bounds","history","device","logging","sim","multi"])` plus `--ic-path` and W&B flags (`--wandb`, `--wandb-project`, `--wandb-name`).

## Configuration and setup
- A `Config` is constructed from CLI args and validated.
- Device/dtype are resolved; `torch.set_default_dtype` is applied.
- Torch settings (tf32, anomaly detection) are applied via `Config.apply_torch_settings`.
- Random seeds for PyTorch and NumPy are set if `--seed` is provided.
- A `ModelAdapter` is created to handle feature construction and optional history buffers.

## Optional W&B logging
- If `--wandb` is provided, `wandb.init` is called with:
  - `project`: `--wandb-project` (default: `AITimeStepper`)
  - `name`: `--wandb-name` or `save_name` or `runner_train`
  - `config`: `config.as_wandb_dict()`
- Per-epoch metrics logged: `loss`, `dt`, `E0`, `rel_dE`, `rel_dE_mean`, `rel_dL_mean`.
- `wandb.finish()` is called after the training loop.

## Model/input construction
- Initial conditions are generated via `generate_random_ic`.
- If `num_orbits > 1`, multiple independent particle states are built:
  - Each orbit uses its own `HistoryBuffer` when history is enabled.
  - Particles are stacked with `stack_particles` to form a batch.
- Otherwise a single particle state is created.
- Input dimension is inferred using `ModelAdapter.input_dim_from_state(...)`.
- A `FullyConnectedNN` is instantiated with fixed hidden dims `[200, 1000, 1000, 200]`, `tanh` activations, dropout 0.2, and positive outputs.

## Training loop
- Optimizer: `torch.optim.Adam` with `--lr` and `--weight-decay`.
- Loop runs for `--epochs` or stops early if `--duration` (seconds) is exceeded.
- Loss computation depends on history and batch mode:
  - `loss_fn_batch_history_batch` for multi-orbit + history
  - `loss_fn_batch_batch` for multi-orbit without history
  - `loss_fn_batch_history` for single-orbit + history
  - `loss_fn_batch` for single-orbit without history
- Each iteration: forward pass, `loss.backward()`, `optimizer.step()`.

## Checkpointing and outputs
- Every 10 epochs (and final epoch) a checkpoint is saved:
  - Path: `data/<save_name|run_nbody>/model/model_epoch_XXXX.pt`
  - `save_checkpoint` stores model/optimizer state plus config and logs.
- Progress is printed each checkpoint save.

## Notes
- `--ic-path` is accepted but currently unused in training; training always uses generated ICs.
- History integration requires `--history-len > 0` when `integrator_mode=history`.
