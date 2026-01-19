# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the core PyTorch models, feature construction, losses, and integrators used for ML time-stepping.
- `simulators/` holds NumPy-based physics simulators and utility classes for non-training runs.
- `run/` provides runnable scripts for sanity checks, training/evaluation, and Slurm/interactive helpers.
- `optuna/` contains sweep entry points (see `optuna/main.py` and `optuna/run.sh`).
- `jupyter_notebooks/` is for exploration and visualization.
- `data/` is the default output location for models, logs, plots, and movies and is git-ignored.

## Build, Test, and Development Commands
- `python -c "import torch, numpy, matplotlib, wandb"` — quick import sanity check for the environment.
- `python run/simulator_test.py` — two-body simulator sanity run (plots + optional movie output).
- `python run/ML_history_wandb.py --epochs 1000 --n-steps 10 --history-len 3 --feature-type basic` — trains the history-aware time-stepper with W&B logging.
- `python run/ML_test.py` or `python run/ML_history_wandb.py --debug ...` — local evaluation/debugging scripts.
- `bash optuna/run.sh` — launch Optuna sweeps (adapt to your environment/cluster).

## Coding Style & Naming Conventions
- Python code follows standard 4-space indentation; avoid tabs.
- Use `snake_case` for functions/variables and `CapWords` for classes (as in `src/` and `simulators/`).
- No formatter or linter is configured; keep changes consistent with nearby code.
- Prefer explicit device handling (`cpu`/`cuda`) and deterministic parameter names (e.g., `history_len`, `feature_type`).

## Testing Guidelines
- There is no dedicated test framework; validation is done via runnable scripts in `run/`.
- Keep new sanity scripts named `*_test.py` or `*_sanity.py`, and document key flags in the script’s `argparse` help.
- When adding features, include a minimal reproducible run in `run/` and note expected outputs.

## Commit & Pull Request Guidelines
- Commit messages in history are concise, sentence case, and often end with a period; follow that style.
- PRs should include: a short summary, commands run, and any relevant plots/metrics.
- Do not commit generated outputs under `data/` (ignored); attach artifacts to the PR if needed.
