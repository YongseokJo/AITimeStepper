#!/usr/bin/env python3
"""Run Optuna hyperparameter search while logging each trial to Weights & Biases.

This script wraps the existing `optuna/main.py` `Optimization` class and runs
trials with a separate `wandb` run per Optuna trial for clear experiment tracking.

Usage:
    python run/optuna_wandb.py --study-name three_body_preliminary --n-trials 50

Key flags:
  --study-name    Name of the Optuna study (default: three_body_preliminary)
  --n-trials      Number of Optuna trials
  --storage       SQLite storage path for Optuna (default: ./optuna.db)
  --wandb-project W&B project name
  --wandb-entity  W&B entity/team (optional)
  --n-gpus        Number of GPUs available (used to select gpu_id by job index)
"""

import argparse
import sys
import pathlib
import optuna
import wandb
from functools import partial
import importlib.util


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def load_local_optuna_main():
    path = project_root / "optuna" / "main.py"
    spec = importlib.util.spec_from_file_location("local_optuna_main", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="three_body_preliminary")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--storage", default="./optuna.db")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--wandb-project", default="AITimeStepper")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    local = load_local_optuna_main()

    # Construct Optimization object from existing local module
    opt = local.Optimization(args.study_name, args.storage, load_if_exists=True, directions=["minimize", "maximize"])

    # Create study if not already created (the class creates it too, but do it explicitly)
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=args.study_name,
        storage="sqlite:///" + str(args.storage),
        load_if_exists=True,
        directions=["minimize", "maximize"],
        sampler=sampler,
    )

    gpu_id = 0  # simple single-job default; you can adapt for multi-job / slurm


    def wrapped_objective(gpu_id, trial):
        # Suggest a small search space here and call the ML_test_wandb script as a subprocess.
        lr = trial.suggest_float("lr", 1e-9, 1e-3, log=True)
        n_steps = trial.suggest_int("n_steps", 1, 20)
        epochs = trial.suggest_int("epochs", 10, 200)

        run_name = args.wandb_name or f"{args.study_name}-trial-{trial.number}"

        cmd = [
            sys.executable,
            str(project_root / "run" / "ML_test_wandb.py"),
            "--optuna",
            "--lr", str(lr),
            "--n-steps", str(n_steps),
            "--epochs", str(epochs),
            "--wandb-project", args.wandb_project,
            "--wandb-name", run_name,
        ]

        # Run the training script and capture its stdout
        import subprocess, json
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # For debugging: write child stdout/stderr to this process's stdout
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)

        # Parse OPTUNA_RESULT_JSON line
        loss_val = None
        time_step_val = None
        for line in proc.stdout.splitlines():
            if line.startswith("OPTUNA_RESULT_JSON:"):
                try:
                    payload = json.loads(line.split("OPTUNA_RESULT_JSON:", 1)[1])
                    loss_val = float(payload.get("loss", 1e6))
                    time_step_val = float(payload.get("time_step", 0.0))
                except Exception:
                    pass

        if loss_val is None:
            # failed to parse result: penalize
            return 1e6, 0.0

        # Optionally log trial params to a lightweight W&B run here (skipped)
        return loss_val, time_step_val


    # Run optimization using the wrapped objective
    study.optimize(partial(wrapped_objective, gpu_id), n_trials=args.n_trials, n_jobs=args.n_jobs)

    print("Optuna + W&B optimization finished.")


if __name__ == "__main__":
    main()
