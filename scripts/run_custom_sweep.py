#!/usr/bin/env python3
"""
Simple script to run a grid search (sweep) over hyperparameters for AITimeStepper.
It generates temporary TOML configuration files and launches 'run/runner.py'.

Usage:
    python scripts/run_custom_sweep.py --base_config configs/example.toml --output_dir data/sweeps
"""

import argparse
import itertools
import json
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, List

import tomli_w  # Requires tomli-w for writing TOML (or just use simple string formatting if dep missing)

# If tomli_w is missing, fallback to basic string formatting (less robust but works for simple dicts)
try:
    import tomli_w
    HAS_TOMLI_W = True
except ImportError:
    HAS_TOMLI_W = False
    print("Warning: 'tomli_w' not found. Using simple TOML generator.")

def load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            raise RuntimeError("Python 3.11+ or 'tomli' installed required.")
    
    with open(path, "rb") as f:
        return tomllib.load(f)

def write_toml(data: Dict[str, Any], path: str):
    if HAS_TOMLI_W:
        with open(path, "wb") as f:
            tomli_w.dump(data, f)
    else:
        # Very basic TOML writer fallback
        with open(path, "w") as f:
            for section, content in data.items():
                if isinstance(content, dict):
                    f.write(f"[{section}]\n")
                    for k, v in content.items():
                        if isinstance(v, bool):
                            val = str(v).lower()
                        elif isinstance(v, str):
                            val = f'"{v}"'
                        elif isinstance(v, list):
                            val = str(v) # JSON-like list style is valid in TOML
                        else:
                            val = str(v)
                        f.write(f"{k} = {val}\n")
                    f.write("\n")
                else:
                    # Top level
                    if isinstance(content, bool):
                        val = str(content).lower()
                    elif isinstance(content, str):
                        val = f'"{content}"'
                    elif isinstance(content, list):
                        val = str(content)
                    else:
                        val = str(content)
                    f.write(f"{section} = {val}\n")

def main():
    parser = argparse.ArgumentParser(description="Run AITimeStepper Parameter Sweep")
    parser.add_argument("--base-config", type=str, required=True, help="Path to base TOML config")
    parser.add_argument("--output-dir", type=str, default="data/sweeps", help="Directory to store sweep results")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but do not run")
    args = parser.parse_args()

    base_config_path = pathlib.Path(args.base_config)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_toml(str(base_config_path))

    # =========================================================================
    # DEFINE HYPERPARAMETER GRID HERE
    # =========================================================================
    grid = {
        "lr": [1e-3, 1e-4, 5e-5],
        "hidden_dims": [[200, 200, 200], [200, 1000, 1000, 200]],
        "history_len": [0, 5, 10],
        "feature_type": ["basic", "delta_mag"],
    }
    # =========================================================================

    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    print(f"Base config: {base_config_path}")
    print(f"Found {len(combinations)} combinations to test.")
    print("-" * 60)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        run_name = f"sweep_{int(time.time())}_{i:03d}"
        print(f"Preparing Run {i+1}/{len(combinations)}: {run_name}")
        print(f"  Params: {params}")

        # Create new config dict based on base
        new_config = base_config.copy()
        
        # We need to inject params into the right sections
        # Config structure in TOML is flatter or has [train] sections.
        # Our runner uses a flat Config object but TOML can be nested.
        # runner.py flattens it somewhat.
        
        # Let's update top-level or sections.
        # 'hidden_dims' is top-level/Config.
        # 'lr' is top-level/Config.
        # 'history_len' is top-level/Config.
        # 'feature_type' is top-level/Config. 
        
        for k, v in params.items():
            # If the key exists in [train] or [history], update it there?
            # Or just put it at top level if runner handles it.
            # runner.py's _config_from_sources merges base, train, simulate.
            # Safe bet: update if exists in top, else put in top.
            new_config[k] = v
        
        new_config["save_name"] = run_name
        
        # Write temporary config
        temp_config_path = output_dir / f"{run_name}.toml"
        write_toml(new_config, str(temp_config_path))

        if not args.dry_run:
            cmd = ["python", "run/runner.py", "--config", str(temp_config_path)]
            print(f"  Executing: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  Error running {run_name}: {e}")
            print("-" * 60)
        else:
            print(f"  Dry run: generated {temp_config_path}")

if __name__ == "__main__":
    main()
