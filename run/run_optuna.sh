#!/bin/bash

# Example Optuna entrypoint for unified runner
# Adjust study name, trials, and storage as needed.

python optuna/main.py --runner run/runner.py
