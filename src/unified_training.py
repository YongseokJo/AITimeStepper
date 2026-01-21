"""
Unified epoch training for two-phase training system.

This module provides the epoch orchestrator that combines:
- Part 1: Trajectory collection (collect_trajectory)
- Part 2: Generalization training (generalize_on_trajectory)

One epoch = Part 1 + Part 2. The trajectory from Part 1 is passed
directly to Part 2 for generalization training.

Phase 5 of AITimeStepper training refactor.
Requirements: TRAIN-08, TRAIN-09
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import Config
from .history_buffer import HistoryBuffer
from .model_adapter import ModelAdapter
from .particle import ParticleTorch
from .trajectory_collection import collect_trajectory
from .generalization_training import generalize_on_trajectory


def train_epoch_two_phase(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Dict[str, Any]:
    """
    Execute one complete epoch: Part 1 (collection) + Part 2 (generalization).

    Implements TRAIN-08: One epoch = Part 1 + Part 2.

    The epoch proceeds as follows:
    1. Part 1: Collect validated trajectory using collect_trajectory()
       - Predicts dt, integrates, accepts/rejects based on energy threshold
       - Collects config.steps_per_epoch accepted samples
       - Discards warmup steps (first history_len)
    2. Part 2: Generalize on trajectory using generalize_on_trajectory()
       - Samples random minibatches from collected trajectory
       - Trains until all samples pass OR max iterations (config.replay_steps)

    The epoch completes after Part 2 finishes (either converged or max iterations).

    Args:
        model: Neural network for dt prediction
        particle: Starting particle state for this epoch
        optimizer: PyTorch optimizer for model
        config: Config with energy_threshold, steps_per_epoch, replay_steps, etc.
        adapter: ModelAdapter for feature construction
        history_buffer: Optional history buffer for temporal features

    Returns:
        epoch_result: dict with:
            'trajectory_metrics': dict from collect_trajectory()
                - 'total_steps', 'warmup_discarded', 'trajectory_length'
                - 'mean_retrain_iterations', 'mean_energy_error', 'max_retrain_iterations'
            'generalization_metrics': dict from generalize_on_trajectory()
                - 'mean_rel_dE', 'max_rel_dE', 'final_pass_rate'
            'converged': bool - True if Part 2 converged before max iterations
            'part2_iterations': int - Number of iterations in Part 2
            'epoch_time': float - Wall clock time for epoch (seconds)

    Example:
        >>> epoch_result = train_epoch_two_phase(
        ...     model, particle, optimizer, config, adapter, history_buffer
        ... )
        >>> print(f"Trajectory: {epoch_result['trajectory_metrics']['trajectory_length']} samples")
        >>> print(f"Converged: {epoch_result['converged']} in {epoch_result['part2_iterations']} iterations")
    """
    epoch_start = time.perf_counter()

    # Part 1: Collect trajectory (TRAIN-04, HIST-02)
    trajectory, traj_metrics = collect_trajectory(
        model=model,
        particle=particle,
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=history_buffer,
    )

    # Log warning if trajectory is empty (edge case)
    if not trajectory:
        warnings.warn(
            f"train_epoch_two_phase: Empty trajectory collected "
            f"(steps_per_epoch={config.steps_per_epoch}, history_len={config.history_len})",
            UserWarning,
        )

    # Part 2: Generalize on trajectory (TRAIN-05, TRAIN-06, TRAIN-07)
    converged, part2_iters, gen_metrics = generalize_on_trajectory(
        model=model,
        trajectory=trajectory,
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=history_buffer,
    )

    epoch_time = time.perf_counter() - epoch_start

    return {
        'trajectory_metrics': traj_metrics,
        'generalization_metrics': gen_metrics,
        'converged': converged,
        'part2_iterations': part2_iters,
        'epoch_time': epoch_time,
    }
