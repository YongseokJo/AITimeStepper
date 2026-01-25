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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .checkpoint import save_checkpoint
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
    wandb_run: Optional[Any] = None,
    retrain_log_every: int = 100,
    part2_log_every: int = 100,
    return_trajectory: bool = False,
    include_warmup: bool = False,
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
            'trajectory': list of (ParticleTorch, dt) tuples if return_trajectory is True

    Example:
        >>> epoch_result = train_epoch_two_phase(
        ...     model, particle, optimizer, config, adapter, history_buffer
        ... )
        >>> print(f"Trajectory: {epoch_result['trajectory_metrics']['trajectory_length']} samples")
        >>> print(f"Converged: {epoch_result['converged']} in {epoch_result['part2_iterations']} iterations")
    """
    epoch_start = time.perf_counter()

    # Part 1: Collect trajectory (TRAIN-04, HIST-02)
    trajectory, traj_metrics, final_particle = collect_trajectory(
        model=model,
        particle=particle,
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=history_buffer,
        wandb_run=wandb_run,
        retrain_log_every=retrain_log_every,
        include_warmup=include_warmup,
        return_final_particle=True,
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
        wandb_run=wandb_run,
        log_every=part2_log_every,
    )

    epoch_time = time.perf_counter() - epoch_start

    result = {
        'trajectory_metrics': traj_metrics,
        'generalization_metrics': gen_metrics,
        'converged': converged,
        'part2_iterations': part2_iters,
        'epoch_time': epoch_time,
        'final_particle': final_particle,
    }
    if return_trajectory:
        result['trajectory'] = trajectory
    return result


def run_two_phase_training(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    save_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
    checkpoint_interval: int = 10,
    checkpoint_extra: Optional[Dict[str, Any]] = None,
    return_final_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    Run N epochs of two-phase training with checkpointing and logging.

    Implements TRAIN-09: Run for fixed N epochs (config.epochs).

    Training loop:
    1. For each epoch in range(config.epochs):
       a. Call train_epoch_two_phase() to execute Part 1 + Part 2
       b. Log metrics to W&B (if enabled)
       c. Save checkpoint every checkpoint_interval epochs and on final epoch
    2. Return aggregated training results

    IMPORTANT: The same history_buffer instance is passed to all epochs.
    This preserves temporal context across epochs (per 05-RESEARCH.md).

    IMPORTANT: The particle state is NOT reset between epochs by default.
    Each epoch continues from the final state of the previous epoch.
    For fresh ICs each epoch, caller should handle particle reset.

    Args:
        model: Neural network for dt prediction
        particle: Initial particle state
        optimizer: PyTorch optimizer for model
        config: Config with epochs, energy_threshold, steps_per_epoch, etc.
        adapter: ModelAdapter for feature construction
        history_buffer: Optional history buffer (persists across epochs)
        save_dir: Directory for checkpoint saves (None = no saves)
        wandb_run: W&B run object from wandb.init() (None = no logging)
        checkpoint_interval: Save checkpoint every N epochs (default: 10)
        checkpoint_extra: Extra payload to save in each checkpoint

    Returns:
        training_result: dict with:
            'epochs_completed': int - Number of epochs run
            'total_time': float - Total training time (seconds)
            'final_metrics': dict - Metrics from last epoch
            'convergence_rate': float - Fraction of epochs where Part 2 converged
            'results': List[dict] - Per-epoch results (if config.debug)
            'final_trajectory': cumulative List[(ParticleTorch, dt)] when return_final_trajectory is True

    Example:
        >>> result = run_two_phase_training(
        ...     model, particle, optimizer, config, adapter,
        ...     history_buffer=history_buffer,
        ...     save_dir=Path("data/my_run/model"),
        ...     wandb_run=wandb_run,
        ... )
        >>> print(f"Trained for {result['epochs_completed']} epochs")
        >>> print(f"Convergence rate: {result['convergence_rate']:.1%}")
    """
    # Import wandb only if needed (avoid import error if not installed)
    wandb = None
    if wandb_run is not None:
        try:
            import wandb as wandb_lib
            wandb = wandb_lib
        except ImportError:
            warnings.warn("wandb_run provided but wandb not installed; logging disabled")
            wandb_run = None

    # Track overall training
    training_start = time.perf_counter()
    all_results: List[Dict[str, Any]] = []
    converged_count = 0

    # Current particle (will be updated each epoch if continuing trajectory)
    current_particle = particle

    # Track last epoch result for final return
    epoch_result: Dict[str, Any] = {}
    final_trajectory: Optional[List[Tuple[ParticleTorch, float]]] = [] if return_final_trajectory else None

    part2_log_every = int(config.extra.get("part2_log_every", 100)) if isinstance(config.extra, dict) else 100

    for epoch in range(config.epochs):
        # Execute one epoch (Part 1 + Part 2)
        want_trajectory = return_final_trajectory
        epoch_result = train_epoch_two_phase(
            model=model,
            particle=current_particle,
            optimizer=optimizer,
            config=config,
            adapter=adapter,
            history_buffer=history_buffer,
            wandb_run=wandb_run,
            part2_log_every=part2_log_every,
            return_trajectory=want_trajectory,
            include_warmup=True,
        )
        current_particle = epoch_result['final_particle']
        if want_trajectory:
            epoch_trajectory = epoch_result.pop('trajectory', None)
            if epoch_trajectory:
                final_trajectory.extend(epoch_trajectory)

        # Track convergence
        if epoch_result['converged']:
            converged_count += 1

        # Store results (only in debug mode to save memory)
        if config.debug:
            all_results.append(epoch_result)

        # W&B logging
        if wandb_run is not None and wandb is not None:
            traj_m = epoch_result['trajectory_metrics']
            gen_m = epoch_result['generalization_metrics']

            # Compute acceptance rate: 1 / (1 + mean_retrain_iterations)
            mean_retrain = traj_m.get('mean_retrain_iterations', 0.0)
            acceptance_rate = 1.0 / (1.0 + mean_retrain) if mean_retrain >= 0 else 0.0

            log_data = {
                'epoch': epoch,
                'epoch_time': epoch_result['epoch_time'],
                # Part 1 metrics
                'part1/trajectory_length': traj_m.get('trajectory_length', 0),
                'part1/mean_retrain_iterations': mean_retrain,
                'part1/max_retrain_iterations': traj_m.get('max_retrain_iterations', 0),
                'part1/mean_energy_error': traj_m.get('mean_energy_error', 0.0),
                'part1/min_energy_error': traj_m.get('min_energy_error', 0.0),
                'part1/max_energy_error': traj_m.get('max_energy_error', 0.0),
                'part1/warmup_discarded': traj_m.get('warmup_discarded', 0),
                'part1/acceptance_rate': acceptance_rate,
                'part1/mean_retrain_loss': traj_m.get('mean_retrain_loss', 0.0),
                'part1/max_retrain_loss': traj_m.get('max_retrain_loss', 0.0),
                'part1/mean_retrain_rel_dE': traj_m.get('mean_retrain_rel_dE', 0.0),
                'part1/max_retrain_rel_dE': traj_m.get('max_retrain_rel_dE', 0.0),
                # Part 2 metrics
                'part2/converged': int(epoch_result['converged']),
                'part2/iterations': epoch_result['part2_iterations'],
                'part2/final_pass_rate': gen_m.get('final_pass_rate', 0.0),
                'part2/mean_rel_dE': gen_m.get('mean_rel_dE', 0.0),
                'part2/max_rel_dE': gen_m.get('max_rel_dE', 0.0),
            }
            wandb.log(log_data)

        # Checkpointing: every checkpoint_interval epochs and on final epoch
        if save_dir is not None:
            is_checkpoint_epoch = (epoch % checkpoint_interval == 0) or (epoch == config.epochs - 1)
            if is_checkpoint_epoch:
                save_path = Path(save_dir) / f"model_epoch_{epoch:04d}.pt"
                save_checkpoint(
                    save_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    logs=epoch_result,
                    config=config,
                    extra=checkpoint_extra,
                )
                if config.debug:
                    print(f"Checkpoint saved: {save_path}")

        # Print progress (every 10 epochs or in debug mode)
        if config.debug or (epoch % 10 == 0) or (epoch == config.epochs - 1):
            traj_len = epoch_result['trajectory_metrics']['trajectory_length']
            conv_str = "converged" if epoch_result['converged'] else f"max_iter({epoch_result['part2_iterations']})"
            print(f"Epoch {epoch}: traj_len={traj_len}, part2={conv_str}, time={epoch_result['epoch_time']:.2f}s")

    # Compute final results
    total_time = time.perf_counter() - training_start
    convergence_rate = converged_count / config.epochs if config.epochs > 0 else 0.0

    result = {
        'epochs_completed': config.epochs,
        'total_time': total_time,
        'final_metrics': epoch_result if config.epochs > 0 else {},
        'convergence_rate': convergence_rate,
        'results': all_results if config.debug else [],
        'final_particle': current_particle,
    }
    if return_final_trajectory:
        result['final_trajectory'] = final_trajectory or []
    return result
