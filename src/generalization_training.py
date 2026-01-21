"""
Generalization training loop for two-phase training (Part 2).

This module implements the second phase of the two-phase training routine:
- Phase 1 (Part 1): collect_trajectory() - Collect validated trajectory samples
- Phase 2 (Part 2): generalize_on_trajectory() - Train on random minibatches until convergence

The generalization loop:
1. Sample random minibatch from collected trajectory
2. Evaluate each sample with single-step prediction and integration
3. Compute losses for samples that fail energy threshold
4. Update model with aggregated loss
5. Repeat until all samples pass OR max iterations reached

Key properties:
- Trajectory particles are never modified (immutability via clone_detached)
- Convergence when all minibatch samples pass energy threshold
- Safety limit prevents infinite loops (config.replay_steps)
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import Config
from .history_buffer import HistoryBuffer
from .model_adapter import ModelAdapter
from .particle import ParticleTorch
from .trajectory_collection import (
    attempt_single_step,
    check_energy_threshold,
    compute_single_step_loss,
)


def sample_minibatch(
    trajectory: List[Tuple[ParticleTorch, float]],
    batch_size: int,
) -> List[Tuple[ParticleTorch, float]]:
    """
    Sample random minibatch from trajectory.

    Uses random.sample() for uniform sampling without replacement from the
    trajectory list. If batch_size exceeds trajectory length, returns all
    samples (capped at trajectory length).

    Args:
        trajectory: List of (particle, dt) tuples from collect_trajectory()
        batch_size: Number of samples to draw (capped at trajectory length)

    Returns:
        List of (particle, dt) tuples

    Example:
        >>> trajectory = [(p1, 0.01), (p2, 0.02), (p3, 0.015)]
        >>> batch = sample_minibatch(trajectory, batch_size=2)
        >>> len(batch)
        2
    """
    batch_size = min(batch_size, len(trajectory))
    return random.sample(trajectory, k=batch_size)


def evaluate_minibatch(
    model: torch.nn.Module,
    minibatch: List[Tuple[ParticleTorch, float]],
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Tuple[bool, List[torch.Tensor], List[float], Dict[str, Any]]:
    """
    Evaluate all samples in minibatch with single-step predictions.

    For each sample in the minibatch:
    1. Call attempt_single_step() to predict dt and integrate
    2. Check energy threshold with check_energy_threshold()
    3. If failed, compute loss with compute_single_step_loss()

    IMPORTANT: Trajectory particles are never modified. attempt_single_step()
    creates clone_detached() internally to preserve immutability.

    Args:
        model: Neural network for dt prediction
        minibatch: List of (particle, dt) tuples to evaluate
        config: Config with energy_threshold
        adapter: ModelAdapter for feature construction
        history_buffer: Optional history buffer for temporal features

    Returns:
        (all_pass, losses, rel_dE_list, metrics) where:
        - all_pass: True if all samples passed energy threshold
        - losses: List of loss tensors for failed samples
        - rel_dE_list: List of relative energy errors (floats) for all samples
        - metrics: Dict with 'pass_count', 'fail_count', 'mean_rel_dE', 'max_rel_dE'

    Example:
        >>> all_pass, losses, rel_dEs, metrics = evaluate_minibatch(
        ...     model, minibatch, config, adapter
        ... )
        >>> if not all_pass:
        ...     total_loss = torch.stack(losses).mean()
        ...     total_loss.backward()
    """
    losses: List[torch.Tensor] = []
    rel_dE_list: List[float] = []
    pass_count = 0
    fail_count = 0

    for particle, _stored_dt in minibatch:
        # Predict dt, integrate one step (creates clone_detached internally)
        _advanced, _dt, E0, E1 = attempt_single_step(
            model, particle, config, adapter, history_buffer
        )

        # Check energy threshold
        passed, rel_dE = check_energy_threshold(E0, E1, config.energy_threshold)

        # Track relative energy error (convert to float)
        rel_dE_value = rel_dE.item() if rel_dE.numel() == 1 else rel_dE.mean().item()
        rel_dE_list.append(rel_dE_value)

        if passed:
            pass_count += 1
        else:
            fail_count += 1
            # Compute loss for failed sample
            loss = compute_single_step_loss(E0, E1, config)
            losses.append(loss)

    # Compute metrics
    all_pass = (fail_count == 0)
    metrics: Dict[str, Any] = {
        'pass_count': pass_count,
        'fail_count': fail_count,
        'mean_rel_dE': sum(rel_dE_list) / len(rel_dE_list) if rel_dE_list else 0.0,
        'max_rel_dE': max(rel_dE_list) if rel_dE_list else 0.0,
    }

    return all_pass, losses, rel_dE_list, metrics


def generalize_on_trajectory(
    model: torch.nn.Module,
    trajectory: List[Tuple[ParticleTorch, float]],
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Train on random minibatches from trajectory until convergence.

    Implements the generalization training loop (Part 2 of two-phase training):
    1. Sample random minibatch from trajectory
    2. Evaluate all samples with single-step predictions
    3. If all pass energy threshold: converged, return success
    4. Otherwise: aggregate losses, backprop, update model
    5. Repeat until convergence or max iterations (config.replay_steps)

    IMPORTANT: Trajectory particles are never modified during replay.
    The attempt_single_step() function creates clone_detached() internally
    to preserve immutability.

    Args:
        model: Neural network for dt prediction
        trajectory: List of (particle, dt) tuples from collect_trajectory()
        optimizer: PyTorch optimizer for model
        config: Config with replay_steps, replay_batch_size, min_replay_size
        adapter: ModelAdapter for feature construction
        history_buffer: Optional history buffer for temporal features

    Returns:
        (converged, iterations, final_metrics) where:
        - converged: True if all samples passed before max iterations
        - iterations: Number of iterations completed
        - final_metrics: Dict with 'mean_rel_dE', 'max_rel_dE', 'final_pass_rate',
                        optionally 'skipped' if trajectory was too small

    Example:
        >>> converged, iters, metrics = generalize_on_trajectory(
        ...     model, trajectory, optimizer, config, adapter
        ... )
        >>> if converged:
        ...     print(f"Converged in {iters} iterations")
        >>> else:
        ...     print(f"Did not converge after {config.replay_steps} iterations")
    """
    # Edge case: empty trajectory
    if not trajectory:
        return True, 0, {
            'mean_rel_dE': 0.0,
            'max_rel_dE': 0.0,
            'final_pass_rate': 1.0,
        }

    # Edge case: trajectory below min_replay_size
    if len(trajectory) < config.min_replay_size:
        return True, 0, {
            'mean_rel_dE': 0.0,
            'max_rel_dE': 0.0,
            'final_pass_rate': 1.0,
            'skipped': True,
        }

    # Track metrics across iterations
    final_metrics: Dict[str, Any] = {
        'mean_rel_dE': 0.0,
        'max_rel_dE': 0.0,
        'final_pass_rate': 0.0,
    }

    # Main convergence loop
    for iteration in range(config.replay_steps):
        # 1. Sample minibatch
        minibatch = sample_minibatch(trajectory, config.replay_batch_size)

        # 2. Evaluate minibatch
        all_pass, losses, rel_dE_list, batch_metrics = evaluate_minibatch(
            model, minibatch, config, adapter, history_buffer
        )

        # Update final metrics
        final_metrics['mean_rel_dE'] = batch_metrics['mean_rel_dE']
        final_metrics['max_rel_dE'] = batch_metrics['max_rel_dE']
        pass_rate = batch_metrics['pass_count'] / len(minibatch) if minibatch else 1.0
        final_metrics['final_pass_rate'] = pass_rate

        # 3. Check convergence
        if all_pass:
            return True, iteration + 1, final_metrics

        # 4. Aggregate losses and backprop
        if losses:
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # Did not converge within max iterations
    return False, config.replay_steps, final_metrics
