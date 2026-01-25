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
from typing import Any, Dict, List, Optional, Tuple, Union

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
    trajectory: List[Tuple[ParticleTorch, float, Optional[torch.Tensor]]],
    batch_size: int,
) -> List[Tuple[ParticleTorch, float, Optional[torch.Tensor]]]:
    """
    Sample random minibatch from trajectory.

    Uses random.sample() for uniform sampling without replacement from the
    trajectory list. If batch_size exceeds trajectory length, returns all
    samples (capped at trajectory length).

    Args:
        trajectory: List of (particle, dt, mask) tuples from collect_trajectory()
        batch_size: Number of samples to draw (capped at trajectory length)

    Returns:
        List of (particle, dt, mask) tuples

    Example:
        >>> trajectory = [(p1, 0.01, None), (p2, 0.02, None)]
        >>> batch = sample_minibatch(trajectory, batch_size=2)
        >>> len(batch)
        2
    """
    batch_size = min(batch_size, len(trajectory))
    return random.sample(trajectory, k=batch_size)


def evaluate_minibatch(
    model: torch.nn.Module,
    minibatch: List[Tuple[ParticleTorch, float, Optional[torch.Tensor]]],
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
        minibatch: List of (particle, dt, mask) tuples to evaluate
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

    for item in minibatch:
        if len(item) == 3:
            particle, _stored_dt, active_mask = item
        else:
            particle, _stored_dt = item
            active_mask = None

        # Predict dt, integrate one step (creates clone_detached internally)
        _advanced, _dt, E0, E1 = attempt_single_step(
            model, particle, config, adapter, history_buffer, active_mask=active_mask
        )

        # Check energy threshold
        passed, rel_dE = check_energy_threshold(E0, E1, config.energy_threshold, active_mask=active_mask)

        # Track relative energy error (convert to float)
        rel_dE_value = rel_dE.item() if rel_dE.numel() == 1 else rel_dE.mean().item()
        rel_dE_list.append(rel_dE_value)

        if passed:
            pass_count += 1
        else:
            fail_count += 1
            # Compute loss for failed sample
            loss = compute_single_step_loss(E0, E1, config, active_mask=active_mask)
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
    trajectory: List[Tuple[ParticleTorch, float, Optional[torch.Tensor]]],
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    wandb_run: Optional[Any] = None,
    log_every: int = 100,
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
        trajectory: List of (particle, dt, mask) tuples from collect_trajectory()
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
    wandb = None
    if wandb_run is not None:
        try:
            import wandb as wandb_lib
            wandb = wandb_lib
        except ImportError:
            wandb_run = None

    if config.debug:
        print(f"part2: trajectory_len={len(trajectory)} min_replay_size={config.min_replay_size}")

    # Edge case: empty trajectory
    if not trajectory:
        if wandb_run is not None and wandb is not None:
            wandb.log({"part2/skip_reason": 1, "part2/trajectory_len": 0})
        if config.debug:
            print("part2: skipped (empty trajectory)")
        return True, 0, {
            'mean_rel_dE': 0.0,
            'max_rel_dE': 0.0,
            'final_pass_rate': 1.0,
        }

    # Edge case: trajectory below min_replay_size
    if len(trajectory) < config.min_replay_size:
        if wandb_run is not None and wandb is not None:
            wandb.log({"part2/skip_reason": 2, "part2/trajectory_len": len(trajectory)})
        if config.debug:
            print("part2: skipped (trajectory below min_replay_size)")
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

        loss_mean = 0.0
        loss_max = 0.0
        if losses:
            loss_stack = torch.stack(losses)
            loss_mean = loss_stack.mean().item()
            loss_max = loss_stack.max().item()

        grad_norm = 0.0
        lr = optimizer.param_groups[0].get("lr", 0.0) if optimizer.param_groups else 0.0

        # Update final metrics
        final_metrics['mean_rel_dE'] = batch_metrics['mean_rel_dE']
        final_metrics['max_rel_dE'] = batch_metrics['max_rel_dE']
        pass_rate = batch_metrics['pass_count'] / len(minibatch) if minibatch else 1.0
        final_metrics['final_pass_rate'] = pass_rate

        # 3. Check convergence
        if all_pass:
            if (
                wandb_run is not None
                and wandb is not None
                and log_every > 0
                and iteration % log_every == 0
            ):
                wandb.log(
                    {
                        "part2/replay/iteration": iteration,
                        "part2/replay/pass_count": batch_metrics["pass_count"],
                        "part2/replay/fail_count": batch_metrics["fail_count"],
                        "part2/replay/mean_rel_dE": batch_metrics["mean_rel_dE"],
                        "part2/replay/max_rel_dE": batch_metrics["max_rel_dE"],
                        "part2/replay/loss_mean": loss_mean,
                        "part2/replay/loss_max": loss_max,
                        "part2/replay/grad_norm": grad_norm,
                        "part2/replay/lr": lr,
                        "part2/replay/trajectory_len": len(trajectory),
                    }
                )
            if config.debug and iteration % config.debug_replay_every == 0:
                print(
                    "part2/replay "
                    f"iter={iteration} pass={batch_metrics['pass_count']} fail={batch_metrics['fail_count']} "
                    f"mean_rel_dE={batch_metrics['mean_rel_dE']:.4e} max_rel_dE={batch_metrics['max_rel_dE']:.4e} "
                    f"loss_mean={loss_mean:.4e} lr={lr:.4e}"
                )
            return True, iteration + 1, final_metrics

        # 4. Aggregate losses and backprop
        if losses:
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm_sq += param.grad.detach().pow(2).sum().item()
            grad_norm = grad_norm_sq ** 0.5
            optimizer.step()

            if (
                wandb_run is not None
                and wandb is not None
                and log_every > 0
                and iteration % log_every == 0
            ):
                wandb.log(
                    {
                        "part2/replay/iteration": iteration,
                        "part2/replay/pass_count": batch_metrics["pass_count"],
                        "part2/replay/fail_count": batch_metrics["fail_count"],
                        "part2/replay/mean_rel_dE": batch_metrics["mean_rel_dE"],
                        "part2/replay/max_rel_dE": batch_metrics["max_rel_dE"],
                        "part2/replay/loss_mean": loss_mean,
                        "part2/replay/loss_max": loss_max,
                        "part2/replay/grad_norm": grad_norm,
                        "part2/replay/lr": lr,
                        "part2/replay/trajectory_len": len(trajectory),
                    }
                )
            if config.debug and iteration % config.debug_replay_every == 0:
                print(
                    "part2/replay "
                    f"iter={iteration} pass={batch_metrics['pass_count']} fail={batch_metrics['fail_count']} "
                    f"mean_rel_dE={batch_metrics['mean_rel_dE']:.4e} max_rel_dE={batch_metrics['max_rel_dE']:.4e} "
                    f"loss_mean={loss_mean:.4e} grad_norm={grad_norm:.4e} lr={lr:.4e}"
                )

    # Did not converge within max iterations
    return False, config.replay_steps, final_metrics
