"""
Trajectory collection primitives for two-phase training.

This module provides the core accept/reject loop components:
- attempt_single_step: Predict dt, integrate, return state and energies
- check_energy_threshold: Verify energy conservation
- compute_single_step_loss: Loss for retrain loop
- collect_trajectory_step: Accept/reject retrain loop
- collect_trajectory: N steps per epoch orchestrator (TRAIN-04, HIST-02)

Phase 3 of AITimeStepper training refactor.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .config import Config
from .history_buffer import HistoryBuffer
from .losses import band_loss_zero_inside_where
from .model_adapter import ModelAdapter
from .particle import ParticleTorch

# Numerical stability constant
EPS = 1e-12


def attempt_single_step(
    model: torch.nn.Module,
    particle: ParticleTorch,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> Tuple[ParticleTorch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict dt, integrate one step, return new state and energy info.

    IMPORTANT: Creates clone_detached() at start to prevent graph accumulation.

    This function is the core primitive for trajectory collection. It takes the
    current particle state, predicts a timestep using the model, integrates the
    system forward, and returns the new state along with energy measurements.

    Args:
        model: Neural network that predicts dt
        particle: Current particle state (will NOT be modified)
        config: Configuration with energy_threshold
        adapter: ModelAdapter for feature construction
        history_buffer: Optional history for temporal features
        active_mask: Optional boolean/float mask (1.0=active, 0.0=inactive).
                     Inactive particles get dt=0.

    Returns:
        (advanced_particle, dt, E0, E1) where:
        - advanced_particle: ParticleTorch after integration (in computation graph)
        - dt: predicted timestep tensor
        - E0: initial energy (before integration)
        - E1: final energy (after integration)

    Example:
        >>> p_new, dt, E0, E1 = attempt_single_step(model, particle, config, adapter)
        >>> passed, rel_dE = check_energy_threshold(E0, E1, config.energy_threshold)
        >>> if passed:
        ...     # Accept step, use p_new for next iteration
    """
    # Clone particle to prevent graph accumulation
    p = particle.clone_detached()

    # Build features
    feats = adapter.build_feature_tensor(p, history_buffer=history_buffer)

    # Handle feature dimension: ensure batch dimension exists
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)

    # Get model prediction
    params = model(feats)
    dt_raw = params[:, 0]

    # Add epsilon to ensure positive dt
    dt = dt_raw + EPS

    # Mask dt for inactive particles
    if active_mask is not None:
        # active_mask should be broadcastable to dt
        if dt.dim() == 0 and active_mask.numel() > 1:
            dt = dt.expand_as(active_mask)
        # Ensure mask is float/compatible
        mask = active_mask.to(dt.dtype)
        dt = dt * mask

    # Compute initial energy
    E0 = p.total_energy_batch(G=1.0)
    # Normalize to 1D if scalar (single system case)
    if E0.dim() == 0:
        E0 = E0.unsqueeze(0)

    # Update particle dt
    p.update_dt(dt)

    # Integrate for configured number of steps
    n_steps = max(1, int(config.n_steps))
    for _ in range(n_steps):
        p.evolve_batch(G=1.0)

    # Compute final energy
    E1 = p.total_energy_batch(G=1.0)
    if E1.dim() == 0:
        E1 = E1.unsqueeze(0)

    return p, dt, E0, E1


def check_energy_threshold(
    E0: torch.Tensor,
    E1: torch.Tensor,
    threshold: float,
    active_mask: Optional[torch.Tensor] = None,
) -> Tuple[bool, torch.Tensor]:
    """
    Check if relative energy error is within threshold.

    This function computes the relative energy error between two states and
    checks if it satisfies the acceptance criterion for trajectory collection.

    Args:
        E0: Initial energy (tensor, possibly batched)
        E1: Final energy (tensor, same shape as E0)
        threshold: Relative energy error threshold (e.g., 2e-4)
        active_mask: Optional mask to ignore energy error of inactive particles.

    Returns:
        (passed, rel_dE) where:
        - passed: True if rel_dE < threshold
        - rel_dE: Relative energy error tensor

    Example:
        >>> E0 = torch.tensor([1.0])
        >>> E1 = torch.tensor([1.0001])
        >>> passed, rel_dE = check_energy_threshold(E0, E1, threshold=2e-4)
        >>> print(f"Energy conserved: {passed}, error: {rel_dE.item():.6f}")
    """
    # Safe division: avoid division by zero
    E0_safe = E0 + EPS * E0.detach().abs() + EPS

    # Compute relative error
    rel_dE = torch.abs((E1 - E0) / E0_safe)

    if active_mask is not None:
        mask = active_mask.to(rel_dE.dtype)
        # Zero out error for inactive particles
        rel_dE = rel_dE * mask

    # For single-step collection, compare scalar value
    # Handle both single system and batched cases
    if rel_dE.numel() == 1:
        passed = rel_dE.item() < threshold
    else:
        # For batched case, all must pass
        passed = (rel_dE < threshold).all().item()

    return passed, rel_dE


def compute_single_step_loss(
    E0: torch.Tensor,
    E1: torch.Tensor,
    config: Config,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute loss for a single integration step.
    Uses band loss pattern from existing losses.py.

    This function is used during the retrain loop (Part 1, Phase B) to compute
    the loss for failed trajectory steps. It penalizes energy errors outside
    the acceptable band [E_lower, E_upper].

    Args:
        E0: Initial energy
        E1: Final energy
        config: Config with E_lower, E_upper bounds
        active_mask: Optional mask to compute loss only for active particles.

    Returns:
        Scalar loss tensor (mean over batch if batched)

    Example:
        >>> E0 = torch.tensor([1.0])
        >>> E1 = torch.tensor([1.001])
        >>> loss = compute_single_step_loss(E0, E1, config)
        >>> loss.backward()  # Gradient flows through model parameters
    """
    # Safe division for relative error computation
    E0_safe = E0 + EPS * E0.detach().abs() + EPS
    rel_dE = torch.abs((E1 - E0) / E0_safe)

    # Replace inf/nan with 1.0 penalty (integration exploded)
    rel_dE = torch.where(
        torch.isfinite(rel_dE),
        rel_dE,
        torch.full_like(rel_dE, 1.0)
    )

    # Add epsilon before log to avoid log(0)
    rel_dE_safe = rel_dE + EPS

    # Compute band loss: penalize values outside [E_lower, E_upper] in log space
    loss = band_loss_zero_inside_where(
        torch.log(rel_dE_safe),
        math.log(config.E_lower),
        math.log(config.E_upper)
    )

    if active_mask is not None:
        mask = active_mask.to(loss.dtype)
        loss = loss * mask
        # Mean over active particles only
        active_count = mask.sum()
        if active_count > 0:
            return loss.sum() / active_count
        else:
            return torch.tensor(0.0, device=loss.device)

    # Return mean for batched case
    return loss.mean()


def collect_trajectory_step(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    max_retrain_warn: int = 1000,
    wandb_run: Optional[Any] = None,
    retrain_log_every: int = 100,
    step_idx: Optional[int] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> Tuple[ParticleTorch, float, Dict[str, Any]]:
    """
    Collect one accepted trajectory step with retrain loop.

    Implements TRAIN-01, TRAIN-02, TRAIN-03:
    - Predict dt, integrate one step, check energy
    - If energy exceeds threshold: reject, retrain on same state
    - Loop until energy within threshold
    - Return accepted particle state and dt

    IMPORTANT: No retry limit by design (user requirement).
    For debugging, check metrics['retrain_iterations'].

    Args:
        model: Neural network for dt prediction
        particle: Starting particle state
        optimizer: PyTorch optimizer for model
        config: Config with energy_threshold
        adapter: ModelAdapter for feature construction
        history_buffer: Optional history buffer
        max_retrain_warn: Issue warning every N iterations (default=1000)
        active_mask: Optional mask for multi-orbit stopping

    Returns:
        (accepted_particle, accepted_dt, metrics) where:
        - accepted_particle: ParticleTorch that passed energy check
        - accepted_dt: float value of accepted timestep
        - metrics: dict with 'retrain_iterations', 'reject_count', 'final_energy_error'

    Example:
        >>> p_new, dt, metrics = collect_trajectory_step(
        ...     model, particle, optimizer, config, adapter
        ... )
        >>> print(f"Accepted step with dt={dt:.6f} after {metrics['retrain_iterations']} retrains")
    """
    retrain_iterations = 0
    retrain_loss_sum = 0.0
    retrain_loss_max = 0.0
    retrain_rel_dE_sum = 0.0
    retrain_rel_dE_max = 0.0
    retrain_rel_dE_count = 0
    last_loss_value = 0.0

    wandb = None
    if wandb_run is not None:
        try:
            import wandb as wandb_lib
            wandb = wandb_lib
        except ImportError:
            warnings.warn("wandb_run provided but wandb not installed; logging disabled")
            wandb_run = None

    while True:
        # 1. Attempt a step (creates fresh clone internally)
        p_attempt, dt, E0, E1 = attempt_single_step(
            model, particle, config, adapter, history_buffer, active_mask=active_mask
        )

        # 2. Check energy threshold
        passed, rel_dE = check_energy_threshold(E0, E1, config.energy_threshold, active_mask=active_mask)
        rel_dE_value = rel_dE.item() if rel_dE.numel() == 1 else rel_dE.mean().item()
        retrain_rel_dE_sum += rel_dE_value
        retrain_rel_dE_max = max(retrain_rel_dE_max, rel_dE_value)
        retrain_rel_dE_count += 1

        if passed:
            # ACCEPT: energy within threshold
            metrics = {
                'retrain_iterations': retrain_iterations,
                'reject_count': retrain_iterations,
                'final_energy_error': rel_dE_value,
                'retrain_loss_mean': (
                    retrain_loss_sum / retrain_iterations if retrain_iterations > 0 else 0.0
                ),
                'retrain_loss_max': retrain_loss_max,
                'retrain_rel_dE_mean': (
                    retrain_rel_dE_sum / retrain_rel_dE_count
                    if retrain_rel_dE_count > 0
                    else 0.0
                ),
                'retrain_rel_dE_max': retrain_rel_dE_max,
                'last_retrain_loss': last_loss_value,
            }
            # Extract dt as float for return
            dt_value = dt.item() if dt.numel() == 1 else dt.mean().item()
            if wandb_run is not None and wandb is not None:
                log_data = {
                    'part1/step/retrain_iterations': retrain_iterations,
                    'part1/step/final_energy_error': rel_dE_value,
                    'part1/step/retrain_loss_mean': metrics['retrain_loss_mean'],
                    'part1/step/retrain_loss_max': metrics['retrain_loss_max'],
                    'part1/step/retrain_rel_dE_mean': metrics['retrain_rel_dE_mean'],
                    'part1/step/retrain_rel_dE_max': metrics['retrain_rel_dE_max'],
                    'part1/step/accepted_dt': dt_value,
                }
                if step_idx is not None:
                    log_data['part1/step/step_idx'] = step_idx
                wandb.log(log_data)
            if config.debug:
                print(
                    "accept_step "
                    f"step_idx={step_idx} retrain_iters={retrain_iterations} "
                    f"final_rel_dE={rel_dE_value:.4e} last_loss={last_loss_value:.4e} "
                    f"dt={dt_value:.4e}"
                )
            return p_attempt, dt_value, metrics

        # REJECT: retrain on same state
        loss = compute_single_step_loss(E0, E1, config, active_mask=active_mask)
        loss_value = loss.item()
        last_loss_value = loss_value
        retrain_loss_sum += loss_value
        retrain_loss_max = max(retrain_loss_max, loss_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        retrain_iterations += 1

        if (
            wandb_run is not None
            and wandb is not None
            and retrain_log_every > 0
            and retrain_iterations % retrain_log_every == 0
        ):
            grad_norm_sq = 0.0
            param_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm_sq += param.grad.detach().pow(2).sum().item()
                param_norm_sq += param.detach().pow(2).sum().item()
            grad_norm = math.sqrt(grad_norm_sq)
            param_norm = math.sqrt(param_norm_sq)
            lr = optimizer.param_groups[0].get("lr", 0.0) if optimizer.param_groups else 0.0
            log_data = {
                'part1/retrain/loss': loss_value,
                'part1/retrain/rel_dE': rel_dE_value,
                'part1/retrain/iteration': retrain_iterations,
                'part1/retrain/grad_norm': grad_norm,
                'part1/retrain/param_norm': param_norm,
                'part1/retrain/lr': lr,
            }
            if step_idx is not None:
                log_data['part1/retrain/step_idx'] = step_idx
            wandb.log(log_data)
            if config.debug:
                print(
                    "retrain_log "
                    f"step_idx={step_idx} iter={retrain_iterations} "
                    f"loss={loss_value:.4e} rel_dE={rel_dE_value:.4e} "
                    f"grad_norm={grad_norm:.4e} lr={lr:.4e}"
                )

        # Issue warning for long-running loops
        if retrain_iterations > 0 and retrain_iterations % max_retrain_warn == 0:
            val = rel_dE.item() if rel_dE.numel() == 1 else rel_dE.mean().item()
            warnings.warn(
                f"collect_trajectory_step: {retrain_iterations} retrain iterations "
                f"(rel_dE={val:.2e}, threshold={config.energy_threshold:.2e})",
                RuntimeWarning,
            )
        if config.max_retrain_steps and retrain_iterations >= config.max_retrain_steps:
            val = rel_dE.item() if rel_dE.numel() == 1 else rel_dE.mean().item()
            raise RuntimeError(
                "collect_trajectory_step exceeded max_retrain_steps="
                f"{config.max_retrain_steps} (rel_dE={val:.2e})."
            )


def collect_trajectory(
    model: torch.nn.Module,
    particle: ParticleTorch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    adapter: ModelAdapter,
    history_buffer: Optional[HistoryBuffer] = None,
    wandb_run: Optional[Any] = None,
    retrain_log_every: int = 100,
    include_warmup: bool = False,
    return_final_particle: bool = False,
) -> Union[
    Tuple[List[Tuple[ParticleTorch, float, Optional[torch.Tensor]]], Dict[str, Any]],
    Tuple[List[Tuple[ParticleTorch, float, Optional[torch.Tensor]]], Dict[str, Any], ParticleTorch]
]:
    """
    Collect a trajectory of N validated steps for one epoch.

    Implements TRAIN-04 (steps_per_epoch) and HIST-02 (warmup discard):
    - Collects config.steps_per_epoch accepted steps
    - Always pushes to history buffer (for feature computation)
    - Discards first history_len steps from returned trajectory (warmup)

    The warmup phase (first history_len steps) fills the history buffer
    with meaningful state data. These steps are computed and the buffer
    is populated, but they are not included in the returned trajectory
    since the model's predictions during warmup may be based on incomplete
    history.

    Args:
        model: Neural network for dt prediction
        particle: Starting particle state
        optimizer: PyTorch optimizer for model
        config: Config with steps_per_epoch, history_len, energy_threshold
        adapter: ModelAdapter for feature construction
        history_buffer: Optional history buffer (required if history_enabled)

    Returns:
        (trajectory, epoch_metrics) where:
        - trajectory: List of (ParticleTorch, dt, mask) tuples (warmup excluded by default)
        - epoch_metrics: dict with 'total_steps', 'warmup_discarded',
                         'trajectory_length', 'mean_retrain_iterations',
                         'mean_energy_error', 'max_retrain_iterations'
        - final_particle: last accepted particle if return_final_particle is True

    Example:
        >>> trajectory, metrics = collect_trajectory(
        ...     model, particle, optimizer, config, adapter, history_buffer
        ... )
        >>> print(f"Collected {len(trajectory)} steps (discarded {metrics['warmup_discarded']} warmup)")

    Raises:
        ValueError: If steps_per_epoch < 1
    """
    # Validate steps_per_epoch
    if config.steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be >= 1")

    # Determine warmup length: only applies when history is enabled
    warmup_len = config.history_len if (history_buffer is not None and config.history_len > 0) else 0

    # Warn if all steps will be warmup (empty trajectory)
    if warmup_len >= config.steps_per_epoch:
        warnings.warn(
            f"steps_per_epoch ({config.steps_per_epoch}) <= history_len ({warmup_len}): "
            f"trajectory will be empty (all steps are warmup)",
            UserWarning,
        )

    trajectory: List[Tuple[ParticleTorch, float, Optional[torch.Tensor]]] = []
    all_metrics: List[Dict[str, Any]] = []

    # Current particle state (will be updated each step)
    current_particle = particle

    for step_idx in range(config.steps_per_epoch):
        active_mask = None
        if config.stop_at_period:
            t = current_particle.current_time
            T = current_particle.period
            # Ensure proper shapes for comparison
            # If t is scalar and T is tensor (batched), or vice versa
            if t.dim() == 0 and T.dim() > 0:
                t = t.expand_as(T)
            if T.dim() == 0 and t.dim() > 0:
                T = T.expand_as(t)
            
            # Create boolean mask: active where current_time < period
            active_mask = (t < T).float()
            
            # If all particles have finished their periods, stop collection
            if active_mask.sum() == 0:
                break

        # Collect one accepted step
        accepted_particle, accepted_dt, step_metrics = collect_trajectory_step(
            model=model,
            particle=current_particle,
            optimizer=optimizer,
            config=config,
            adapter=adapter,
            history_buffer=history_buffer,
            wandb_run=wandb_run,
            retrain_log_every=retrain_log_every,
            step_idx=step_idx,
            active_mask=active_mask,
        )

        # Always push to history buffer (needed for next step's features)
        # This happens BEFORE warmup check - we always update the buffer
        if history_buffer is not None:
            # Push detached copy to avoid graph accumulation
            history_buffer.push(accepted_particle.clone_detached())

        # Update current particle for next iteration (detach to prevent graph accumulation)
        current_particle = accepted_particle.clone_detached()

        # Track metrics from all steps (including warmup)
        all_metrics.append(step_metrics)

        # HIST-02: Discard warmup steps (first history_len) unless requested
        if include_warmup or step_idx >= warmup_len:
            trajectory.append((accepted_particle.clone_detached(), accepted_dt, active_mask))

    # Aggregate epoch metrics
    energy_error_list = [m['final_energy_error'] for m in all_metrics]
    epoch_metrics = {
        'total_steps': config.steps_per_epoch,
        'warmup_discarded': warmup_len,
        'trajectory_length': len(trajectory),
        'mean_retrain_iterations': (
            sum(m['retrain_iterations'] for m in all_metrics) / len(all_metrics)
            if all_metrics else 0.0
        ),
        'mean_energy_error': (
            sum(energy_error_list) / len(energy_error_list)
            if all_metrics else 0.0
        ),
        'min_energy_error': min(energy_error_list, default=0.0),
        'max_energy_error': max(energy_error_list, default=0.0),
        'max_retrain_iterations': max((m['retrain_iterations'] for m in all_metrics), default=0),
        'mean_retrain_loss': (
            sum(m['retrain_loss_mean'] for m in all_metrics) / len(all_metrics)
            if all_metrics else 0.0
        ),
        'max_retrain_loss': max((m['retrain_loss_max'] for m in all_metrics), default=0.0),
        'mean_retrain_rel_dE': (
            sum(m['retrain_rel_dE_mean'] for m in all_metrics) / len(all_metrics)
            if all_metrics else 0.0
        ),
        'max_retrain_rel_dE': max((m['retrain_rel_dE_max'] for m in all_metrics), default=0.0),
    }

    if return_final_particle:
        return trajectory, epoch_metrics, current_particle
    return trajectory, epoch_metrics
