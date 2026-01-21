"""
Trajectory collection primitives for two-phase training.

This module provides the core accept/reject loop components:
- attempt_single_step: Predict dt, integrate, return state and energies
- check_energy_threshold: Verify energy conservation
- compute_single_step_loss: Loss for retrain loop

Phase 3 of AITimeStepper training refactor.
"""

import math
from typing import Optional, Tuple

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

    # Compute initial energy
    E0 = p.total_energy_batch(G=1.0)
    # Normalize to 1D if scalar (single system case)
    if E0.dim() == 0:
        E0 = E0.unsqueeze(0)

    # Update particle dt
    p.update_dt(dt)

    # Integrate one step
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
) -> Tuple[bool, torch.Tensor]:
    """
    Check if relative energy error is within threshold.

    This function computes the relative energy error between two states and
    checks if it satisfies the acceptance criterion for trajectory collection.

    Args:
        E0: Initial energy (tensor, possibly batched)
        E1: Final energy (tensor, same shape as E0)
        threshold: Relative energy error threshold (e.g., 2e-4)

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

    # Return mean for batched case
    return loss.mean()
