from __future__ import annotations

from typing import Dict, Optional

import torch

from .nbody_features import _compute_acceleration, _pairwise_distance_stats, _normalize_mass
import warnings


def compute_norm_scales(
    position: torch.Tensor,
    velocity: torch.Tensor,
    mass: torch.Tensor,
    *,
    softening: float = 0.0,
    G: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute characteristic scales for input normalization.

    Returns dict with L0, V0, A0, T0.
    """
    if position.dim() < 2:
        raise ValueError(f"position must have shape (..., N, D), got {position.shape}")
    if velocity.shape != position.shape:
        raise ValueError(f"velocity shape {velocity.shape} must match position shape {position.shape}")

    mass_t = _normalize_mass(mass, position)

    total_mass = mass_t.sum(dim=-1) + eps
    com = (mass_t.unsqueeze(-1) * position).sum(dim=-2) / total_mass.unsqueeze(-1)
    vcom = (mass_t.unsqueeze(-1) * velocity).sum(dim=-2) / total_mass.unsqueeze(-1)
    pos_rel = position - com.unsqueeze(-2)
    vel_rel = velocity - vcom.unsqueeze(-2)

    _, pair_mean, _ = _pairwise_distance_stats(position, softening=softening, eps=eps)
    radius_rms = torch.sqrt((pos_rel ** 2).sum(dim=-1).mean(dim=-1) + eps)
    speed_rms = torch.sqrt((vel_rel ** 2).sum(dim=-1).mean(dim=-1) + eps)

    if torch.is_tensor(pair_mean):
        pair_mean_val = pair_mean.mean().item()
    else:
        pair_mean_val = float(pair_mean)

    L0 = pair_mean_val if pair_mean_val > eps else float(radius_rms.mean().item())
    V0 = float(speed_rms.mean().item())

    acc = _compute_acceleration(position, mass_t, softening=softening, G=G)
    acc_rms = torch.sqrt((acc ** 2).sum(dim=-1).mean(dim=-1) + eps)
    A0 = float(acc_rms.mean().item())

    def _sanitize(val: float, name: str) -> float:
        if not torch.isfinite(torch.tensor(val)):
            warnings.warn(f"Normalization scale {name} is non-finite ({val}); using 1.0", RuntimeWarning)
            return 1.0
        if val <= 0:
            warnings.warn(f"Normalization scale {name} <= 0 ({val}); using 1.0", RuntimeWarning)
            return 1.0
        return float(val)

    L0 = _sanitize(L0, "L0")
    V0 = _sanitize(V0, "V0")
    A0 = _sanitize(A0, "A0")

    T0 = L0 / V0 if V0 > eps else 1.0
    T0 = _sanitize(T0, "T0")

    return {
        "L0": float(L0),
        "V0": float(V0),
        "A0": float(A0),
        "T0": float(T0),
    }


def derive_norm_scales_from_config(
    config,
    *,
    particle: Optional["ParticleTorch"] = None,
    eps: float = 1e-12,
) -> Optional[Dict[str, float]]:
    if not getattr(config, "normalize_inputs", False):
        return None

    if getattr(config, "norm_mode", "auto") == "manual":
        L0 = float(config.norm_L0) if config.norm_L0 is not None else 1.0
        V0 = float(config.norm_V0) if config.norm_V0 is not None else 1.0
        if L0 <= 0:
            L0 = 1.0
        if V0 <= 0:
            V0 = 1.0
        A0 = float(config.norm_A0) if config.norm_A0 is not None else (V0 ** 2) / max(L0, eps)
        T0 = float(config.norm_T0) if config.norm_T0 is not None else (L0 / max(V0, eps))
        return {"L0": L0, "V0": V0, "A0": A0, "T0": T0}

    if particle is None:
        raise ValueError("auto normalization requires a ParticleTorch")

    return compute_norm_scales(
        position=particle.position,
        velocity=particle.velocity,
        mass=particle.mass,
        softening=getattr(particle, "softening", 0.0),
        G=1.0,
    )


def apply_norm_scales_to_config(config, scales: Optional[Dict[str, float]]) -> None:
    if not scales:
        return
    config.norm_L0 = float(scales.get("L0", config.norm_L0 or 1.0))
    config.norm_V0 = float(scales.get("V0", config.norm_V0 or 1.0))
    config.norm_A0 = float(scales.get("A0", config.norm_A0 or 1.0))
    config.norm_T0 = float(scales.get("T0", config.norm_T0 or 1.0))
