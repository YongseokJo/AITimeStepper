import torch


def _normalize_mass(mass: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """
    Broadcast mass to match position.shape[:-1] (i.e., (..., N)).
    Accepts scalar, (N,), or (..., N).
    """
    target_shape = position.shape[:-1]
    if not torch.is_tensor(mass):
        mass = torch.as_tensor(mass, dtype=position.dtype, device=position.device)

    if mass.dim() == 0:
        return mass.expand(target_shape)
    if mass.shape == target_shape:
        return mass
    if mass.dim() == 1 and mass.shape[0] == target_shape[-1]:
        view_shape = (1,) * (len(target_shape) - 1) + (target_shape[-1],)
        return mass.reshape(view_shape).expand(target_shape)
    if mass.dim() < len(target_shape):
        view_shape = (1,) * (len(target_shape) - mass.dim()) + tuple(mass.shape)
        return mass.reshape(view_shape).expand(target_shape)
    if mass.shape[-1] != target_shape[-1]:
        raise ValueError(f"mass shape {mass.shape} incompatible with position shape {position.shape}")
    return mass


def _compute_acceleration(
    position: torch.Tensor,
    mass: torch.Tensor,
    softening: float = 0.0,
    G: float = 1.0,
) -> torch.Tensor:
    """
    Vectorized N-body acceleration for arbitrary leading dims.
    position: (..., N, D)
    mass:     (..., N)
    """
    pos = position
    soft2 = float(softening) ** 2

    pos_i = pos.unsqueeze(-2)
    pos_j = pos.unsqueeze(-3)
    r_ij = pos_j - pos_i  # (..., N, N, D)

    dist2 = (r_ij ** 2).sum(dim=-1) + soft2
    N = pos.shape[-2]
    eye = torch.eye(N, device=pos.device, dtype=torch.bool)
    eye = eye.view((1,) * (dist2.dim() - 2) + (N, N))
    dist2 = dist2.masked_fill(eye, float("inf"))

    inv_r3 = dist2.pow(-1.5)  # (..., N, N)
    m_j = mass.unsqueeze(-2)  # (..., 1, N)
    inv_r3_m = inv_r3 * m_j

    accel = G * (r_ij * inv_r3_m.unsqueeze(-1)).sum(dim=-2)
    return accel


def _stat_features(x: torch.Tensor, dim: int, eps: float = 1e-12):
    mean = x.mean(dim=dim)
    maxv = x.max(dim=dim).values
    minv = x.min(dim=dim).values
    rms = torch.sqrt((x ** 2).mean(dim=dim) + eps)
    return mean, maxv, minv, rms


def _pairwise_distance_stats(
    position: torch.Tensor,
    softening: float = 0.0,
    eps: float = 1e-12,
):
    """
    Return (min, mean, max) pairwise distances for each leading-dim system.
    position: (..., N, D)
    """
    pos = position
    N = pos.shape[-2]
    lead_shape = pos.shape[:-2]

    if N < 2:
        zeros = torch.zeros(lead_shape, device=pos.device, dtype=pos.dtype)
        return zeros, zeros, zeros

    pos_i = pos.unsqueeze(-2)
    pos_j = pos.unsqueeze(-3)
    r_ij = pos_j - pos_i
    dist2 = (r_ij ** 2).sum(dim=-1) + float(softening) ** 2 + eps

    eye = torch.eye(N, device=pos.device, dtype=torch.bool)
    eye = eye.view((1,) * (dist2.dim() - 2) + (N, N))

    dist = torch.sqrt(dist2)
    dist_no_diag = dist.masked_fill(eye, 0.0)
    dist_for_min = dist.masked_fill(eye, float("inf"))

    pair_min = dist_for_min.min(dim=-1).values.min(dim=-1).values
    pair_max = dist_no_diag.max(dim=-1).values.max(dim=-1).values

    pair_count = N * (N - 1)
    pair_sum = dist_no_diag.sum(dim=(-2, -1))
    pair_mean = pair_sum / float(pair_count)
    return pair_min, pair_mean, pair_max


def system_features(
    position: torch.Tensor,
    velocity: torch.Tensor,
    mass,
    softening: float = 0.0,
    G: float = 1.0,
    mode: str = "basic",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build fixed-size, permutation-invariant features for an N-body system.

    Returns shape (..., F), where the leading dims match position.shape[:-2].
    """
    pos = position
    vel = velocity

    if pos.dim() < 2:
        raise ValueError(f"position must have shape (..., N, D), got {pos.shape}")
    if vel.shape != pos.shape:
        raise ValueError(f"velocity shape {vel.shape} must match position shape {pos.shape}")

    N = pos.shape[-2]
    D = pos.shape[-1]

    mass_t = _normalize_mass(mass, pos)

    total_mass = mass_t.sum(dim=-1)
    total_mass_safe = total_mass + eps

    com = (mass_t.unsqueeze(-1) * pos).sum(dim=-2) / total_mass_safe.unsqueeze(-1)
    vcom = (mass_t.unsqueeze(-1) * vel).sum(dim=-2) / total_mass_safe.unsqueeze(-1)

    pos_rel = pos - com.unsqueeze(-2)
    vel_rel = vel - vcom.unsqueeze(-2)

    radius = torch.linalg.norm(pos_rel, dim=-1)
    speed = torch.linalg.norm(vel_rel, dim=-1)

    acc = _compute_acceleration(pos, mass_t, softening=softening, G=G)
    acc_mag = torch.linalg.norm(acc, dim=-1)

    r_mean, r_max, r_min, r_rms = _stat_features(radius, dim=-1, eps=eps)
    v_mean, v_max, v_min, v_rms = _stat_features(speed, dim=-1, eps=eps)
    a_mean, a_max, a_min, a_rms = _stat_features(acc_mag, dim=-1, eps=eps)
    m_mean, m_max, m_min, m_rms = _stat_features(mass_t, dim=-1, eps=eps)

    pair_min, pair_mean, pair_max = _pairwise_distance_stats(pos, softening=softening, eps=eps)

    base_shape = total_mass.shape
    n_val = torch.full(base_shape, float(N), device=pos.device, dtype=pos.dtype)
    d_val = torch.full(base_shape, float(D), device=pos.device, dtype=pos.dtype)
    soft_val = torch.full(base_shape, float(softening), device=pos.device, dtype=pos.dtype)

    if mode == "basic":
        feats = [
            n_val,
            d_val,
            total_mass,
            r_mean,
            r_max,
            v_mean,
            v_max,
            a_mean,
            a_max,
            pair_min,
            pair_mean,
        ]
    elif mode == "rich":
        feats = [
            n_val,
            d_val,
            total_mass,
            m_mean,
            m_min,
            m_max,
            m_rms,
            r_mean,
            r_min,
            r_max,
            r_rms,
            v_mean,
            v_min,
            v_max,
            v_rms,
            a_mean,
            a_min,
            a_max,
            a_rms,
            pair_min,
            pair_mean,
            pair_max,
            soft_val,
        ]
    else:
        raise ValueError(f"Unsupported mode={mode!r}. Use 'basic' or 'rich'.")

    return torch.stack(feats, dim=-1)
