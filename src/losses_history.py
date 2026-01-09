import math
from typing import List

import torch
import torch.nn.functional as F

from .particle import ParticleTorch
from .history_buffer import HistoryBuffer
from .losses import band_loss_zero_inside_where


def loss_fn_batch_history(
    model,
    particle: ParticleTorch,
    history: HistoryBuffer,
    n_steps: int = 1,
    rel_loss_bound: float = 1e-3,
    dt_bound: float = 1e-3,  # not used if you keep your dt reg variant
    lambda_dt: float = 1.0,
    lambda_pred: float = 1.0,
    E_lower: float = 1e-8,
    E_upper: float = 1e-4,
    L_lower: float = 1e-8,
    L_upper: float = 1e-4,
    return_particle: bool = False,
):
    """
    History-aware forward pass + physics loss.

    Model input is the time-concatenated feature vector provided by
    `history.features_for(p)` where p is the current ParticleTorch.
    Model is expected to output [dt_raw, E_hat_raw] per sample.
    """
    # 0) fresh copy
    p = particle.clone_detached()

    # 1) build history features (B, F_total) or (F_total,)
    batch = history.features_for(p)
    if batch.dim() == 1:
        batch = batch.unsqueeze(0)  # (1, F_total)

    # 2) model predict raw dt and raw E_hat
    params = model(batch)  # (B, 2)
    if params.dim() == 1:
        params = params.unsqueeze(0)

    dt_raw = params[:, 0]
    E_hat_raw = F.softplus(params[:, 1])

    eps = 1e-12
    dt = dt_raw + eps
    E_hat = E_hat_raw + eps

    # 3) initial energy (B,)
    E0 = p.total_energy_batch(G=1.0)
    if E0.dim() == 0:
        E0 = E0.unsqueeze(0)

    # 4) evolve with predicted dt for n_steps
    p.update_dt(dt)

    def _angular_momentum_mag(particle_state: ParticleTorch) -> torch.Tensor:
        pos = particle_state.position
        vel = particle_state.velocity
        m = particle_state.mass

        # normalize to (B, N, D)
        if pos.dim() == 2:
            pos_b = pos.unsqueeze(0)
            vel_b = vel.unsqueeze(0)
        else:
            pos_b = pos
            vel_b = vel

        B, N, D = pos_b.shape

        m_t = torch.as_tensor(m, device=pos_b.device, dtype=pos_b.dtype)
        if m_t.dim() == 0:
            m_b = m_t.expand(B, N)
        elif m_t.dim() == 1:
            m_b = m_t.unsqueeze(0)
        else:
            m_b = m_t

        if D == 2:
            Lz = (m_b * (pos_b[..., 0] * vel_b[..., 1] - pos_b[..., 1] * vel_b[..., 0])).sum(dim=1)
            L_mag = torch.abs(Lz)
        else:
            Lvec = (m_b.unsqueeze(-1) * torch.cross(pos_b, vel_b, dim=-1)).sum(dim=1)
            L_mag = torch.linalg.norm(Lvec, dim=-1)

        return L_mag.squeeze(0) if pos.dim() == 2 else L_mag

    E_list = [E0]
    L_mag_list = [_angular_momentum_mag(p)]
    for _ in range(n_steps):
        p.evolve_batch(G=1.0)
        Ei = p.total_energy_batch(G=1.0)
        if Ei.dim() == 0:
            Ei = Ei.unsqueeze(0)
        E_list.append(Ei)
        L_mag_list.append(_angular_momentum_mag(p))

    E_all = torch.stack(E_list, dim=0)  # (n_steps+1, B)
    E_mean = E_all.mean(dim=0)
    E1 = E_all[1]
    E_last = E_all[-1]

    E0_safe = E0 + eps * E0.detach().abs() + eps
    rel_dE_mean = torch.abs((E_mean - E0) / E0_safe)
    rel_dE_next = torch.abs((E1 - E0) / E0_safe)
    rel_dE_last = torch.abs((E_last - E0) / E0_safe)
    rel_dE_max = torch.abs((E_all - E0) / E0_safe).max(dim=0).values
    rel_dE = rel_dE_mean

    # --- Angular momentum (magnitude) and its relative change ---
    L_mag_all = torch.stack(
        [x if torch.is_tensor(x) else torch.as_tensor(x, device=E0.device, dtype=E0.dtype) for x in L_mag_list],
        dim=0,
    )
    if L_mag_all.dim() == 1:
        L_mag_all = L_mag_all.unsqueeze(1)

    L0_mag = L_mag_all[0]
    L_mean = L_mag_all.mean(dim=0)
    L1 = L_mag_all[1]
    L_last = L_mag_all[-1]

    L0_safe = L0_mag + eps * L0_mag.detach().abs() + eps
    rel_dL_mean = torch.abs((L_mean - L0_mag) / L0_safe)
    rel_dL_next = torch.abs((L1 - L0_mag) / L0_safe)
    rel_dL_last = torch.abs((L_last - L0_mag) / L0_safe)
    rel_dL_max = torch.abs((L_mag_all - L0_mag) / L0_safe).max(dim=0).values

    rel_dL_mean = torch.where(torch.isfinite(rel_dL_mean), rel_dL_mean, torch.full_like(rel_dL_mean, 1.0))
    rel_dL_next = torch.where(torch.isfinite(rel_dL_next), rel_dL_next, torch.full_like(rel_dL_next, 1.0))
    rel_dL_last = torch.where(torch.isfinite(rel_dL_last), rel_dL_last, torch.full_like(rel_dL_last, 1.0))
    rel_dL_max = torch.where(torch.isfinite(rel_dL_max), rel_dL_max, torch.full_like(rel_dL_max, 1.0))

    rel_dE = torch.where(torch.isfinite(rel_dE), rel_dE, torch.full_like(rel_dE, 1.0))
    rel_dE_safe = rel_dE + eps
    E_hat_safe = E_hat.clamp(min=eps)

    # --- Angular momentum for batched state ---
    def _compute_angular_momentum_batch(particle: ParticleTorch):
        pos = particle.position
        vel = particle.velocity
        m = particle.mass

        # normalize to batched shapes: (B, N, D)
        if pos.dim() == 2:
            pos_b = pos.unsqueeze(0)
            vel_b = vel.unsqueeze(0)
        else:
            pos_b = pos
            vel_b = vel

        if torch.is_tensor(m) and m.dim() == 1:
            m_b = m.unsqueeze(0)
        else:
            m_b = m

        B, N, D = pos_b.shape
        if D == 2:
            Lz = (m_b * (pos_b[..., 0] * vel_b[..., 1] - pos_b[..., 1] * vel_b[..., 0])).sum(dim=1)
            return Lz
        else:
            Lvec = (m_b.unsqueeze(-1) * torch.cross(pos_b, vel_b, dim=-1)).sum(dim=1)
            return Lvec

    # initial and (approximate) per-step L magnitudes
    L0 = _compute_angular_momentum_batch(p)
    if not torch.is_tensor(L0):
        L0 = torch.as_tensor(L0)
    if L0.dim() == 0:
        L0 = L0.unsqueeze(0)

    # We didn't store per-step particle states; approximate by using final state's L for steps
    L_list = [L0]
    for _ in range(1, E_all.shape[0]):
        L_list.append(_compute_angular_momentum_batch(p))

    L_all = torch.stack(L_list, dim=0)
    if L_all.dim() == 3:
        L_mag_all = torch.linalg.norm(L_all, dim=-1)
    else:
        L_mag_all = torch.abs(L_all)

    L_mean = L_mag_all.mean(dim=0)
    L1 = L_mag_all[1]
    L_last = L_mag_all[-1]

    L0_safe = L0 + eps * L0.detach().abs() + eps
    rel_dL_mean = torch.abs((L_mean - L0) / L0_safe)
    rel_dL_next = torch.abs((L1 - L0) / L0_safe)
    rel_dL_last = torch.abs((L_last - L0) / L0_safe)
    rel_dL_max = torch.abs((L_mag_all - L0) / L0_safe).max(dim=0).values


    
    # 6) loss terms
    loss_energy_mean = band_loss_zero_inside_where(torch.log(rel_dE_mean), math.log(E_lower), math.log(E_upper))
    loss_energy_last = band_loss_zero_inside_where(torch.log(rel_dE_last), math.log(E_lower), math.log(E_upper))
    loss_energy_next = band_loss_zero_inside_where(torch.log(rel_dE_next), math.log(E_lower), math.log(E_upper))
    loss_energy_max = band_loss_zero_inside_where(torch.log(rel_dE_max), math.log(E_lower), math.log(E_upper))
    loss_energy = loss_energy_mean.mean() + loss_energy_last.mean() + loss_energy_max.mean() + loss_energy_next.mean()
    # angular momentum loss terms (separate bounds)
    loss_ang_mean = band_loss_zero_inside_where(torch.log(rel_dL_mean.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang_last = band_loss_zero_inside_where(torch.log(rel_dL_last.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang_next = band_loss_zero_inside_where(torch.log(rel_dL_next.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang_max = band_loss_zero_inside_where(torch.log(rel_dL_max.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang = loss_ang_mean.mean() + loss_ang_last.mean() + loss_ang_max.mean() + loss_ang_next.mean()

    loss_dt = dt  # keep same as your batch loss variant
    loss_pred = (torch.log(rel_dE_safe) - torch.log(E_hat_safe)) ** 2

    # combine energy and angular momentum losses (equal weighting)
    loss = loss_energy + loss_ang

    metrics = {
        "rel_dE": rel_dE.mean(),
        "dt": dt.mean(),
        "E0": E0.mean(),
        "loss_energy": loss_energy.mean(),
        "loss_energy_mean": loss_energy_mean.mean(),
        "loss_energy_last": loss_energy_last.mean(),
        "loss_energy_next": loss_energy_next.mean(),
        "loss_energy_max": loss_energy_max.mean(),
        "rel_dE_mean": rel_dE_mean.mean(),
        "rel_dE_next": rel_dE_next.mean(),
        "rel_dE_last": rel_dE_last.mean(),
        "rel_dE_max": rel_dE_max.mean(),
        "loss_pred": loss_pred.mean(),
        "loss_dt": loss_dt.mean(),
        "loss_ang": loss_ang.mean(),
        "rel_dL_mean": rel_dL_mean.mean(),
        "rel_dL_next": rel_dL_next.mean(),
        "rel_dL_last": rel_dL_last.mean(),
        "rel_dL_max": rel_dL_max.mean(),
        "rel_dE_full": rel_dE,
    }

    if return_particle:
        return loss, metrics, p
    else:
        return loss, metrics, _


def loss_fn_batch_history_batch(
    model,
    batch_state: ParticleTorch,
    histories: "List[HistoryBuffer]",
    n_steps: int = 1,
    rel_loss_bound: float = 1e-3,
    dt_bound: float = 1e-3,
    lambda_dt: float = 1.0,
    lambda_pred: float = 1.0,
    E_lower: float = 1e-8,
    E_upper: float = 1e-4,
    L_lower: float = 1e-8,
    L_upper: float = 1e-4,
    return_particle: bool = False,
):
    """
    Batched, history-aware loss. Builds inputs per sample from corresponding
    HistoryBuffer and computes physics losses over the batch.
    """
    import torch
    import math
    import torch.nn.functional as F

    p = batch_state.clone_detached()

    # Build per-sample concatenated features → (B, F_total)
    # histories list length should match batch size
    B = p.position.shape[0] if p.position.dim() == 3 else 1
    assert len(histories) == B, "histories length must match batch size"

    feats = HistoryBuffer.features_for_histories(histories, p)
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)

    params = model(feats)  # (B, 2)
    if params.dim() == 1:
        params = params.unsqueeze(0)

    dt_raw = params[:, 0]
    E_hat_raw = F.softplus(params[:, 1])

    eps = 1e-12
    dt = dt_raw + eps
    E_hat = E_hat_raw + eps

    E0 = p.total_energy_batch(G=1.0)
    if E0.dim() == 0:
        E0 = E0.unsqueeze(0)

    p.update_dt(dt)

    def _angular_momentum_mag(particle_state: ParticleTorch) -> torch.Tensor:
        pos = particle_state.position
        vel = particle_state.velocity
        m = particle_state.mass

        if pos.dim() == 2:
            pos_b = pos.unsqueeze(0)
            vel_b = vel.unsqueeze(0)
        else:
            pos_b = pos
            vel_b = vel

        B, N, D = pos_b.shape

        m_t = torch.as_tensor(m, device=pos_b.device, dtype=pos_b.dtype)
        if m_t.dim() == 0:
            m_b = m_t.expand(B, N)
        elif m_t.dim() == 1:
            m_b = m_t.unsqueeze(0)
        else:
            m_b = m_t

        if D == 2:
            Lz = (m_b * (pos_b[..., 0] * vel_b[..., 1] - pos_b[..., 1] * vel_b[..., 0])).sum(dim=1)
            L_mag = torch.abs(Lz)
        else:
            Lvec = (m_b.unsqueeze(-1) * torch.cross(pos_b, vel_b, dim=-1)).sum(dim=1)
            L_mag = torch.linalg.norm(Lvec, dim=-1)

        return L_mag.squeeze(0) if pos.dim() == 2 else L_mag

    E_list = [E0]
    L_mag_list = [_angular_momentum_mag(p)]
    for _ in range(n_steps):
        p.evolve_batch(G=1.0)
        Ei = p.total_energy_batch(G=1.0)
        if Ei.dim() == 0:
            Ei = Ei.unsqueeze(0)
        E_list.append(Ei)
        L_mag_list.append(_angular_momentum_mag(p))

    E_all = torch.stack(E_list, dim=0)  # (n_steps+1, B)
    E_mean = E_all.mean(dim=0)
    E1 = E_all[1]
    E_last = E_all[-1]

    E0_safe = E0 + eps * E0.detach().abs() + eps
    rel_dE_mean = torch.abs((E_mean - E0) / E0_safe)
    rel_dE_next = torch.abs((E1 - E0) / E0_safe)
    rel_dE_last = torch.abs((E_last - E0) / E0_safe)
    rel_dE_max = torch.abs((E_all - E0) / E0_safe).max(dim=0).values
    rel_dE = rel_dE_mean

    rel_dE = torch.where(torch.isfinite(rel_dE), rel_dE, torch.full_like(rel_dE, 1.0))
    rel_dE_safe = rel_dE + eps
    E_hat_safe = E_hat.clamp(min=eps)

    L_mag_all = torch.stack(
        [x if torch.is_tensor(x) else torch.as_tensor(x, device=E0.device, dtype=E0.dtype) for x in L_mag_list],
        dim=0,
    )
    if L_mag_all.dim() == 1:
        L_mag_all = L_mag_all.unsqueeze(1)

    L0_mag = L_mag_all[0]
    L_mean = L_mag_all.mean(dim=0)
    L1 = L_mag_all[1]
    L_last = L_mag_all[-1]

    L0_safe = L0_mag + eps * L0_mag.detach().abs() + eps
    rel_dL_mean = torch.abs((L_mean - L0_mag) / L0_safe)
    rel_dL_next = torch.abs((L1 - L0_mag) / L0_safe)
    rel_dL_last = torch.abs((L_last - L0_mag) / L0_safe)
    rel_dL_max = torch.abs((L_mag_all - L0_mag) / L0_safe).max(dim=0).values

    rel_dL_mean = torch.where(torch.isfinite(rel_dL_mean), rel_dL_mean, torch.full_like(rel_dL_mean, 1.0))
    rel_dL_next = torch.where(torch.isfinite(rel_dL_next), rel_dL_next, torch.full_like(rel_dL_next, 1.0))
    rel_dL_last = torch.where(torch.isfinite(rel_dL_last), rel_dL_last, torch.full_like(rel_dL_last, 1.0))
    rel_dL_max = torch.where(torch.isfinite(rel_dL_max), rel_dL_max, torch.full_like(rel_dL_max, 1.0))

    loss_energy_mean = band_loss_zero_inside_where(torch.log(rel_dE_mean), math.log(E_lower), math.log(E_upper))
    loss_energy_last = band_loss_zero_inside_where(torch.log(rel_dE_last), math.log(E_lower), math.log(E_upper))
    loss_energy_next = band_loss_zero_inside_where(torch.log(rel_dE_next), math.log(E_lower), math.log(E_upper))
    loss_energy_max = band_loss_zero_inside_where(torch.log(rel_dE_max), math.log(E_lower), math.log(E_upper))
    loss_energy = loss_energy_mean.mean() + loss_energy_last.mean() + loss_energy_max.mean() + loss_energy_next.mean()
    # angular momentum loss terms (separate bounds)
    loss_ang_mean = band_loss_zero_inside_where(torch.log(rel_dL_mean.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang_last = band_loss_zero_inside_where(torch.log(rel_dL_last.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang_next = band_loss_zero_inside_where(torch.log(rel_dL_next.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang_max = band_loss_zero_inside_where(torch.log(rel_dL_max.clamp(min=eps)), math.log(L_lower), math.log(L_upper))
    loss_ang = loss_ang_mean.mean() + loss_ang_last.mean() + loss_ang_max.mean() + loss_ang_next.mean()

    loss_dt = dt
    loss_pred = (torch.log(rel_dE_safe) - torch.log(E_hat_safe)) ** 2

    loss = loss_energy + loss_ang

    metrics = {
        "rel_dE": rel_dE.mean(),
        "dt": dt.mean(),
        "E0": E0.mean(),
        "loss_energy": loss_energy.mean(),
        "loss_energy_mean": loss_energy_mean.mean(),
        "loss_energy_last": loss_energy_last.mean(),
        "loss_energy_next": loss_energy_next.mean(),
        "loss_energy_max": loss_energy_max.mean(),
        "rel_dE_mean": rel_dE_mean.mean(),
        "rel_dE_next": rel_dE_next.mean(),
        "rel_dE_last": rel_dE_last.mean(),
        "rel_dE_max": rel_dE_max.mean(),
        "loss_pred": loss_pred.mean(),
        "loss_dt": loss_dt.mean(),
        "loss_ang": loss_ang.mean(),
        "rel_dL_mean": rel_dL_mean.mean(),
        "rel_dL_next": rel_dL_next.mean(),
        "rel_dL_last": rel_dL_last.mean(),
        "rel_dL_max": rel_dL_max.mean(),
        "rel_dE_full": rel_dE,
    }

    if return_particle:
        return loss, metrics, p
    else:
        return loss, metrics, _
