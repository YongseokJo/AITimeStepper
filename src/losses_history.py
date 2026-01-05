import math
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

    E_list = [E0]
    for _ in range(n_steps):
        p.acceleration = p.get_acceleration(G=1.0)
        p.evolve_batch(G=1.0)
        Ei = p.total_energy_batch(G=1.0)
        if Ei.dim() == 0:
            Ei = Ei.unsqueeze(0)
        E_list.append(Ei)

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

    # 6) loss terms
    loss_energy_mean = band_loss_zero_inside_where(torch.log(rel_dE_mean), math.log(E_lower), math.log(E_upper))
    loss_energy_last = band_loss_zero_inside_where(torch.log(rel_dE_last), math.log(E_lower), math.log(E_upper))
    loss_energy_next = band_loss_zero_inside_where(torch.log(rel_dE_next), math.log(E_lower), math.log(E_upper))
    loss_energy_max = band_loss_zero_inside_where(torch.log(rel_dE_max), math.log(E_lower), math.log(E_upper))
    loss_energy = loss_energy_mean.mean() + loss_energy_last.mean() + loss_energy_max.mean() + loss_energy_next.mean()

    loss_dt = dt  # keep same as your batch loss variant
    loss_pred = (torch.log(rel_dE_safe) - torch.log(E_hat_safe)) ** 2

    loss = loss_energy

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

    # Use first history instance to call features_for_batch (it uses self for config)
    feats = histories[0].features_for_batch(p)
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

    E_list = [E0]
    for _ in range(n_steps):
        p.acceleration = p.get_acceleration(G=1.0)
        p.evolve_batch(G=1.0)
        Ei = p.total_energy_batch(G=1.0)
        if Ei.dim() == 0:
            Ei = Ei.unsqueeze(0)
        E_list.append(Ei)

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

    loss_energy_mean = band_loss_zero_inside_where(torch.log(rel_dE_mean), math.log(E_lower), math.log(E_upper))
    loss_energy_last = band_loss_zero_inside_where(torch.log(rel_dE_last), math.log(E_lower), math.log(E_upper))
    loss_energy_next = band_loss_zero_inside_where(torch.log(rel_dE_next), math.log(E_lower), math.log(E_upper))
    loss_energy_max = band_loss_zero_inside_where(torch.log(rel_dE_max), math.log(E_lower), math.log(E_upper))
    loss_energy = loss_energy_mean.mean() + loss_energy_last.mean() + loss_energy_max.mean() + loss_energy_next.mean()

    loss_dt = dt
    loss_pred = (torch.log(rel_dE_safe) - torch.log(E_hat_safe)) ** 2

    loss = loss_energy

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
        "rel_dE_full": rel_dE,
    }

    if return_particle:
        return loss, metrics, p
    else:
        return loss, metrics, _
