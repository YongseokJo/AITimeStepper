import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .particle import ParticleTorch
torch.autograd.set_detect_anomaly(True)



def loss_fn(model, particle: ParticleTorch):
    batch = particle.get_batch()
    params = model(batch)  # (batch, 2)
    print("params: ", params, "params shape: ", params.shape)
    dt = params[0]
    E  = params[1]

    E0 = particle.total_energy(G=1.0)
    particle.update_dt(dt)
    particle.get_acceleration()
    particle.evolve(G=1.0)  # how many periods?
    E1 = particle.total_energy(G=1.0)
    print("E0: ", E0, "E1: ", E1, "dt: ", dt, "E: ", E)

    # I have to get both instant energy loss and final energy loss.
    #while (particle.current_time < particle.period):
    #    particle.get_acceleration()
    #    particle.evolve(G=1.0)
    
    # Reuse same object, but swap in new tensors
    # particle.reset_state(pos0, vel0, dt=1e-3)
    loss = ((E1 - E0)/E0 - 1e-3)**2 - dt**2 + ((E1 - E0)/E0 - E)**2

    return loss


def loss_fn_1(model, particle: ParticleTorch, n_steps: int = 1,
              rel_loss_bound=1e-3, dt_bound=1e-3,
              lambda_dt=1.0, lambda_pred=1.0,
              E_lower=1e-8, E_upper=1e-4,
              return_particle: bool = False):
    """
    One forward pass + loss.

    model: maps batch -> [dt_raw, E_hat_raw]
    particle: a ParticleTorch object with some initial state
    n_steps: how many integration steps to evolve with the predicted dt.
    """

    # --- 0. Work on a fresh copy of the particle state ---
    p = particle.clone_detached()


    #print("pre-pos: ", p.position)
    #print("pre-vel", p.position, p.velocity)

    # --- 1. Build model input ---
    #batch = p.get_batch()          # shape (2,) for a single system
    batch = p._get_batch_()
    batch = batch.unsqueeze(0)     # (1, 2) -> batch dimension

    #print(batch.shape)

    # --- 2. Model predicts raw dt and raw E_hat ---
    params = model(batch)          # (1, 2)
    params = params.squeeze(0)     # (2,)

    dt_raw  = params[0]
    E_hat_raw = F.softplus(params[1])
    #print(E_hat_raw)

    #print(batch, dt_raw)
    # ----------------------------------------------------
    # 2a. Constrain dt and E_hat to be positive & well-behaved
    # ----------------------------------------------------
    eps = 1e-12
    #eps = 1e-6

    dt = dt_raw + eps
    E_hat = E_hat_raw + eps 

    # --- 3. Initial energy ---
    E0 = p.total_energy(G=1.0)

    # --- 4. Use predicted dt, evolve for n_steps ---
    p.update_dt(dt)


    #print("pre-pos: ", p.position)
    #print("pre-vel", p.position, p.velocity)
    for _ in range(n_steps):
        p.acceleration = p.get_acceleration(G=1.0)
        p.evolve(G=1.0)

    #print("post-pos: ", p.position)
    #print("post-vel: ", p.velocity)
    E1 = p.total_energy(G=1.0)

    # --- 5. Relative energy change with guards ---
    # avoid division by 0: add small epsilon relative to |E0|
    E0_safe = E0 + eps * E0.detach().abs() + eps
    rel_dE = torch.abs((E1 - E0) / E0_safe)
    #print(E0, E1, rel_dE, dt)

    # If integrator exploded and we got inf/nan, replace by a large penalty value
    rel_dE = torch.where(
        torch.isfinite(rel_dE),
        rel_dE,
        torch.full_like(rel_dE, 1.0)
    )

    # clamp before log to avoid log(0)
    #rel_dE_safe = rel_dE.clamp(min=eps)
    rel_dE_safe = rel_dE + eps
    E_hat_safe  = E_hat.clamp(min=eps)

    # --- 6. Loss components ---
    # (a) We want rel_dE close to some small rel_loss_bound (both > 0)
    target_log_rel = math.log(rel_loss_bound)
    #loss_energy = (torch.log(rel_dE_safe) - target_log_rel) ** 2
    loss_energy = band_loss_zero_inside_where(torch.log(rel_dE_safe), math.log(E_lower), math.log(E_upper))
    #print("rel_dE_safe:", torch.log(rel_dE_safe).item(), "target_log_rel:", target_log_rel)

    # (b) We want dt near dt_bound (in log space)
    #target_log_dt = math.log(dt_bound)
    #loss_dt = (torch.log(dt) - target_log_dt) ** 2
    loss_dt = -10*dt**2
    #loss_dt = - dt
    
    # (c) We want model’s E_hat to approximate the actual rel_dE
    loss_pred = (torch.log(rel_dE_safe) - torch.log(E_hat_safe)) ** 2

    loss = loss_energy #+ lambda_dt * loss_dt #+ lambda_pred * loss_pred

    metrics = {
        "rel_dE": rel_dE,
        "dt": dt,
        "E0": E0,
        "loss_energy": loss_energy,
        "loss_pred": loss_pred,
        "loss_dt": loss_dt,
    }
    if return_particle:
        return loss, metrics, p  # p is the *advanced* particle
    else:
        return loss, metrics



def loss_fn_batch(
    model,
    particle: "ParticleTorch",   # can be single or batched
    n_steps: int = 1,
    rel_loss_bound=1e-3,
    dt_bound=1e-3,               # not used if you keep the -10*dt**2 version
    lambda_dt=1.0,
    lambda_pred=1.0,
    E_lower=1e-8,
    E_upper=1e-4, 
    return_particle: bool = False,
):
    """
    One forward pass + loss.

    model: maps batch -> [dt_raw, E_hat_raw]
            If batch has shape (B, D), model(batch) has shape (B, 2).

    particle: ParticleTorch object that may represent:
        - a single system: position shape (2,)
        - a batch of systems: position shape (B, 2)

    n_steps: how many integration steps to evolve with the predicted dt.
    """

    # --- 0. Work on a fresh copy of the particle state ---
    p = particle.clone_detached()

    # --- 1. Build model input ---
    # p.get_batch() should return:
    #   (2,) for a single system  -> we add batch dim
    #   (B, 2) for batched input  -> we keep as is
    #batch = p._get_batch()
    batch = p._get_batch_()
    if batch.dim() == 1:
        # (2,) -> (1, 2)
        batch = batch.unsqueeze(0)

    # --- 2. Model predicts raw dt and raw E_hat ---
    params = model(batch)  # (B, 2) or (1, 2)
    if params.dim() == 1:
        # (2,) -> (1, 2) for safety
        params = params.unsqueeze(0)

    # params: (B, 2)
    dt_raw     = params[:, 0]           # (B,)
    E_hat_raw  = F.softplus(params[:, 1])  # (B,)

    # ----------------------------------------------------
    # 2a. Constrain dt and E_hat to be positive & well-behaved
    # ----------------------------------------------------
    eps = 1e-12

    dt    = dt_raw  + eps              # (B,)
    E_hat = E_hat_raw + eps            # (B,)

    # --- 3. Initial energy ---
    # Expecting total_energy to return scalar for single system,
    # or shape (B,) for batched systems.
    E0 = p.total_energy_batch(G=1.0)
    if E0.dim() == 0:
        E0 = E0.unsqueeze(0)           # make it (1,) for consistency

    # --- 4. Use predicted dt, evolve for n_steps ---
    # ParticleTorch.update_dt should support dt with shape (B,) or (1,)
    p.update_dt(dt)

    E_list = [E0]   # store energies

    for _ in range(n_steps):
        p.acceleration = p.get_acceleration(G=1.0)
        p.evolve_batch(G=1.0)

        Ei = p.total_energy_batch(G=1.0)
        if Ei.dim() == 0:
            Ei = Ei.unsqueeze(0)
        E_list.append(Ei)

    # now E_list is length n_steps+1, each element shape (B,)
    E_all = torch.stack(E_list, dim=0)  # (n_steps+1, B)

    # use mean energy difference (can choose max or last)
    E_mean = E_all.mean(dim=0)  # (B,)
    E1 = E_all[1]              # (B,)
    E_last = E_all[-1]              # (B,)

    # Relative energy error using mean:
    E0_safe = E0 + eps * E0.detach().abs() + eps
    rel_dE_mean = torch.abs((E_mean - E0) / E0_safe)  # (B,)
    rel_dE_next = torch.abs((E1 - E0) / E0_safe)      # (B,)
    rel_dE_last = torch.abs((E_last - E0) / E0_safe)      # (B,)
    rel_dE_max = torch.abs((E_all - E0) / E0_safe).max(dim=0).values

    rel_dE = rel_dE_mean  # choose which one to use


    # Replace inf/nan by a large penalty
    rel_dE = torch.where(
        torch.isfinite(rel_dE),
        rel_dE,
        torch.full_like(rel_dE, 1.0)
    )

    rel_dE_safe = rel_dE + eps               # (B,)
    E_hat_safe  = E_hat.clamp(min=eps)       # (B,)

    # --- 6. Loss components (vectorized over batch) ---
    # (a) We want rel_dE close to rel_loss_bound (both > 0)
    target_log_rel = math.log(rel_loss_bound)
    #loss_energy = (torch.log(rel_dE_safe) - target_log_rel) ** 2   # (B,)
    #loss_energy = band_loss_zero_inside_where(torch.log(rel_dE_safe), math.log(E_lower), math.log(E_upper))
    loss_energy_mean = band_loss_zero_inside_where(torch.log(rel_dE_mean), math.log(E_lower), math.log(E_upper))
    loss_energy_last = band_loss_zero_inside_where(torch.log(rel_dE_last), math.log(E_lower), math.log(E_upper))
    loss_energy_next = band_loss_zero_inside_where(torch.log(rel_dE_next), math.log(E_lower), math.log(E_upper))
    loss_energy_max = band_loss_zero_inside_where(torch.log(rel_dE_max), math.log(E_lower), math.log(E_upper))
    loss_energy = loss_energy_mean.mean() + loss_energy_last.mean() + loss_energy_max.mean() + loss_energy_next.mean()

    # (b) dt regularization — still per-sample
    #target_log_dt = math.log(dt_bound)
    #loss_dt = (torch.log(dt) - target_log_dt) ** 2
    loss_dt = dt                                        # (B,)
    #loss_dt = -dt

    # (c) We want E_hat to approximate rel_dE in log-space
    loss_pred = (torch.log(rel_dE_safe) - torch.log(E_hat_safe)) ** 2  # (B,)

    # Final scalar loss: average (or sum) over the batch
    #loss = loss_energy.mean()  # + lambda_dt * loss_dt.mean() + lambda_pred * loss_pred.mean()
    loss = loss_energy

    # --- 7. Metrics (use batch-averaged values so .item() still works) ---
    metrics = {
        "rel_dE":       rel_dE.mean(),          # scalar tensor
        "dt":           dt.mean(),              # scalar tensor
        "E0":           E0.mean(),           # scalar tensor
        "loss_energy":  loss_energy.mean(),
        "loss_energy_mean": loss_energy_mean.mean(),
        "loss_energy_last": loss_energy_last.mean(),
        "loss_energy_next": loss_energy_next.mean(),
        "loss_energy_max":  loss_energy_max.mean(),
        # relative energy change summaries (exposed for logging)
        "rel_dE_mean":  rel_dE_mean.mean(),
        "rel_dE_next":  rel_dE_next.mean(),
        "rel_dE_last":  rel_dE_last.mean(),
        "rel_dE_max":   rel_dE_max.mean(),
        "loss_pred":    loss_pred.mean(),
        "loss_dt":      loss_dt.mean(),
        # if you ever want full per-sample arrays:
         "rel_dE_full": rel_dE,
        # "dt_full":     dt,
    }

    if return_particle:
        return loss, metrics, p  # p now represents batched advanced particle(s)
    else:
        return loss, metrics, _


def band_loss_zero_inside_where(rel_dE, E_lower, E_upper):
    loss_below = (E_lower - rel_dE).clamp(min=0)**2
    loss_above = (rel_dE - E_upper).clamp(min=0)**2
    return loss_below + loss_above

float_info = torch.finfo(torch.float)
double_info = torch.finfo(torch.double)
double_tiny = double_info.tiny
float_tiny = float_info.tiny
eps = float_tiny




