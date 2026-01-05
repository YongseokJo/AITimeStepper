import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb

import sys
import pathlib
from pathlib import Path
import argparse
# Add parent directory of this file (your_project/)
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from src import *

# ===== CLI / Config =====
parser = argparse.ArgumentParser(description="Train ML time-step predictor with W&B logging")
parser.add_argument("--epochs", "-n", type=int, default=1000, help="number of training epochs")
parser.add_argument("--optimizer", "-o", type=str, default="LBFGS", help="optimizer type")
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--weight-decay", type=float, default=1e-2, help="optimizer weight decay")
parser.add_argument("--n-steps", type=int, default=10, help="number of integration steps per update")
parser.add_argument("--dt-bound", type=float, default=1e-8, help="dt bound (for loss heuristics)")
parser.add_argument("--rel-loss-bound", type=float, default=1e-5, help="relative loss bound")
parser.add_argument("--energy-threshold", type=float, default=2e-4, help="accept/reject energy threshold")
parser.add_argument("--replay-batch", type=int, default=512, help="replay buffer batch size")
parser.add_argument("--E_lower", type=float, default=1e-6, help="lower energy bound for loss calculation")
parser.add_argument("--E_upper", type=float, default=1e-4, help="upper energy bound for loss calculation")
parser.add_argument("--min-replay-size", type=int, default=2, help="min replay buffer size before training")
parser.add_argument("--eccentricity", "-e", type=float, default=0.9, help="eccentricity for generate_IC")
parser.add_argument("--semi-major", "-a", type=float, default=1.0, help="semi-major axis for generate_IC")
parser.add_argument("--wandb-project", type=str, default="AITimeStepper", help="W&B project name")
parser.add_argument("--wandb-name", type=str, default="two_body_ML_integrator", help="W&B run name")
parser.add_argument("--optuna", action="store_true", help="run in optuna mode and print final metrics as JSON to stdout")
parser.add_argument("--adam-epochs", type=int, default=0, help="number of epochs to run Adam before switching to L-BFGS (0 disables Adam phase)")
parser.add_argument("--adam-lr", type=float, default=None, help="learning rate for Adam (defaults to --lr)")
parser.add_argument("--lbfgs-lr", type=float, default=1.0, help="learning rate for L-BFGS")
parser.add_argument("--lbfgs-max-iter", type=int, default=500, help="max_iter for L-BFGS")
parser.add_argument("--lbfgs-history-size", type=int, default=50, help="history_size for L-BFGS")
parser.add_argument("--lbfgs-line-search", type=str, default="strong_wolfe", help="line_search_fn for L-BFGS (or 'none')")
args = parser.parse_args()

# Normalize optimizer option to be case-insensitive
if isinstance(args.optimizer, str):
    args.optimizer = args.optimizer.strip().lower()

# ===== Initialize Weights & Biases =====
wandb.init(
    project=args.wandb_project,
    name=args.wandb_name,
    config={
        "input_size": 2,
        "hidden_dims": [200, 1000, 1000, 200],
        "output_size": 2,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "activation": "tanh",
        "dropout": 0.2,
        "num_epochs": args.epochs,
        "dt_bound": args.dt_bound,
        "rel_loss_bound": args.rel_loss_bound,
        "energy_threshold": args.energy_threshold,
        "n_steps": args.n_steps,
        "replay_batch_size": args.replay_batch,
        "min_replay_size": args.min_replay_size,
        "eccentricity": args.eccentricity,
        "semi_major_axis": args.semi_major,
    }
)

dtype = torch.float32
dtype = torch.double
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)
# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
wandb.log({"device": str(device)})

import random

# ===== Setup =====
ptcls, T = generate_IC(e=args.eccentricity, a=args.semi_major)
print("ptcls shape: ", ptcls.shape)
print("ptcls: ", ptcls)
ptcls = torch.tensor(ptcls, device=device, dtype=dtype)
particle = make_particle(ptcls, device=device, dtype=dtype)

particle.period       = torch.tensor(T, device=device, dtype=dtype)
particle.current_time = torch.tensor(0, device=device, dtype=dtype)

# Model configuration
input_size = 2
hidden_dim = [200, 1000, 1000, 200]
output_size = 2

model = FullyConnectedNN(
    input_dim=input_size, 
    output_dim=output_size, 
    hidden_dims=hidden_dim, 
    activation='tanh', 
    dropout=0.2, 
    output_positive=True
).to(device)
model.to(dtype=dtype)

# Optimizer selection (supports phased specs like "adam+lbfgs")
opt_spec = args.optimizer
if "+" in opt_spec:
    phases = [p.strip().lower() for p in opt_spec.split("+") if p.strip()]
elif "," in opt_spec:
    phases = [p.strip().lower() for p in opt_spec.split(",") if p.strip()]
else:
    phases = [opt_spec.strip().lower()]

# Prepare optimizer objects for possible phases
adam_opt = None
sgd_opt = None
lbfgs_opt = None
adam_scheduler = None

if 'adam' in phases:
    adam_lr = args.adam_lr if args.adam_lr is not None else args.lr
    adam_opt = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=args.weight_decay)
    adam_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        adam_opt, factor=0.5, patience=500, min_lr=1e-8
    )

if 'sgd' in phases or 'sgd-momentum' in phases:
    sgd_opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if 'adamw' in phases and adam_opt is None:
    adam_opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if 'rmsprop' in phases:
    sgd_opt = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if 'lbfgs' in phases:
    line_search = None if str(args.lbfgs_line_search).lower() in ('none', 'null', '') else args.lbfgs_line_search
    lbfgs_opt = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        line_search_fn=line_search
    )

# Choose initial optimizer according to first phase
first_phase = phases[0] if phases else 'sgd'
if first_phase == 'adam' and adam_opt is not None:
    optimizer = adam_opt
elif first_phase == 'lbfgs' and lbfgs_opt is not None:
    optimizer = lbfgs_opt
elif first_phase.startswith('sgd') and sgd_opt is not None:
    optimizer = sgd_opt
elif adam_opt is not None:
    optimizer = adam_opt
elif lbfgs_opt is not None:
    optimizer = lbfgs_opt
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Phase flags
has_adam_phase = ('adam' in phases)
has_lbfgs_phase = ('lbfgs' in phases)
adam_epochs = args.adam_epochs

# Training hyperparameters
best_val_loss = float("inf")
best_largest_timestep = 0
num_epochs = args.epochs
dt_bound = args.dt_bound
rel_loss_bound = args.rel_loss_bound
energy_threshold = args.energy_threshold
n_steps = args.n_steps

particle_state = particle.clone_detached()
history = []

accepted_states = []
replay_batch_size = 512
min_replay_size = 2

# ===== Training Loop =====
for epoch in range(num_epochs):
    # ----- log position before this epoch's update -----
    pos_before = particle_state.position.detach().cpu().clone()

    # determine current training phase (adam then lbfgs if configured)
    if has_adam_phase and epoch < adam_epochs:
        current_phase = 'adam'
    elif has_lbfgs_phase and epoch >= adam_epochs and has_lbfgs_phase:
        current_phase = 'lbfgs'
    else:
        current_phase = phases[0]

    # pick optimizer object for this phase
    if current_phase == 'adam' and adam_opt is not None:
        optimizer = adam_opt
    elif current_phase == 'lbfgs' and lbfgs_opt is not None:
        optimizer = lbfgs_opt
    elif current_phase.startswith('sgd') and sgd_opt is not None:
        optimizer = sgd_opt

    # always start loss from the *current* state
    if current_phase == "lbfgs":
        stored = {}

        def closure():
            stored['loss'], stored['logs'], stored['p_next'] = loss_fn_batch(
                model,
                particle_state,
                n_steps=n_steps,
                rel_loss_bound=rel_loss_bound,
                E_lower=args.E_lower,
                E_upper=args.E_upper,
                return_particle=True,
            )
            optimizer.zero_grad()
            stored['loss'].backward()
            return stored['loss']

        optimizer.step(closure)
        loss = stored['loss']
        logs = stored['logs']
        p_next = stored['p_next']
    else:
        loss, logs, p_next = loss_fn_batch(
            model,
            particle_state,
            n_steps=n_steps,
            rel_loss_bound=rel_loss_bound,
            E_lower=args.E_lower,
            E_upper=args.E_upper,
            return_particle=True,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if using Adam with a scheduler, step scheduler on this metric
        if current_phase == 'adam' and adam_scheduler is not None:
            try:
                adam_scheduler.step(loss.item())
            except Exception:
                adam_scheduler.step(loss)

    # scalar for decision
    rel_dE_val = logs["rel_dE"].item()
    accepted = rel_dE_val <= energy_threshold

    # ----- log proposed position -----
    pos_after = p_next.position.detach().cpu().clone()

    # store everything you care about for this epoch
    history.append({
        "pos_before": pos_before,
        "pos_after": pos_after,
        "accepted": accepted,
        "rel_dE": rel_dE_val,
        "E0": logs["E0"].item(),
    })

    # update current state if accepted
    if accepted:
        particle_state = p_next.clone_detached()
        accepted_states.append(particle_state.clone_detached())
        print(f"Epoch {epoch}: Accepted step.")
    else:
        print(f"Epoch {epoch}: Rejected step. dt = {logs['dt'].item():.6e}")

    # ===== Replay Buffer Training =====
    if len(accepted_states) >= min_replay_size:
        batch_states = random.sample(
            accepted_states,
            k=min(replay_batch_size, len(accepted_states))
        )

        batch_states_detached = [p.clone_detached() for p in batch_states]
        batch_state = stack_particles(batch_states_detached)

        max_replay_steps = 1000

        for inner_step in range(max_replay_steps):
            if current_phase == "lbfgs":
                stored_rep = {}

                def closure_rep():
                    stored_rep['loss'], stored_rep['logs'], _ = loss_fn_batch(
                        model,
                        batch_state,
                        n_steps=n_steps,
                        rel_loss_bound=rel_loss_bound,
                        return_particle=False,
                    )
                    optimizer.zero_grad()
                    stored_rep['loss'].backward()
                    return stored_rep['loss']

                optimizer.step(closure_rep)
                replay_loss = stored_rep['loss']
                logs_rep = stored_rep['logs']
            else:
                optimizer.zero_grad()
                replay_loss, logs_rep, _ = loss_fn_batch(
                    model,
                    batch_state,
                    n_steps=n_steps,
                    rel_loss_bound=rel_loss_bound,
                    return_particle=False,
                )
                replay_loss.backward()
                optimizer.step()

            rel_dE_full = logs_rep["rel_dE_full"].detach()

            if (rel_dE_full <= energy_threshold).all():
                print(
                    f"[Replay] Converged in {inner_step+1} steps "
                    f"(max rel_dE = {rel_dE_full.max().item():.3e})"
                )
                # Log convergence
                wandb.log({
                    "replay/converged_step": inner_step + 1,
                    "replay/max_rel_dE": rel_dE_full.max().item(),
                })
                break

            if inner_step % 10 == 0:
                mean_rel_dE = logs_rep['rel_dE'].item()
                max_rel_dE = rel_dE_full.max().item()
                print(
                    f"[Replay] step {inner_step:03d} | "
                    f"loss = {replay_loss.item():.3e} | "
                    f"mean rel_dE = {mean_rel_dE:.3e} | "
                    f"max rel_dE = {max_rel_dE:.3e}"
                )
                wandb.log({
                    "replay/loss": replay_loss.item(),
                    "replay/mean_rel_dE": mean_rel_dE,
                    "replay/max_rel_dE": max_rel_dE,
                    "replay/step": inner_step,
                })

    # ===== Main Epoch Logging =====
    if epoch % 1 == 0:
        log_dict = {
            "epoch": epoch,
            "loss": loss.item(),
            "rel_dE": logs['rel_dE'].item(),
            "dt": logs['dt'].item(),
            "E0": logs['E0'].item(),
            "loss_energy": logs['loss_energy'].item(),
            "loss_energy_mean": logs.get('loss_energy_mean', torch.tensor(float('nan'))).item(),
            "loss_energy_last": logs.get('loss_energy_last', torch.tensor(float('nan'))).item(),
            "loss_energy_next": logs.get('loss_energy_next', torch.tensor(float('nan'))).item(),
            "loss_energy_max": logs.get('loss_energy_max', torch.tensor(float('nan'))).item(),
            "rel_dE_mean": logs.get('rel_dE_mean', torch.tensor(float('nan'))).item(),
            "rel_dE_last": logs.get('rel_dE_last', torch.tensor(float('nan'))).item(),
            "rel_dE_next": logs.get('rel_dE_next', torch.tensor(float('nan'))).item(),
            "rel_dE_max": logs.get('rel_dE_max', torch.tensor(float('nan'))).item(),
            "loss_pred": logs['loss_pred'].item(),
            "loss_dt": logs['loss_dt'].item(),
            "accepted": int(accepted),
            "num_accepted_states": len(accepted_states),
        }
        
        print(
            f"Epoch {epoch:4d} | "
            f"loss = {loss.item():.6e} | "
            f"rel_dE = {logs['rel_dE'].item():.6e} | "
            f"dt = {logs['dt'].item():.6e} | "
            f"E0 = {logs['E0'].item():.6e} | "
            f"loss_energy = {logs['loss_energy'].item():.6e} | "
            f"loss_energy_mean = {logs.get('loss_energy_mean', torch.tensor(float('nan'))).item():.6e} | "
            f"loss_energy_last = {logs.get('loss_energy_last', torch.tensor(float('nan'))).item():.6e} | "
            f"loss_energy_next = {logs.get('loss_energy_next', torch.tensor(float('nan'))).item():.6e} | "
            f"loss_energy_max = {logs.get('loss_energy_max', torch.tensor(float('nan'))).item():.6e} | "
            f"rel_dE_mean = {logs.get('rel_dE_mean', torch.tensor(float('nan'))).item():.6e} | "
            f"rel_dE_last = {logs.get('rel_dE_last', torch.tensor(float('nan'))).item():.6e} | "
            f"rel_dE_next = {logs.get('rel_dE_next', torch.tensor(float('nan'))).item():.6e} | "
            f"rel_dE_max = {logs.get('rel_dE_max', torch.tensor(float('nan'))).item():.6e} | "
            f"loss_pred = {logs['loss_pred'].item():.6e} | "
            f"loss_dt = {logs['loss_dt'].item():.6e}"
        )
        
        wandb.log(log_dict)

    # ===== Checkpoint Saving =====
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        checkpoint_path = f"../data/model/epoch_{epoch:04d}.pt"
        save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loss=loss,
            info=logs,
            extra={"n_steps": n_steps, "dt_bound": dt_bound, "rel_loss_bound": rel_loss_bound},
        )
        
        # Log model checkpoint to wandb using absolute path
        abs_checkpoint_path = (project_root / checkpoint_path).resolve()
        if abs_checkpoint_path.exists():
            wandb.save(str(abs_checkpoint_path), base_path=str(abs_checkpoint_path.parent))
        print(f"Saved checkpoint at epoch {epoch}")

# ===== Post-Training Visualization =====
print("Generating visualizations...")

# Compile position sequence
pos_seq = []
for h in history:
    if h["accepted"]:
        pos_seq.append(h["pos_after"])
    else:
        pos_seq.append(h["pos_before"])

pos_seq = torch.stack(pos_seq, dim=0).numpy()

# Extract particle trajectories
T, N, dim = pos_seq.shape
assert N == 2 and dim == 2, "Expecting 2 particles in 2D."

x1, y1 = pos_seq[:, 0, 0], pos_seq[:, 0, 1]
x2, y2 = pos_seq[:, 1, 0], pos_seq[:, 1, 1]

# ===== Figure 1: Trajectory with acceptance markers =====
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(x1, y1, '-', label="Particle 1", alpha=0.8)
ax.plot(x2, y2, '-', label="Particle 2", alpha=0.8)

for t, h in enumerate(history):
    if h["accepted"]:
        m = "o"
        c1 = "tab:blue"
        c2 = "tab:orange"
    else:
        m = "x"
        c1 = "gray"
        c2 = "gray"

    ax.scatter(x1[t], y1[t], marker=m, s=35, color=c1)
    ax.scatter(x2[t], y2[t], marker=m, s=35, color=c2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Two-Particle Trajectories\n○ accepted, × rejected")
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.savefig("../data/plot/two_body_ML_integrator.png", dpi=100, bbox_inches='tight')
wandb.log({"trajectory": wandb.Image("../data/plot/two_body_ML_integrator.png")})
print("Saved trajectory plot")
plt.close()

# ===== Figure 2: Trajectory with energy =====
E_all = np.array([float(h["E0"]) for h in history])
accepted_all = np.array([bool(h["accepted"]) for h in history])
steps_idx = np.arange(T)

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.3)

ax_traj = fig.add_subplot(gs[0, 0])
ax_E = fig.add_subplot(gs[0, 1])

# Trajectory subplot
all_x = pos_seq[:, :, 0]
all_y = pos_seq[:, :, 1]
xmin, xmax = all_x.min(), all_x.max()
ymin, ymax = all_y.min(), all_y.max()
pad = 0.1 * max(xmax - xmin, ymax - ymin)

ax_traj.set_xlim(xmin - pad, xmax + pad)
ax_traj.set_ylim(ymin - pad, ymax + pad)
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.set_xlabel("x")
ax_traj.set_ylabel("y")
ax_traj.set_title("Trajectory (colored = accepted, grey = rejected)")

ax_traj.plot(x1, y1, '-', alpha=0.6, color='tab:blue')
ax_traj.plot(x2, y2, '-', alpha=0.6, color='tab:orange')

# Energy subplot
ax_E.set_xlim(steps_idx[0], steps_idx[-1])
ax_E.set_xlabel("Iteration")
ax_E.set_ylabel("E0")
ax_E.set_title("Energy (black = accepted, grey = rejected)")

Emin, Emax = E_all.min(), E_all.max()
dE = Emax - Emin if Emax > Emin else 1.0
ax_E.set_ylim(Emin - 0.1 * dE, Emax + 0.1 * dE)
ax_E.grid(True)

# Plot energy with acceptance markers
x_acc = steps_idx[accepted_all]
y_acc = E_all[accepted_all]
x_rej = steps_idx[~accepted_all]
y_rej = E_all[~accepted_all]

ax_E.scatter(x_acc, y_acc, s=15, color="black", label="accepted", alpha=0.7)
ax_E.scatter(x_rej, y_rej, s=15, color="gray", label="rejected", alpha=0.5)
ax_E.legend(loc="best")

plt.tight_layout()
plt.savefig("../data/plot/two_body_ML_trajectory_energy.png", dpi=100, bbox_inches='tight')
wandb.log({"trajectory_energy": wandb.Image("../data/plot/two_body_ML_trajectory_energy.png")})
print("Saved trajectory+energy plot")
plt.close()

# ===== Statistics Summary =====
acceptance_rate = np.mean(accepted_all)
num_accepted = np.sum(accepted_all)
num_rejected = np.sum(~accepted_all)
final_energy = E_all[-1]
initial_energy = E_all[0]
energy_drift = (final_energy - initial_energy) / initial_energy if initial_energy != 0 else 0

summary_stats = {
    "summary/acceptance_rate": acceptance_rate,
    "summary/num_accepted": num_accepted,
    "summary/num_rejected": num_rejected,
    "summary/final_energy": final_energy,
    "summary/initial_energy": initial_energy,
    "summary/energy_drift": energy_drift,
    "summary/total_epochs": num_epochs,
}

print(f"\n===== Training Summary =====")
print(f"Acceptance rate: {acceptance_rate:.2%}")
print(f"Accepted steps: {num_accepted} / Rejected steps: {num_rejected}")
print(f"Initial energy: {initial_energy:.6e}")
print(f"Final energy: {final_energy:.6e}")
print(f"Energy drift: {energy_drift:.6e}")

wandb.log(summary_stats)
wandb.finish()

print("Training complete! Logs saved to Weights & Biases.")

if args.optuna:
    # Print a compact JSON line for Optuna wrapper to parse
    try:
        import json
        final_metrics = {"loss": float(loss.item()), "time_step": float(logs['dt'].item())}
        print("OPTUNA_RESULT_JSON:" + json.dumps(final_metrics), flush=True)
    except Exception:
        # ensure we don't fail the process because of logging
        pass
