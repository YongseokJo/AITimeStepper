import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

import sys
import pathlib
import argparse
# Add parent directory of this file (your_project/)
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from src import *

dtype = torch.float32
dtype = torch.double
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)
# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
import random



ptcls, T = generate_IC(e=0.9, a=1.0)
print("ptcls sahpe: ", ptcls.shape)
print("ptcls: ", ptcls)
ptcls = torch.tensor(ptcls, device=device, dtype=dtype)
particle = make_particle(ptcls, device=device, dtype=dtype)

particle.period       = torch.tensor(T, device=device, dtype=dtype)
particle.current_time = torch.tensor(0, device=device, dtype=dtype)

#input_size = 10     # Number of input features: force + masses + velocities
input_size = 2     # Number of input features: force + masses + velocities
hidden_dim = [200,1000, 1000, 200]     # Number of hidden neurons
output_size = 2     # Number of output two time steps


#model = SimpleNN(input_size, hidden_size, output_size).to(device)
model = FullyConnectedNN(input_dim=input_size, output_dim=output_size, 
                         hidden_dims=hidden_dim, activation='tanh', 
                         dropout=0.2, output_positive=True).to(device)
model.to(dtype=dtype) 


# Define the loss function (CrossEntropyLoss is common for classification tasks)
#criterion = CustomizableLoss(nParticle=2, nAttribute=13, nBatch=batch_size,alpha=0.1, beta=0.1, gamma=10.0, TargetEnergyError=TargetEnergyError,
#                            data_min=data_min, data_max=data_max,device=device)
#criterion = loss_fn
#criterion(model, particle)

#raise

# Define the optimizer (Stochastic Gradient Descent in this example)
#optimizer = optim.SGD(model.parameters(), lr=0.0001)
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-6,           # learning rate
    momentum=0.9,      # typical value
    weight_decay=1e-2   # optional
)

#optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Paths to save best and last models
#base_path = "../data/models/two_body/"
#best_val_path = base_path + "best_model_loss.pth"
#best_timestep_path = base_path + "best_model_timestep.pth"
#last_model_path = base_path + f"last_model_{epoch}.pth"
best_val_loss = float("inf")
best_largest_timestep = 0

#example_input = torch.randn(1, input_size).to(device)

# Training routine
num_epochs = 10000
num_epochs = 4000
num_epochs = 1000
#num_epochs = 500
#num_epochs = 1000
dt_bound = 1e-8
rel_loss_bound = 1e-5
energy_threshold = 2e-4
n_steps = 10
particle_state = particle.clone_detached()  # start from initial condition
history = [] 

accepted_states = []   # replay buffer of ParticleTorch objects
replay_batch_size = 512 # or whatever
min_replay_size = 2   # start replay only after we have enough

for epoch in range(num_epochs):
    # ----- log position before this epoch's update -----
    pos_before = particle_state.position.detach().cpu().clone()

    # always start loss from the *current* state
    loss, logs, p_next = loss_fn_batch(
        model,
        particle_state,     # this is your “current” particle
        n_steps=1, #n_steps,
        rel_loss_bound=rel_loss_bound,
        return_particle=True,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # scalar for decision
    rel_dE_val = logs["rel_dE"].item()
    accepted = rel_dE_val <= energy_threshold

    # ----- log proposed position -----
    pos_after = p_next.position.detach().cpu().clone()

    # store everything you care about for this epoch
    history.append({
        "pos_before": pos_before,   # tensor on CPU
        "pos_after": pos_after,     # tensor on CPU
        "accepted": accepted,
        "rel_dE": rel_dE_val,
        "E0": logs["E0"].item(),
    })

    # update current state if accepted
    if accepted:
        particle_state = p_next.clone_detached()  # or your own helper
        accepted_states.append(particle_state.clone_detached())
        print(f"Epoch {epoch}: Accepted step.")
    else:
        print(f"Epoch {epoch}: Rejected step. dt = {logs['dt'].item():.6e}")


    if len(accepted_states) >= min_replay_size:
        # sample a batch of previously accepted states
        batch_states = random.sample(
            accepted_states,
            k=min(replay_batch_size, len(accepted_states))
        )

        # Important: use detached copies as starting states
        batch_states_detached = [p.clone_detached() for p in batch_states]
        batch_state = stack_particles(batch_states_detached)

        max_replay_steps = 1000

        for inner_step in range(max_replay_steps):
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
                break

            if inner_step % 10 == 0:
                print(
                    f"[Replay] step {inner_step:03d} | "
                    f"loss = {replay_loss.item():.3e} | "
                    f"mean rel_dE = {logs_rep['rel_dE'].item():.3e} | "
                    f"max rel_dE = {rel_dE_full.max().item():.3e}"
            )


    if len(accepted_states) >= min_replay_size:
        # sample a batch of previously accepted states
        batch_states = random.sample(
            accepted_states,
            k=min(replay_batch_size, len(accepted_states))
        )

    if epoch % 1 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"loss = {loss.item():.6e} | "
            f"rel_dE = {logs['rel_dE'].item():.6e} | "
            f"dt = {logs['dt'].item():.6e} | "
            f"E0 = {logs['E0'].item():.6e} | "
            f"loss_energy = {logs['loss_energy'].item():.6e} | "
            f"loss_pred = {logs['loss_pred'].item():.6e} | "
            f"loss_dt = {logs['loss_dt'].item():.6e}"
        )

    # example: save every 100 epochs
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        save_checkpoint(
            path=f"../data/model/epoch_{epoch:04d}.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loss=loss,
            info=logs,
            extra={"n_steps": n_steps, "dt_bound": dt_bound, "rel_loss_bound": rel_loss_bound},
        )


import matplotlib.pyplot as plt
import numpy as np

# sequence of positions representing the actual integrator state over time
pos_seq = []

for h in history:
    if h["accepted"]:
        pos_seq.append(h["pos_after"])
    else:
        # state did not change; still at pos_before
        pos_seq.append(h["pos_before"])

# stack to (T, N, dim)
pos_seq = torch.stack(pos_seq, dim=0).numpy()

T, N, dim = pos_seq.shape
assert N == 2 and dim == 2

fig, ax = plt.subplots(figsize=(6, 6))

# draw continuous trajectory lines (actual integrator path)
ax.plot(pos_seq[:, 0, 0], pos_seq[:, 0, 1], '-', alpha=0.8, label="Particle 1 path")
ax.plot(pos_seq[:, 1, 0], pos_seq[:, 1, 1], '-', alpha=0.8, label="Particle 2 path")

for t, h in enumerate(history):
    # before = actual state when step t started
    pb = h["pos_before"]     # shape (2,2) → particle idx 0/1, x/y
    # after = proposal
    pa = h["pos_after"]

    if h["accepted"]:
        # → plot "after" as normal colored point
        ax.scatter(pa[0,0], pa[0,1], s=40, marker="o", color="tab:blue")
        ax.scatter(pa[1,0], pa[1,1], s=40, marker="o", color="tab:orange")
    else:
        # REJECTED CASE:
        # → plot before (normal color)
        ax.scatter(pb[0,0], pb[0,1], s=40, marker="o", color="tab:blue")
        ax.scatter(pb[1,0], pb[1,1], s=40, marker="o", color="tab:orange")

        # → plot proposed after as grey
        ax.scatter(pa[0,0], pa[0,1], s=40, marker="o", color="gray", alpha=0.5)
        ax.scatter(pa[1,0], pa[1,1], s=40, marker="o", color="gray", alpha=0.5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Two-Particle Trajectories\nAccepted: colored | Rejected: after-state is gray")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("../data/plot/two_body_ML_integrator_new.png", dpi=100, bbox_inches='tight')
plt.close()


# sequence of positions representing the actual integrator state over time
pos_seq = []

for h in history:
    if h["accepted"]:
        pos_seq.append(h["pos_after"])
    else:
        # state did not change; still at pos_before
        pos_seq.append(h["pos_before"])

# stack to (T, N, dim)
pos_seq = torch.stack(pos_seq, dim=0).numpy()

import matplotlib.pyplot as plt
import numpy as np

# pos_seq: (T, 2, 2)  → T timesteps, 2 particles, x,y
T, N, dim = pos_seq.shape
assert N == 2 and dim == 2, "Expecting 2 particles in 2D."

# unpack trajectories
x1, y1 = pos_seq[:, 0, 0], pos_seq[:, 0, 1]   # particle 0
x2, y2 = pos_seq[:, 1, 0], pos_seq[:, 1, 1]   # particle 1

fig, ax = plt.subplots(figsize=(6, 6))

# continuous trajectory lines
ax.plot(x1, y1, '-', label="Particle 1", alpha=0.8)
ax.plot(x2, y2, '-', label="Particle 2", alpha=0.8)

# mark acceptance vs rejection
for t, h in enumerate(history):
    if h["accepted"]:
        m = "o"    # circle mark
        c1 = "tab:blue"
        c2 = "tab:orange"
    else:
        m = "x"    # rejected
        c1 = "gray"
        c2 = "gray"

    # particle 1 point
    ax.scatter(x1[t], y1[t], marker=m, s=35, color=c1)
    # particle 2 point
    ax.scatter(x2[t], y2[t], marker=m, s=35, color=c2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Two-Particle Trajectories\n○ accepted, × rejected")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("../data/plot/two_body_ML_integrator.png", dpi=100, bbox_inches='tight')
plt.close()



# --------------------------------------------------------
# Build pos_seq only to get global limits
# --------------------------------------------------------
pos_seq = []
for h in history:
    if h["accepted"]:
        pos_seq.append(h["pos_after"])   # state after accepted step
    else:
        pos_seq.append(h["pos_before"])  # unchanged state if rejected

# (T, N, dim) on CPU → NumPy
pos_seq = torch.stack(pos_seq, dim=0).cpu().numpy()  # (T, 2, 2)
T, N, dim = pos_seq.shape

# Energy and acceptance arrays
E_all        = np.array([float(h["E0"]) for h in history])
accepted_all = np.array([bool(h["accepted"]) for h in history])
steps_idx    = np.arange(T)

# --------------------------------------------------------
# Figure with 2 panels (trajectory + energy)
# --------------------------------------------------------
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.3, 1.0],
    wspace=0.3
)

ax_traj = fig.add_subplot(gs[0, 0])
ax_E    = fig.add_subplot(gs[0, 1])

# ── Trajectory axis limits ─────────────────────────────
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
ax_traj.set_title("Trajectory (colored = accepted/before, grey = rejected-after)")

# iteration text inside the trajectory axes
iter_text = ax_traj.text(
    0.5, 1.05, "Iteration 0",
    transform=ax_traj.transAxes,
    ha="center", va="bottom",
    fontsize=16, fontweight="bold"
)

# ── Energy axis limits ────────────────────────────────
ax_E.set_xlim(steps_idx[0], steps_idx[-1])
ax_E.set_xlabel("Iteration")
ax_E.set_ylabel("E0")
ax_E.set_title("Energy (black = accepted, grey = rejected)")

Emin, Emax = E_all.min(), E_all.max()
dE = Emax - Emin if Emax > Emin else 1.0
ax_E.set_ylim(Emin - 0.1 * dE, Emax + 0.1 * dE)
ax_E.grid(True)

# --------------------------------------------------------
# Artists to update
# --------------------------------------------------------
# Trajectories
scat_before = ax_traj.scatter([], [], s=50)                # colored (before/persistent)
scat_after  = ax_traj.scatter([], [], s=50, alpha=0.5)     # grey (rejected-after)

# Energy: two scatters (accepted = black, rejected = grey)
E_acc_scat = ax_E.scatter([], [], s=15, color="black", label="accepted")
E_rej_scat = ax_E.scatter([], [], s=15, color="gray",  label="rejected")
ax_E.legend(loc="best")

def init():
    # reset trajectory scatters
    scat_before.set_offsets(np.empty((0, 2)))
    scat_after.set_offsets(np.empty((0, 2)))
    scat_before.set_color(["tab:blue", "tab:orange"])

    # reset energy scatters
    E_acc_scat.set_offsets(np.empty((0, 2)))
    E_rej_scat.set_offsets(np.empty((0, 2)))

    iter_text.set_text("Iteration 0")
    return scat_before, scat_after, E_acc_scat, E_rej_scat, iter_text

def update(frame):
    # ----- particle positions for this iteration -----
    h = history[frame]
    pb = h["pos_before"].cpu().numpy()  # (2, 2)
    pa = h["pos_after"].cpu().numpy()   # (2, 2)
    accepted = h["accepted"]

    iter_text.set_text(f"Iteration {frame}")

    if accepted:
        # accepted: only show the "current" (after) positions in color
        before_xy = pa  # this is the state we actually keep
        scat_before.set_offsets(before_xy)
        scat_before.set_color(["tab:blue", "tab:orange"])

        # no rejected-after points
        scat_after.set_offsets(np.empty((0, 2)))
    else:
        # rejected: keep pb as colored (persistent state)
        before_xy = pb
        scat_before.set_offsets(before_xy)
        scat_before.set_color(["tab:blue", "tab:orange"])

        # show proposed (rejected) pa in grey
        after_xy = pa
        scat_after.set_offsets(after_xy)
        scat_after.set_color(["gray", "gray"])

    # ----- energy scatter up to this iteration -----
    x_hist = steps_idx[:frame+1]
    E_hist = E_all[:frame+1]
    acc_mask = accepted_all[:frame+1]

    # accepted points
    x_acc = x_hist[acc_mask]
    y_acc = E_hist[acc_mask]
    if len(x_acc) > 0:
        acc_offsets = np.column_stack([x_acc, y_acc])
    else:
        acc_offsets = np.empty((0, 2))
    E_acc_scat.set_offsets(acc_offsets)

    # rejected points
    x_rej = x_hist[~acc_mask]
    y_rej = E_hist[~acc_mask]
    if len(x_rej) > 0:
        rej_offsets = np.column_stack([x_rej, y_rej])
    else:
        rej_offsets = np.empty((0, 2))
    E_rej_scat.set_offsets(rej_offsets)

    return scat_before, scat_after, E_acc_scat, E_rej_scat, iter_text

from matplotlib.animation import FuncAnimation

ani = FuncAnimation(
    fig,
    update,
    frames=len(history),
    init_func=init,
    interval=20,
    blit=False,   # easier with text + multiple axes
)

ani.save("../data/movie/trajectory_ML.mp4", fps=30)
plt.close(fig)

raise
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6, 6))

# put title inside axes at top center
iter_text = ax.text(
    0.5, 1.05, "Iteration 0",
    transform=ax.transAxes,
    ha="center", va="bottom",
    fontsize=20, fontweight="bold"
)

scat_before = ax.scatter([], [], s=50)
scat_after  = ax.scatter([], [], s=50, alpha=0.5)

def init():
    all_x = pos_seq[:, :, 0]
    all_y = pos_seq[:, :, 1]
    ax.set_xlim(all_x.min() - 0.1, all_x.max() + 0.1)
    ax.set_ylim(all_y.min() - 0.1, all_y.max() + 0.1)

    ax.set_aspect("equal")
    ax.set_title("Trajectory Movie (colored = accepted/before, grey = rejected-after)")

    iter_text.set_text("Iteration 0")
    return scat_before, scat_after, iter_text

def update(frame):
    pb = history[frame]["pos_before"]
    pa = history[frame]["pos_after"]
    accepted = history[frame]["accepted"]

    # update iteration text
    iter_text.set_text(f"Iteration {frame}")

    if accepted:
        before_xy = np.vstack([pa[0], pa[1]])
        scat_before.set_offsets(before_xy)
        scat_before.set_color(["tab:blue", "tab:orange"])
        scat_after.set_offsets(np.empty((0, 2)))

    else:
        before_xy = np.vstack([pb[0], pb[1]])
        scat_before.set_offsets(before_xy)
        scat_before.set_color(["tab:blue", "tab:orange"])

        after_xy = np.vstack([pa[0], pa[1]])
        scat_after.set_offsets(after_xy)
        scat_after.set_color(["gray", "gray"])

    return scat_before, scat_after, iter_text

ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True)
ani.save("../data/movie/trajectory_ML.mp4", fps=30)

raise
for epoch in range(num_epochs):

    train_loss = \
        train_one_epoch(model, optimizer, criterion, train_loader, input_size, device)
    test_loss, energy_error, energy_error_fiducial, time_step, time_step_fiducial = \
        validate(model, criterion, test_loader, input_size, device)

    if epoch > 5:
        # Save the best model based on validation loss
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            #torch.save(model.state_dict(), best_val_path)
            #traced_model = torch.jit.trace(model, example_input)
            #traced_model.save(base_path + f"c++/best_model_loss.pt")
            print(f"--> Best model saved at epoch {epoch} with val loss {best_val_loss:.4e}")
        
        # Save the best model based on validation largest timestep
        if time_step > best_largest_timestep:
            best_largest_timestep = time_step
            torch.save(model.state_dict(), best_timestep_path)
            traced_model = torch.jit.trace(model, example_input)
            #traced_model.save(base_path + f"c++/best_model_timestep.pt")
            print(f"--> Best model saved at epoch {epoch} with largest timestep {best_largest_timestep:.4e}")

    # Save the last model after every epoch (or just once at the end)
    #torch.save(model.state_dict(), base_path + f"model_{epoch}.pth")
    #traced_model = torch.jit.trace(model, example_input)
    #traced_model.save(base_path + f"c++/model_{epoch}.pt")

    print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4e}, Test Loss: {test_loss:.4e}, Energy Loss: {energy_error:.4e}/{energy_error_fiducial:.4e}, Time step: {time_step:.4e}/{time_step_fiducial:.4e}")