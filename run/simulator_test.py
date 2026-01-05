import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import pathlib
import argparse
# Add parent directory of this file (your_project/)
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from simulators import *
from src import FullyConnectedNN, HistoryBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=-1,
                    help="Time step for integrator")
parser.add_argument("--steps", type=int, default=-1,
                    help="Number of steps to evolve per period")
parser.add_argument("--Nperiod", type=int, default=10,
                    help="Number of period")
parser.add_argument("--e", type=float, default=0.7,
                    help="Eccentricity")
parser.add_argument("--a", type=float, default=1.0,
                    help="Semi-major axis")
parser.add_argument("--eps", type=float, default=0.1,
                    help="time constant for dt update")
parser.add_argument("--ML", action="store_true", default=False,
                    help="Use machine learning–based timestep" )
parser.add_argument("--save-name", "-s", type=str, default=None,
                    help="base name (directory) under data/ to save outputs")
parser.add_argument("--model-path", type=str,
                    default=str(project_root / "data" / "model" / "epoch_0399.pt"),
                    help="Path to the model checkpoint to load (torch .pt)")
parser.add_argument("--history-len", type=int, default=0,
                    help="Length of history (in steps) for history-aware ML model")
parser.add_argument("--history-feature-type", choices=["basic", "rich"], default="basic",
                    help="HistoryBuffer feature type used during training")
args = parser.parse_args()
dt = args.dt
steps_per_period = args.steps
Nperiod = args.Nperiod
e = args.e
a = args.a
eps = args.eps
isML = args.ML
history_len = max(0, args.history_len)
history_feature_type = args.history_feature_type
history_buffer = None

if isML:
    suffix = f"e{e}_a{a}_ML"
else:
    suffix = f"e{e}_eps{eps}"
save_name = args.save_name if args.save_name is not None else suffix
print(save_name)

# prepare output directories under data/<save_name>/plot and /movie
out_base = project_root / "data" #/save_name
plot_dir = out_base / "plot"
movie_dir = out_base / "movie"
print("plot_dir:", plot_dir)
print("movie_dir:", movie_dir)
import os
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(movie_dir, exist_ok=True)


# NumPy backend
p1, p2, T = generate_IC(e=e, a=a, dt=dt)

if isML:
    # 1. Rebuild the model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 2*(history_len+1) # Number of input features: force + masses + velocities
    #hidden_dim = [8,32,8]     # Number of hidden neurons
    hidden_dim = [200,1000, 1000, 200]     # Number of hidden neurons
    output_size = 2     # Number of output two time steps


    model = FullyConnectedNN(input_dim=input_size, output_dim=output_size, 
                            hidden_dims=hidden_dim, activation='tanh', 
                            dropout=0.2, output_positive=True).to(device)

    # load checkpoint from CLI-provided path
    model_path = args.model_path
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(torch.double)
    print("Loaded ML model for dt prediction.")

    p1.update_model(model, device)
    p2.update_model(model, device)

    if history_len > 0:
        history_buffer = HistoryBuffer(history_len=history_len,
                                       feature_type=history_feature_type)
        p1.attach_history_buffer(history_buffer)
        p2.attach_history_buffer(history_buffer)
        print(f"Attached HistoryBuffer len={history_len}, type={history_feature_type}.")

final_time = Nperiod*T

if dt == -1:
    dt = T/steps_per_period
    steps = steps_per_period*Nperiod
else:
    steps = int(final_time / dt)

if isML:
    use_history_model = history_len > 0 and history_buffer is not None
    if use_history_model:
        p1.update_dt_from_history_model(p2)
        p2.update_dt_from_history_model(p1)
    else:
        p1.update_dt_from_model(p2)
        p2.update_dt_from_model(p1)
else:
    p1.dt = dt
    p2.dt = dt

#p1 = Particle(1.0, [ -0.5, 0.0 ], [0.0,  0.5], backend="numpy", dt=dt)
#p2 = Particle(1.0, [  0.5, 0.0 ], [0.0, -0.5], backend="numpy", dt=dt)

print(p1["position"], p1["velocity"], p2["position"], p2["velocity"])

# Storage for trajectories
traj1 = []
traj2 = []
energies = []
momenta = []
angular_momenta = []


#for _ in range(steps):
steps = 0
current_dt = 0.0
while current_dt < final_time:
    evolve_dt(p1, p2, eps, isML=isML)
    #print(p1, p2)

    traj1.append(p1.position.copy())
    traj2.append(p2.position.copy())
    energies.append(total_energy(p1, p2))
    momenta.append(total_momentum(p1, p2))
    angular_momenta.append(total_angular_momentum_com(p1, p2))
    current_dt = min(p1.current_time, p2.current_time)
    steps += 1

traj1 = np.array(traj1)
traj2 = np.array(traj2)

# Trajectory Plot
plt.figure(figsize=(6, 6))
plt.plot(traj1[:,0], traj1[:,1], label="Particle 1")
plt.plot(traj2[:,0], traj2[:,1], label="Particle 2")

plt.scatter([traj1[0,0]], [traj1[0,1]], color='blue', s=50)   # starting points
plt.scatter([traj2[0,0]], [traj2[0,1]], color='orange', s=50)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.title("Two-Body Trajectories")
plt.savefig(str(plot_dir / f"two_body_{save_name}.png"), dpi=100, bbox_inches='tight')
plt.close()


# Energy Plot
# Convert to arrays
E = np.array(energies)
E0 = E[0]
residual = E - E0

# total momentum per step: list/array of shape (n_steps, ndim)
P = np.array(momenta)        # e.g. from total_momentum(p1, p2) each step
P_norm = np.linalg.norm(P, axis=1)

# total angular momentum per step: can be scalar (2D) or vector (3D)
L = np.array(angular_momenta)    # e.g. from total_angular_momentum(p1, p2)
if L.ndim == 1:
    L_mag = np.abs(L)              # 2D: scalar L_z
else:
    L_mag = np.linalg.norm(L, axis=1)  # 3D: |L|

fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    4, 1, figsize=(8, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 2, 2]}
)

# --- Total energy ---
ax1.plot(E, linewidth=1)
ax1.set_ylabel("Total Energy")
ax1.set_title("Energy, Momentum, and Angular Momentum")

# --- Energy residuals ---
ax2.plot(residual, linewidth=1)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("ΔE")
ax2.set_xlabel("Step")

# --- Total momentum magnitude ---
ax3.plot(P_norm, linewidth=1)
ax3.set_ylabel(r"$|\mathbf{P}|$")
ax3.set_xlabel("Step")

# --- Total angular momentum magnitude ---
ax4.plot(L_mag, linewidth=1)
ax4.set_ylabel(r"$|\mathbf{L}|$")
ax4.set_xlabel("Step")

# Grids
for ax in (ax1, ax2, ax3, ax4):
    ax.grid(True, linewidth=0.5)

plt.tight_layout()
plt.savefig(str(plot_dir / f"energy_mom_L_{save_name}.png"),
            dpi=100, bbox_inches="tight")
plt.close()



# For movie generation, 
steps = len(E)

max_frames = 1000  # or 500, 200, etc.
stride = max(1, steps // max_frames)

traj1_plot = traj1[::stride]
traj2_plot = traj2[::stride]
E_plot     = E[::stride]
res_plot   = residual[::stride]
P_plot     = P_norm[::stride]
L_plot     = L_mag[::stride]

steps_plot = np.arange(steps)[::stride]
nframes    = len(E_plot)

# --------------------------------------------------------------------
# Figure with 1 big trajectory panel + 4 time-series panels
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(
    4, 2,
    width_ratios=[1.3, 1.0],
    height_ratios=[1.0, 1.0, 1.0, 1.0],
    hspace=0.4,
    wspace=0.3,
)

ax_traj = fig.add_subplot(gs[:, 0])          # left: spans all 4 rows
ax_E    = fig.add_subplot(gs[0, 1])          # right, row 0
ax_res  = fig.add_subplot(gs[1, 1], sharex=ax_E)  # right, row 1
ax_P    = fig.add_subplot(gs[2, 1], sharex=ax_E)  # right, row 2
ax_L    = fig.add_subplot(gs[3, 1], sharex=ax_E)  # right, row 3

# ── fixed limits: trajectory ──
xmin = min(traj1[:,0].min(), traj2[:,0].min())
xmax = max(traj1[:,0].max(), traj2[:,0].max())
ymin = min(traj1[:,1].min(), traj2[:,1].min())
ymax = max(traj1[:,1].max(), traj2[:,1].max())
pad  = 0.1 * max(xmax - xmin, ymax - ymin)

ax_traj.set_xlim(xmin - pad, xmax + pad)
ax_traj.set_ylim(ymin - pad, ymax + pad)
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.set_xlabel("x")
ax_traj.set_ylabel("y")
ax_traj.set_title("Two-Body Trajectories")

# ── fixed limits: energy ──
ax_E.set_xlim(steps_plot[0], steps_plot[-1])
ax_E.set_ylabel("Total Energy")
ax_E.set_title("Energy, Momentum, Angular Momentum")

Emin, Emax = E_plot.min(), E_plot.max()
dE = Emax - Emin if Emax > Emin else 1.0
ax_E.set_ylim(Emin - 0.1 * dE, Emax + 0.1 * dE)
ax_E.grid(True)

# ── residual ──
ax_res.set_xlabel("")  # shared x label at bottom only
ax_res.set_ylabel("ΔE")
rmin, rmax = res_plot.min(), res_plot.max()
dr = rmax - rmin if rmax > rmin else 1.0
ax_res.set_ylim(rmin - 0.1 * dr, rmax + 0.1 * dr)
ax_res.axhline(0.0, color="black", linewidth=0.8)
ax_res.grid(True)

# ── |P| ──
ax_P.set_ylabel(r"$|\mathbf{P}|$")
Pmin, Pmax = P_plot.min(), P_plot.max()
dP = Pmax - Pmin if Pmax > Pmin else 1.0
ax_P.set_ylim(Pmin - 0.1 * dP, Pmax + 0.1 * dP)
ax_P.grid(True)

# ── |L| ──
ax_L.set_xlabel("Step")
ax_L.set_ylabel(r"$|\mathbf{L}|$")
Lmin, Lmax = L_plot.min(), L_plot.max()
dL = Lmax - Lmin if Lmax > Lmin else 1.0
ax_L.set_ylim(Lmin - 0.1 * dL, Lmax + 0.1 * dL)
ax_L.grid(True)

# --------------------------------------------------------------------
# Artists
# --------------------------------------------------------------------
traj1_line, = ax_traj.plot([], [], label="Particle 1")
traj2_line, = ax_traj.plot([], [], label="Particle 2")
p1_point,   = ax_traj.plot([], [], marker="o", linestyle="None")
p2_point,   = ax_traj.plot([], [], marker="o", linestyle="None")
ax_traj.legend(loc="upper right")

E_line,   = ax_E.plot([], [], lw=1.0)
res_line, = ax_res.plot([], [], lw=1.0)
P_line,   = ax_P.plot([], [], lw=1.0)
L_line,   = ax_L.plot([], [], lw=1.0)

def init():
    traj1_line.set_data([], [])
    traj2_line.set_data([], [])
    p1_point.set_data([], [])
    p2_point.set_data([], [])
    E_line.set_data([], [])
    res_line.set_data([], [])
    P_line.set_data([], [])
    L_line.set_data([], [])
    return (
        traj1_line, traj2_line, p1_point, p2_point,
        E_line, res_line, P_line, L_line
    )

def update(frame):
    i = frame  # 0 .. nframes-1

    # trajectories up to this frame
    traj1_line.set_data(traj1_plot[:i+1, 0], traj1_plot[:i+1, 1])
    traj2_line.set_data(traj2_plot[:i+1, 0], traj2_plot[:i+1, 1])

    # current positions
    p1_point.set_data([traj1_plot[i, 0]], [traj1_plot[i, 1]])
    p2_point.set_data([traj2_plot[i, 0]], [traj2_plot[i, 1]])

    # time-series
    x = steps_plot[:i+1]
    E_line.set_data(x, E_plot[:i+1])
    res_line.set_data(x, res_plot[:i+1])
    P_line.set_data(x, P_plot[:i+1])
    L_line.set_data(x, L_plot[:i+1])

    return (
        traj1_line, traj2_line, p1_point, p2_point,
        E_line, res_line, P_line, L_line
    )

ani = animation.FuncAnimation(
    fig,
    update,
    frames=nframes,
    init_func=init,
    interval=20,
    blit=False,
)

from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save(str(movie_dir / f"two_body_movie_{save_name}.mp4"), writer=writer, dpi=120)
plt.close(fig)

raise

# ── set up figure with 3 panels ────────────────────────
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.3, 1.0],
    height_ratios=[1.0, 1.0],
    hspace=0.3,
    wspace=0.3
)

ax_traj = fig.add_subplot(gs[:, 0])   # left, spanning 2 rows
ax_E    = fig.add_subplot(gs[0, 1])  # top-right
ax_res  = fig.add_subplot(gs[1, 1], sharex=ax_E)  # bottom-right

# ── pre-set limits so they don't jump around ───────────
# Trajectory limits
xmin = min(traj1[:,0].min(), traj2[:,0].min())
xmax = max(traj1[:,0].max(), traj2[:,0].max())
ymin = min(traj1[:,1].min(), traj2[:,1].min())
ymax = max(traj1[:,1].max(), traj2[:,1].max())

padding = 0.1 * max(xmax - xmin, ymax - ymin)
ax_traj.set_xlim(xmin - padding, xmax + padding)
ax_traj.set_ylim(ymin - padding, ymax + padding)
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.set_xlabel("x")
ax_traj.set_ylabel("y")
ax_traj.set_title("Two-Body Trajectories")

# Energy limits
ax_E.set_xlim(0, steps)
ax_E.set_ylabel("Total Energy")
ax_E.set_title("Energy Conservation")

Emin, Emax = E.min(), E.max()
dE = Emax - Emin if Emax > Emin else 1.0
ax_E.set_ylim(Emin - 0.1*dE, Emax + 0.1*dE)
ax_E.grid(True)

# Residual limits
ax_res.set_xlim(0, steps)
ax_res.set_xlabel("Step")
ax_res.set_ylabel("Residual")
rmin, rmax = residual.min(), residual.max()
dr = rmax - rmin if rmax > rmin else 1.0
ax_res.set_ylim(rmin - 0.1*dr, rmax + 0.1*dr)
ax_res.axhline(0.0, color="black", linewidth=0.8)
ax_res.grid(True)

# ── lines / artists to update ─────────────────────────
# Trajectories (history)


max_frames = 1000  # or 500, 200, etc.
stride = max(1, steps // max_frames)

traj1_plot = traj1[::stride]
traj2_plot = traj2[::stride]
E_plot     = E[::stride]
res_plot   = residual[::stride]
nframes    = len(E_plot)
steps_plot = np.arange(steps)[::stride] 

E0 = E_plot[0]

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.3, 1.0],
    height_ratios=[1.0, 1.0],
    hspace=0.3,
    wspace=0.3
)

ax_traj = fig.add_subplot(gs[:, 0])
ax_E    = fig.add_subplot(gs[0, 1])
ax_res  = fig.add_subplot(gs[1, 1], sharex=ax_E)

# ── fixed limits ──
xmin = min(traj1[:,0].min(), traj2[:,0].min())
xmax = max(traj1[:,0].max(), traj2[:,0].max())
ymin = min(traj1[:,1].min(), traj2[:,1].min())
ymax = max(traj1[:,1].max(), traj2[:,1].max())
pad  = 0.1 * max(xmax - xmin, ymax - ymin)

ax_traj.set_xlim(xmin - pad, xmax + pad)
ax_traj.set_ylim(ymin - pad, ymax + pad)
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.set_xlabel("x")
ax_traj.set_ylabel("y")
ax_traj.set_title("Two-Body Trajectories")

ax_E.set_xlim(steps_plot[0], steps_plot[-1])
ax_E.set_ylabel("Total Energy")
ax_E.set_title("Energy Conservation")
Emin, Emax = E_plot.min(), E_plot.max()
dE = Emax - Emin if Emax > Emin else 1.0
ax_E.set_ylim(Emin - 0.1*dE, Emax + 0.1*dE)
ax_E.grid(True)

ax_res.set_xlim(steps_plot[0], steps_plot[-1])
ax_res.set_xlabel("Step")
ax_res.set_ylabel("Residual")
rmin, rmax = res_plot.min(), res_plot.max()
dr = rmax - rmin if rmax > rmin else 1.0
ax_res.set_ylim(rmin - 0.1*dr, rmax + 0.1*dr)
ax_res.axhline(0.0, color="black", linewidth=0.8)
ax_res.grid(True)

traj1_line, = ax_traj.plot([], [], label="Particle 1")
traj2_line, = ax_traj.plot([], [], label="Particle 2")
p1_point,   = ax_traj.plot([], [], marker="o", linestyle="None")
p2_point,   = ax_traj.plot([], [], marker="o", linestyle="None")
ax_traj.legend(loc="upper right")

E_line,   = ax_E.plot([], [], lw=1.0)
res_line, = ax_res.plot([], [], lw=1.0)

def init():
    traj1_line.set_data([], [])
    traj2_line.set_data([], [])
    p1_point.set_data([], [])
    p2_point.set_data([], [])
    E_line.set_data([], [])
    res_line.set_data([], [])
    return traj1_line, traj2_line, p1_point, p2_point, E_line, res_line

def update(frame):
    i = frame  # 0 .. nframes-1

    # trajectory up to this frame (still smooth since traj*_plot are dense enough)
    traj1_line.set_data(traj1_plot[:i+1, 0], traj1_plot[:i+1, 1])
    traj2_line.set_data(traj2_plot[:i+1, 0], traj2_plot[:i+1, 1])

    # current positions
    p1_point.set_data([traj1_plot[i, 0]], [traj1_plot[i, 1]])
    p2_point.set_data([traj2_plot[i, 0]], [traj2_plot[i, 1]])

    # energy & residual
    x = steps_plot[:i+1]
    E_line.set_data(x, E_plot[:i+1])
    res_line.set_data(x, res_plot[:i+1])

    return traj1_line, traj2_line, p1_point, p2_point, E_line, res_line

ani = animation.FuncAnimation(
    fig,
    update,
    frames=nframes,
    init_func=init,
    interval=20,
    #blit=True,
    blit=False,
)

from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save(f"../data/movie/two_body_movie_{suffix}.mp4", writer=writer, dpi=120)
plt.close(fig)

