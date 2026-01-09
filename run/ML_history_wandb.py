import argparse
import pathlib
import sys
import random
import time

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb

# Add parent directory of this file (project root)
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
print(f"Project root: {project_root}")
from src import (
    FullyConnectedNN,
    generate_IC,
    make_particle,
    stack_particles,
    loss_fn_batch,
    loss_fn_batch_history,
    loss_fn_batch_history_batch,
    HistoryBuffer,
)


parser = argparse.ArgumentParser(description="History-aware ML time-step predictor with W&B logging")
parser.add_argument("--epochs", "-n", type=int, default=1000, help="number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-2, help="optimizer weight decay")
parser.add_argument("--n-steps", type=int, default=10, help="integration steps per loss eval")
parser.add_argument("--rel-loss-bound", type=float, default=1e-5, help="relative loss bound")
parser.add_argument("--energy-threshold", type=float, default=2e-4, help="accept/reject energy threshold")
parser.add_argument("--E_lower", type=float, default=1e-6, help="lower energy bound for loss calculation")
parser.add_argument("--E_upper", type=float, default=1e-4, help="upper energy bound for loss calculation")
parser.add_argument("--L_lower", type=float, default=1e-4, help="lower angular momentum bound for loss calculation")
parser.add_argument("--L_upper", type=float, default=1e-2, help="upper angular momentum bound for loss calculation")
parser.add_argument("--eccentricity", "-e", type=float, default=0.9, help="eccentricity for generate_IC")
parser.add_argument("--semi-major", "-a", type=float, default=1.0, help="semi-major axis for generate_IC")
parser.add_argument("--history-len", type=int, default=3, help="number of past states to include")
parser.add_argument("--feature-type", type=str, choices=["basic", "rich", "delta_mag"], default="delta_mag", help="feature type per state")
parser.add_argument("--wandb-project", type=str, default="AITimeStepper", help="W&B project name")
parser.add_argument("--wandb-name", type=str, default="two_body_ML_integrator_history", help="W&B run name")
parser.add_argument("--optuna", action="store_true", help="optuna mode: print final metrics as JSON to stdout")
parser.add_argument("--save-name", "-s", type=str, default=None, help="base filename/dir under data/ to save outputs")
parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="compute device to use (auto, cpu, or cuda)")
parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64", help="tensor dtype for training")
parser.add_argument("--tf32", action="store_true", help="enable TF32 matmul on CUDA (faster, less precise)")
parser.add_argument("--compile", action="store_true", help="use torch.compile for model")
parser.add_argument("--detect-anomaly", action="store_true", help="enable autograd anomaly detection (slow)")
parser.add_argument("--replay-steps", type=int, default=1000, help="max replay optimization steps per epoch")
parser.add_argument("--replay-batch-size", type=int, default=512, help="replay batch size")
parser.add_argument("--debug", action="store_true", help="enable debug printouts (timing + progress)")
parser.add_argument("--debug-every", type=int, default=1, help="print debug info every N epochs")
parser.add_argument("--debug-replay-every", type=int, default=10, help="print debug info every N replay steps")
args = parser.parse_args()

dtype_map = {"float32": torch.float32, "float64": torch.float64}
dtype = dtype_map[args.dtype]
torch.set_default_dtype(dtype)
if args.detect_anomaly:
    torch.autograd.set_detect_anomaly(True)
# Choose device with graceful fallback
if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if args.tf32 and device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _sync_if_cuda():
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def _dbg(msg: str):
    if not args.debug:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DEBUG {ts}] {msg}", flush=True)

wandb.init(
    project=args.wandb_project,
    name=args.wandb_name,
    config={
        "optimizer": "adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.epochs,
        "n_steps": args.n_steps,
        "rel_loss_bound": args.rel_loss_bound,
        "energy_threshold": args.energy_threshold,
        "E_lower": args.E_lower,
        "E_upper": args.E_upper,
        "L_lower": args.L_lower,
        "L_upper": args.L_upper,
        "eccentricity": args.eccentricity,
        "semi_major": args.semi_major,
        "history_len": args.history_len,
        "feature_type": args.feature_type,
    },
)

_dbg(
    "run config: "
    f"epochs={args.epochs} n_steps={args.n_steps} lr={args.lr} wd={args.weight_decay} "
    f"E_bounds=({args.E_lower},{args.E_upper}) L_bounds=({args.L_lower},{args.L_upper}) "
    f"history_len={args.history_len} feature_type={args.feature_type} device={device}"
)

# prepare save directories
import os, json, datetime

if args.save_name is None:
    # create a timestamped folder name
    args.save_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

base_out = project_root / "data" / args.save_name
model_out = base_out / "model"
plot_out = base_out / "plot"
log_out = base_out / "logs"
os.makedirs(model_out, exist_ok=True)
os.makedirs(plot_out, exist_ok=True)
os.makedirs(log_out, exist_ok=True)

wandb.log({"save/base_out": str(base_out)})

ptcls, T = generate_IC(e=args.eccentricity, a=args.semi_major)
# attempt to place tensors on the chosen device; on failure, fall back to CPU
try:
    ptcls = torch.tensor(ptcls, device=device, dtype=dtype)
except RuntimeError as e:
    print(f"Warning: device {device} unavailable ({e}). Falling back to CPU.", file=sys.stderr)
    device = torch.device("cpu")
    ptcls = torch.tensor(ptcls, device=device, dtype=dtype)
particle = make_particle(ptcls, device=device, dtype=dtype)
particle.period = torch.tensor(T, device=device, dtype=dtype)
particle.current_time = torch.tensor(0, device=device, dtype=dtype)

# history buffer
history = HistoryBuffer(history_len=args.history_len, feature_type=args.feature_type)

# build a dummy feature vector to infer input_dim
with torch.no_grad():
    dummy_feat = history.features_for(particle)
    if dummy_feat.dim() == 1:
        input_dim = dummy_feat.numel()
    else:
        input_dim = dummy_feat.shape[-1]

hidden_dim = [200, 1000, 1000, 200]
output_dim = 2

model = FullyConnectedNN(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dims=hidden_dim,
    activation="tanh",
    dropout=0.2,
    output_positive=True,
).to(device)
model.to(dtype=dtype)
if args.compile:
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}); continuing without compile.", file=sys.stderr)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

num_epochs = args.epochs
energy_threshold = args.energy_threshold
n_steps = args.n_steps
rel_loss_bound = args.rel_loss_bound

# training history
epoch_log = []
accepted_states = []  # list of (particle, history_snapshot)
replay_batch_size = args.replay_batch_size
min_replay_size = 2

for epoch in range(num_epochs):
    epoch_t0 = time.perf_counter()
    if args.debug and (epoch % max(args.debug_every, 1) == 0):
        _dbg(f"epoch {epoch} start | accepted_states={len(accepted_states)}")

    # build loss using history-aware features
    loss_t0 = time.perf_counter()
    _sync_if_cuda()
    loss, logs, p_next = loss_fn_batch_history(
        model,
        particle,
        history,
        n_steps=n_steps,
        rel_loss_bound=rel_loss_bound,
        E_lower=args.E_lower,
        E_upper=args.E_upper,
        L_lower=args.L_lower,
        L_upper=args.L_upper,
        return_particle=True,
    )
    _sync_if_cuda()
    loss_t1 = time.perf_counter()
    if args.debug and (epoch % max(args.debug_every, 1) == 0):
        try:
            dt_val = float(logs.get("dt").item())
        except Exception:
            dt_val = float("nan")
        try:
            rel_dE_val_dbg = float(logs.get("rel_dE").item())
        except Exception:
            rel_dE_val_dbg = float("nan")
        try:
            rel_dL_val_dbg = float(logs.get("rel_dL_mean", torch.tensor(float('nan'))).item())
        except Exception:
            rel_dL_val_dbg = float("nan")
        _dbg(
            f"epoch {epoch} forward+loss done in {loss_t1 - loss_t0:.3f}s "
            f"| loss={float(loss.item()):.6e} dt={dt_val:.6e} rel_dE={rel_dE_val_dbg:.6e} rel_dL_mean={rel_dL_val_dbg:.6e}"
        )

    opt_t0 = time.perf_counter()
    _sync_if_cuda()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _sync_if_cuda()
    opt_t1 = time.perf_counter()
    if args.debug and (epoch % max(args.debug_every, 1) == 0):
        _dbg(f"epoch {epoch} backward+step done in {opt_t1 - opt_t0:.3f}s")

    rel_dE_val = logs["rel_dE"].item()
    accepted = rel_dE_val <= energy_threshold

    if args.debug and (epoch % max(args.debug_every, 1) == 0):
        _dbg(f"epoch {epoch} accept={int(accepted)} | rel_dE={float(rel_dE_val):.6e} threshold={energy_threshold:.6e}")

    # record
    epoch_log.append({
        "epoch": epoch,
        "rel_dE": rel_dE_val,
        "dt": logs["dt"].item(),
        "E0": logs["E0"].item(),
        "loss": loss.item(),
        "accepted": int(accepted),
        # additional rel_dE / loss metrics
        "rel_dE_mean": float(logs.get("rel_dE_mean", torch.tensor(float('nan'))).item()),
        "rel_dE_next": float(logs.get("rel_dE_next", torch.tensor(float('nan'))).item()),
        "rel_dE_last": float(logs.get("rel_dE_last", torch.tensor(float('nan'))).item()),
        "rel_dE_max": float(logs.get("rel_dE_max", torch.tensor(float('nan'))).item()),
        "loss_energy": float(logs.get("loss_energy", torch.tensor(float('nan'))).item()),
        "loss_energy_mean": float(logs.get("loss_energy_mean", torch.tensor(float('nan'))).item()),
        "loss_energy_next": float(logs.get("loss_energy_next", torch.tensor(float('nan'))).item()),
        "loss_energy_last": float(logs.get("loss_energy_last", torch.tensor(float('nan'))).item()),
        "loss_energy_max": float(logs.get("loss_energy_max", torch.tensor(float('nan'))).item()),
        # angular momentum metrics
        "rel_dL_mean": float(logs.get("rel_dL_mean", torch.tensor(float('nan'))).item()),
        "rel_dL_next": float(logs.get("rel_dL_next", torch.tensor(float('nan'))).item()),
        "rel_dL_last": float(logs.get("rel_dL_last", torch.tensor(float('nan'))).item()),
        "rel_dL_max": float(logs.get("rel_dL_max", torch.tensor(float('nan'))).item()),
        "loss_ang": float(logs.get("loss_ang", torch.tensor(float('nan'))).item()),
        "loss_pred": float(logs.get("loss_pred", torch.tensor(float('nan'))).item()),
        "loss_dt": float(logs.get("loss_dt", torch.tensor(float('nan'))).item()),
    })

    # update particle if accepted, then push the pre-update state into history
    # so next epoch sees it as part of the past.
    if accepted:
        # push current state BEFORE update into history
        history.push(particle)
        particle = p_next.clone_detached()
        # store accepted state with its own history snapshot for replay
        accepted_states.append((particle.clone_detached(), history.clone()))

    # ===== Replay Buffer Training (train on batch of accepted states) =====
    if len(accepted_states) >= min_replay_size:
        rep_outer_t0 = time.perf_counter()
        if args.debug and (epoch % max(args.debug_every, 1) == 0):
            _dbg("enter replay buffer training")

        batch_states = random.sample(
            accepted_states,
            k=min(replay_batch_size, len(accepted_states))
        )

        batch_states_detached = [p.clone_detached() for (p, _) in batch_states]
        stack_t0 = time.perf_counter()
        batch_state = stack_particles(batch_states_detached)
        stack_t1 = time.perf_counter()
        batch_histories = [h for (_, h) in batch_states]

        if args.debug and (epoch % max(args.debug_every, 1) == 0):
            try:
                pos_shape = tuple(batch_state.position.shape)
            except Exception:
                pos_shape = None
            _dbg(
                f"replay batch built in {stack_t1 - stack_t0:.3f}s "
                f"| batch_size={len(batch_states_detached)} pos_shape={pos_shape} histories={len(batch_histories)}"
            )

        max_replay_steps = args.replay_steps

        for inner_step in range(max_replay_steps):
            # LBFGS requires a closure
            if isinstance(optimizer, torch.optim.LBFGS):
                stored_rep = {}

                if args.debug and (epoch % max(args.debug_every, 1) == 0) and inner_step == 0:
                    _dbg("replay optimizer is LBFGS (closure may run multiple times per step)")

                def closure_rep():
                    stored_rep['loss'], stored_rep['logs'], _ = loss_fn_batch_history_batch(
                        model,
                        batch_state,
                        batch_histories,
                        n_steps=n_steps,
                        rel_loss_bound=rel_loss_bound,
                        E_lower=args.E_lower,
                        E_upper=args.E_upper,
                        L_lower=args.L_lower,
                        L_upper=args.L_upper,
                        return_particle=False,
                    )
                    optimizer.zero_grad()
                    stored_rep['loss'].backward()
                    return stored_rep['loss']

                optimizer.step(closure_rep)
                replay_loss = stored_rep['loss']
                logs_rep = stored_rep['logs']
            else:
                rep_step_t0 = time.perf_counter()
                _sync_if_cuda()
                optimizer.zero_grad()
                replay_loss, logs_rep, _ = loss_fn_batch_history_batch(
                    model,
                    batch_state,
                    batch_histories,
                    n_steps=n_steps,
                    rel_loss_bound=rel_loss_bound,
                    E_lower=args.E_lower,
                    E_upper=args.E_upper,
                    L_lower=args.L_lower,
                    L_upper=args.L_upper,
                    return_particle=False,
                )
                replay_loss.backward()
                optimizer.step()
                _sync_if_cuda()
                rep_step_t1 = time.perf_counter()

                if args.debug and (epoch % max(args.debug_every, 1) == 0) and (inner_step % max(args.debug_replay_every, 1) == 0):
                    try:
                        rep_rel_dE = float(logs_rep.get('rel_dE', torch.tensor(float('nan'))).item())
                    except Exception:
                        rep_rel_dE = float('nan')
                    try:
                        rep_rel_dL = float(logs_rep.get('rel_dL_mean', torch.tensor(float('nan'))).item())
                    except Exception:
                        rep_rel_dL = float('nan')
                    _dbg(
                        f"replay step {inner_step}/{max_replay_steps} done in {rep_step_t1 - rep_step_t0:.3f}s "
                        f"| replay_loss={float(replay_loss.item()):.6e} mean_rel_dE={rep_rel_dE:.6e} mean_rel_dL={rep_rel_dL:.6e}"
                    )

            rel_dE_full = logs_rep.get('rel_dE_full', logs_rep.get('rel_dE'))

            # check convergence
            try:
                if torch.is_tensor(rel_dE_full) and (rel_dE_full <= energy_threshold).all():
                    wandb.log({
                        "replay/converged_step": inner_step + 1,
                        "replay/max_rel_dE": float(rel_dE_full.max().item()),
                    })
                    break
            except Exception:
                pass

            if inner_step % 10 == 0:
                try:
                    mean_rel_dE = logs_rep.get('rel_dE', torch.tensor(float('nan'))).item()
                except Exception:
                    mean_rel_dE = float('nan')
                try:
                    max_rel = float(rel_dE_full.max().item()) if torch.is_tensor(rel_dE_full) else float('nan')
                except Exception:
                    max_rel = float('nan')

                wandb.log({
                    "replay/loss": replay_loss.item(),
                    "replay/mean_rel_dE": mean_rel_dE,
                    "replay/max_rel_dE": max_rel,
                    "replay/mean_rel_dL": float(logs_rep.get('rel_dL_mean', torch.tensor(float('nan'))).item()) if torch.is_tensor(logs_rep.get('rel_dL_mean', None)) else float('nan'),
                    "replay/loss_ang": float(logs_rep.get('loss_ang', torch.tensor(float('nan'))).item()) if torch.is_tensor(logs_rep.get('loss_ang', None)) else float('nan'),
                    "replay/step": inner_step,
                })

        if args.debug and (epoch % max(args.debug_every, 1) == 0):
            rep_outer_t1 = time.perf_counter()
            _dbg(f"exit replay buffer training | elapsed={rep_outer_t1 - rep_outer_t0:.3f}s")

    # log to wandb
    if args.debug and (epoch % max(args.debug_every, 1) == 0):
        _dbg("wandb.log(epoch metrics) start")
    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        "rel_dE": rel_dE_val,
        "dt": logs["dt"].item(),
        "E0": logs["E0"].item(),
        "accepted": int(accepted),
        "history/len": args.history_len,
        # extra rel_dE summaries
        "rel_dE_mean": logs.get("rel_dE_mean", torch.tensor(float('nan'))).item(),
        "rel_dE_next": logs.get("rel_dE_next", torch.tensor(float('nan'))).item(),
        "rel_dE_last": logs.get("rel_dE_last", torch.tensor(float('nan'))).item(),
        "rel_dE_max": logs.get("rel_dE_max", torch.tensor(float('nan'))).item(),
        # loss energy summaries
        "loss_energy": logs.get("loss_energy", torch.tensor(float('nan'))).item(),
        "loss_energy_mean": logs.get("loss_energy_mean", torch.tensor(float('nan'))).item(),
        "loss_energy_next": logs.get("loss_energy_next", torch.tensor(float('nan'))).item(),
        "loss_energy_last": logs.get("loss_energy_last", torch.tensor(float('nan'))).item(),
        "loss_energy_max": logs.get("loss_energy_max", torch.tensor(float('nan'))).item(),
        # angular momentum summaries
        "rel_dL_mean": logs.get("rel_dL_mean", torch.tensor(float('nan'))).item(),
        "rel_dL_next": logs.get("rel_dL_next", torch.tensor(float('nan'))).item(),
        "rel_dL_last": logs.get("rel_dL_last", torch.tensor(float('nan'))).item(),
        "rel_dL_max": logs.get("rel_dL_max", torch.tensor(float('nan'))).item(),
        "loss_ang": logs.get("loss_ang", torch.tensor(float('nan'))).item(),
        "loss_pred": logs.get("loss_pred", torch.tensor(float('nan'))).item(),
        "loss_dt": logs.get("loss_dt", torch.tensor(float('nan'))).item(),
    })
    if args.debug and (epoch % max(args.debug_every, 1) == 0):
        _dbg("wandb.log(epoch metrics) done")

    if args.debug and (epoch % max(args.debug_every, 1) == 0):
        epoch_t1 = time.perf_counter()
        _dbg(f"epoch {epoch} end | elapsed={epoch_t1 - epoch_t0:.3f}s")

    # try logging full distribution as histogram (if available)
    try:
        rel_full = logs.get("rel_dE_full", None)
        if rel_full is not None:
            wandb.log({"rel_dE_full_hist": wandb.Histogram(rel_full.detach().cpu().numpy())})
    except Exception:
        pass

    # checkpoint every 100 epochs and final epoch
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        ckpt_path = model_out / f"model_epoch_{epoch:04d}.pt"
        try:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "logs": {k: (v.item() if hasattr(v, 'item') else v) for k,v in logs.items()},
            }, str(ckpt_path))
            try:
                art = wandb.Artifact(name=f"model_epoch_{epoch:04d}", type="model")
                art.add_file(str(ckpt_path))
                wandb.log_artifact(art)
            except Exception:
                # fallback to save (may symlink); keep silent
                try:
                    wandb.save(str(ckpt_path), base_path=str(model_out))
                except Exception:
                    pass
        except Exception:
            pass

# summary
accepted_all = np.array([x["accepted"] for x in epoch_log], dtype=bool)
acceptance_rate = float(accepted_all.mean()) if len(accepted_all) else 0.0

wandb.log({
    "summary/acceptance_rate": acceptance_rate,
    "summary/total_epochs": num_epochs,
})
wandb.finish()

if args.optuna:
    try:
        import json
        final = epoch_log[-1] if epoch_log else {"loss": float("inf"), "dt": float("nan")}
        out = {"loss": float(final["loss"]), "time_step": float(final["dt"])}
        print("OPTUNA_RESULT_JSON:" + json.dumps(out), flush=True)
    except Exception:
        pass

print("History-aware training complete.")

# save epoch log
try:
    hist_path = log_out / "history.json"
    with open(hist_path, 'w') as fh:
        json.dump(epoch_log, fh, indent=2)
    print(f"Saved history JSON to {hist_path}")
    try:
        art = wandb.Artifact(name=f"history_{args.save_name}", type="logs")
        art.add_file(str(hist_path))
        wandb.log_artifact(art)
    except Exception:
        try:
            wandb.save(str(hist_path), base_path=str(log_out))
        except Exception:
            pass
except Exception:
    pass
