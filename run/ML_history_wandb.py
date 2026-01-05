import argparse
import pathlib
import sys
import random

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
parser.add_argument("--eccentricity", "-e", type=float, default=0.9, help="eccentricity for generate_IC")
parser.add_argument("--semi-major", "-a", type=float, default=1.0, help="semi-major axis for generate_IC")
parser.add_argument("--history-len", type=int, default=3, help="number of past states to include")
parser.add_argument("--feature-type", type=str, choices=["basic", "rich"], default="basic", help="feature type per state")
parser.add_argument("--wandb-project", type=str, default="AITimeStepper", help="W&B project name")
parser.add_argument("--wandb-name", type=str, default="two_body_ML_integrator_history", help="W&B run name")
parser.add_argument("--optuna", action="store_true", help="optuna mode: print final metrics as JSON to stdout")
parser.add_argument("--save-name", "-s", type=str, default=None, help="base filename/dir under data/ to save outputs")
parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="compute device to use (auto, cpu, or cuda)")
args = parser.parse_args()

dtype = torch.double
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)
# Choose device with graceful fallback
if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

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
        "eccentricity": args.eccentricity,
        "semi_major": args.semi_major,
        "history_len": args.history_len,
        "feature_type": args.feature_type,
    },
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

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

num_epochs = args.epochs
energy_threshold = args.energy_threshold
n_steps = args.n_steps
rel_loss_bound = args.rel_loss_bound

# training history
epoch_log = []
accepted_states = []  # list of (particle, history_snapshot)
replay_batch_size = 512
min_replay_size = 2

for epoch in range(num_epochs):
    # build loss using history-aware features
    loss, logs, p_next = loss_fn_batch_history(
        model,
        particle,
        history,
        n_steps=n_steps,
        rel_loss_bound=rel_loss_bound,
        E_lower=args.E_lower,
        E_upper=args.E_upper,
        return_particle=True,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    rel_dE_val = logs["rel_dE"].item()
    accepted = rel_dE_val <= energy_threshold

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
        batch_states = random.sample(
            accepted_states,
            k=min(replay_batch_size, len(accepted_states))
        )

        batch_states_detached = [p.clone_detached() for (p, _) in batch_states]
        batch_state = stack_particles(batch_states_detached)
        batch_histories = [h for (_, h) in batch_states]

        max_replay_steps = 1000

        for inner_step in range(max_replay_steps):
            # LBFGS requires a closure
            if isinstance(optimizer, torch.optim.LBFGS):
                stored_rep = {}

                def closure_rep():
                    stored_rep['loss'], stored_rep['logs'], _ = loss_fn_batch_history_batch(
                        model,
                        batch_state,
                        batch_histories,
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
                replay_loss, logs_rep, _ = loss_fn_batch_history_batch(
                    model,
                    batch_state,
                    batch_histories,
                    n_steps=n_steps,
                    rel_loss_bound=rel_loss_bound,
                    return_particle=False,
                )
                replay_loss.backward()
                optimizer.step()

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
                    "replay/step": inner_step,
                })

    # log to wandb
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
        "loss_pred": logs.get("loss_pred", torch.tensor(float('nan'))).item(),
        "loss_dt": logs.get("loss_dt", torch.tensor(float('nan'))).item(),
    })

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

