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
    Config,
    FullyConnectedNN,
    generate_IC,
    make_particle,
    stack_particles,
    loss_fn_batch_history_batch,
    HistoryBuffer,
    save_checkpoint,
    ModelAdapter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="History-aware ML time-step predictor with batched multi-orbit training")
    Config.add_cli_args(
        parser,
        include=["train", "bounds", "history", "device", "logging", "multi"],
    )
    parser.add_argument("--wandb-project", type=str, default="AITimeStepper", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default="two_body_ML_integrator_history_multi", help="W&B run name")
    parser.add_argument("--optuna", action="store_true", help="optuna mode: print final metrics as JSON to stdout")
    return parser.parse_args()


def _sync_if_cuda(device: torch.device):
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def _dbg(config, msg: str):
    if not config.debug:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DEBUG {ts}] {msg}", flush=True)


def _split_particle_from_batch(batch_p: "ParticleTorch", idx: int) -> "ParticleTorch":
    pos = batch_p.position[idx]
    vel = batch_p.velocity[idx]
    mass = batch_p.mass[idx] if batch_p.mass.dim() > 1 else batch_p.mass
    dt_field = batch_p.dt
    if torch.is_tensor(dt_field) and dt_field.dim() > 0:
        dt_i = dt_field[idx]
    else:
        dt_i = dt_field
    soft = getattr(batch_p, "softening", 0.0)
    ct = getattr(batch_p, "current_time", torch.tensor(0.0, device=pos.device, dtype=pos.dtype))
    out = type(batch_p).from_tensors(
        mass=mass.clone(),
        position=pos.clone(),
        velocity=vel.clone(),
        dt=dt_i.clone() if torch.is_tensor(dt_i) else torch.tensor(dt_i, device=pos.device, dtype=pos.dtype),
        softening=soft,
    )
    out.current_time = ct.clone() if torch.is_tensor(ct) else torch.tensor(ct, device=pos.device, dtype=pos.dtype)
    return out


def main():
    args = parse_args()
    config = Config.from_dict(vars(args))
    config.validate()
    if config.seed is None:
        config.seed = 42
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dtype = config.resolve_dtype()
    torch.set_default_dtype(dtype)
    device = config.resolve_device()
    config.apply_torch_settings(device)

    print(f"Using device: {device}")
    if config.tf32 and device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config.as_wandb_dict(),
    )

    _dbg(config, "starting multi-orbit setup")

    # prepare save directories
    import os, json, datetime

    if config.save_name is None:
        config.save_name = f"run_multi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    base_out = project_root / "data" / config.save_name
    model_out = base_out / "model"
    log_out = base_out / "logs"
    os.makedirs(model_out, exist_ok=True)
    os.makedirs(log_out, exist_ok=True)

    # Verify directories are writable
    try:
        test_file = model_out / ".write_test"
        test_file.touch()
        test_file.unlink()
        _dbg(args, f"save directories verified: base={base_out}")
    except Exception as exc:
        print(f"WARNING: save directory {model_out} may not be writable: {exc}", file=sys.stderr)

    wandb.log({"save/base_out": str(base_out)})

    adapter = ModelAdapter(config, device=device, dtype=dtype)

    # build initial batch of orbits
    particles = []
    histories = []
    orbit_meta = []
    for i in range(config.num_orbits):
        e_i = float(np.random.uniform(config.e_min, config.e_max))
        a_i = float(np.random.uniform(config.a_min, config.a_max))
        ptcls, T_i = generate_IC(e=e_i, a=a_i)
        try:
            ptcls_t = torch.tensor(ptcls, device=device, dtype=dtype)
        except RuntimeError as exc:
            print(f"Warning: device {device} unavailable ({exc}). Falling back to CPU.", file=sys.stderr)
            device = torch.device("cpu")
            ptcls_t = torch.tensor(ptcls, device=device, dtype=dtype)
        particle = make_particle(ptcls_t, device=device, dtype=dtype)
        particle.period = torch.tensor(T_i, device=device, dtype=dtype)
        particle.current_time = torch.tensor(0.0, device=device, dtype=dtype)
        hb = HistoryBuffer(history_len=config.history_len, feature_type=config.feature_type)
        hb.push(particle.clone_detached())
        particles.append(particle)
        histories.append(hb)
        orbit_meta.append({"e": e_i, "a": a_i, "T": float(T_i)})

    batch_state_init = stack_particles(particles)
    input_dim = adapter.input_dim_from_state(batch_state_init, histories=histories)

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
    if config.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"Warning: torch.compile failed ({exc}); continuing without compile.", file=sys.stderr)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    num_epochs = config.epochs
    energy_threshold = config.energy_threshold
    n_steps = config.n_steps
    rel_loss_bound = config.rel_loss_bound

    epoch_log = []

    _dbg(config, f"initialized {len(particles)} orbits | input_dim={input_dim}")

    for epoch in range(num_epochs):
        epoch_t0 = time.perf_counter()
        if config.debug and (epoch % max(config.debug_every, 1) == 0):
            _dbg(config, f"epoch {epoch} start")

        batch_state = stack_particles([p.clone_detached() for p in particles])

        step_t0 = time.perf_counter()
        _sync_if_cuda(device)
        optimizer.zero_grad()
        loss, logs, p_next = loss_fn_batch_history_batch(
            model,
            batch_state,
            histories,
            n_steps=n_steps,
            rel_loss_bound=rel_loss_bound,
            E_lower=config.E_lower,
            E_upper=config.E_upper,
            L_lower=config.L_lower,
            L_upper=config.L_upper,
            return_particle=True,
        )
        loss.backward()
        optimizer.step()
        _sync_if_cuda(device)
        step_t1 = time.perf_counter()

        rel_dE_full = logs.get("rel_dE_full", None)
        if rel_dE_full is None:
            accept_mask = torch.ones(len(particles), dtype=torch.bool, device=device)
        else:
            if rel_dE_full.dim() == 0:
                accept_mask = torch.full((len(particles),), bool(rel_dE_full <= energy_threshold), device=device)
            else:
                accept_mask = rel_dE_full <= energy_threshold
        acceptance_rate = float(accept_mask.float().mean().item())

        p_next_det = p_next.clone_detached()
        for idx in range(len(particles)):
            histories[idx].push(particles[idx].clone_detached())
            if bool(accept_mask[idx].item()):
                particles[idx] = _split_particle_from_batch(p_next_det, idx)

        try:
            mean_rel_dE = float(logs.get("rel_dE", torch.tensor(float("nan"))).mean().item())
        except Exception:
            mean_rel_dE = float("nan")

        epoch_log.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "rel_dE": mean_rel_dE,
            "acceptance_rate": acceptance_rate,
        })

        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "rel_dE": mean_rel_dE,
            "dt": float(logs.get("dt", torch.tensor(float("nan"))).mean().item()) if hasattr(logs.get("dt", None), "mean") else float("nan"),
            "rel_dE_mean": float(logs.get("rel_dE_mean", torch.tensor(float("nan"))).mean().item()) if hasattr(logs.get("rel_dE_mean", None), "mean") else float("nan"),
            "rel_dE_max": float(logs.get("rel_dE_max", torch.tensor(float("nan"))).max().item()) if hasattr(logs.get("rel_dE_max", None), "max") else float("nan"),
            "rel_dL_mean": float(logs.get("rel_dL_mean", torch.tensor(float("nan"))).mean().item()) if hasattr(logs.get("rel_dL_mean", None), "mean") else float("nan"),
            "loss_energy": float(logs.get("loss_energy", torch.tensor(float("nan"))).mean().item()) if hasattr(logs.get("loss_energy", None), "mean") else float("nan"),
            "loss_ang": float(logs.get("loss_ang", torch.tensor(float("nan"))).mean().item()) if hasattr(logs.get("loss_ang", None), "mean") else float("nan"),
            "acceptance_rate": acceptance_rate,
            "batch_time": step_t1 - step_t0,
        })

        if config.debug and (epoch % max(config.debug_every, 1) == 0):
            _dbg(config, f"epoch {epoch} loss={float(loss.item()):.6e} accept_rate={acceptance_rate:.3f} step_time={step_t1 - step_t0:.3f}s")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ckpt_path = model_out / f"model_epoch_{epoch:04d}.pt"
            try:
                _dbg(config, f"saving checkpoint to {ckpt_path}")
                # Convert logs to saveable format (scalars -> item(), batched tensors -> list)
                logs_save = {}
                for k, v in logs.items():
                    if torch.is_tensor(v):
                        if v.numel() == 1:
                            logs_save[k] = v.item()
                        else:
                            logs_save[k] = v.detach().cpu().tolist()
                    else:
                        logs_save[k] = v
                
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    loss=loss,
                    logs=logs_save,
                    config=config,
                    extra={"orbit_meta": orbit_meta},
                )
                _dbg(config, f"checkpoint saved successfully to {ckpt_path}")
                try:
                    art = wandb.Artifact(name=f"model_epoch_{epoch:04d}", type="model")
                    art.add_file(str(ckpt_path))
                    wandb.log_artifact(art)
                except Exception as art_exc:
                    _dbg(config, f"wandb artifact upload failed: {art_exc}; trying fallback save")
                    try:
                        wandb.save(str(ckpt_path), base_path=str(model_out))
                        _dbg(config, f"wandb fallback save successful")
                    except Exception as fallback_exc:
                        print(f"WARNING: wandb save failed: {fallback_exc}", file=sys.stderr)
            except Exception as save_exc:
                print(f"ERROR: failed to save checkpoint {ckpt_path}: {save_exc}", file=sys.stderr)

        if config.debug and (epoch % max(config.debug_every, 1) == 0):
            epoch_t1 = time.perf_counter()
            _dbg(config, f"epoch {epoch} end | elapsed={epoch_t1 - epoch_t0:.3f}s")

    acceptance_values = np.array([x.get("acceptance_rate", 0.0) for x in epoch_log], dtype=float)
    overall_acceptance = float(acceptance_values.mean()) if len(acceptance_values) else 0.0

    wandb.log({
        "summary/acceptance_rate": overall_acceptance,
        "summary/total_epochs": num_epochs,
    })
    wandb.finish()

    if args.optuna:
        try:
            import json
            final = epoch_log[-1] if epoch_log else {"loss": float("inf"), "dt": float("nan")}
            out = {"loss": float(final.get("loss", float("inf"))), "time_step": float(final.get("dt", float("nan")))}
            print("OPTUNA_RESULT_JSON:" + json.dumps(out), flush=True)
        except Exception:
            pass

    print("Multi-orbit history-aware training complete.")

    try:
        import json
        hist_path = log_out / "history.json"
        with open(hist_path, "w") as fh:
            json.dump(epoch_log, fh, indent=2)
        print(f"Saved history JSON to {hist_path}")
        try:
            art = wandb.Artifact(name=f"history_{config.save_name}", type="logs")
            art.add_file(str(hist_path))
            wandb.log_artifact(art)
        except Exception as art_exc:
            print(f"WARNING: wandb history artifact upload failed: {art_exc}", file=sys.stderr)
            try:
                wandb.save(str(hist_path), base_path=str(log_out))
            except Exception as fallback_exc:
                print(f"WARNING: wandb history fallback save failed: {fallback_exc}", file=sys.stderr)
    except Exception as hist_exc:
        print(f"ERROR: failed to save history JSON: {hist_exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
