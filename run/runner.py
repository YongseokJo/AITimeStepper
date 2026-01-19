import argparse
import json
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import torch

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import (
    Config,
    FullyConnectedNN,
    HistoryBuffer,
    ModelAdapter,
    load_model_state,
    load_config_from_checkpoint,
    loss_fn_batch,
    loss_fn_batch_history,
    loss_fn_batch_history_batch,
    make_particle,
    save_checkpoint,
    stack_particles,
)
from src.external_potentials import PointMassTidalField
from simulators.nbody_simulator import (
    evolve_particles,
    evolve_particles_ml,
    generate_random_ic,
    total_energy,
    total_momentum,
    total_angular_momentum,
)
from simulators.particle import Particle as SimParticle


def _load_ic(path: str) -> np.ndarray:
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    return np.loadtxt(p)


def _make_sim_particles(ptcls: np.ndarray) -> list[SimParticle]:
    particles = []
    for row in ptcls:
        mass = float(row[0])
        dim = (len(row) - 1) // 2
        pos = row[1 : 1 + dim]
        vel = row[1 + dim : 1 + 2 * dim]
        particles.append(SimParticle(mass=mass, position=pos, velocity=vel))
    return particles


def _external_accel_from_config(config: Config):
    if config.external_field_mass is None or config.external_field_position is None:
        return None
    field = PointMassTidalField(M=config.external_field_mass, R0=config.external_field_position)

    def _accel(pos: np.ndarray, time: float):
        pos_t = torch.as_tensor(pos, dtype=torch.float64)
        acc = field.acceleration(pos_t.unsqueeze(0), time).squeeze(0)
        return acc.detach().cpu().numpy()

    return _accel


def run_simulation(config: Config) -> None:
    if config.integrator_mode == "history" and (config.history_len is None or config.history_len <= 0):
        raise ValueError("history integrator_mode requires history_len > 0")
    if config.model_path:
        ckpt_config = load_config_from_checkpoint(config.model_path)
        if ckpt_config is not None:
            if config.history_len == 0 and ckpt_config.history_len:
                config.history_len = ckpt_config.history_len
            if config.feature_type == Config.feature_type and ckpt_config.feature_type:
                config.feature_type = ckpt_config.feature_type
            if config.dtype == Config.dtype and ckpt_config.dtype:
                config.dtype = ckpt_config.dtype

    config.validate()
    rng_seed = config.seed

    if config.extra.get("ic_path"):
        ptcls = _load_ic(config.extra["ic_path"])
    else:
        ptcls = generate_random_ic(
            num_particles=config.num_particles,
            dim=config.dim,
            mass=config.mass,
            pos_scale=config.pos_scale,
            vel_scale=config.vel_scale,
            seed=rng_seed,
        )

    particles = _make_sim_particles(ptcls)
    device = config.resolve_device()
    dtype = config.resolve_dtype()
    adapter = ModelAdapter(config, device=device, dtype=dtype)

    if config.integrator_mode in ("ml", "history"):
        model = FullyConnectedNN(
            input_dim=adapter.input_dim_from_state(
                make_particle(ptcls, device=device, dtype=dtype),
                history_buffer=adapter.history_buffer,
            ),
            output_dim=2,
            hidden_dims=[200, 1000, 1000, 200],
            activation="tanh",
            dropout=0.2,
            output_positive=True,
        ).to(device)
        load_model_state(model, config.model_path, map_location=device)
        model = model.to(dtype=dtype)
        for p in particles:
            p.update_model(model, device)
            if adapter.history_buffer is not None:
                p.attach_history_buffer(adapter.history_buffer)
    else:
        model = None

    external_accel = _external_accel_from_config(config)

    steps = config.steps if config.steps and config.steps > 0 else 1000
    dt = config.dt if config.dt and config.dt > 0 else None

    energies = []
    momenta = []
    angular = []

    pos0 = np.stack([p.position for p in particles], axis=0)
    vel0 = np.stack([p.velocity for p in particles], axis=0)
    mass0 = np.array([p.mass for p in particles], dtype=float)
    energy_initial = total_energy(pos0, vel0, mass0)
    momentum_initial = total_momentum(vel0, mass0)
    angular_initial = total_angular_momentum(pos0, vel0, mass0)

    max_steps = steps if config.duration is None else None
    step_count = 0
    while True:
        if max_steps is not None and step_count >= max_steps:
            break
        if config.duration is not None:
            current_time = float(getattr(particles[0], "current_time", 0.0))
            if current_time >= config.duration:
                break
        if config.integrator_mode in ("ml", "history"):
            history_buffer = adapter.history_buffer if config.integrator_mode == "history" else None
            evolve_particles_ml(
                particles,
                model,
                history_buffer=history_buffer,
                feature_mode=adapter.feature_mode(),
                eps=1e-6,
                G=1.0,
                adapter=adapter,
                config=config,
                external_acceleration=external_accel,
            )
        else:
            evolve_particles(
                particles,
                dt=dt,
                G=1.0,
                external_acceleration=external_accel,
            )

        pos = np.stack([p.position for p in particles], axis=0)
        vel = np.stack([p.velocity for p in particles], axis=0)
        mass = np.array([p.mass for p in particles], dtype=float)
        energies.append(total_energy(pos, vel, mass))
        momenta.append(total_momentum(vel, mass))
        angular.append(total_angular_momentum(pos, vel, mass))
        step_count += 1

    energy_final = energies[-1] if energies else None
    momentum_final = momenta[-1] if momenta else None
    angular_final = angular[-1] if angular else None
    if energy_initial is not None and energy_initial != 0.0 and energy_final is not None:
        energy_residual = (energy_final - energy_initial) / energy_initial
    else:
        energy_residual = None
    if momentum_final is not None:
        mom_init_norm = float(np.linalg.norm(momentum_initial))
        mom_diff_norm = float(np.linalg.norm(momentum_final - momentum_initial))
        momentum_residual = mom_diff_norm / (mom_init_norm + 1e-12)
    else:
        momentum_residual = None
    if angular_final is not None:
        if np.isscalar(angular_initial):
            ang_init = float(angular_initial)
            ang_final = float(angular_final)
            if ang_init != 0.0:
                angular_residual = (ang_final - ang_init) / ang_init
            else:
                angular_residual = None
        else:
            ang_init_norm = float(np.linalg.norm(angular_initial))
            ang_diff_norm = float(np.linalg.norm(angular_final - angular_initial))
            angular_residual = ang_diff_norm / (ang_init_norm + 1e-12)
    else:
        angular_residual = None
    print(
        json.dumps(
            {
                "steps": step_count,
                "energy_initial": energy_initial,
                "energy_final": energy_final,
                "energy_residual": energy_residual,
                "momentum_initial": momentum_initial.tolist() if momentum_initial is not None else None,
                "momentum_final": momentum_final.tolist() if momentum_final is not None else None,
                "momentum_residual": momentum_residual,
                "angular_momentum_initial": angular_initial.tolist() if hasattr(angular_initial, "tolist") else angular_initial,
                "angular_momentum_final": angular_final.tolist() if hasattr(angular_final, "tolist") else angular_final,
                "angular_momentum_residual": angular_residual,
            },
            indent=2,
        )
    )


def run_training(config: Config) -> None:
    if config.integrator_mode == "history" and (config.history_len is None or config.history_len <= 0):
        raise ValueError("history integrator_mode requires history_len > 0")
    config.validate()
    device = config.resolve_device()
    dtype = config.resolve_dtype()
    torch.set_default_dtype(dtype)
    config.apply_torch_settings(device)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    adapter = ModelAdapter(config, device=device, dtype=dtype)

    def _build_particle() -> torch.Tensor:
        ptcls = generate_random_ic(
            num_particles=config.num_particles,
            dim=config.dim,
            mass=config.mass,
            pos_scale=config.pos_scale,
            vel_scale=config.vel_scale,
            seed=config.seed,
        )
        return ptcls

    if config.num_orbits > 1:
        particles = []
        histories = []
        for i in range(config.num_orbits):
            ptcls = _build_particle()
            particle = make_particle(ptcls, device=device, dtype=dtype)
            particle.current_time = torch.tensor(0.0, device=device, dtype=dtype)
            particles.append(particle)
            if adapter.history_enabled:
                hb = HistoryBuffer(history_len=config.history_len, feature_type=config.feature_type)
                hb.push(particle.clone_detached())
                histories.append(hb)
        batch_state = stack_particles(particles)
        input_dim = adapter.input_dim_from_state(batch_state, histories=histories) if adapter.history_enabled else adapter.input_dim_from_state(batch_state)
    else:
        ptcls = _build_particle()
        particle = make_particle(ptcls, device=device, dtype=dtype)
        particle.current_time = torch.tensor(0.0, device=device, dtype=dtype)
        input_dim = adapter.input_dim_from_state(particle, history_buffer=adapter.history_buffer)
        histories = None
        batch_state = None

    model = FullyConnectedNN(
        input_dim=input_dim,
        output_dim=2,
        hidden_dims=[200, 1000, 1000, 200],
        activation="tanh",
        dropout=0.2,
        output_positive=True,
    ).to(device)
    model.to(dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_time = time.perf_counter()
    epoch = 0
    while epoch < config.epochs:
        if config.duration is not None and (time.perf_counter() - start_time) >= config.duration:
            break
        if config.num_orbits > 1:
            if adapter.history_enabled:
                loss, logs, _ = loss_fn_batch_history_batch(
                    model,
                    batch_state,
                    histories,
                    n_steps=config.n_steps,
                    rel_loss_bound=config.rel_loss_bound,
                    E_lower=config.E_lower,
                    E_upper=config.E_upper,
                    L_lower=config.L_lower,
                    L_upper=config.L_upper,
                    return_particle=True,
                )
            else:
                loss, logs, _ = loss_fn_batch(
                    model,
                    batch_state,
                    n_steps=config.n_steps,
                    rel_loss_bound=config.rel_loss_bound,
                    E_lower=config.E_lower,
                    E_upper=config.E_upper,
                    return_particle=True,
                )
        else:
            if adapter.history_enabled:
                loss, logs, _ = loss_fn_batch_history(
                    model,
                    particle,
                    adapter.history_buffer,
                    n_steps=config.n_steps,
                    rel_loss_bound=config.rel_loss_bound,
                    E_lower=config.E_lower,
                    E_upper=config.E_upper,
                    L_lower=config.L_lower,
                    L_upper=config.L_upper,
                    return_particle=True,
                )
            else:
                loss, logs = loss_fn_batch(
                    model,
                    particle,
                    n_steps=config.n_steps,
                    rel_loss_bound=config.rel_loss_bound,
                    E_lower=config.E_lower,
                    E_upper=config.E_upper,
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"
            save_path = save_dir / f"model_epoch_{epoch:04d}.pt"
            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch=epoch,
                loss=loss,
                logs=logs,
                config=config,
            )
            print(f"epoch {epoch} loss={float(loss.item()):.6e} saved={save_path}")
        epoch += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified N-body ML + simulation runner")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Train ML time-stepper for N-body")
    Config.add_cli_args(train, include=["train", "bounds", "history", "device", "logging", "sim", "multi"])
    train.add_argument("--ic-path", type=str, default=None, help="path to ICs (npy/txt) for training")

    sim = sub.add_parser("simulate", help="Run N-body simulation")
    Config.add_cli_args(sim, include=["sim", "history", "device", "logging", "external"])
    sim.add_argument("--ic-path", type=str, default=None, help="path to ICs (npy/txt) for simulation")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = Config.from_dict(vars(args))
    if getattr(args, "ic_path", None):
        config.extra["ic_path"] = args.ic_path

    if args.mode == "simulate":
        run_simulation(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
