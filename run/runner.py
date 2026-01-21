import argparse
import json
import pathlib
import sys
import time
import warnings
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
    load_config_from_checkpoint,
    load_model_state,
    make_particle,
    run_two_phase_training,
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
    """
    Train ML time-stepper using two-phase training system.

    This function handles:
    1. Validation and device/dtype setup
    2. Seed initialization
    3. ModelAdapter creation
    4. W&B setup (if enabled)
    5. Unsupported feature warnings (num_orbits > 1, duration)
    6. Particle initialization (single orbit)
    7. Model and optimizer creation
    8. Delegation to run_two_phase_training()
    9. Training summary and W&B cleanup
    """
    # SECTION 1: Validation
    if config.integrator_mode == "history" and (config.history_len is None or config.history_len <= 0):
        raise ValueError("history integrator_mode requires history_len > 0")
    config.validate()

    # SECTION 2: Device/dtype setup
    device = config.resolve_device()
    dtype = config.resolve_dtype()
    torch.set_default_dtype(dtype)
    config.apply_torch_settings(device)

    # SECTION 3: Seed setup
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # SECTION 4: Adapter creation
    adapter = ModelAdapter(config, device=device, dtype=dtype)

    # SECTION 5: W&B setup
    wandb_run = None
    wandb = None
    if config.extra.get("wandb", False):
        try:
            import wandb as wandb_lib
        except ImportError as exc:
            raise RuntimeError("wandb is not installed; install it or disable --wandb") from exc
        wandb = wandb_lib
        wandb_project = config.extra.get("wandb_project") or "AITimeStepper"
        wandb_name = config.extra.get("wandb_name") or config.save_name or "runner_train"
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config.as_wandb_dict(),
        )

    # SECTION 6: Unsupported feature warnings
    if config.num_orbits > 1:
        warnings.warn(
            f"Multi-orbit training (num_orbits={config.num_orbits}) not yet supported "
            "in two-phase training. Using single orbit.",
            UserWarning,
        )
    if config.duration is not None:
        warnings.warn(
            f"Duration-based training (duration={config.duration}) not yet supported "
            "in two-phase training. Using epoch count.",
            UserWarning,
        )

    # SECTION 7: Particle initialization (single orbit)
    ptcls = generate_random_ic(
        num_particles=config.num_particles,
        dim=config.dim,
        mass=config.mass,
        pos_scale=config.pos_scale,
        vel_scale=config.vel_scale,
        seed=config.seed,
    )
    particle = make_particle(ptcls, device=device, dtype=dtype)
    particle.current_time = torch.tensor(0.0, device=device, dtype=dtype)
    input_dim = adapter.input_dim_from_state(particle, history_buffer=adapter.history_buffer)

    # SECTION 8: Model creation
    model = FullyConnectedNN(
        input_dim=input_dim,
        output_dim=2,
        hidden_dims=[200, 1000, 1000, 200],
        activation="tanh",
        dropout=0.2,
        output_positive=True,
    ).to(device)
    model.to(dtype=dtype)

    # SECTION 9: Optimizer creation
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # SECTION 10: Training with run_two_phase_training()
    save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"
    result = run_two_phase_training(
        model=model,
        particle=particle,
        optimizer=optimizer,
        config=config,
        adapter=adapter,
        history_buffer=adapter.history_buffer,
        save_dir=save_dir,
        wandb_run=wandb_run,
        checkpoint_interval=10,
    )

    # SECTION 11: Training summary
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"  Epochs completed: {result['epochs_completed']}")
    print(f"  Total time: {result['total_time']:.2f}s")
    print(f"  Convergence rate: {result['convergence_rate']:.1%}")
    if result.get('final_metrics'):
        final = result['final_metrics']
        if 'trajectory_metrics' in final:
            traj = final['trajectory_metrics']
            print(f"  Final trajectory length: {traj.get('trajectory_length', 'N/A')}")
        if 'generalization_metrics' in final:
            gen = final['generalization_metrics']
            print(f"  Final pass rate: {gen.get('final_pass_rate', 'N/A'):.1%}" if isinstance(gen.get('final_pass_rate'), (int, float)) else f"  Final pass rate: {gen.get('final_pass_rate', 'N/A')}")
    print(f"{'='*60}\n")

    # SECTION 12: W&B cleanup
    if wandb_run is not None:
        wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified N-body ML + simulation runner")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Train ML time-stepper for N-body")
    Config.add_cli_args(train, include=["train", "bounds", "history", "device", "logging", "sim", "multi"])
    train.add_argument("--ic-path", type=str, default=None, help="path to ICs (npy/txt) for training")
    train.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    train.add_argument("--wandb-project", type=str, default="AITimeStepper", help="W&B project name")
    train.add_argument("--wandb-name", type=str, default=None, help="W&B run name (defaults to save_name)")

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
