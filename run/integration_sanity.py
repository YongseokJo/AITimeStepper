import json
import pathlib
import sys

import numpy as np
import torch

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import Config, FullyConnectedNN, ModelAdapter, HistoryBuffer, make_particle, save_checkpoint, load_model_state
from src.losses_history import loss_fn_batch_history
from simulators.nbody_simulator import generate_random_ic, evolve_particles_ml, total_energy, total_momentum, total_angular_momentum
from simulators.particle import Particle as SimParticle


def _make_sim_particles(ptcls: np.ndarray) -> list[SimParticle]:
    particles = []
    for row in ptcls:
        mass = float(row[0])
        dim = (len(row) - 1) // 2
        pos = row[1 : 1 + dim]
        vel = row[1 + dim : 1 + 2 * dim]
        particles.append(SimParticle(mass=mass, position=pos, velocity=vel))
    return particles


def main() -> None:
    config = Config(
        epochs=3,
        n_steps=2,
        history_len=3,
        feature_type="delta_mag",
        num_particles=4,
        dim=2,
        mass=1.0,
        pos_scale=0.1,
        vel_scale=1.0,
        save_name="integration_sanity",
    )
    config.validate()

    device = config.resolve_device()
    dtype = config.resolve_dtype()
    torch.set_default_dtype(dtype)
    config.apply_torch_settings(device)

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

    adapter = ModelAdapter(config, device=device, dtype=dtype)
    input_dim = adapter.input_dim_from_state(particle, history_buffer=adapter.history_buffer)

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

    for epoch in range(config.epochs):
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch} loss={float(loss.item()):.6e}")

    save_dir = project_root / "data" / config.save_name / "model"
    save_path = save_dir / "model_epoch_0000.pt"
    save_checkpoint(save_path, model, optimizer, epoch=config.epochs - 1, loss=loss, logs=logs, config=config)

    # Simulator sanity
    sim_particles = _make_sim_particles(ptcls)
    load_model_state(model, save_path, map_location=device)
    for p in sim_particles:
        p.update_model(model, device)
        p.attach_history_buffer(adapter.history_buffer)

    energies = []
    momenta = []
    angular = []
    for _ in range(20):
        evolve_particles_ml(
            sim_particles,
            model,
            history_buffer=adapter.history_buffer,
            feature_mode=adapter.feature_mode(),
            eps=1e-6,
            G=1.0,
            adapter=adapter,
            config=config,
        )
        pos = np.stack([p.position for p in sim_particles], axis=0)
        vel = np.stack([p.velocity for p in sim_particles], axis=0)
        mass = np.array([p.mass for p in sim_particles], dtype=float)
        energies.append(total_energy(pos, vel, mass))
        momenta.append(total_momentum(vel, mass))
        angular.append(total_angular_momentum(pos, vel, mass))

    summary = {
        "energy_final": energies[-1] if energies else None,
        "momentum_final": momenta[-1].tolist() if momenta else None,
        "angular_momentum_final": angular[-1].tolist() if angular else None,
    }

    log_path = project_root / "data" / config.save_name / "sanity.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"sanity summary saved: {log_path}")


if __name__ == "__main__":
    main()
