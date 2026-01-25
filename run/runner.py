import argparse
import json
import pathlib
import sys
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import (
    Config,
    FullyConnectedNN,
    HistoryBuffer,
    ModelAdapter,
    load_checkpoint,
    load_config_from_checkpoint,
    load_model_state,
    make_particle,
    run_two_phase_training,
    save_checkpoint,
    stack_particles,
)
from src.normalization import apply_norm_scales_to_config, derive_norm_scales_from_config
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


def _load_train_ic(config: Config) -> np.ndarray:
    if not config.model_path:
        raise ValueError("use_train_ic requires model_path to load training ICs.")
    ckpt = load_checkpoint(config.model_path, map_location="cpu")
    train_ic = None
    if isinstance(ckpt.get("extra"), dict):
        train_ic = ckpt["extra"].get("train_ic")
    if train_ic is None:
        raise ValueError("Checkpoint missing training ICs; retrain or provide --ic-path.")
    return np.array(train_ic, dtype=float)


def _normalize_orbit_masses(mass) -> tuple[float, float]:
    if np.isscalar(mass):
        m = float(mass)
        return m, m
    mass_arr = np.asarray(mass, dtype=float).reshape(-1)
    if mass_arr.shape[0] != 2:
        raise ValueError("Orbit sampling requires mass scalar or length-2 array.")
    return float(mass_arr[0]), float(mass_arr[1])


def _generate_orbit_ic(eccentricity: float, semi_major: float, mass, dim: int) -> tuple[np.ndarray, float]:
    if dim < 2:
        raise ValueError("Orbit sampling requires dim >= 2.")
    m1, m2 = _normalize_orbit_masses(mass)
    G = 1.0
    r_p = semi_major * (1.0 - eccentricity)
    v_p = np.sqrt(G * (m1 + m2) * (1.0 + eccentricity) / (semi_major * (1.0 - eccentricity)))
    pos = np.zeros((2, dim), dtype=float)
    vel = np.zeros((2, dim), dtype=float)
    pos[0, 0] = -0.5 * r_p
    pos[1, 0] = 0.5 * r_p
    vel[0, 1] = 0.5 * v_p
    vel[1, 1] = -0.5 * v_p
    masses = np.array([m1, m2], dtype=float).reshape(2, 1)
    
    T = 2 * np.pi * np.sqrt(semi_major**3 / (G * (m1 + m2)))

    return np.concatenate([masses, pos, vel], axis=1), T


def _estimate_orbit_period(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> Optional[float]:
    if positions.shape[0] != 2 or velocities.shape[0] != 2:
        return None
    r = positions[1] - positions[0]
    v = velocities[1] - velocities[0]
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        return None
    m = np.asarray(masses, dtype=float).reshape(-1)
    if m.shape[0] < 2:
        return None
    mu = float(m[0] + m[1])
    if mu <= 0.0:
        return None
    energy = 0.5 * float(np.dot(v, v)) - mu / r_norm
    if energy >= 0.0:
        return None
    semi_major = -mu / (2.0 * energy)
    if semi_major <= 0.0:
        return None
    return 2 * np.pi * np.sqrt(semi_major**3 / mu)


def _sample_orbit_ics(config: Config, count: int, rng: np.random.Generator) -> list[tuple[np.ndarray, float]]:
    if count <= 0:
        return []
    if config.num_particles != 2:
        raise ValueError("Multi-orbit simulation requires num_particles=2.")
    if config.dim < 2:
        raise ValueError("Multi-orbit simulation requires dim >= 2.")
    if config.e_min > config.e_max:
        raise ValueError("e_min must be <= e_max for orbit sampling.")
    if config.a_min > config.a_max:
        raise ValueError("a_min must be <= a_max for orbit sampling.")
    out = []
    for _ in range(count):
        e_val = float(rng.uniform(config.e_min, config.e_max))
        a_val = float(rng.uniform(config.a_min, config.a_max))
        out.append(_generate_orbit_ic(e_val, a_val, config.mass, config.dim))
    return out


def _build_simulation_orbits(config: Config, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    if config.extra.get("multi_orbit_sim"):
        total = int(config.num_orbits)
        if total < 1:
            raise ValueError("num_orbits must be >= 1 for multi-orbit simulation.")
        if config.num_particles != 2 or config.dim < 2:
            raise ValueError("Multi-orbit simulation supports 2-body systems in >=2D.")
        entries: list[tuple[str, np.ndarray]] = []
        if config.extra.get("ic_path"):
            entries.append(("ic", _load_ic(config.extra["ic_path"])))
        if config.extra.get("use_train_ic"):
            entries.append(("train", _load_train_ic(config)))
        remaining = total - len(entries)
        if remaining < 0:
            raise ValueError("num_orbits is smaller than the number of requested base orbits.")
        sampled = _sample_orbit_ics(config, remaining, rng)
        for idx, (ptcls, T) in enumerate(sampled, start=1):
            entries.append((f"orbit_{idx}", ptcls))
        return entries
    if config.extra.get("ic_path"):
        return [("ic", _load_ic(config.extra["ic_path"]))]
    if config.extra.get("use_train_ic"):
        return [("train_ic", _load_train_ic(config))]
    return [
        (
            "orbit",
            generate_random_ic(
                num_particles=config.num_particles,
                dim=config.dim,
                mass=config.mass,
                pos_scale=config.pos_scale,
                vel_scale=config.vel_scale,
                seed=config.seed,
            ),
        )
    ]

def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise RuntimeError(
                "TOML config requires Python 3.11+ (tomllib) or install tomli."
            ) from exc
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _split_toml_config(data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    base = {k: v for k, v in data.items() if k not in {"train", "simulate"}}
    train = data.get("train", {}) or {}
    simulate = data.get("simulate", {}) or {}
    return base, train, simulate


def _subparser_for_mode(parser: argparse.ArgumentParser, mode: str) -> Optional[argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices.get(mode)
    return None


def _cli_overrides_for_mode(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    mode: str,
    argv: list[str],
) -> Dict[str, Any]:
    option_strings = {arg for arg in argv if arg.startswith("-")}
    overrides: Dict[str, Any] = {}
    actions = list(parser._actions)
    subparser = _subparser_for_mode(parser, mode)
    if subparser is not None:
        actions.extend(subparser._actions)
    for action in actions:
        if not action.option_strings:
            continue
        if any(opt in option_strings for opt in action.option_strings):
            if action.dest not in {"config", "mode"}:
                overrides[action.dest] = getattr(args, action.dest, None)
    return overrides


def _config_from_sources(
    base: Dict[str, Any],
    section: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Config:
    payload: Dict[str, Any] = {}
    payload.update(base)
    payload.update(section)
    payload.update(overrides)
    return Config.from_dict(payload)


def _default_model_path(config: Config) -> pathlib.Path:
    save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"
    return save_dir / f"model_epoch_{config.epochs - 1:04d}.pt"


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


def _compute_stride(total_steps: int, max_frames: int) -> int:
    if max_frames is None or max_frames <= 0:
        return 1
    return max(1, total_steps // max_frames)


def _set_axis_limits(ax, data: np.ndarray, pad_frac: float = 0.1) -> None:
    if data.size == 0:
        return
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    pad = (dmax - dmin) * pad_frac if dmax > dmin else 1.0
    ax.set_ylim(dmin - pad, dmax + pad)


def _save_movie(
    positions: np.ndarray,
    energies: np.ndarray,
    residuals: np.ndarray,
    momentum_norm: np.ndarray,
    angular_norm: np.ndarray,
    dt_array: np.ndarray,
    config: Config,
    save_name: Optional[str] = None,
) -> pathlib.Path:
    """
    Generate an animation of the N-body simulation.

    Args:
        positions: Array of shape (steps, N, dim) with particle positions.
                   Only the first 2 dimensions are used for 2D plotting.
        energies: Total energy at each step.
        residuals: Relative energy error at each step.
        momentum_norm: Magnitude of total momentum at each step.
        angular_norm: Magnitude of total angular momentum at each step.
        dt_array: Array of shape (steps,) with timestep values.
        config: Configuration object with movie settings.

    Returns:
        Path to the saved movie file.
    """
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --movie") from exc

    steps, n_particles, dim = positions.shape
    if dim < 2:
        raise ValueError("Movie generation requires at least 2D positions.")

    # Use only x, y for plotting
    positions_2d = positions[:, :, :2]

    series_len = min(
        steps,
        energies.shape[0],
        residuals.shape[0],
        momentum_norm.shape[0],
        angular_norm.shape[0],
        dt_array.shape[0],
    )
    if series_len == 0:
        raise ValueError("Movie generation requires at least one frame.")
    positions_2d = positions_2d[:series_len]
    energies = energies[:series_len]
    residuals = residuals[:series_len]
    momentum_norm = momentum_norm[:series_len]
    angular_norm = angular_norm[:series_len]
    dt_array = dt_array[:series_len]

    time_full = np.cumsum(dt_array)
    if time_full.size > 0:
        time_full = time_full - time_full[0]
    if config.movie_max_frames is None or config.movie_max_frames <= 0 or series_len <= config.movie_max_frames:
        frame_idx = np.arange(series_len)
    else:
        if time_full.size == 0 or time_full[-1] == time_full[0]:
            frame_idx = np.arange(series_len)
        else:
            target_times = np.linspace(time_full[0], time_full[-1], num=config.movie_max_frames)
            frame_idx = np.searchsorted(time_full, target_times, side="left")
            frame_idx = np.clip(frame_idx, 0, series_len - 1)
            frame_idx = np.unique(frame_idx)

    pos_plot = positions_2d[frame_idx]
    E_plot = energies[frame_idx]
    res_plot = residuals[frame_idx]
    P_plot = momentum_norm[frame_idx]
    L_plot = angular_norm[frame_idx]
    dt_plot = dt_array[frame_idx]
    # Avoid log scale error if dt <= 0 (should not happen in valid sim, but safety first)
    dt_plot_safe = np.maximum(dt_plot, 1e-12)
    time_plot = time_full[frame_idx]
    steps_plot = np.arange(series_len)[frame_idx]
    nframes = len(time_plot)

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(
        5,
        2,
        width_ratios=[1.3, 1.0],
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
        hspace=0.4,
        wspace=0.3,
    )

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_E = fig.add_subplot(gs[0, 1])
    ax_res = fig.add_subplot(gs[1, 1], sharex=ax_E)
    ax_P = fig.add_subplot(gs[2, 1], sharex=ax_E)
    ax_L = fig.add_subplot(gs[3, 1], sharex=ax_E)
    ax_dt = fig.add_subplot(gs[4, 1], sharex=ax_E)

    # Compute axis limits from all particles
    xmin = positions_2d[:, :, 0].min()
    xmax = positions_2d[:, :, 0].max()
    ymin = positions_2d[:, :, 1].min()
    ymax = positions_2d[:, :, 1].max()
    pad = 0.1 * max(xmax - xmin, ymax - ymin, 1e-6)

    ax_traj.set_xlim(xmin - pad, xmax + pad)
    ax_traj.set_ylim(ymin - pad, ymax + pad)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_title(f"{n_particles}-Body Trajectories")

    ax_E.set_xlim(time_plot[0], time_plot[-1])
    ax_E.set_ylabel("Total Energy")
    ax_E.set_title("Energy, Momentum, Angular Momentum, dt")
    _set_axis_limits(ax_E, E_plot)
    ax_E.grid(True)

    ax_res.set_ylabel("ΔE")
    _set_axis_limits(ax_res, res_plot)
    ax_res.axhline(0.0, color="black", linewidth=0.8)
    ax_res.grid(True)

    ax_P.set_ylabel(r"$|℄|$ ")
    _set_axis_limits(ax_P, P_plot)
    ax_P.grid(True)

    ax_L.set_ylabel(r"$|ℇ|$ ")
    _set_axis_limits(ax_L, L_plot)
    ax_L.grid(True)

    ax_dt.set_xlabel("Time (code units)")
    ax_dt.set_ylabel("dt")
    # If dt is empty or non-positive, keep linear to avoid log-scale draw errors.
    if dt_plot_safe.size == 0 or not np.any(dt_plot_safe > 0.0):
        _set_axis_limits(ax_dt, dt_plot_safe)
        ax_dt.set_yscale("linear")
    else:
        ax_dt.set_yscale("log")
        # Log scale cannot include <= 0 limits, so compute positive bounds explicitly.
        positive_dt = dt_plot_safe[dt_plot_safe > 0.0]
        dmin = float(np.min(positive_dt))
        dmax = float(np.max(positive_dt))
        if dmin == dmax:
            lower = dmin / 10.0
            upper = dmax * 10.0
        else:
            pad = 10 ** 0.1
            lower = dmin / pad
            upper = dmax * pad
        ax_dt.set_ylim(lower, upper)
    ax_dt.grid(True)

    if time_full.size > 1 and time_full[-1] != time_full[0]:
        def _time_to_step(t: np.ndarray) -> np.ndarray:
            return np.interp(t, time_full, np.arange(series_len))

        def _step_to_time(s: np.ndarray) -> np.ndarray:
            return np.interp(s, np.arange(series_len), time_full)

        ax_step = ax_E.secondary_xaxis("top", functions=(_time_to_step, _step_to_time))
        ax_step.set_xlabel("Step")

    # Create trajectory lines and marker points for each particle
    colors = plt.cm.tab10.colors
    traj_lines = []
    markers = []
    for i in range(n_particles):
        color = colors[i % len(colors)]
        line, = ax_traj.plot([], [], color=color, label=f"Particle {i + 1}")
        marker, = ax_traj.plot([], [], marker="o", linestyle="None", color=color, markersize=6)
        traj_lines.append(line)
        markers.append(marker)
    ax_traj.legend(loc="upper right", fontsize="small", ncol=max(1, n_particles // 5))

    E_line, = ax_E.plot([], [], lw=1.0)
    res_line, = ax_res.plot([], [], lw=1.0)
    P_line, = ax_P.plot([], [], lw=1.0)
    L_line, = ax_L.plot([], [], lw=1.0)
    dt_line, = ax_dt.plot([], [], lw=1.0, color="purple")

    def init():
        for line in traj_lines:
            line.set_data([], [])
        for marker in markers:
            marker.set_data([], [])
        E_line.set_data([], [])
        res_line.set_data([], [])
        P_line.set_data([], [])
        L_line.set_data([], [])
        dt_line.set_data([], [])
        return traj_lines + markers + [E_line, res_line, P_line, L_line, dt_line]

    def update(frame):
        for p_idx in range(n_particles):
            traj_lines[p_idx].set_data(
                pos_plot[: frame + 1, p_idx, 0],
                pos_plot[: frame + 1, p_idx, 1],
            )
            markers[p_idx].set_data(
                [pos_plot[frame, p_idx, 0]],
                [pos_plot[frame, p_idx, 1]],
            )
        x = time_plot[: frame + 1]
        E_line.set_data(x, E_plot[: frame + 1])
        res_line.set_data(x, res_plot[: frame + 1])
        P_line.set_data(x, P_plot[: frame + 1])
        L_line.set_data(x, L_plot[: frame + 1])
        dt_line.set_data(x, dt_plot_safe[: frame + 1])
        return traj_lines + markers + [E_line, res_line, P_line, L_line, dt_line]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=nframes,
        init_func=init,
        interval=20,
        blit=True,
    )

    movie_dir = pathlib.Path(config.movie_dir) if config.movie_dir else pathlib.Path(project_root) / "data" / "movie"
    movie_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = save_name or config.save_name or "run_nbody"
    movie_path = movie_dir / f"nbody_movie_{resolved_name}.mp4"
    writer = animation.FFMpegWriter(fps=config.movie_fps, bitrate=1800)
    ani.save(str(movie_path), writer=writer, dpi=config.movie_dpi)
    plt.close(fig)
    return movie_path


def _trajectory_to_movie_arrays(
    trajectory: list[tuple["ParticleTorch", float]],
    config: Config,
    initial_particle: Optional["ParticleTorch"] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not trajectory:
        return None
    frames = [step[0] for step in trajectory]
    if initial_particle is not None:
        frames = [initial_particle] + frames

    positions = np.stack([p.position.detach().cpu().numpy() for p in frames], axis=0)
    velocities = np.stack([p.velocity.detach().cpu().numpy() for p in frames], axis=0)
    masses = frames[0].mass.detach().cpu().numpy()
    
    # Extract dt: check if it's a tensor or float
    dt_list = []
    if initial_particle is not None:
        dt_list.append(0.0)
    for step in trajectory:
        # step[0] is particle, but step[1] might be the dt used for *that* step?
        # The trajectory format is [(Particle, dt_used, mask), ...] usually or similar.
        # Check collect_trajectory return signature in trajectory_collection.py:
        # trajectory.append((accepted_particle.clone_detached(), accepted_dt, active_mask))
        # So step[1] is the float dt.
        dt_val = step[1]
        if torch.is_tensor(dt_val):
            dt_val = dt_val.item()
        dt_list.append(dt_val)
    dt_array = np.array(dt_list, dtype=float)

    # Handle batched trajectory (T, B, N, D): select first environment for visualization
    if positions.ndim == 4:
        print(f"Batched trajectory detected (shape {positions.shape}). Visualizing first batch item only.")
        B = positions.shape[1]
        positions = positions[:, 0, :, :]
        velocities = velocities[:, 0, :, :]
        # If masses has batch dimension matching B, slice it too
        if masses.ndim > 0 and masses.shape[0] == B:
            masses = masses[0]

    if positions.shape[2] < 2:
        print(
            f"Warning: Movie generation skipped. Requires at least 2D simulation, "
            f"but got {positions.shape[2]}D."
        )
        return None
    energies = np.array([total_energy(positions[i], velocities[i], masses) for i in range(len(trajectory))])
    energy_initial = energies[0] if len(energies) > 0 else None
    if energy_initial not in (None, 0.0):
        residuals = (energies - energy_initial) / energy_initial
    else:
        residuals = np.zeros_like(energies)
    momenta = np.stack([total_momentum(velocities[i], masses) for i in range(len(trajectory))], axis=0)
    angular = np.stack(
        [total_angular_momentum(positions[i], velocities[i], masses) for i in range(len(trajectory))],
        axis=0,
    )
    momentum_norm = np.linalg.norm(momenta, axis=-1)
    if np.ndim(angular) <= 1:
        angular_norm = np.abs(angular)
    else:
        angular_norm = np.linalg.norm(angular, axis=-1)
    return positions, energies, residuals, momentum_norm, angular_norm, dt_array


def _run_single_simulation(config: Config, ptcls: np.ndarray, *, save_name: Optional[str] = None) -> None:
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
    original_name = config.save_name
    if save_name is not None:
        config.save_name = save_name
    try:
        particles = _make_sim_particles(ptcls)
        device = config.resolve_device()
        dtype = config.resolve_dtype()
        adapter = ModelAdapter(config, device=device, dtype=dtype)
        norm_state = make_particle(ptcls, device=device, dtype=dtype)
        norm_scales = derive_norm_scales_from_config(config, particle=norm_state)
        if norm_scales:
            adapter.set_norm_scales(norm_scales)
            apply_norm_scales_to_config(config, norm_scales)
            if config.debug:
                print(f"norm_scales={norm_scales}")

        if config.integrator_mode in ("ml", "history"):
            model = FullyConnectedNN(
                input_dim=adapter.input_dim_from_state(
                    make_particle(ptcls, device=device, dtype=dtype),
                    history_buffer=adapter.history_buffer,
                ),
                output_dim=2,
                hidden_dims=list(config.hidden_dims),
                activation=config.activation,
                dropout=config.dropout,
                output_positive=True,
                fourier_scale=config.fourier_scale,
                fourier_dim=config.fourier_dim,
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
        positions_history = []
        dt_history = []

        pos0 = np.stack([p.position for p in particles], axis=0)
        vel0 = np.stack([p.velocity for p in particles], axis=0)
        mass0 = np.array([p.mass for p in particles], dtype=float)
        energy_initial = total_energy(pos0, vel0, mass0)
        momentum_initial = total_momentum(vel0, mass0)
        angular_initial = total_angular_momentum(pos0, vel0, mass0)

        duration_limit = config.duration
        if config.Nperiod and config.Nperiod > 0:
            period = _estimate_orbit_period(pos0, vel0, mass0)
            if period is not None:
                duration_limit = float(config.Nperiod * period)
        max_steps = steps if duration_limit is None else None
        step_count = 0
        while True:
            if max_steps is not None and step_count >= max_steps:
                break
            if duration_limit is not None:
                current_time = float(getattr(particles[0], "current_time", 0.0))
                if current_time >= duration_limit:
                    break
            if config.integrator_mode in ("ml", "history"):
                history_buffer = adapter.history_buffer if config.integrator_mode == "history" else None
                step_dt = evolve_particles_ml(
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
                step_dt = evolve_particles(
                    particles,
                    dt=dt,
                    G=1.0,
                    external_acceleration=external_accel,
                )

            pos = np.stack([p.position for p in particles], axis=0)
            vel = np.stack([p.velocity for p in particles], axis=0)
            mass = np.array([p.mass for p in particles], dtype=float)
            if config.movie:
                if step_count == 0 and not positions_history:
                    positions_history.append(pos0.copy())
                    energies.append(energy_initial)
                    momenta.append(momentum_initial)
                    angular.append(angular_initial)
                    dt_history.append(0.0)
                positions_history.append(pos.copy())
            energies.append(total_energy(pos, vel, mass))
            momenta.append(total_momentum(vel, mass))
            angular.append(total_angular_momentum(pos, vel, mass))
            dt_history.append(step_dt)
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
        if config.movie:
            if step_count == 0:
                raise ValueError("Movie generation requires at least one simulation step.")
            positions = np.stack(positions_history, axis=0)
            # Movie generation requires at least 2D positions
            if positions.shape[2] < 2:
                print(
                    f"Warning: Movie generation skipped. Requires at least 2D simulation, "
                    f"but got {positions.shape[2]}D."
                )
            else:
                energies_arr = np.asarray(energies, dtype=float)
                residuals_arr = (
                    (energies_arr - energy_initial) / energy_initial
                    if energy_initial not in (None, 0.0)
                    else np.zeros_like(energies_arr)
                )
                momenta_arr = np.stack(momenta, axis=0)
                angular_arr = np.stack(angular, axis=0)
                P_norm = np.linalg.norm(momenta_arr, axis=-1)
                if np.ndim(angular_arr) <= 1:
                    L_norm = np.abs(angular_arr)
                else:
                    L_norm = np.linalg.norm(angular_arr, axis=-1)
                dt_array = np.array(dt_history, dtype=float)
                movie_path = _save_movie(positions, energies_arr, residuals_arr, P_norm, L_norm, dt_array, config)
                print(f"Movie saved: {movie_path}")
    finally:
        config.save_name = original_name


def run_simulation(config: Config) -> None:
    rng_seed = config.seed
    rng = np.random.default_rng(rng_seed)
    entries = _build_simulation_orbits(config, rng)
    if config.extra.get("multi_orbit_sim"):
        print(f"Running {len(entries)} independent orbit simulations.")
    for idx, (label, ptcls) in enumerate(entries, start=1):
        if len(entries) > 1:
            print(f"Orbit {idx}/{len(entries)}: {label}")
        
        # Use label as suffix if it's train-related or if multiple orbits are present
        if label in ("train", "train_ic"):
            suffix = "train_ic"
        elif len(entries) > 1:
            suffix = label
        else:
            suffix = None

        save_name = f"{config.save_name or 'run_nbody'}_{suffix}" if suffix else config.save_name
        _run_single_simulation(config, ptcls, save_name=save_name)

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
    # (Removed multi-orbit warning as we support it now)
    if config.duration is not None:
        warnings.warn(
            f"Duration-based training (duration={config.duration}) not yet supported "
            "in two-phase training. Using epoch count.",
            UserWarning,
        )

    # SECTION 7: Particle initialization
    particles = []
    
    use_multi = (config.num_orbits > 1) or config.stop_at_period
    ptcls_for_norm = None

    if config.extra.get("ic_path"):
        ptcls = _load_ic(config.extra["ic_path"])
        p = make_particle(ptcls, device=device, dtype=dtype)
        particles.append(p)
        ptcls_for_norm = ptcls
    elif use_multi:
        rng = np.random.default_rng(config.seed)
        sampled = _sample_orbit_ics(config, config.num_orbits, rng)
        for ptcls, T in sampled:
            p = make_particle(ptcls, device=device, dtype=dtype)
            p.period = torch.tensor(T, device=device, dtype=dtype)
            particles.append(p)
        if sampled:
            ptcls_for_norm = sampled[0][0]
    else:
        ptcls = generate_random_ic(
            num_particles=config.num_particles,
            dim=config.dim,
            mass=config.mass,
            pos_scale=config.pos_scale,
            vel_scale=config.vel_scale,
            seed=config.seed,
        )
        p = make_particle(ptcls, device=device, dtype=dtype)
        particles.append(p)
        ptcls_for_norm = ptcls

    if len(particles) == 1:
        particle = particles[0]
    else:
        particle = stack_particles(particles)
        # Ensure current_time is initialized as vector if multi
        if particle.current_time.dim() == 0:
             # Force vector
             particle.current_time = torch.zeros(len(particles), device=device, dtype=dtype)
    initial_particle = particle.clone_detached()

    norm_scales = derive_norm_scales_from_config(config, particle=particle)
    if norm_scales:
        adapter.set_norm_scales(norm_scales)
        apply_norm_scales_to_config(config, norm_scales)
        if wandb_run is not None:
            wandb_run.log({f"norm/{k}": v for k, v in norm_scales.items()})
        if config.debug:
            print(f"norm_scales={norm_scales}")
    input_dim = adapter.input_dim_from_state(particle, history_buffer=adapter.history_buffer)

    # SECTION 8: Model creation
    model = FullyConnectedNN(
        input_dim=input_dim,
        output_dim=2,
        hidden_dims=list(config.hidden_dims),
        activation=config.activation,
        dropout=config.dropout,
        output_positive=True,
        fourier_scale=config.fourier_scale,
        fourier_dim=config.fourier_dim,
    ).to(device)
    model.to(dtype=dtype)
    if config.debug:
        print(f"DEBUG: Model initialized with hidden_dims={list(config.hidden_dims)}")
        if config.fourier_scale > 0:
            print(f"DEBUG: Fourier Features ENABLED (scale={config.fourier_scale}, dim={config.fourier_dim})")

    # SECTION 9: Optimizer creation
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # SECTION 10: Training with run_two_phase_training()
    save_dir = pathlib.Path(project_root) / "data" / (config.save_name or "run_nbody") / "model"
    # Note: saving all ICs might be large for multi-orbit, but useful.
    checkpoint_extra = {"train_ic": ptcls_for_norm.tolist() if ptcls_for_norm is not None else []} 
    
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
        checkpoint_extra=checkpoint_extra,
        return_final_trajectory=config.movie,
    )

    # SECTION 11: Training summary
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{ '='*60}")
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
    if config.movie:
        trajectory = result.get("final_trajectory")
        if not trajectory:
            warnings.warn("Training movie skipped: empty trajectory.", UserWarning)
        else:
            movie_arrays = _trajectory_to_movie_arrays(trajectory, config, initial_particle=initial_particle)
            if movie_arrays is not None:
                positions, energies_arr, residuals_arr, P_norm, L_norm, dt_array = movie_arrays
                movie_path = _save_movie(
                    positions,
                    energies_arr,
                    residuals_arr,
                    P_norm,
                    L_norm,
                    dt_array,
                    config,
                    save_name=f"{config.save_name or 'run_nbody'}_train",
                )
                print(f"Training movie saved: {movie_path}")
    print(f"{ '='*60}\n")

    # SECTION 12: W&B cleanup
    if wandb_run is not None:
        wandb.finish()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified N-body ML + simulation runner")
    parser.add_argument("--config", type=str, default=None, help="path to TOML config")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Train ML time-stepper for N-body")
    train.add_argument("--config", type=str, default=None, help=argparse.SUPPRESS)
    Config.add_cli_args(train, include=["train", "bounds", "history", "device", "logging", "sim", "multi"])
    train.add_argument("--ic-path", type=str, default=None, help="path to ICs (npy/txt) for training")
    train.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    train.add_argument("--wandb-project", type=str, default="AITimeStepper", help="W&B project name")
    train.add_argument("--wandb-name", type=str, default=None, help="W&B run name (defaults to save_name)")

    sim = sub.add_parser("simulate", help="Run N-body simulation")
    sim.add_argument("--config", type=str, default=None, help=argparse.SUPPRESS)
    Config.add_cli_args(sim, include=["sim", "history", "device", "logging", "external", "multi"])
    sim.add_argument("--ic-path", type=str, default=None, help="path to ICs (npy/txt) for simulation")
    sim.add_argument("--multi-orbit-sim", action="store_true", help="run multiple independent orbits in simulation")
    sim.add_argument(
        "--use-train-ic",
        action="store_true",
        help="use training ICs stored in the model checkpoint (overrides random ICs)",
    )

    both = sub.add_parser("both", help="Run training then simulation")
    both.add_argument("--config", type=str, default=None, help=argparse.SUPPRESS)
    Config.add_cli_args(both, include=["train", "bounds", "history", "device", "logging", "sim", "multi", "external"])
    both.add_argument("--train-ic-path", type=str, default=None, help="path to ICs (npy/txt) for training")
    both.add_argument("--sim-ic-path", type=str, default=None, help="path to ICs (npy/txt) for simulation")
    both.add_argument("--multi-orbit-sim", action="store_true", help="run multiple independent orbits in simulation")
    both.add_argument(
        "--use-train-ic",
        action="store_true",
        help="use training ICs stored in the model checkpoint for simulation",
    )
    both.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    both.add_argument("--wandb-project", type=str, default="AITimeStepper", help="W&B project name")
    both.add_argument("--wandb-name", type=str, default=None, help="W&B run name (defaults to save_name)")

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_data = {}
    if args.config:
        config_data = _load_toml(args.config)
    base_cfg, train_cfg, sim_cfg = _split_toml_config(config_data)
    argv = sys.argv[1:]

    if args.mode == "simulate":
        overrides = _cli_overrides_for_mode(args, parser, "simulate", argv)
        if getattr(args, "ic_path", None):
            overrides["ic_path"] = args.ic_path
        if getattr(args, "multi_orbit_sim", False):
            overrides["multi_orbit_sim"] = True
        if getattr(args, "use_train_ic", False):
            overrides["use_train_ic"] = True
        config = _config_from_sources(base_cfg, sim_cfg, overrides)
        if getattr(args, "ic_path", None):
            config.extra["ic_path"] = args.ic_path
        if getattr(args, "multi_orbit_sim", False):
            config.extra["multi_orbit_sim"] = True
        if getattr(args, "use_train_ic", False):
            config.extra["use_train_ic"] = True
        run_simulation(config)
        return

    if args.mode == "train":
        overrides = _cli_overrides_for_mode(args, parser, "train", argv)
        if getattr(args, "ic_path", None):
            overrides["ic_path"] = args.ic_path
        config = _config_from_sources(base_cfg, train_cfg, overrides)
        if getattr(args, "ic_path", None):
            config.extra["ic_path"] = args.ic_path
        run_training(config)
        return

    train_overrides = _cli_overrides_for_mode(args, parser, "both", argv)
    sim_overrides = dict(train_overrides)
    if getattr(args, "train_ic_path", None):
        train_overrides["ic_path"] = args.train_ic_path
    if getattr(args, "sim_ic_path", None):
        sim_overrides["ic_path"] = args.sim_ic_path
    if getattr(args, "multi_orbit_sim", False):
        sim_overrides["multi_orbit_sim"] = True
    if getattr(args, "use_train_ic", False):
        sim_overrides["use_train_ic"] = True
    explicit_integrator = (
        "integrator_mode" in base_cfg
        or "integrator_mode" in sim_cfg
        or "integrator_mode" in sim_overrides
    )

    train_config = _config_from_sources(base_cfg, train_cfg, train_overrides)
    if getattr(args, "train_ic_path", None):
        train_config.extra["ic_path"] = args.train_ic_path
    run_training(train_config)

    sim_config = _config_from_sources(base_cfg, sim_cfg, sim_overrides)
    if sim_config.model_path is None:
        sim_config.model_path = str(_default_model_path(train_config))
    if not explicit_integrator and sim_config.integrator_mode == "analytic":
        sim_config.integrator_mode = "history" if sim_config.history_len and sim_config.history_len > 0 else "ml"
    if getattr(args, "sim_ic_path", None):
        sim_config.extra["ic_path"] = args.sim_ic_path
    if getattr(args, "multi_orbit_sim", False):
        sim_config.extra["multi_orbit_sim"] = True
    if getattr(args, "use_train_ic", False):
        sim_config.extra["use_train_ic"] = True
    run_simulation(sim_config)


if __name__ == "__main__":
    main()
