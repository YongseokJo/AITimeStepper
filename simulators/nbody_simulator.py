from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .particle import Particle, predict_dt_from_model_system, predict_dt_from_history_model_system
from src.config import Config
from src.model_adapter import ModelAdapter


def generate_random_ic(
    num_particles: int,
    dim: int = 2,
    mass: float | np.ndarray = 1.0,
    pos_scale: float = 0.1,
    vel_scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=pos_scale, size=(num_particles, dim))
    vel = rng.normal(scale=vel_scale, size=(num_particles, dim))
    pos = pos - pos.mean(axis=0, keepdims=True)
    vel = vel - vel.mean(axis=0, keepdims=True)
    if np.isscalar(mass):
        masses = np.full((num_particles, 1), float(mass))
    else:
        mass_arr = np.asarray(mass, dtype=float).reshape(-1)
        if mass_arr.shape[0] != num_particles:
            raise ValueError("mass array length must match num_particles")
        masses = mass_arr[:, None]
    ptcls = np.concatenate([masses, pos, vel], axis=1)
    return ptcls


def _stack_system(particles: Sequence[Particle]):
    if not particles:
        raise ValueError("particles must be a non-empty sequence")
    pos = np.stack([p.position for p in particles], axis=0)
    vel = np.stack([p.velocity for p in particles], axis=0)
    mass = np.array([p.mass for p in particles], dtype=pos.dtype)
    softening = max(p.softening for p in particles)
    return pos, vel, mass, softening


def compute_accelerations(
    positions,
    masses,
    softening: float = 0.0,
    G: float = 1.0,
    external_acceleration: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    time: float = 0.0,
):
    """
    Vectorized N-body accelerations for positions (N, D) and masses (N,).
    """
    pos = np.asarray(positions)
    m = np.asarray(masses)
    r_ij = pos[None, :, :] - pos[:, None, :]  # x_j - x_i
    dist2 = np.sum(r_ij ** 2, axis=-1) + softening ** 2

    N = pos.shape[0]
    np.fill_diagonal(dist2, np.inf)
    inv_r3 = dist2 ** -1.5

    accel = G * np.sum(r_ij * inv_r3[..., None] * m[None, :, None], axis=1)

    if external_acceleration is not None:
        accel = accel + external_acceleration(pos, float(time))

    return accel


def evolve_step(
    positions,
    velocities,
    masses,
    dt,
    softening: float = 0.0,
    G: float = 1.0,
    external_acceleration: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    time: float = 0.0,
):
    """
    One leapfrog step for an N-body system.
    """
    a_n = compute_accelerations(
        positions,
        masses,
        softening=softening,
        G=G,
        external_acceleration=external_acceleration,
        time=time,
    )
    v_half = velocities + 0.5 * a_n * dt
    x_new = positions + v_half * dt
    a_new = compute_accelerations(
        x_new,
        masses,
        softening=softening,
        G=G,
        external_acceleration=external_acceleration,
        time=time + float(dt),
    )
    v_new = v_half + 0.5 * a_new * dt
    return x_new, v_new, a_new


def evolve_particles(
    particles: Sequence[Particle],
    dt=None,
    softening: float | None = None,
    G: float = 1.0,
    external_acceleration: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
):
    """
    In-place N-body evolution for a list of Particle objects.
    """
    pos, vel, mass, soft_default = _stack_system(particles)
    if dt is None:
        dt = min(p.dt for p in particles)
    if softening is None:
        softening = soft_default

    time = float(getattr(particles[0], "current_time", 0.0))
    x_new, v_new, a_new = evolve_step(
        pos,
        vel,
        mass,
        dt,
        softening=softening,
        G=G,
        external_acceleration=external_acceleration,
        time=time,
    )
    for i, p in enumerate(particles):
        p.position = x_new[i]
        p.velocity = v_new[i]
        p.acceleration = a_new[i]
        p.update_time(dt)
    return dt


def evolve_particles_ml(
    particles: Sequence[Particle],
    model,
    history_buffer=None,
    feature_mode: str = "basic",
    eps: float = 1e-6,
    G: float = 1.0,
    adapter: Optional[ModelAdapter] = None,
    config: Optional[Config] = None,
    external_acceleration: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
):
    """
    N-body evolution using ML-predicted dt (history-aware if provided).
    """
    if history_buffer is not None:
        dt = predict_dt_from_history_model_system(
            model,
            system=particles,
            history_buffer=history_buffer,
            eps=eps,
            device=getattr(particles[0], "device", None),
            adapter=adapter,
            config=config,
        )
    else:
        dt = predict_dt_from_model_system(
            model,
            system=particles,
            eps=eps,
            device=getattr(particles[0], "device", None),
            feature_mode=feature_mode,
            adapter=adapter,
            config=config,
        )

    for p in particles:
        p.dt = dt
    return evolve_particles(
        particles,
        dt=dt,
        G=G,
        external_acceleration=external_acceleration,
    )


def total_energy(positions, velocities, masses, softening: float = 0.0, G: float = 1.0):
    """
    Total energy = kinetic + potential for an N-body system.
    """
    pos = np.asarray(positions)
    vel = np.asarray(velocities)
    m = np.asarray(masses)

    KE = 0.5 * np.sum(m[:, None] * vel ** 2)

    r_ij = pos[:, None, :] - pos[None, :, :]
    dist2 = np.sum(r_ij ** 2, axis=-1) + softening ** 2
    np.fill_diagonal(dist2, np.inf)
    inv_dist = 1.0 / np.sqrt(dist2 + 1e-30)
    PE = -0.5 * G * np.sum(m[:, None] * m[None, :] * inv_dist)
    return float(KE + PE)


def total_momentum(velocities, masses):
    vel = np.asarray(velocities)
    m = np.asarray(masses)
    return np.sum(m[:, None] * vel, axis=0)


def total_angular_momentum(positions, velocities, masses):
    pos = np.asarray(positions)
    vel = np.asarray(velocities)
    m = np.asarray(masses)
    if pos.shape[1] == 2:
        Lz = np.sum(m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
        return Lz
    return np.sum(np.cross(pos, vel) * m[:, None], axis=0)
