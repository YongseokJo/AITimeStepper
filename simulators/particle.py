from typing import Optional, Sequence

import torch
import numpy as np
import torch.nn.functional as F

from src.history_buffer import HistoryBuffer
from src.particle import ParticleTorch

def get_xp(backend):
    if backend == "torch":
        return torch
    elif backend == "numpy":
        return np
    else:
        raise ValueError("backend must be 'torch' or 'numpy'")


class Particle:
    def __init__(self, mass, position, velocity, dt=1e-2, softening=0.0):
        self.mass = float(mass)
        self.softening = float(softening)
        self.current_time = 0.0

        # store in NumPy
        self.position = np.asarray(position, dtype=np.float32)
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.acceleration = np.zeros_like(self.position, dtype=np.float32)

        self.dt = float(dt)
        self.model = None   # optional torch model
        self.device = torch.device("cpu")
        self.history_buffer: Optional[HistoryBuffer] = None

    # ----------------------------------

    def kinetic_energy(self):
        return 0.5 * self.mass * np.sum(self.velocity ** 2)

    # ----------------------------------

    def update_dt(self, secondary, eps=0.1):
        """Analytic dt update (NumPy only)."""
        dist = np.linalg.norm(self.position - secondary.position)
        acc_norm = np.linalg.norm(self.acceleration)

        if acc_norm == 0.0:
            self.dt = float(eps * np.sqrt(dist + 1e-12))
        else:
            self.dt = float(eps * np.sqrt(dist / acc_norm))

        return self.dt

    # ----------------------------------

    def update_model(self, model, device):
        """Attach a trained PyTorch model."""
        self.model = model
        self.device = device

    # ----------------------------------

    def attach_history_buffer(self, history_buffer: Optional[HistoryBuffer]):
        """Register a HistoryBuffer that matches the training setup."""
        if history_buffer is not None and not isinstance(history_buffer, HistoryBuffer):
            raise TypeError("attach_history_buffer expects a HistoryBuffer instance")
        self.history_buffer = history_buffer

    # ----------------------------------

    def update_dt_from_model(self, secondary=None, eps=1e-6, system=None, feature_mode: str = "basic"):
        """
        Use a PyTorch model to compute dt while all particle data stays NumPy.
        """
        if self.model is None:
            raise RuntimeError("No model attached. Use update_model().")

        if system is not None:
            dt = predict_dt_from_model_system(
                self.model,
                system=system,
                eps=eps,
                device=self.device,
                feature_mode=feature_mode,
            )
            for p in system:
                p.dt = dt
            self.dt = dt
            return dt

        dt = predict_dt_from_model(
            self.model,
            self,
            secondary,
            eps=eps,
            device=self.device,
            feature_mode=feature_mode,
        )
        self.dt = dt
        return dt

    # ----------------------------------

    def update_dt_from_history_model(
        self,
        secondary,
        history_buffer: Optional[HistoryBuffer] = None,
        eps: float = 1e-6,
        push_history: bool = True,
        system: Optional[Sequence["Particle"]] = None,
    ):
        """Predict dt using a history-aware model trained with HistoryBuffer."""
        if self.model is None:
            raise RuntimeError("No model attached. Use update_model().")

        hb = history_buffer or self.history_buffer
        if hb is None:
            raise RuntimeError(
                "No HistoryBuffer available. Pass one explicitly or call attach_history_buffer()."
            )

        device = getattr(self, "device", torch.device("cpu"))
        if system is not None:
            state = _system_to_history_state(system, device=device)
        else:
            if secondary is None:
                raise ValueError("A secondary particle is required.")
            state = _pair_to_history_state(self, secondary, device=device)
        feats = hb.features_for(state)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        feats = feats.to(device=device, dtype=torch.double)

        model = self.model.to(device)
        model.eval()
        with torch.no_grad():
            params = model(feats)
            if params.dim() == 1:
                params = params.unsqueeze(0)
            dt_raw = params[:, 0]

        dt_tensor = dt_raw + eps
        dt_value = float(dt_tensor.squeeze(0).item())
        self.dt = dt_value

        if system is not None:
            for p in system:
                p.dt = dt_value
            if push_history:
                _maybe_push_history_state_system(hb, state, system)
        else:
            if push_history:
                _maybe_push_history_state(hb, state, self, secondary)

        return dt_value

    # ----------------------------------

    def update_time(self, dt):
        self.current_time += float(dt)

    # ----------------------------------

    def __repr__(self):
        return f"Particle(m={self.mass}, pos={self.position}, vel={self.velocity})"

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Particle has no attribute '{key}'")


def system_from_particles(
    particles: Sequence["Particle"],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> ParticleTorch:
    if not particles:
        raise ValueError("system_from_particles requires a non-empty particle sequence")
    if dtype is None:
        dtype = torch.double

    pos = torch.tensor(
        np.stack([p.position for p in particles], axis=0),
        dtype=dtype,
        device=device,
    )
    vel = torch.tensor(
        np.stack([p.velocity for p in particles], axis=0),
        dtype=dtype,
        device=device,
    )
    mass = torch.tensor([p.mass for p in particles], dtype=dtype, device=device)

    dt_value = min(p.dt for p in particles)
    softening = max(p.softening for p in particles)
    dt_tensor = torch.tensor(dt_value, dtype=dtype, device=device)

    state = ParticleTorch.from_tensors(
        mass=mass,
        position=pos,
        velocity=vel,
        dt=dt_tensor,
        softening=softening,
    )
    return state


def predict_dt_from_model_system(
    model,
    system: Sequence["Particle"],
    eps: float = 1e-6,
    device: torch.device | None = None,
    feature_mode: str = "basic",
):
    if device is None:
        device = torch.device("cpu")

    state = system_from_particles(system, device=device, dtype=torch.double)
    feats = state.system_features(mode=feature_mode)
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        params = model(feats)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        dt_raw = params[:, 0]

    dt_tensor = dt_raw + eps
    if dt_tensor.numel() == 1:
        return float(dt_tensor.squeeze(0).item())
    return dt_tensor


def predict_dt_from_history_model_system(
    model,
    system: Sequence["Particle"],
    history_buffer: HistoryBuffer,
    eps: float = 1e-6,
    device: torch.device | None = None,
    push_history: bool = True,
):
    if device is None:
        device = torch.device("cpu")
    if history_buffer is None:
        raise ValueError("history_buffer is required for history-based prediction")

    state = _system_to_history_state(system, device=device)
    feats = history_buffer.features_for(state)
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)
    feats = feats.to(device=device, dtype=torch.double)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        params = model(feats)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        dt_raw = params[:, 0]

    dt_value = float((dt_raw + eps).squeeze(0).item())
    if push_history:
        _maybe_push_history_state_system(history_buffer, state, system)
    return dt_value


def predict_dt_from_model(
    model,
    p1,
    p2=None,
    eps=1e-6,
    device=None,
    system=None,
    feature_mode: str = "basic",
):
    """
    Compute dt from a PyTorch model while inputs are NumPy arrays.
    Supports either a 2-body pair (p1, p2) or an N-body system list.
    """
    if device is None:
        device = torch.device("cpu")

    if system is not None:
        return predict_dt_from_model_system(
            model,
            system=system,
            eps=eps,
            device=device,
            feature_mode=feature_mode,
        )
    if p2 is None:
        raise ValueError("predict_dt_from_model requires both p1 and p2 or a system list")

    return predict_dt_from_model_system(
        model,
        system=[p1, p2],
        eps=eps,
        device=device,
        feature_mode=feature_mode,
    )

def predict_dt_from_model_torch(
    model,
    p1,
    p2=None,
    eps=1e-6,
    device=None,
    system=None,
    feature_mode: str = "basic",
):
    """
    Predict dt using torch-based N-body system features.
    """
    if device is None:
        device = torch.device("cpu")
    if system is not None:
        return predict_dt_from_model_system(
            model,
            system=system,
            eps=eps,
            device=device,
            feature_mode=feature_mode,
        )
    if p2 is None:
        raise ValueError("predict_dt_from_model_torch requires both p1 and p2 or a system list")
    return predict_dt_from_model_system(
        model,
        system=[p1, p2],
        eps=eps,
        device=device,
        feature_mode=feature_mode,
    )

def predict_dt_from_model_(
    model,
    p1,
    p2=None,
    eps=1e-6,
    device=None,
    system=None,
    feature_mode: str = "basic",
):
    """
    Legacy wrapper that forwards to N-body system features.
    """
    if device is None:
        device = torch.device("cpu")
    if system is not None:
        return predict_dt_from_model_system(
            model,
            system=system,
            eps=eps,
            device=device,
            feature_mode=feature_mode,
        )
    if p2 is None:
        raise ValueError("predict_dt_from_model_ requires both p1 and p2 or a system list")
    return predict_dt_from_model_system(
        model,
        system=[p1, p2],
        eps=eps,
        device=device,
        feature_mode=feature_mode,
    )


def _system_to_history_state(particles: Sequence[Particle], device: torch.device) -> ParticleTorch:
    pos = torch.tensor(
        np.stack([p.position for p in particles], axis=0),
        dtype=torch.double,
        device=device,
    )
    vel = torch.tensor(
        np.stack([p.velocity for p in particles], axis=0),
        dtype=torch.double,
        device=device,
    )
    mass = torch.tensor([p.mass for p in particles], dtype=torch.double, device=device)
    dt_value = min(p.dt for p in particles)
    dt_tensor = torch.tensor(dt_value, dtype=torch.double, device=device)
    softening = max(p.softening for p in particles)

    state = ParticleTorch.from_tensors(
        mass=mass,
        position=pos,
        velocity=vel,
        dt=dt_tensor,
        softening=softening,
    )
    return state


def _pair_to_history_state(primary: Particle, secondary: Particle, device: torch.device) -> ParticleTorch:
    return _system_to_history_state([primary, secondary], device=device)


def _maybe_push_history_state(
    history_buffer: HistoryBuffer,
    state: ParticleTorch,
    primary: Particle,
    secondary: Particle,
):
    token = (float(primary.current_time), float(secondary.current_time))
    last_token = getattr(history_buffer, "_last_push_token", None)
    if last_token == token:
        return
    history_buffer.push(state)
    history_buffer._last_push_token = token


def _maybe_push_history_state_system(
    history_buffer: HistoryBuffer,
    state: ParticleTorch,
    particles: Sequence[Particle],
):
    token = tuple(float(p.current_time) for p in particles)
    last_token = getattr(history_buffer, "_last_push_token", None)
    if last_token == token:
        return
    history_buffer.push(state)
    history_buffer._last_push_token = token
