from typing import Optional

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

    def update_dt_from_model(self, secondary=None, eps=1e-6):
        """
        Use a PyTorch model to compute dt while all particle data stays NumPy.
        """
        if self.model is None:
            raise RuntimeError("No model attached. Use update_model().")


        dt = predict_dt_from_model(
            self.model, self, secondary,
            eps=eps, device=self.device
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
    ):
        """Predict dt using a history-aware model trained with HistoryBuffer."""
        if self.model is None:
            raise RuntimeError("No model attached. Use update_model().")
        if secondary is None:
            raise ValueError("A secondary particle is required.")

        hb = history_buffer or self.history_buffer
        if hb is None:
            raise RuntimeError(
                "No HistoryBuffer available. Pass one explicitly or call attach_history_buffer()."
            )

        device = getattr(self, "device", torch.device("cpu"))
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


def predict_dt_from_model(model, p1, p2=None, eps=1e-6, device=None):
    """
    Compute dt from a PyTorch model while inputs are NumPy arrays.
    Returns a Python float.
    """
    if device is None:
        device = torch.device("cpu")

    if p2 is None:
        raise ValueError("predict_dt_from_model requires both p1 and p2")

    # ---------------------------------------------------------
    # 1. Distance between the two particles (NumPy)
    # ---------------------------------------------------------
    # r = x1 - x0
    r = p2.position - p1.position          # shape (ndim,)
    dist = np.linalg.norm(r)               # scalar

    # ---------------------------------------------------------
    # 2. Acceleration magnitude of chosen particle (here: p1)
    # ---------------------------------------------------------
    acc = p1.acceleration                  # shape (ndim,)
    a_mag = np.linalg.norm(acc)           # scalar

    # ---------------------------------------------------------
    # 3. Build feature vector [dist, a_mag] to match NN training
    # ---------------------------------------------------------
    feat_np = np.array([dist, a_mag], dtype=np.double)  # shape (2,)
    x = torch.from_numpy(feat_np).to(device=device, dtype=torch.double).unsqueeze(0)  # shape (1, 2)


    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dt, E = model(x).squeeze(0)      # scalar

    #print(x, dt)

    return float(dt.item())              # return pure Python float

def predict_dt_from_model_torch(model, p1, p2=None, G=1.0, which_accel=0, eps=1e-6, device=None):
    """
    Predict dt (and E_hat) from a PyTorch model using the full
    feature vector produced by ParticleTorch.

    Feature vector format:

        [distance,
         accel_mag,
         pos0_x, pos0_y, ..., pos1_x, pos1_y, ...,
         vel0_x, vel0_y, ..., vel1_x, vel1_y, ...]

    Returns: dt (python float)
    """
    if device is None:
        device = torch.device("cpu")

    if p2 is None:
        raise ValueError("predict_dt_from_model requires both p1 and p2")

    # ---------------------------------------------------------
    # 1. Build a temporary ParticleTorch(batch=1) containing 2 particles
    # ---------------------------------------------------------
    # expected ParticleTorch.from_tensors:
    #   position: (B, N, dim)
    #   velocity: (B, N, dim)
    #   mass:     (B, N)
    # p1 and p2 are single particles → shape them into batch=1

    pos = torch.stack([p1.position, p2.position], dim=0).unsqueeze(0)   # (1, 2, dim)
    vel = torch.stack([p1.velocity, p2.velocity], dim=0).unsqueeze(0)   # (1, 2, dim)
    mass = torch.tensor([p1.mass, p2.mass], dtype=pos.dtype).unsqueeze(0)  # (1, 2)

    # accelerations as well
    acc = torch.stack([p1.acceleration, p2.acceleration], dim=0).unsqueeze(0)  # (1, 2, dim)

    # Build a tiny ParticleTorch object matching training format
    tmp = ParticleTorch.from_tensors(
        mass=mass.to(device),
        position=pos.to(device),
        velocity=vel.to(device),
    )
    tmp.acceleration = acc.to(device)

    # ---------------------------------------------------------
    # 2. Compute feature vector exactly like training
    # ---------------------------------------------------------
    feats = tmp.features(G=G, which_accel=which_accel)   # shape (1, F)

    # ---------------------------------------------------------
    # 3. Run model
    # ---------------------------------------------------------
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        params = model(feats)    # (1, 2)
        dt_raw     = params[0, 0]
        E_hat_raw  = params[0, 1]

    # enforce positivity with softplus-like lower bound
    dt = dt_raw + eps

    return float(dt.item())

def predict_dt_from_model_(model, p1, p2=None, G=1.0, which_accel=0, eps=1e-6, device=None):
    """
    Predict dt from a trained PyTorch model using NumPy-based feature
    extraction matching the NN training format:

        features = [
            distance,
            accel_mag,
            pos0_x, pos0_y, ...,
            pos1_x, pos1_y, ...,
            vel0_x, vel0_y, ...,
            vel1_x, vel1_y, ...
        ]

    Inputs p1, p2 must have:
        p.position      -> numpy array (dim,)
        p.velocity      -> numpy array (dim,)
        p.acceleration -> numpy array (dim,)
    """
    if device is None:
        device = torch.device("cpu")

    if p2 is None:
        raise ValueError("predict_dt_from_model requires both p1 and p2")

    # ---------------------------------------------------------
    # 1. Positions, velocities, accelerations (NumPy)
    # ---------------------------------------------------------
    pos0 = np.asarray(p1.position, dtype=np.float64)   # (dim,)
    pos1 = np.asarray(p2.position, dtype=np.float64)
    vel0 = np.asarray(p1.velocity, dtype=np.float64)
    vel1 = np.asarray(p2.velocity, dtype=np.float64)
    acc0 = np.asarray(p1.acceleration, dtype=np.float64)
    acc1 = np.asarray(p2.acceleration, dtype=np.float64)

    # dimension
    dim = pos0.shape[0]

    # ---------------------------------------------------------
    # 2. Distance between the two particles
    # ---------------------------------------------------------
    r = pos1 - pos0                   # (dim,)
    dist = np.linalg.norm(r)          # scalar

    # ---------------------------------------------------------
    # 3. Acceleration magnitude (choose particle 0 or 1)
    # ---------------------------------------------------------
    if which_accel == 0:
        a_mag = np.linalg.norm(acc0)
    else:
        a_mag = np.linalg.norm(acc1)

    # ---------------------------------------------------------
    # 4. Flatten position and velocity info
    # ---------------------------------------------------------
    # pos0 = (dim,), pos1 = (dim,) → pos_flat shape: (2*dim,)
    pos_flat = np.concatenate([pos0, pos1], axis=0)

    # vel0 + vel1 → (2*dim,)
    vel_flat = np.concatenate([vel0, vel1], axis=0)

    # ---------------------------------------------------------
    # 5. Build final feature vector
    # ---------------------------------------------------------
    # [ dist, a_mag, pos0..., pos1..., vel0..., vel1... ]
    feat_np = np.concatenate([
        np.array([dist, a_mag], dtype=np.float64),
        pos_flat,
        vel_flat
    ], axis=0)

    # final shape = (2 + 2*dim + 2*dim,) → (2 + 4*dim,)
    # example for dim=2 → shape (10,)

    # ---------------------------------------------------------
    # 6. Convert to PyTorch batch = (1, F)
    # ---------------------------------------------------------
    x = torch.from_numpy(feat_np).to(device=device, dtype=torch.double)
    x = x.unsqueeze(0)   # (1, F)

    # ---------------------------------------------------------
    # 7. Run model
    # ---------------------------------------------------------
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        params = model(x)        # (1, 2)
        dt_raw = params[0, 0]
        E_raw  = params[0, 1]

    # safety: enforce positivity
    dt = dt_raw + eps

    return float(dt.item())


def _pair_to_history_state(primary: Particle, secondary: Particle, device: torch.device) -> ParticleTorch:
    pos = torch.tensor(
        np.stack([primary.position, secondary.position], axis=0),
        dtype=torch.double,
        device=device,
    )
    vel = torch.tensor(
        np.stack([primary.velocity, secondary.velocity], axis=0),
        dtype=torch.double,
        device=device,
    )
    mass = torch.tensor([primary.mass, secondary.mass], dtype=torch.double, device=device)
    dt_value = min(primary.dt, secondary.dt)
    dt_tensor = torch.tensor(dt_value, dtype=torch.double, device=device)

    state = ParticleTorch.from_tensors(
        mass=mass,
        position=pos,
        velocity=vel,
        dt=dt_tensor,
        softening=max(primary.softening, secondary.softening),
    )
    return state


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