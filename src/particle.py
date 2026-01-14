import torch
import numpy as np

from typing import Optional

from .nbody_features import system_features as nbody_system_features
from .external_potentials import ExternalField

class ParticleTorch:
    """
    Batched, differentiable particle state for PyTorch.

    Shapes:
      position: (..., ndim)
      velocity: (..., ndim)
      acceleration: (..., ndim)
      mass: scalar or tensor broadcastable to position[..., 0]
      dt: scalar or tensor broadcastable
    """

    def __init__(self, mass, position, velocity,
                 dt=1e-3, softening=0.0, device=None):
        """
        For *non-training* use (e.g. quick tests). For training, prefer
        the `from_tensors` constructor below so you don't break gradients.
        """
        if device is None:
            device = torch.device("cpu")

        if not torch.is_tensor(position):
            position = torch.tensor(position, dtype=torch.float32, device=device)
        else:
            position = position.to(device)

        if not torch.is_tensor(velocity):
            velocity = torch.tensor(velocity, dtype=torch.float32, device=device)
        else:
            velocity = velocity.to(device)

        self.position = position
        self.velocity = velocity
        self.acceleration = torch.zeros_like(self.position)

        # mass & dt as tensors (can be scalars or broadcastable)
        self.mass = torch.as_tensor(mass, dtype=torch.float32, device=device)
        self.dt   = torch.as_tensor(dt,   dtype=torch.float32, device=device)

        self.softening = float(softening)
        self.current_time = torch.as_tensor(0.0, dtype=torch.float32, device=device)
        self.period = torch.as_tensor(0.0, dtype=torch.float32, device=device)
        #self.current_time = 0.0

        # Optional analytic external field (e.g. tides) applied on top of N-body gravity.
        self.external_field: Optional[ExternalField] = None

    @classmethod
    def from_tensors(cls, mass, position, velocity,
                     dt=0.0, softening=0.0):
        """
        Constructor for TRAINING: takes tensors and stores them directly
        without wrapping them in new tensors (keeps gradients intact).
        """
        obj = cls.__new__(cls)  # create uninitialized instance

        # position, velocity are tensors coming from your model
        assert torch.is_tensor(position)
        assert torch.is_tensor(velocity)

        device = position.device
        obj.position = position
        obj.velocity = velocity
        obj.acceleration = torch.zeros_like(position)

        dtype = position.dtype
        obj.mass = torch.as_tensor(mass, dtype=dtype, device=device)
        obj.dt   = torch.as_tensor(dt,   dtype=dtype, device=device)

        obj.softening = float(softening)
        obj.current_time = 0.0
        obj.external_field = None
        return obj

    def set_external_field(self, field: Optional[ExternalField]):
        """Attach an analytic external field (e.g. tidal) to this particle state."""
        self.external_field = field

    def clone_detached(self):
        """
        Make a copy whose state is detached from the current graph.
        Useful if you want a 'fresh' particle for a new forward pass.
        """
        pos = self.position.detach().clone()
        vel = self.velocity.detach().clone()
        acc = self.acceleration.detach().clone()
        dt  = self.dt.detach().clone()
        mass = self.mass.detach().clone()

        cloned = ParticleTorch.from_tensors(
            mass=mass, position=pos, velocity=vel,
            dt=dt, softening=self.softening
        )
        cloned.external_field = getattr(self, "external_field", None)
        return cloned

    def reset_state(self, position, velocity, dt=None):
        """
        Reuse the same Python object but swap in new tensors (from model
        outputs) for a new forward pass.
        """
        assert torch.is_tensor(position)
        assert torch.is_tensor(velocity)
        self.position = position
        self.velocity = velocity
        self.acceleration = torch.zeros_like(position)

        if dt is not None:
            self.dt = torch.as_tensor(dt, dtype=torch.float32,
                                      device=position.device)

        self.current_time = 0.0

    def get_acceleration(self, G=1.0):
        """
        Compute gravitational acceleration on each particle from all others.

        Assumes:
          position shape: (..., N, ndim)
          mass shape: scalar or broadcastable to (..., N)

        Uses softened Newtonian gravity:
          a_i = G * sum_j m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2)
        """
        pos = self.position  # (..., N, ndim)
        soft2 = self.softening ** 2

        # Build pairwise separation r_ij = x_j - x_i
        # pos_i: (..., N, 1, ndim), pos_j: (..., 1, N, ndim)
        pos_i = pos.unsqueeze(-2)
        pos_j = pos.unsqueeze(-3)
        r_ij = pos_j - pos_i          # (..., N, N, ndim), i = -3, j = -2

        # Squared distances with softening
        dist2 = (r_ij ** 2).sum(dim=-1) + soft2  # (..., N, N)

        # Zero self-interaction by sending diagonal to infinity
        N = pos.shape[-2]
        #idx = torch.arange(N, device=pos.device)
        #dist2[..., idx, idx] = float('inf')
        I = torch.eye(N, device=pos.device, dtype=torch.bool)
        dist2 = dist2.masked_fill(I, float('inf'))

        inv_r3 = dist2.pow(-1.5)  # (..., N, N)

        # Handle masses: scalar or per-particle (..., N)
        m = self.mass
        if torch.is_tensor(m) and m.ndim > 0:
            # mass shape (..., N) -> (..., 1, N) so it is m_j along j index
            m_j = m.unsqueeze(-2)
            # (..., N, N) * (..., 1, N) -> (..., N, N)
            inv_r3_m = inv_r3 * m_j
        else:
            # scalar mass (shared by all particles)
            inv_r3_m = inv_r3 * m

        # Combine: r_ij * m_j / |r_ij|^3, sum over j
        # r_ij:    (..., N, N, ndim)
        # inv_r3_m:(..., N, N)
        accel = G * (r_ij * inv_r3_m.unsqueeze(-1)).sum(dim=-2)  # (..., N, ndim)

        field = getattr(self, "external_field", None)
        if field is not None:
            t = getattr(self, "current_time", 0.0)
            accel = accel + field.acceleration(pos, t)

        self.acceleration = accel
        return accel

    def evolve(self, G=1.0):
        """
        One step of symplectic leapfrog / velocity-Verlet integration:

        v_{n+1/2} = v_n + 0.5 * a(x_n) * dt
        x_{n+1}   = x_n + v_{n+1/2} * dt
        a_{n+1}   = a(x_{n+1})
        v_{n+1}   = v_{n+1/2} + 0.5 * a_{n+1} * dt

        Updates self.position, self.velocity, self.acceleration in place
        (out-of-place tensor ops, so autograd is fine).
        """
        dt = self.dt

        # Acceleration at current positions
        a_n = self.get_acceleration(G=G)

        # Kick to half step in velocity
        v_half = self.velocity + 0.5 * a_n * dt

        # Drift positions
        x_new = self.position + v_half * dt

        # Update position
        self.position = x_new

        # New acceleration at updated positions
        a_new = self.get_acceleration(G=G)

        # Complete velocity kick
        v_new = v_half + 0.5 * a_new * dt

        self.velocity = v_new
        self.acceleration = a_new

        # current_time is bookkeeping only, not part of the graph
        self.current_time = self.current_time + dt.detach()


    def evolve_batch(self, G=1.0):
        """
        One step of symplectic leapfrog / velocity-Verlet integration.

        Supports both:
        - single system: position (N, D), velocity (N, D), dt scalar
        - batched systems: position (B, N, D), velocity (B, N, D),
            dt scalar or dt (B,)

        dt is broadcast over particles and spatial dims.
        """

        pos = self.position        # (N, D) or (B, N, D)
        vel = self.velocity        # same shape as pos
        dt  = self.dt              # scalar or (B,)

        # ---- Make dt a tensor, on the right device, and broadcastable ----
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, dtype=pos.dtype, device=pos.device)

        # Now ensure dt has at least as many dims as pos minus the last two (N, D)
        # For batched case:
        #   pos: (B, N, D), dt: (B,)  -> we reshape dt -> (B, 1, 1)
        # For single case:
        #   pos: (N, D), dt: ()       -> we keep it scalar, broadcasting works
        if dt.dim() == 1 and pos.dim() == 3:
            # (B,) -> (B, 1, 1)
            dt = dt.view(-1, 1, 1)
        # If dt is 0-dim (scalar), broadcasting already works against (N,D) or (B,N,D)

        # ---- 1. Acceleration at current positions ----
        a_n = self.get_acceleration(G=G)  # same shape as pos

        # ---- 2. Kick to half step in velocity ----
        v_half = vel + 0.5 * a_n * dt     # broadcast dt

        # ---- 3. Drift positions ----
        x_new = pos + v_half * dt

        # Update position
        self.position = x_new

        # ---- 4. New acceleration at updated positions ----
        a_new = self.get_acceleration(G=G)

        # ---- 5. Complete velocity kick ----
        v_new = v_half + 0.5 * a_new * dt

        self.velocity = v_new
        self.acceleration = a_new

        # current_time is bookkeeping only, not part of the graph
        # You can keep it scalar; for batched dt we just advance by the
        # mean (or max) dt, depending on your convention.
        dt_for_time = dt
        if dt_for_time.dim() > 0:
            dt_for_time = dt_for_time.mean()   # or .max()

        # Make sure current_time is a tensor on same device
        if not torch.is_tensor(self.current_time):
            self.current_time = torch.tensor(
                float(self.current_time),
                dtype=pos.dtype,
                device=pos.device,
            )

        self.current_time = self.current_time + dt_for_time.detach()


    def total_energy(self, G=1.0):
        """
        Total energy = Kinetic + Potential

        K = 0.5 * m * v^2  (sum over particles)
        U = - G * sum_{i < j} m_i m_j / sqrt(r_ij^2 + eps^2)

        All operations are:
          • differentiable
          • GPU friendly
          • batch friendly
        """
        KE = self.kinetic_energy()

        pos = self.position  # (N, D)
        m = self.mass        # could be scalar or (N,)

        # Pairwise differences
        dx = pos[:, None, :] - pos[None, :, :]   # (N, N, 3)
        dist2 = (dx ** 2).sum(dim=-1)            # (N, N)

        # Add softening^2 and small epsilon
        eps = 1e-12
        dist2 = dist2 + (self.softening ** 2) + eps

        # Safe sqrt
        dist = torch.sqrt(dist2)

        # Inverse distance, but zero out self interactions
        N = pos.shape[0]
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        inv_dist = torch.where(eye, torch.zeros_like(dist), 1.0 / dist)

        # Potential energy: -G * sum_{i<j} m_i m_j / r_ij
        # handle scalar or per-particle masses
        if torch.is_tensor(m):
            if m.dim() == 0:
                m_i = m.expand(N)
            elif m.dim() == 1 and m.shape[0] == N:
                m_i = m
            else:
                m_i = m.reshape(-1)
                if m_i.shape[0] != N:
                    raise ValueError(f"mass shape {m.shape} incompatible with position shape {pos.shape}")
        else:
            m_i = torch.tensor(m, device=pos.device, dtype=pos.dtype).expand(N)
        m_j = m_i[None, :]
        m_i = m_i[:, None]

        pot = -0.5 * G * (m_i * m_j * inv_dist).sum()
        self.KE = KE
        self.pot = pot

        #print("KE =", KE, "PE =", pot)

        E = KE + pot

        field = getattr(self, "external_field", None)
        if field is not None:
            t = getattr(self, "current_time", 0.0)
            phi = field.potential(self.position, t)  # (N,)

            # Broadcast mass to per-particle if needed
            m = self.mass
            if torch.is_tensor(m) and m.dim() == 0:
                m = m.expand_as(phi)
            elif not torch.is_tensor(m):
                m = torch.tensor(float(m), device=phi.device, dtype=phi.dtype).expand_as(phi)

            E = E + (m * phi).sum()

        return E

 
    def total_energy_batch(self, G: float = 1.0):
        """
        Compute total energy (kinetic + potential) for possibly batched systems.

        Supports:
          position: (N, D)       or (B, N, D)
          velocity: (N, D)       or (B, N, D)
          mass:     (N,) or (B, N)
        Returns:
          E: shape (B,) for batched, or scalar tensor for single system.
        """
        pos = self.position    # (N, D) or (B, N, D)
        vel = self.velocity    # same
        mass = self.mass       # (N,) or (B, N)
        soft = getattr(self, "softening", 0.0)

        # --- 1. Normalize to batched shape (B, N, D) and (B, N) ---
        if pos.dim() == 2:
            # (N, D) -> (1, N, D)
            pos = pos.unsqueeze(0)
            vel = vel.unsqueeze(0)
        if mass.dim() == 1:
            # (N,) -> (1, N)
            mass = mass.unsqueeze(0)

        B, N, D = pos.shape

        # --- 2. Kinetic energy ---
        # v^2 = vx^2 + vy^2 (+ vz^2 ...)
        v2 = (vel ** 2).sum(dim=-1)        # (B, N)
        KE = 0.5 * (mass * v2).sum(dim=-1) # (B,)

        # --- 3. Potential energy (pairwise, gravity) ---
        # pairwise distances r_ij
        # pos_i: (B, N, 1, D), pos_j: (B, 1, N, D)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)   # (B, N, N, D)
        dist2 = (diff ** 2).sum(dim=-1)              # (B, N, N)
        if soft != 0.0:
            dist2 = dist2 + soft ** 2

        dist = torch.sqrt(dist2 + 1e-30)             # avoid sqrt(0)
        # mask self-interactions i == j
        eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)  # (1, N, N)
        dist = torch.where(eye, torch.full_like(dist, float("inf")), dist)

        inv_r = 1.0 / dist                           # (B, N, N), diagonal ~ 0

        # masses broadcast to pairwise (i, j)
        m_i = mass.unsqueeze(2)   # (B, N, 1)
        m_j = mass.unsqueeze(1)   # (B, 1, N)
        pair_E = -G * m_i * m_j * inv_r  # (B, N, N)

        # each pair i-j appears twice (i,j) and (j,i) → divide by 2
        PE = 0.5 * pair_E.sum(dim=(1, 2))            # (B,)

        E = KE + PE                                  # (B,)

        field = getattr(self, "external_field", None)
        if field is not None:
            t = getattr(self, "current_time", 0.0)
            phi = field.potential(pos, t)  # (B, N)
            E = E + (mass * phi).sum(dim=1)

        # If you want scalar for non-batch case:
        if self.position.dim() == 2:
            return E.squeeze(0)  # scalar tensor
        else:
            return E             # (B,)



    def kinetic_energy(self, sum_over_all=True):
        """
        KE = 0.5 m v^2
        If sum_over_all=True, returns a scalar.
        Otherwise, returns per-particle energies with same leading shape
        as position[..., 0].
        """
        v2 = (self.velocity ** 2).sum(dim=-1)          # (...,)
        # broadcast mass if needed
        ke = 0.5 * self.mass * v2
        if sum_over_all:
            return ke.sum()
        return ke

    def update_dt(self, dt, eps=0.0):
        self.dt = dt

    def advance_free(self):
        """
        Simple free-particle drift: x <- x + v * dt.
        You can replace this with your own integrator that also updates
        acceleration from forces, etc.
        """
        # Use out-of-place updates to keep autograd happy
        self.position = self.position + self.velocity * self.dt
        self.current_time += float(self.dt.detach().cpu())

    def system_features(self, G=1.0, mode: str = "basic"):
        """
        Fixed-size, permutation-invariant features for an N-body system.
        mode="basic" or "rich".
        """
        return nbody_system_features(
            position=self.position,
            velocity=self.velocity,
            mass=self.mass,
            softening=self.softening,
            G=G,
            mode=mode,
        )

    def features(self, G=1.0, mode: str = "basic"):
        """Alias for system_features, kept for simulator compatibility."""
        return self.system_features(G=G, mode=mode)

    def get_batch(self, which_accel=0, G=1.0):
        """
        Backwards-compatible wrapper for rich, N-body features.
        which_accel is ignored (kept for API compatibility).
        """
        _ = which_accel
        return self.system_features(G=G, mode="rich")


    def _get_batch_(self, which_accel=0, G=1.0):
        """
        Backwards-compatible wrapper for basic, N-body features.
        which_accel is ignored (kept for API compatibility).
        """
        _ = which_accel
        return self.system_features(G=G, mode="basic")


    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Particle has no attribute '{key}'")

    def __repr__(self):
        return (f"ParticleTorch(m={self.mass}, "
                f"pos={self.position}, vel={self.velocity}, "
                f"dt={self.dt}, softening={self.softening})")




def make_particle(ptcls, device, dtype):
    """
    ptcls: shape (N, 1 + 2*D) -> [m, x..., v...]
    Returns a ParticleTorch whose position/velocity are tied to ptcls.
    """
    ptcls_t = torch.as_tensor(ptcls, dtype=dtype, device=device)
    if ptcls_t.dim() != 2:
        raise ValueError(f"ptcls must be 2D (N, 1+2*D), got {ptcls_t.shape}")

    n_cols = ptcls_t.shape[1]
    if (n_cols - 1) % 2 != 0:
        raise ValueError("ptcls must have 1 + 2*D columns ([m, x..., v...])")
    dim = (n_cols - 1) // 2

    position = ptcls_t[:, 1 : 1 + dim]
    velocity = ptcls_t[:, 1 + dim : 1 + 2 * dim]

    particle = ParticleTorch.from_tensors(
        mass=ptcls_t[:, 0],
        position=position,
        velocity=velocity,
    )
    return particle


def generate_IC(e=0.,a=1.0):
    G  = 1.0
    m1 = 1.0
    m2 = 1.0
    M  = m1 + m2

    r_p = a * (1.0 - e)                             # pericenter separation
    v_p = np.sqrt(G * M * (1.0 + e) / (a * (1.0 - e)))  # relative speed at pericenter

    print("r_p =", r_p)
    print("v_p =", v_p)

    # Positions (at pericenter)
    x1, y1 = -r_p / 2.0, 0.0   # ≈ -0.05
    x2, y2 =  r_p / 2.0, 0.0   # ≈ +0.05

    # Velocities (purely tangential)
    vx1, vy1 = 0.0,  +v_p / 2.0   # ≈ +3.08
    vx2, vy2 = 0.0,  -v_p / 2.0   # ≈ -3.08

    T = 2 * np.pi * np.sqrt(a**3 / (G * M))
    print("Orbital period T =", T)

    ptcls = np.zeros((2,5))
    ptcls[0,:] = [m1, x1, y1, vx1, vy1]
    ptcls[1,:] = [m2, x2, y2, vx2, vy2]

    return ptcls, T


def stack_particles(particles):
    pos = torch.stack([p.position for p in particles], dim=0)   # (B, N, D)
    vel = torch.stack([p.velocity for p in particles], dim=0)   # (B, N, D)
    mass = torch.stack([p.mass for p in particles], dim=0)      # (B, N) or (B,)
    out = ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel)
    fields = [getattr(p, "external_field", None) for p in particles]
    if len(fields) > 0 and all(f is fields[0] for f in fields):
        out.external_field = fields[0]
    return out
