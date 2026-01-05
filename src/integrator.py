import torch
from .particle import *

def simple_integrator(p: ParticleTorch, n_steps: int):
    """
    Toy example: free drift for n_steps. Replace with your real
    two-body or N-body integrator, but keep it torch-only.
    """
    pos = p.position
    vel = p.velocity
    dt  = p.dt

    # do NOT write pos += ...
    for _ in range(n_steps):
        pos = pos + vel * dt

    # we don't touch p in-place here; we just return the final tensor
    return pos


