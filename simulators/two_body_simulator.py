import torch
import numpy as np
from .particle import *


# Gravitational constant (SI units). You can set G = 1.0 for code units.
#G = 6.67430e-11
G = 1

def acceleration(p1, p2):
    """
    Acceleration on body p1 due to body p2.

    p1, p2: Particle
    Uses their backend (numpy or torch) and p1.softening.
    """
    r = p2.position - p1.position  # vector from 1 -> 2
    # xp.sum works for both numpy and torch
    dist2 = np.sum(r * r) + p1.softening ** 2
    dist = np.sqrt(dist2)
    dist3 = dist2 * dist

    # G * m2 * r / |r|^3
    a = G * p2.mass * r / dist3
    return a

def total_energy(p1, p2):
    # Kinetic
    K = p1.kinetic_energy() + p2.kinetic_energy()

    # Gravitational potential
    r = np.linalg.norm(p1.position - p2.position)
    U = - p1.mass * p2.mass / r     # softened version if needed

    return float(K + U)

def total_momentum(p1, p2):
    """
    Returns the total linear momentum vector of a two-particle system.
    """
    P = p1.mass * p1.velocity + p2.mass * p2.velocity
    return P.copy()   # or np.array(P)


def total_angular_momentum_com(p1, p2):
    M = p1.mass + p2.mass
    R_com = (p1.mass * p1.position + p2.mass * p2.position) / M
    V_com = (p1.mass * p1.velocity + p2.mass * p2.velocity) / M

    L1 = np.cross(p1.position - R_com, p1.mass * (p1.velocity - V_com))
    L2 = np.cross(p2.position - R_com, p2.mass * (p2.velocity - V_com))
    return L1 + L2



def evolve_dt(p1, p2, eps, isML=False, adapter=None):
    """
    Advance the two-body system by one shared timestep using symplectic
    leapfrog / velocity-Verlet in kick-drift-kick form:

        v_{n+1/2} = v_n + 0.5 * a(x_n) * dt
        x_{n+1}   = x_n + v_{n+1/2} * dt
        a_{n+1}   = a(x_{n+1})
        v_{n+1}   = v_{n+1/2} + 0.5 * a_{n+1} * dt

    Uses dt = min(p1.dt, p2.dt).
    Updates p1.position, p1.velocity, p2.position, p2.velocity (and acceleration) in-place.
    Returns the dt used.
    """

    # --- accelerations at current positions a(x_n) ---
    a1 = acceleration(p1, p2)
    a2 = acceleration(p2, p1)

    # store old accel
    p1.acceleration = a1
    p2.acceleration = a2

    # choose shared timestep (you can plug in ITS logic inside update_dt)
    if not isML:
        p1.update_dt(p2, eps)
        p2.update_dt(p1, eps)
        dt = min(p1.dt, p2.dt)
        #print("dt =", dt)
    else:
        uses_history = (
            getattr(p1, "history_buffer", None) is not None and
            getattr(p2, "history_buffer", None) is not None
        )
        if uses_history:
            dt1 = p1.update_dt_from_history_model(secondary=p2, adapter=adapter)
            dt2 = p2.update_dt_from_history_model(secondary=p1, adapter=adapter)
        else:
            dt1 = p1.update_dt_from_model(secondary=p2, adapter=adapter)
            dt2 = p2.update_dt_from_model(secondary=p1, adapter=adapter)
        dt = min(dt1, dt2)
        prefix = "ML history dt =" if uses_history else "ML dt ="
        #print(prefix, dt)

    # --- kick: v_{n+1/2} = v_n + 0.5 * a_n * dt ---
    v1_half = p1.velocity + 0.5 * a1 * dt
    v2_half = p2.velocity + 0.5 * a2 * dt

    # --- drift: x_{n+1} = x_n + v_{n+1/2} * dt ---
    x1_new = p1.position + v1_half * dt
    x2_new = p2.position + v2_half * dt

    # update positions to x_{n+1} before recomputing acceleration
    p1.position = x1_new
    p2.position = x2_new

    # --- new accelerations a_{n+1} = a(x_{n+1}) ---
    a1_new = acceleration(p1, p2)
    a2_new = acceleration(p2, p1)

    # --- final kick: v_{n+1} = v_{n+1/2} + 0.5 * a_{n+1} * dt ---
    v1_new = v1_half + 0.5 * a1_new * dt
    v2_new = v2_half + 0.5 * a2_new * dt

    # store updated velocities and accelerations
    p1.velocity = v1_new
    p2.velocity = v2_new

    p1.acceleration = a1_new
    p2.acceleration = a2_new

    # advance times
    p1.update_time(dt)
    p2.update_time(dt)

    return dt


def generate_IC(e=0,a=1.0,dt=1e-4):
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

    p1 = Particle(m1, [x1, y1], [vx1, vy1], dt=dt)  # use small dt!
    p2 = Particle(m2, [x2, y2], [vx2, vy2], dt=dt)

    T = 2 * np.pi * np.sqrt(a**3 / (G * M))
    print("Orbital period T =", T)

    return p1, p2, T
