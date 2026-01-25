#!/usr/bin/env python3
"""
Generate a collection of standard 2-body Initial Conditions (ICs) for training and simulation.
These files can be loaded using the `--ic-path` argument in runner.py.
"""

import numpy as np
import pathlib
import argparse

def generate_orbit_ic(eccentricity: float, semi_major: float, mass: float = 1.0, dim: int = 2) -> np.ndarray:
    """
    Generate 2-body ICs for a given eccentricity and semi-major axis.
    Aligns the orbit such that particles start at pericenter (closest approach).
    """
    m1 = m2 = mass
    G = 1.0
    
    # Separation and velocity at pericenter
    r_p = semi_major * (1.0 - eccentricity)
    v_p = np.sqrt(G * (m1 + m2) * (1.0 + eccentricity) / (semi_major * (1.0 - eccentricity)))
    
    # Initialize arrays
    # Shape: (2, 1 + 2*dim) -> [mass, x, y, ..., vx, vy, ...]
    ic = np.zeros((2, 1 + 2 * dim), dtype=float)
    
    # Masses
    ic[0, 0] = m1
    ic[1, 0] = m2
    
    # Positions (centered at origin)
    # Particle 1 at -r_p/2, Particle 2 at +r_p/2 along x-axis
    ic[0, 1] = -0.5 * r_p
    ic[1, 1] = +0.5 * r_p
    
    # Velocities (perpendicular to position)
    # Particle 1 moves +y, Particle 2 moves -y
    ic[0, dim + 2] = +0.5 * v_p  # vy is at index dim + 2 (1 for mass + dim for pos + 1 for vx)
    ic[1, dim + 2] = -0.5 * v_p
    
    return ic

def main():
    parser = argparse.ArgumentParser(description="Generate standard 2-body ICs.")
    parser.add_argument("--out-dir", type=str, default="data/ics", help="Output directory")
    args = parser.parse_args()
    
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    configs = [
        {"name": "circular", "e": 0.0, "a": 1.0},
        {"name": "elliptic_mild", "e": 0.3, "a": 1.0},
        {"name": "elliptic_medium", "e": 0.6, "a": 1.0},
        {"name": "elliptic_high", "e": 0.9, "a": 1.0},
        {"name": "elliptic_extreme", "e": 0.95, "a": 1.0},
    ]
    
    print(f"Generating ICs in {out_dir}...")
    
    for cfg in configs:
        ic = generate_orbit_ic(cfg["e"], cfg["a"])
        filename = f"2body_{cfg['name']}_e{cfg['e']:.2f}.npy"
        path = out_dir / filename
        np.save(path, ic)
        print(f"  Saved {filename}")
        
    print("\nTo use these in training or simulation:")
    print(f"  python run/runner.py both --config configs/fourier_example.toml --ic-path {out_dir}/2body_elliptic_medium_e0.60.npy")

if __name__ == "__main__":
    main()
