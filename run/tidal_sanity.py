import torch

from src.particle import ParticleTorch
from src.external_potentials import PointMassTidalField


def main():
    torch.set_default_dtype(torch.double)
    device = torch.device("cpu")

    # Simple 2-body in 2D
    mass = torch.tensor([1.0, 1.0], device=device)
    pos = torch.tensor([[-0.5, 0.0], [0.5, 0.0]], device=device)
    vel = torch.tensor([[0.0, 0.6], [0.0, -0.6]], device=device)
    dt = torch.tensor(1e-3, device=device)

    p = ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel, dt=dt, softening=0.0)

    # Distant perturber at R=(10,0), M=5.0 (code units)
    tide = PointMassTidalField(M=5.0, R0=[10.0, 0.0], G=1.0, softening=0.0)
    p.set_external_field(tide)

    a_int = ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel, dt=dt, softening=0.0).get_acceleration(G=1.0)
    a_tot = p.get_acceleration(G=1.0)
    a_ext = a_tot - a_int

    print("a_int norm per particle:", torch.linalg.norm(a_int, dim=-1))
    print("a_ext norm per particle:", torch.linalg.norm(a_ext, dim=-1))
    print("a_tot norm per particle:", torch.linalg.norm(a_tot, dim=-1))

    E0 = p.total_energy(G=1.0).detach().cpu().item()
    for _ in range(100):
        p.evolve_batch(G=1.0)
    E1 = p.total_energy(G=1.0).detach().cpu().item()

    print(f"Energy including external (should be ~conserved for static tide): E0={E0:.6e} E1={E1:.6e} dE={E1-E0:.6e}")


if __name__ == "__main__":
    main()
