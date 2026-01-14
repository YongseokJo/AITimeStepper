from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union

import torch


class ExternalField(Protocol):
    """Analytic external field applied on top of N-body forces.

    Implementations should be torch-only and support broadcasting.

    Conventions:
    - `position` has shape (..., N, D)
    - `time` is a scalar tensor/float in the same units as your simulation
    """

    def acceleration(self, position: torch.Tensor, time: Union[float, torch.Tensor]) -> torch.Tensor:
        ...

    def potential(self, position: torch.Tensor, time: Union[float, torch.Tensor]) -> torch.Tensor:
        """Return per-particle external potential Φ(x).

        Shape should be (..., N). The total external potential energy is
        U_ext = sum_i m_i Φ(x_i).
        """
        ...


def _as_time_tensor(time: Union[float, torch.Tensor], *, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(time):
        return time.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(float(time), device=ref.device, dtype=ref.dtype)


def _as_vec(vec: Union[torch.Tensor, list, tuple], *, ref: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(vec, device=ref.device, dtype=ref.dtype)
    return v


RFunc = Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PointMassTidalField:
    """Tidal field from a distant point-mass perturber.

    This adds a *differential* (tidal) acceleration relative to the origin:

        a(x) = G M [ (R - x)/|R - x|^3 - R/|R|^3 ]

    which removes the uniform acceleration of the origin.

    The potential is defined so that a = -∇Φ and matches the above:

        Φ(x) = -G M [ 1/|R - x| - 1/|R| - x·R/|R|^3 ]

    Notes:
    - Works in 2D or 3D.
    - `R` can be constant (`R0`) or time-dependent via `R_of_t`.
    """

    M: float
    R0: Optional[Union[torch.Tensor, list, tuple]] = None
    R_of_t: Optional[RFunc] = None
    G: float = 1.0
    softening: float = 0.0

    def _R(self, position: torch.Tensor, time: Union[float, torch.Tensor]) -> torch.Tensor:
        if self.R_of_t is not None:
            # Signature: R_of_t(time, ref_tensor) -> (D,) tensor
            return self.R_of_t(time, position)
        if self.R0 is None:
            raise ValueError("PointMassTidalField requires either R0 or R_of_t")
        return _as_vec(self.R0, ref=position)

    def acceleration(self, position: torch.Tensor, time: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
        if position.dim() < 2:
            raise ValueError(f"position must have shape (..., N, D), got {tuple(position.shape)}")

        R = self._R(position, time)  # (D,)
        if R.dim() != 1:
            raise ValueError(f"R must be 1D (D,), got {tuple(R.shape)}")

        # Broadcast R to (..., 1, D)
        while R.dim() < position.dim():
            R = R.unsqueeze(0)
        R = R.unsqueeze(-2) if R.shape[-2] != 1 else R

        x = position
        eps2 = float(self.softening) ** 2

        d = R - x  # (..., N, D)
        d2 = (d * d).sum(dim=-1) + eps2  # (..., N)
        d_norm3 = d2.pow(1.5)  # (..., N)

        R2 = (R * R).sum(dim=-1) + eps2  # (..., 1)
        R_norm3 = R2.pow(1.5)  # (..., 1)

        a = (self.G * float(self.M)) * (d / d_norm3.unsqueeze(-1) - R / R_norm3.unsqueeze(-1))
        return a

    def potential(self, position: torch.Tensor, time: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
        if position.dim() < 2:
            raise ValueError(f"position must have shape (..., N, D), got {tuple(position.shape)}")

        R = self._R(position, time)  # (D,)
        if R.dim() != 1:
            raise ValueError(f"R must be 1D (D,), got {tuple(R.shape)}")

        while R.dim() < position.dim():
            R = R.unsqueeze(0)
        R = R.unsqueeze(-2) if R.shape[-2] != 1 else R

        x = position
        eps = float(self.softening)

        # |R-x|
        d = R - x
        d_norm = torch.sqrt((d * d).sum(dim=-1) + eps * eps)

        # |R|
        R_norm = torch.sqrt((R * R).sum(dim=-1) + eps * eps)

        # x·R
        x_dot_R = (x * R).sum(dim=-1)

        Phi = -(self.G * float(self.M)) * (1.0 / d_norm - 1.0 / R_norm - x_dot_R / (R_norm.pow(3)))
        return Phi
