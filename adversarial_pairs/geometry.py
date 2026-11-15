"""Sample and measurement geometry.

All user-facing lengths are in nanometres; the ``*_m`` properties give SI
values for the physics code.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Geometry:
    """Thin-film sample discretised on a regular grid, measured at a standoff.

    Parameters
    ----------
    grid : (nx, ny)
        Number of cells along x and y.
    sample_size_nm : (Lx, Ly)
        Lateral sample size in nm. Cell size is ``Lx / nx`` by ``Ly / ny``.
    thickness_nm : float
        Film thickness t in nm (magnetization uniform through the thickness).
    height_nm : float
        Standoff h of the measurement plane above the film, in nm.
    Ms : float
        Initial saturation magnetization in A/m. Fields are reported in tesla
        for this Ms; optimized saturation values are reported relative to it
        and in A/m.
    """

    grid: Tuple[int, int] = (32, 32)
    sample_size_nm: Tuple[float, float] = (64.0, 64.0)
    thickness_nm: float = 1.0
    height_nm: float = 20.0
    Ms: float = 1e6

    def __post_init__(self):
        if len(self.grid) != 2 or min(self.grid) < 2:
            raise ValueError(f"grid must be two integers >= 2, got {self.grid}")
        if min(self.sample_size_nm) <= 0:
            raise ValueError("sample_size_nm must be positive")
        if self.thickness_nm <= 0 or self.height_nm <= 0:
            raise ValueError("thickness_nm and height_nm must be positive")
        if self.Ms <= 0:
            raise ValueError("Ms must be positive")
        object.__setattr__(self, "grid", (int(self.grid[0]), int(self.grid[1])))
        object.__setattr__(self, "sample_size_nm",
                           (float(self.sample_size_nm[0]), float(self.sample_size_nm[1])))

    @property
    def nx(self) -> int:
        return self.grid[0]

    @property
    def ny(self) -> int:
        return self.grid[1]

    @property
    def dx_nm(self) -> float:
        return self.sample_size_nm[0] / self.nx

    @property
    def dy_nm(self) -> float:
        return self.sample_size_nm[1] / self.ny

    @property
    def dx_m(self) -> float:
        return self.dx_nm * 1e-9

    @property
    def dy_m(self) -> float:
        return self.dy_nm * 1e-9

    @property
    def thickness_m(self) -> float:
        return self.thickness_nm * 1e-9

    @property
    def height_m(self) -> float:
        return self.height_nm * 1e-9

    def replace(self, **changes) -> "Geometry":
        """Return a copy with some fields changed, e.g. ``g.replace(height_nm=30)``."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
