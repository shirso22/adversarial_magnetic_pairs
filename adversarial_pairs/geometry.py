"""Sample and measurement geometry.

All user-facing lengths are in nanometres; the ``*_m`` properties give SI
values for the physics code.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional, Tuple


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
    field_window_nm : (Wx, Wy) or None
        Lateral size in nm of the measurement plane on which the stray field is
        evaluated (and compared in the loss), centered on the sample. ``None``
        (default) uses the sample footprint. A smaller window crops the field;
        a larger one also captures the field outside the sample. The window
        uses the sample's cell size, so its size is rounded to a whole number
        of cells with the same parity as the grid (keeping it exactly centered);
        see :attr:`field_window_size_nm` for the size actually used.
    fft_pad_factor : float
        Size of the zero-padded FFT domain relative to the sample, per axis
        (at least sample + window). The FFT field is that of a periodic array
        of samples, so this sets how isolated the sample is. The default 2
        reproduces the manuscript; larger values converge to an isolated
        film (see the README for numbers).
    """

    grid: Tuple[int, int] = (32, 32)
    sample_size_nm: Tuple[float, float] = (64.0, 64.0)
    thickness_nm: float = 1.0
    height_nm: float = 20.0
    Ms: float = 1e6
    field_window_nm: Optional[Tuple[float, float]] = None
    fft_pad_factor: float = 2.0

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
        if self.fft_pad_factor < 1:
            raise ValueError("fft_pad_factor must be >= 1")
        object.__setattr__(self, "fft_pad_factor", float(self.fft_pad_factor))
        if self.field_window_nm is not None:
            w = tuple(float(v) for v in self.field_window_nm)
            if len(w) != 2 or min(w) <= 0:
                raise ValueError(f"field_window_nm must be two positive sizes in nm, got {self.field_window_nm}")
            object.__setattr__(self, "field_window_nm", w)
            if min(self.field_grid) < 2:
                raise ValueError(f"field_window_nm={w} is smaller than two cells "
                                 f"({self.dx_nm:g} x {self.dy_nm:g} nm)")

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

    # ------------------------------------------------------------------ #
    #  Measurement window                                                 #
    # ------------------------------------------------------------------ #
    @property
    def field_grid(self) -> Tuple[int, int]:
        """Cells (wx, wy) of the measurement window."""
        if self.field_window_nm is None:
            return self.grid
        out = []
        for n, w, d in zip(self.grid, self.field_window_nm, (self.dx_nm, self.dy_nm)):
            # nearest cell count with the parity of n, so the window is exactly centered
            out.append(n + 2 * int(round((w / d - n) / 2.0)))
        return tuple(out)

    @property
    def field_offset_cells(self) -> Tuple[int, int]:
        """Index of the window's first cell relative to the sample's first
        cell (negative when the window is larger than the sample)."""
        return tuple((n - w) // 2 for n, w in zip(self.grid, self.field_grid))

    @property
    def fft_grid(self) -> Tuple[int, int]:
        """Cells of the zero-padded FFT domain: max(pad_factor * n, n + w) per axis."""
        return tuple(max(int(math.ceil(self.fft_pad_factor * n)), n + w)
                     for n, w in zip(self.grid, self.field_grid))

    @property
    def field_window_size_nm(self) -> Tuple[float, float]:
        """Effective window size in nm (after rounding to whole cells)."""
        return (self.field_grid[0] * self.dx_nm, self.field_grid[1] * self.dy_nm)

    @property
    def field_extent_nm(self) -> Tuple[float, float, float, float]:
        """(x0, x1, y0, y1) of the window in the sample frame, where the
        sample occupies [0, Lx] x [0, Ly]."""
        (ox, oy), (wx, wy) = self.field_offset_cells, self.field_grid
        return (ox * self.dx_nm, (ox + wx) * self.dx_nm, oy * self.dy_nm, (oy + wy) * self.dy_nm)

    def replace(self, **changes) -> "Geometry":
        """Return a copy with some fields changed, e.g. ``g.replace(height_nm=30)``."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
