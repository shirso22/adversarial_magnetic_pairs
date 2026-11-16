"""Forward model and the quantities the loss is built from.

* ``StrayFieldOperator`` - thin-film magnetostatic stray field via FFT
  (scalar-potential kernel, zero-padded, float64).
* ``topological_charge`` - Berg-Luscher solid-angle charge (differentiable).
* ``nrmse``, ``smoothness`` - the dimensionless loss ingredients.

Magnetization arrays are torch tensors of shape (nx, ny, 3).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .geometry import Geometry

MU_0 = 4e-7 * math.pi
DTYPE = torch.float64


class StrayFieldOperator:
    """Linear map M (A/m) -> B (T) at the measurement plane of ``geometry``.

    The k-space kernel is built once per (geometry, height, device) and reused,
    so calling the operator inside an optimization loop is cheap.

    Kernel (film occupying -t < z < 0, measured at z = h above its top
    surface; Fourier convention f(k) = sum f(r) e^{-i k.r}):
        F(k)   = (mu0 / 2) (1 - e^{-k t}) e^{-k h}
        phi(k) = F(k) [ Mz / k - i (k . M_par) / k^2 ]
        B      = ( -i kx phi, -i ky phi, +k phi ),  B(k = 0) = 0
    i.e. B = -mu0 grad(Psi) with Psi sourced by the face charges +-Ms mz and
    the volume charge -div(M_par). Checked against a direct dipole sum in
    ``tests/test_core.py``.

    Versions before 1.1.0 had the opposite sign on the in-plane term and on
    Bz, which equals the true field of the texture rotated by pi about z,
    with Bz negated. That rotation preserves Q and the negation preserves the
    NRMSE, so optimized pairs, charges and NRMSE values are unchanged up to
    that rotation; saved textures and Bz maps from those versions are not.

    The output is the field on the measurement window of ``geometry``
    (``geometry.field_grid`` cells, centered on the sample). The FFT domain
    is zero-padded to ``geometry.fft_grid`` cells per axis (by default
    2 * nx, and at least nx + wx), so the computed field is that of a periodic
    array of samples; increase ``geometry.fft_pad_factor`` to approach an
    isolated film.
    """

    def __init__(self, geometry: Geometry, height_nm: Optional[float] = None,
                 device: str = "cpu"):
        self.geometry = geometry
        self.height_nm = geometry.height_nm if height_nm is None else float(height_nm)
        self.device = torch.device(device)
        self._build_kernel()

    def _build_kernel(self):
        g = self.geometry
        nx, ny = g.grid
        wx, wy = g.field_grid
        self.nx_pad, self.ny_pad = g.fft_grid
        # sample position in the padded array; reduces to (nx//2, ny//2) by default
        self.pad_x, self.pad_y = (self.nx_pad - nx) // 2, (self.ny_pad - ny) // 2
        ox, oy = g.field_offset_cells
        self.win_x = slice(self.pad_x + ox, self.pad_x + ox + wx)
        self.win_y = slice(self.pad_y + oy, self.pad_y + oy + wy)

        kx = 2.0 * math.pi * torch.fft.fftfreq(self.nx_pad, d=g.dx_m, dtype=DTYPE)
        ky = 2.0 * math.pi * torch.fft.rfftfreq(self.ny_pad, d=g.dy_m, dtype=DTYPE)
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        k = torch.sqrt(kx ** 2 + ky ** 2)
        zero = k == 0
        k_safe = torch.where(zero, torch.ones_like(k), k)

        F = 0.5 * MU_0 * (-torch.expm1(-k * g.thickness_m)) * torch.exp(-k * self.height_nm * 1e-9)
        F = torch.where(zero, torch.zeros_like(F), F)

        cplx = torch.complex128
        # phi = a_z * Mz + a_x * Mx + a_y * My
        a_z = (F / k_safe).to(cplx)
        a_x = (-1j * F * kx / k_safe ** 2).to(cplx)
        a_y = (-1j * F * ky / k_safe ** 2).to(cplx)
        # B_j = d_j * phi
        d_x = (-1j * kx).to(cplx)
        d_y = (-1j * ky).to(cplx)
        d_z = k.to(cplx)

        dev = self.device
        self._a = torch.stack([a_x, a_y, a_z], dim=-1).to(dev)   # (kx, ky, 3)
        self._d = torch.stack([d_x, d_y, d_z], dim=-1).to(dev)   # (kx, ky, 3)

    def __call__(self, M: torch.Tensor) -> torch.Tensor:
        """Stray field (T) of magnetization ``M`` (A/m), shape (nx, ny, 3).
        Returns shape (wx, wy, 3) on the measurement window."""
        nx, ny = self.geometry.grid
        if tuple(M.shape) != (nx, ny, 3):
            raise ValueError(f"expected M of shape {(nx, ny, 3)}, got {tuple(M.shape)}")
        M = M.to(device=self.device, dtype=DTYPE)
        Mp = torch.zeros((self.nx_pad, self.ny_pad, 3), dtype=DTYPE, device=self.device)
        Mp[self.pad_x:self.pad_x + nx, self.pad_y:self.pad_y + ny] = M
        Mk = torch.fft.rfftn(Mp, dim=(0, 1), norm="forward")
        phi = (self._a * Mk).sum(dim=-1, keepdim=True)
        Bk = self._d * phi
        Bp = torch.fft.irfftn(Bk, s=(self.nx_pad, self.ny_pad), dim=(0, 1), norm="forward")
        return Bp[self.win_x, self.win_y]


def stray_field(M: torch.Tensor, geometry: Geometry, height_nm: Optional[float] = None) -> torch.Tensor:
    """One-off convenience wrapper around :class:`StrayFieldOperator`."""
    return StrayFieldOperator(geometry, height_nm=height_nm, device=str(M.device))(M)


def normalize(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return m / (torch.linalg.norm(m, dim=-1, keepdim=True) + eps)


def topological_charge(m: torch.Tensor) -> torch.Tensor:
    """Berg-Luscher (solid-angle) topological charge of a direction field.

    Each lattice plaquette is split into two triangles whose signed solid
    angles are summed. Exactly quantized for fully wrapped textures, returns
    the correct partial-coverage value for merons and fractional defects, and
    is differentiable. Input is normalized internally, so the charge does not
    depend on Ms.

    Sign convention: matches the finite-difference triple product
    m . (d_x m x d_y m) on the (x = axis 0, y = axis 1) grid.

    Reference: B. Berg and M. Luscher, Nucl. Phys. B 190, 412 (1981).
    """
    n = normalize(m)
    n1, n2 = n[:-1, :-1], n[1:, :-1]
    n3, n4 = n[1:, 1:], n[:-1, 1:]

    def omega(a, b, c):
        num = torch.sum(a * torch.linalg.cross(b, c, dim=-1), dim=-1)
        den = 1.0 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
        return 2.0 * torch.atan2(num, den)

    return -(omega(n1, n2, n3) + omega(n1, n3, n4)).sum() / (4.0 * math.pi)


def topological_charge_finite_difference(m: torch.Tensor) -> torch.Tensor:
    """First-order finite-difference charge. For comparison only: it is not
    quantized and underestimates Q on coarse grids."""
    n = normalize(m)
    cp = torch.linalg.cross(n[:-1, 1:], n[1:, :-1], dim=-1)
    return torch.sum(n[:-1, :-1] * cp) / (4.0 * math.pi)


def nrmse(B1: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
    """RMS of B1 - B2 (over components and cells) divided by the mean field
    magnitude averaged over the two textures (symmetric, scale-invariant)."""
    rmse = torch.sqrt(torch.mean((B1 - B2) ** 2))
    ref = 0.5 * (torch.linalg.norm(B1, dim=-1).mean() + torch.linalg.norm(B2, dim=-1).mean())
    return rmse / (ref + 1e-30)


def smoothness(M: torch.Tensor) -> torch.Tensor:
    """Dimensionless smoothness: mean squared nearest-neighbour difference."""
    dx = M[1:] - M[:-1]
    dy = M[:, 1:] - M[:, :-1]
    return (dx ** 2).mean() + (dy ** 2).mean()
