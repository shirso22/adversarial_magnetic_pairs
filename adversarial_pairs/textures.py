"""Initial magnetization textures.

Every generator returns a unit direction field m of shape (nx, ny, 3)
(float64 torch tensor). Lateral coordinates are normalized to [-1, 1] across
the sample, so ``radius`` is a fraction of the half-width and a texture keeps
the same physical shape when the grid is refined.

In-plane angle convention: psi(phi) = winding * phi + helicity.
helicity = pi/2 gives a Bloch (circulating) texture, 0 a Neel (radial) one.

Use :func:`make_texture` with a registered name, or :func:`parse_texture`
for CLI-style strings such as ``"skyrmion:polarity=-1,radius=0.25"``.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

DTYPE = torch.float64


def _polar_grid(nx: int, ny: int, center=(0.0, 0.0)):
    x = torch.linspace(-1, 1, nx, dtype=DTYPE) - center[0]
    y = torch.linspace(-1, 1, ny, dtype=DTYPE) - center[1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.sqrt(xx ** 2 + yy ** 2), torch.atan2(yy, xx)


def _assemble(theta, psi, mz_sign=1.0):
    st = torch.sin(theta)
    return torch.stack([st * torch.cos(psi), st * torch.sin(psi), mz_sign * torch.cos(theta)], dim=-1)


def skyrmion(nx: int, ny: int, polarity: int = 1, radius: float = 0.3,
             helicity: float = math.pi / 2, winding: int = 1, center=(0.0, 0.0)) -> torch.Tensor:
    """Skyrmion with a tanh profile: theta = pi (1 - tanh(r / radius)).

    The core points along ``+polarity * z``; Q is approximately -polarity * winding.
    Use ``winding=-1`` for an antiskyrmion and ``helicity=0`` for a Neel skyrmion.
    """
    r, phi = _polar_grid(nx, ny, center)
    theta = math.pi * (1 - torch.tanh(r / radius))
    return _assemble(theta, winding * phi + helicity, mz_sign=-polarity)


def meron(nx: int, ny: int, polarity: int = 1, radius: float = 0.2,
          helicity: float = math.pi / 2, winding: int = 1, center=(0.0, 0.0)) -> torch.Tensor:
    """Meron/vortex: core along ``polarity * z``, in-plane far away.
    theta = (pi/2) (1 - exp(-(r / radius)^2)). |Q| = 1/2."""
    r, phi = _polar_grid(nx, ny, center)
    theta = (math.pi / 2) * (1 - torch.exp(-(r / radius) ** 2))
    return _assemble(theta, winding * phi + helicity, mz_sign=polarity)


def fractional(nx: int, ny: int, polarity: int = 1, charge: float = 0.25, radius: float = 0.15,
               helicity: float = math.pi / 2, winding: int = 1, center=(0.0, 0.0)) -> torch.Tensor:
    """Fractional-charge vortex (del Ser et al., PRB 110, 094442 (2024)).

    The far-field tilt theta_FM satisfies |Q| = (1 - cos theta_FM) / 2, so the
    texture covers a fraction ``charge`` of the sphere with integer winding.
    ``charge`` must be in (0, 0.5]; 0.5 reproduces the meron.
    """
    if not 0 < abs(charge) <= 0.5:
        raise ValueError(f"charge must satisfy 0 < |charge| <= 0.5, got {charge}")
    theta_fm = math.acos(1.0 - 2.0 * abs(charge))
    r, phi = _polar_grid(nx, ny, center)
    theta = theta_fm * (1 - torch.exp(-(r / radius) ** 2))
    return _assemble(theta, winding * phi + helicity, mz_sign=polarity)


def uniform(nx: int, ny: int, theta_deg: float = 90.0, phi_deg: float = 0.0) -> torch.Tensor:
    """Uniform magnetization; theta from +z, phi from +x (degrees). Q = 0."""
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    v = torch.tensor([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)], dtype=DTYPE)
    return v.expand(nx, ny, 3).clone()


def antiskyrmion(nx: int, ny: int, **kw) -> torch.Tensor:
    """Skyrmion with winding -1."""
    kw.setdefault("winding", -1)
    return skyrmion(nx, ny, **kw)


TEXTURES: Dict[str, Callable[..., torch.Tensor]] = {
    "skyrmion": skyrmion,
    "antiskyrmion": antiskyrmion,
    "meron": meron,
    "fractional": fractional,
    "uniform": uniform,
}
ALIASES = {"vortex": "meron", "fractional_q": "fractional", "uniform_inplane": "uniform"}


def canonical_name(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in TEXTURES:
        raise KeyError(f"unknown texture {name!r}; available: {sorted(TEXTURES)} (aliases {sorted(ALIASES)})")
    return key


def make_texture(name: str, grid: Tuple[int, int], **params) -> torch.Tensor:
    """Build a registered texture on ``grid``; ``params`` go to the generator."""
    return TEXTURES[canonical_name(name)](grid[0], grid[1], **params)


def register_texture(name: str, fn: Callable[..., torch.Tensor]) -> None:
    """Add a custom generator ``fn(nx, ny, **params) -> (nx, ny, 3)`` tensor."""
    TEXTURES[name.lower()] = fn


def perturb(m: torch.Tensor, strength: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Add Gaussian noise of std ``strength`` to each component, renormalize."""
    if strength <= 0:
        return m.clone()
    noise = torch.randn(m.shape, dtype=m.dtype, generator=generator)
    return torch.nn.functional.normalize(m + strength * noise, dim=-1)


def random_params(name: str, rng: np.random.Generator) -> dict:
    """Randomize the discrete/orientational parameters of a texture class, as in
    the Monte Carlo ensemble: random polarity for cored textures, random
    in-plane direction for the uniform state."""
    key = canonical_name(name)
    if key == "uniform":
        return {"theta_deg": 90.0, "phi_deg": float(rng.uniform(0.0, 360.0))}
    return {"polarity": int(rng.choice([-1, 1]))}


def _parse_value(text: str):
    t = text.strip()
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    # simple multiples of pi, e.g. "pi/2", "-pi", "3*pi/4"
    if "pi" in low and set(low.replace("pi", "")) <= set("0123456789.+-*/ "):
        return float(eval(low, {"__builtins__": {}}, {"pi": math.pi}))
    return t


def parse_texture(spec: str) -> Tuple[str, dict]:
    """Parse ``"name"`` or ``"name:key=value,key=value"`` into (name, params).

    >>> parse_texture("skyrmion:polarity=-1,radius=0.25")
    ('skyrmion', {'polarity': -1, 'radius': 0.25})
    """
    name, _, rest = spec.partition(":")
    params = {}
    for item in filter(None, (s.strip() for s in rest.split(","))):
        key, eq, val = item.partition("=")
        if not eq:
            raise ValueError(f"bad texture parameter {item!r} in {spec!r}; expected key=value")
        params[key.strip()] = _parse_value(val)
    return canonical_name(name), params
