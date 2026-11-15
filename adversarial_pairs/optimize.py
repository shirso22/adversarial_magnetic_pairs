"""Adversarial pair optimization.

Two textures are parameterized by polar angles (theta, phi) per cell, so
|m| = 1 everywhere, plus one saturation-magnetization ratio s per texture
(M = s * Ms_initial * m). Adam minimizes

    L = NRMSE(B1, B2)^2 + beta * E_topo + gamma * E_smooth

with E_topo either anchoring each charge to a target,
(Q1 - Q1*)^2 + (Q2 - Q2*)^2, or targeting a charge difference,
(|Q1 - Q2| - q)^2. beta is raised by penalty continuation (x growth every
``beta_update_every`` iterations while the charge constraint misses
``charge_tol``).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from . import physics
from .geometry import Geometry
from .textures import make_texture, parse_texture, perturb

DTYPE = torch.float64

MS_MODES = {
    "fixed": (False, False),   # both textures keep the initial Ms
    "free": (True, True),      # both saturation magnetizations are optimized
    "first": (True, False),    # only texture 1's Ms may change
    "second": (False, True),   # only texture 2's Ms may change
}


@dataclass
class OptimizerSettings:
    """Hyperparameters of the pair optimization (defaults = manuscript values).

    ms_mode : "fixed" | "free" | "first" | "second"
        Which saturation magnetizations may vary. |m| = 1 is always enforced.
    ms_bounds : (low, high)
        Bounds on Ms / Ms_initial for the textures allowed to vary. A lower
        bound > 0 rules out the trivial Ms -> 0 solution.
    smoothness_on : "M" | "m"
        "M" applies the smoothness penalty to s * m (as in the code that
        produced the manuscript figures); "m" applies it to the unit field
        (as described in Appendix 3 of the manuscript).
    """

    n_iter: int = 450
    lr: float = 1e-2
    gamma: float = 1e-3
    beta0: float = 1.0
    beta_growth: float = 1.5
    beta_max: float = 1e6
    beta_update_every: int = 100
    charge_tol: float = 1e-2
    ms_mode: str = "free"
    ms_bounds: Tuple[float, float] = (1.0, 5.0)
    smoothness_on: str = "M"
    log_every: int = 0          # print progress every N iterations (0 = silent)

    def __post_init__(self):
        if self.ms_mode not in MS_MODES:
            raise ValueError(f"ms_mode must be one of {sorted(MS_MODES)}, got {self.ms_mode!r}")
        if self.smoothness_on not in ("M", "m"):
            raise ValueError("smoothness_on must be 'M' or 'm'")
        lo, hi = self.ms_bounds
        if not 0 < lo <= 1 <= hi:
            raise ValueError("ms_bounds must satisfy 0 < low <= 1 <= high")

    def replace(self, **changes) -> "OptimizerSettings":
        return dataclasses.replace(self, **changes)


@dataclass
class PairResult:
    """Outcome of one adversarial pair optimization.

    m1, m2 : unit direction fields (nx, ny, 3), numpy
    Ms1, Ms2 : final saturation magnetizations in A/m
    B1, B2 : stray fields in tesla at the measurement plane
    """

    geometry: Geometry
    settings: OptimizerSettings
    m1: np.ndarray
    m2: np.ndarray
    Ms1: float
    Ms2: float
    B1: np.ndarray
    B2: np.ndarray
    Q1: float
    Q2: float
    nrmse: float
    initial: Dict[str, float]
    targets: Dict[str, Optional[float]]
    history: Dict[str, List[float]]
    labels: Tuple[str, str] = ("texture 1", "texture 2")

    @property
    def M1(self) -> np.ndarray:
        """Physical magnetization of texture 1 in A/m."""
        return self.Ms1 * self.m1

    @property
    def M2(self) -> np.ndarray:
        return self.Ms2 * self.m2

    @property
    def charge_error(self) -> float:
        """Violation of the topology constraint at the end of the run."""
        t = self.targets
        if t.get("q") is not None:
            return abs(abs(self.Q1 - self.Q2) - t["q"])
        return max(abs(self.Q1 - t["Q1"]), abs(self.Q2 - t["Q2"]))

    @property
    def charge_constraint_met(self) -> bool:
        return self.charge_error <= self.settings.charge_tol

    @property
    def max_field_difference_T(self) -> float:
        return float(np.linalg.norm(self.B1 - self.B2, axis=-1).max())

    @property
    def mean_field_difference_T(self) -> float:
        return float(np.linalg.norm(self.B1 - self.B2, axis=-1).mean())

    def summary(self) -> dict:
        g = self.geometry
        return {
            "textures": list(self.labels),
            "Q1": self.Q1, "Q2": self.Q2, "dQ": abs(self.Q1 - self.Q2),
            "targets": self.targets,
            "charge_error": self.charge_error,
            "charge_constraint_met": self.charge_constraint_met,
            "nrmse": self.nrmse,
            "initial": self.initial,
            "Ms1_A_per_m": self.Ms1, "Ms2_A_per_m": self.Ms2,
            "Ms1_ratio": self.Ms1 / g.Ms, "Ms2_ratio": self.Ms2 / g.Ms,
            "mean_abs_dB_T": self.mean_field_difference_T,
            "max_abs_dB_T": self.max_field_difference_T,
            "geometry": g.to_dict(),
            "settings": dataclasses.asdict(self.settings),
        }

    def __str__(self) -> str:
        s = self.summary()
        text = (f"{s['textures'][0]} vs {s['textures'][1]}: "
                f"Q1 {self.Q1:+.4f}  Q2 {self.Q2:+.4f}  |dQ| {s['dQ']:.4f}  "
                f"NRMSE {self.nrmse * 100:.3f}% (initial {self.initial['nrmse'] * 100:.1f}%)  "
                f"Ms1 {s['Ms1_ratio']:.3f}  Ms2 {s['Ms2_ratio']:.3f} x Ms0")
        if not self.charge_constraint_met:
            text += (f"\n  WARNING: charge constraint violated (error {self.charge_error:.3g} > "
                     f"tol {self.settings.charge_tol:g}, targets {self.targets}); this is not a "
                     f"valid adversarial pair. Try more iterations or a larger beta0.")
        return text

    def save(self, directory: Union[str, Path]) -> Path:
        """Write ``pair.npz`` (arrays + history) and ``summary.json``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "pair.npz", m1=self.m1, m2=self.m2, B1=self.B1, B2=self.B2,
            Ms=np.array([self.Ms1, self.Ms2]),
            **{f"history_{k}": np.asarray(v) for k, v in self.history.items()},
        )
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        return d

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "PairResult":
        d = Path(directory)
        s = json.loads((d / "summary.json").read_text())
        z = np.load(d / "pair.npz")
        geom = s["geometry"]
        geometry = Geometry(grid=tuple(geom["grid"]), sample_size_nm=tuple(geom["sample_size_nm"]),
                            thickness_nm=geom["thickness_nm"], height_nm=geom["height_nm"], Ms=geom["Ms"])
        st = dict(s["settings"])
        st["ms_bounds"] = tuple(st["ms_bounds"])
        history = {k[len("history_"):]: z[k].tolist() for k in z.files if k.startswith("history_")}
        return cls(geometry=geometry, settings=OptimizerSettings(**st),
                   m1=z["m1"], m2=z["m2"], Ms1=float(z["Ms"][0]), Ms2=float(z["Ms"][1]),
                   B1=z["B1"], B2=z["B2"], Q1=s["Q1"], Q2=s["Q2"], nrmse=s["nrmse"],
                   initial=s["initial"], targets=s["targets"], history=history,
                   labels=tuple(s["textures"]))


def _to_angles(m: torch.Tensor):
    m = physics.normalize(m.detach().to(DTYPE))
    theta = torch.acos(m[..., 2].clamp(-1.0, 1.0))
    phi = torch.atan2(m[..., 1], m[..., 0])
    return theta.clone().requires_grad_(True), phi.clone().requires_grad_(True)


def _from_angles(theta, phi):
    st = torch.sin(theta)
    return torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)


def _snap(value: float, grid: Optional[float]) -> float:
    return value if not grid else round(value / grid) * grid


def optimize_pair(m1_init: torch.Tensor, m2_init: torch.Tensor, geometry: Geometry, *,
                  settings: Optional[OptimizerSettings] = None,
                  mode: str = "anchor",
                  Q1_target: Optional[float] = None, Q2_target: Optional[float] = None,
                  q: Optional[float] = None,
                  charge_grid: Optional[float] = None,
                  labels: Tuple[str, str] = ("texture 1", "texture 2"),
                  device: str = "cpu") -> PairResult:
    """Evolve two initial direction fields into an adversarial pair.

    mode="anchor"     : keep each charge at its target. Targets default to the
                        initial charges, snapped to multiples of ``charge_grid``
                        if given (e.g. 0.25 -> -1, -0.5, 0.25, 0 ...).
    mode="difference" : only require |Q1 - Q2| = q. ``q`` defaults to the
                        initial |Q1 - Q2| (snapped likewise).
    """
    settings = settings or OptimizerSettings()
    if mode not in ("anchor", "difference"):
        raise ValueError("mode must be 'anchor' or 'difference'")
    op = physics.StrayFieldOperator(geometry, device=device)

    m1_init = physics.normalize(m1_init.to(device=device, dtype=DTYPE))
    m2_init = physics.normalize(m2_init.to(device=device, dtype=DTYPE))
    with torch.no_grad():
        Q1_0 = physics.topological_charge(m1_init).item()
        Q2_0 = physics.topological_charge(m2_init).item()
        nrmse_0 = physics.nrmse(op(m1_init), op(m2_init)).item()

    if mode == "anchor":
        Q1_target = _snap(Q1_0, charge_grid) if Q1_target is None else float(Q1_target)
        Q2_target = _snap(Q2_0, charge_grid) if Q2_target is None else float(Q2_target)
        targets = {"Q1": Q1_target, "Q2": Q2_target, "q": None}
    else:
        q = abs(_snap(Q1_0 - Q2_0, charge_grid)) if q is None else float(q)
        targets = {"Q1": None, "Q2": None, "q": q}

    theta1, phi1 = _to_angles(m1_init)
    theta2, phi2 = _to_angles(m2_init)
    vary1, vary2 = MS_MODES[settings.ms_mode]
    s1 = torch.tensor(1.0, dtype=DTYPE, device=device, requires_grad=vary1)
    s2 = torch.tensor(1.0, dtype=DTYPE, device=device, requires_grad=vary2)
    params = [theta1, phi1, theta2, phi2] + [s for s, v in ((s1, vary1), (s2, vary2)) if v]
    opt = torch.optim.Adam(params, lr=settings.lr)

    beta = settings.beta0
    hist = {k: [] for k in ("loss", "nrmse", "Q1", "Q2", "Ms1_ratio", "Ms2_ratio", "beta")}

    for it in range(settings.n_iter):
        opt.zero_grad()
        m1, m2 = _from_angles(theta1, phi1), _from_angles(theta2, phi2)
        M1, M2 = s1 * m1, s2 * m2
        field_err = physics.nrmse(op(M1), op(M2))
        Q1, Q2 = physics.topological_charge(m1), physics.topological_charge(m2)
        if mode == "anchor":
            e_topo = (Q1 - Q1_target) ** 2 + (Q2 - Q2_target) ** 2
        else:
            e_topo = (torch.abs(Q1 - Q2) - q) ** 2
        if settings.smoothness_on == "M":
            e_smooth = physics.smoothness(M1) + physics.smoothness(M2)
        else:
            e_smooth = physics.smoothness(m1) + physics.smoothness(m2)
        loss = field_err ** 2 + beta * e_topo + settings.gamma * e_smooth
        loss.backward()
        opt.step()
        with torch.no_grad():
            lo, hi = settings.ms_bounds
            s1.clamp_(lo, hi)
            s2.clamp_(lo, hi)

        q1, q2 = Q1.item(), Q2.item()
        for k, v in (("loss", loss.item()), ("nrmse", field_err.item()), ("Q1", q1), ("Q2", q2),
                     ("Ms1_ratio", s1.item()), ("Ms2_ratio", s2.item()), ("beta", beta)):
            hist[k].append(v)

        if mode == "anchor":
            violation = max(abs(q1 - Q1_target), abs(q2 - Q2_target))
        else:
            violation = abs(abs(q1 - q2) - q)
        if (it + 1) % settings.beta_update_every == 0 and violation > settings.charge_tol:
            beta = min(beta * settings.beta_growth, settings.beta_max)

        if settings.log_every and (it + 1) % settings.log_every == 0:
            print(f"it {it + 1:5d} | NRMSE {field_err.item() * 100:6.3f}% | "
                  f"Q1 {q1:+.4f} Q2 {q2:+.4f} | Ms {s1.item():.3f}/{s2.item():.3f} | beta {beta:.2e}")

    # Final state after the last update (the history lags it by one step).
    with torch.no_grad():
        m1, m2 = _from_angles(theta1, phi1), _from_angles(theta2, phi2)
        M1, M2 = s1 * m1, s2 * m2
        B1, B2 = op(M1), op(M2)
        final_nrmse = physics.nrmse(B1, B2).item()
        Q1f = physics.topological_charge(m1).item()
        Q2f = physics.topological_charge(m2).item()

    Ms0 = geometry.Ms
    return PairResult(
        geometry=geometry, settings=settings,
        m1=m1.cpu().numpy(), m2=m2.cpu().numpy(),
        Ms1=s1.item() * Ms0, Ms2=s2.item() * Ms0,
        B1=(B1 * Ms0).cpu().numpy(), B2=(B2 * Ms0).cpu().numpy(),
        Q1=Q1f, Q2=Q2f, nrmse=final_nrmse,
        initial={"Q1": Q1_0, "Q2": Q2_0, "nrmse": nrmse_0},
        targets=targets, history=hist, labels=tuple(labels),
    )


TextureArg = Union[str, Tuple[str, dict], torch.Tensor]


def _resolve_texture(tex: TextureArg, geometry: Geometry):
    if isinstance(tex, torch.Tensor):
        return tex, "custom"
    if isinstance(tex, str):
        name, params = parse_texture(tex)
    else:
        name, params = tex
    return make_texture(name, geometry.grid, **params), name


def generate_pair(texture1: TextureArg = "skyrmion", texture2: TextureArg = "meron",
                  geometry: Optional[Geometry] = None, *,
                  perturbation: float = 0.0, seed: Optional[int] = None,
                  **kwargs) -> PairResult:
    """Build two textures and optimize them into an adversarial pair.

    ``texture1``/``texture2`` can be a name ("skyrmion"), a spec string
    ("meron:polarity=-1"), a (name, params) tuple, or a (nx, ny, 3) tensor.
    ``perturbation`` adds Gaussian noise (std, in units of |m|) before
    optimizing. Remaining keyword arguments go to :func:`optimize_pair`
    (``settings``, ``mode``, ``Q1_target``, ``Q2_target``, ``q``,
    ``charge_grid``, ``device``).
    """
    geometry = geometry or Geometry()
    m1, name1 = _resolve_texture(texture1, geometry)
    m2, name2 = _resolve_texture(texture2, geometry)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    m1, m2 = perturb(m1, perturbation, gen), perturb(m2, perturbation, gen)
    kwargs.setdefault("labels", (name1, name2))
    return optimize_pair(m1, m2, geometry, **kwargs)
