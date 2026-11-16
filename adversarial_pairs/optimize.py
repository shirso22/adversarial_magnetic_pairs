"""Adversarial pair optimization.

Two textures are parameterized by polar angles (theta, phi) per cell, so
|m| = 1 everywhere, plus one saturation-magnetization ratio s per texture
(M = s * Ms_initial * m). Adam minimizes

    L = D(B1, B2)^2 + beta * E_topo + gamma * E_smooth

where D is the field metric (``OptimizerSettings.metric``): the NRMSE
(default) or the mean |B1 - B2| over a fixed reference field. E_topo either
anchors each charge to a target, (Q1 - Q1*)^2 + (Q2 - Q2*)^2, or targets a
charge difference, (|Q1 - Q2| - q)^2. beta is raised by penalty continuation
(x growth every ``beta_update_every`` iterations while the charge constraint
misses ``charge_tol``).

Variants:
* ``freeze=(1,)`` keeps texture 1 fixed and optimizes only texture 2.
* ``ms_mode="shared"`` ties one Ms to both textures.
* :func:`fit_target_field` matches one texture to a supplied field instead
  of a co-evolving partner (texture 1 is then the given field).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from . import physics
from .geometry import Geometry
from .textures import make_texture, parse_texture, perturb

DTYPE = torch.float64

MS_MODES = {
    "fixed": (False, False),   # both textures keep the initial Ms
    "free": (True, True),      # both saturation magnetizations are optimized independently
    "first": (True, False),    # only texture 1's Ms may change
    "second": (False, True),   # only texture 2's Ms may change
    "shared": (True, True),    # one Ms, optimized, used for both textures
}
MS_ALIASES = {"independent": "free"}
FIELD_METRICS = ("nrmse", "abs")


@dataclass
class OptimizerSettings:
    """Hyperparameters of the pair optimization (defaults = manuscript values).

    ms_mode : "fixed" | "free" (alias "independent") | "first" | "second" | "shared"
        Which saturation magnetizations may vary. "shared" optimizes a single
        Ms used by both textures. With metric="nrmse" (scale-invariant) a
        shared Ms only feels the smoothness term and behaves exactly like
        "fixed"; it differs from it with metric="abs". |m| = 1 is always
        enforced.
    ms_bounds : (low, high)
        Bounds on Ms / Ms_initial for the textures allowed to vary.
    smoothness_on : "M" | "m"
        "M" applies the smoothness penalty to s * m (as in the code that
        produced the manuscript figures); "m" applies it to the unit field
        (as described in Appendix 3 of the manuscript).
    metric : "nrmse" | "abs"
        Field term of the loss. "nrmse" (default) is scale-invariant. "abs" is
        mean |B1 - B2| divided by a fixed reference (the mean initial field
        magnitude), so it penalizes absolute field differences and therefore
        also the overall field scale: with Ms allowed to vary it pushes Ms
        toward the lower bound.
    tol : float or None
        Early stopping: stop as soon as NRMSE <= tol and the charge constraint
        is met. ``None`` runs all ``n_iter`` iterations.
    freeze : tuple of 1 and/or 2
        Textures held fixed (angles and Ms). E.g. ``(1,)`` optimizes only M2
        against the field of the unchanged M1. Freezing both is not allowed.
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
    metric: str = "nrmse"
    tol: Optional[float] = None
    freeze: Tuple[int, ...] = ()
    log_every: int = 0          # print progress every N iterations (0 = silent)

    def __post_init__(self):
        self.ms_mode = MS_ALIASES.get(self.ms_mode, self.ms_mode)
        if self.ms_mode not in MS_MODES:
            raise ValueError(f"ms_mode must be one of {sorted(MS_MODES) + sorted(MS_ALIASES)}, "
                             f"got {self.ms_mode!r}")
        if self.smoothness_on not in ("M", "m"):
            raise ValueError("smoothness_on must be 'M' or 'm'")
        if self.metric not in FIELD_METRICS:
            raise ValueError(f"metric must be one of {FIELD_METRICS}, got {self.metric!r}")
        lo, hi = self.ms_bounds
        if not 0 < lo <= 1 <= hi:
            raise ValueError("ms_bounds must satisfy 0 < low <= 1 <= high")
        self.ms_bounds = (float(lo), float(hi))
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if self.tol is not None and self.tol <= 0:
            raise ValueError("tol must be positive or None")
        freeze = tuple(sorted({int(i) for i in (self.freeze or ())}))
        if not set(freeze) <= {1, 2}:
            raise ValueError(f"freeze may only contain 1 and 2, got {self.freeze}")
        if len(freeze) == 2:
            raise ValueError("cannot freeze both textures")
        self.freeze = freeze

    def replace(self, **changes) -> "OptimizerSettings":
        return dataclasses.replace(self, **changes)


def _json_safe(x):
    """NaN -> None, recursively, so summary.json is strict JSON."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _nan_if_none(x):
    return float("nan") if x is None else x


@dataclass
class PairResult:
    """Outcome of one adversarial pair optimization.

    m1, m2 : unit direction fields (nx, ny, 3), numpy
    Ms1, Ms2 : final saturation magnetizations in A/m
    B1, B2 : stray fields in tesla on the measurement window, shape
             ``geometry.field_grid + (c,)`` with c = 3, or c = 1 (Bz) when
             fitting a Bz-only target field
    target_field : True when texture 1 is a supplied field (see
             :func:`fit_target_field`); m1, Ms1 and Q1 are then NaN
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
    seed: Optional[int] = None
    n_iter_run: Optional[int] = None
    stopped_early: bool = False
    target_field: bool = False

    @property
    def M1(self) -> np.ndarray:
        """Physical magnetization of texture 1 in A/m."""
        return self.Ms1 * self.m1

    @property
    def M2(self) -> np.ndarray:
        return self.Ms2 * self.m2

    @property
    def Ms_ratio(self) -> float:
        """Ms2 / Ms1 (NaN for a target-field fit)."""
        return self.Ms2 / self.Ms1

    @property
    def charge_error(self) -> float:
        """Violation of the topology constraint at the end of the run."""
        t = self.targets
        if t.get("q") is not None:
            return abs(abs(self.Q1 - self.Q2) - t["q"])
        errs = [abs(Q - tq) for Q, tq in ((self.Q1, t.get("Q1")), (self.Q2, t.get("Q2"))) if tq is not None]
        return max(errs) if errs else 0.0

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
        return _json_safe({
            "textures": list(self.labels),
            "Q1": self.Q1, "Q2": self.Q2, "dQ": abs(self.Q1 - self.Q2),
            "targets": self.targets,
            "charge_error": self.charge_error,
            "charge_constraint_met": self.charge_constraint_met,
            "nrmse": self.nrmse,
            "initial": self.initial,
            "Ms1_A_per_m": self.Ms1, "Ms2_A_per_m": self.Ms2,
            "Ms1_ratio": self.Ms1 / g.Ms, "Ms2_ratio": self.Ms2 / g.Ms,
            "Ms2_over_Ms1": self.Ms_ratio,
            "mean_abs_dB_T": self.mean_field_difference_T,
            "max_abs_dB_T": self.max_field_difference_T,
            "seed": self.seed,
            "n_iter_run": self.n_iter_run,
            "stopped_early": self.stopped_early,
            "target_field": self.target_field,
            "geometry": g.to_dict(),
            "settings": dataclasses.asdict(self.settings),
        })

    def __str__(self) -> str:
        s = self.summary()
        ms = (f"Ms2 {self.Ms2 / self.geometry.Ms:.3f} x Ms0" if self.target_field else
              f"Ms1 {s['Ms1_ratio']:.3f}  Ms2 {s['Ms2_ratio']:.3f} x Ms0")
        q1 = "target field" if self.target_field else f"Q1 {self.Q1:+.4f}"
        text = (f"{s['textures'][0]} vs {s['textures'][1]}: {q1}  Q2 {self.Q2:+.4f}  "
                f"NRMSE {self.nrmse * 100:.3f}% (initial {self.initial['nrmse'] * 100:.1f}%)  {ms}")
        if self.stopped_early:
            text += f"  [stopped at iteration {self.n_iter_run}]"
        if not self.charge_constraint_met:
            text += (f"\n  WARNING: charge constraint violated (error {self.charge_error:.3g} > "
                     f"tol {self.settings.charge_tol:g}, targets {self.targets}); this is not a "
                     f"valid adversarial pair. Try more iterations or a larger beta0.")
        return text

    def history_rows(self, every: int = 1) -> List[dict]:
        """Per-iteration log as rows (every ``every``-th iteration plus the last).
        Values at iteration i are those before the i-th optimizer step."""
        h = self.history
        n = len(h["loss"])
        idx = sorted(set(range(0, n, max(1, every))) | ({n - 1} if n else set()))
        Ms0 = self.geometry.Ms
        rows = []
        for i in idx:
            Ms1, Ms2 = h["Ms1_ratio"][i] * Ms0, h["Ms2_ratio"][i] * Ms0
            rows.append({"iteration": i, "loss": h["loss"][i], "nrmse": h["nrmse"][i],
                         "mean_abs_dB_T": h["abs_dB"][i] * Ms0 if "abs_dB" in h else float("nan"),
                         "Q1": h["Q1"][i], "Q2": h["Q2"][i],
                         "Ms1_A_per_m": Ms1, "Ms2_A_per_m": Ms2, "Ms2_over_Ms1": Ms2 / Ms1,
                         "beta": h["beta"][i]})
        return rows

    def save(self, directory: Union[str, Path], history_every: int = 1) -> Path:
        """Write ``pair.npz`` (arrays + full history), ``summary.json`` and
        ``history.csv`` (every ``history_every``-th iteration plus the last)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "pair.npz", m1=self.m1, m2=self.m2, B1=self.B1, B2=self.B2,
            Ms=np.array([self.Ms1, self.Ms2]),
            **{f"history_{k}": np.asarray(v) for k, v in self.history.items()},
        )
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        rows = self.history_rows(history_every)
        if rows:
            with open(d / "history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        return d

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "PairResult":
        d = Path(directory)
        s = json.loads((d / "summary.json").read_text())
        z = np.load(d / "pair.npz")
        geom = s["geometry"]
        window = geom.get("field_window_nm")
        geometry = Geometry(grid=tuple(geom["grid"]), sample_size_nm=tuple(geom["sample_size_nm"]),
                            thickness_nm=geom["thickness_nm"], height_nm=geom["height_nm"], Ms=geom["Ms"],
                            field_window_nm=tuple(window) if window else None,
                            fft_pad_factor=geom.get("fft_pad_factor", 2.0))
        known = {f.name for f in dataclasses.fields(OptimizerSettings)}
        st = {k: v for k, v in s["settings"].items() if k in known}
        st["ms_bounds"] = tuple(st["ms_bounds"])
        st["freeze"] = tuple(st.get("freeze") or ())
        history = {k[len("history_"):]: z[k].tolist() for k in z.files if k.startswith("history_")}
        return cls(geometry=geometry, settings=OptimizerSettings(**st),
                   m1=z["m1"], m2=z["m2"], Ms1=float(z["Ms"][0]), Ms2=float(z["Ms"][1]),
                   B1=z["B1"], B2=z["B2"], Q1=_nan_if_none(s["Q1"]), Q2=s["Q2"], nrmse=s["nrmse"],
                   initial={k: _nan_if_none(v) for k, v in s["initial"].items()},
                   targets=s["targets"], history=history,
                   labels=tuple(s["textures"]), seed=s.get("seed"), n_iter_run=s.get("n_iter_run"),
                   stopped_early=s.get("stopped_early", False), target_field=s.get("target_field", False))


def _to_angles(m: torch.Tensor, trainable: bool = True):
    m = physics.normalize(m.detach().to(DTYPE))
    theta = torch.acos(m[..., 2].clamp(-1.0, 1.0))
    phi = torch.atan2(m[..., 1], m[..., 0])
    return theta.clone().requires_grad_(trainable), phi.clone().requires_grad_(trainable)


def _from_angles(theta, phi):
    st = torch.sin(theta)
    return torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)


def _snap(value: float, grid: Optional[float]) -> float:
    return value if not grid else round(value / grid) * grid


def _mean_magnitude(B: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(B, dim=-1).mean()


def optimize_pair(m1_init: Optional[torch.Tensor], m2_init: torch.Tensor, geometry: Geometry, *,
                  settings: Optional[OptimizerSettings] = None,
                  mode: str = "anchor",
                  Q1_target: Optional[float] = None, Q2_target: Optional[float] = None,
                  q: Optional[float] = None,
                  charge_grid: Optional[float] = None,
                  labels: Tuple[str, str] = ("texture 1", "texture 2"),
                  device: str = "cpu",
                  B_target=None,
                  seed: Optional[int] = None) -> PairResult:
    """Evolve two initial direction fields into an adversarial pair.

    mode="anchor"     : keep each charge at its target. Targets default to the
                        initial charges, snapped to multiples of ``charge_grid``
                        if given (e.g. 0.25 -> -1, -0.5, 0.25, 0 ...). A frozen
                        texture keeps its own charge and is not constrained.
    mode="difference" : only require |Q1 - Q2| = q. ``q`` defaults to the
                        initial |Q1 - Q2| (snapped likewise).

    ``B_target`` (tesla, shape ``geometry.field_grid + (3,)`` or
    ``geometry.field_grid`` for Bz only) replaces texture 1 by a fixed field;
    ``m1_init`` must then be None. Prefer :func:`fit_target_field`.
    """
    settings = settings or OptimizerSettings()
    if mode not in ("anchor", "difference"):
        raise ValueError("mode must be 'anchor' or 'difference'")
    target_mode = B_target is not None
    frozen = set(settings.freeze)
    op = physics.StrayFieldOperator(geometry, device=device)
    Ms0 = geometry.Ms
    wx, wy = geometry.field_grid

    comps = [0, 1, 2]
    if target_mode:
        if m1_init is not None:
            raise ValueError("pass m1_init=None together with B_target")
        if mode != "anchor":
            raise ValueError("a target-field fit supports mode='anchor' only")
        if 2 in frozen:
            raise ValueError("cannot freeze the texture being fitted to the target field")
        if settings.ms_mode == "shared":
            raise ValueError("ms_mode='shared' has no meaning for a target-field fit; use 'fixed' or 'free'")
        if Q1_target is not None:
            raise ValueError("texture 1 is the target field; set Q2_target only")
        Bt = torch.as_tensor(np.asarray(B_target), dtype=DTYPE, device=device)
        if tuple(Bt.shape) == (wx, wy):
            comps, Bt = [2], Bt[..., None]
        elif tuple(Bt.shape) != (wx, wy, 3):
            raise ValueError(f"B_target must have shape {(wx, wy, 3)} or {(wx, wy)} (Bz only) "
                             f"to match the measurement window, got {tuple(Bt.shape)}")
        Bt = Bt / Ms0                    # internal field units (Ms = 1)
        frozen.add(1)
    else:
        m1_init = physics.normalize(m1_init.to(device=device, dtype=DTYPE))
    m2_init = physics.normalize(m2_init.to(device=device, dtype=DTYPE))

    def field(M):
        return op(M)[..., comps]

    with torch.no_grad():
        Q2_0 = physics.topological_charge(m2_init).item()
        B2_0 = field(m2_init)
        if target_mode:
            Q1_0, B1_0 = float("nan"), Bt
            b_ref = _mean_magnitude(Bt)
        else:
            Q1_0, B1_0 = physics.topological_charge(m1_init).item(), field(m1_init)
            b_ref = 0.5 * (_mean_magnitude(B1_0) + _mean_magnitude(B2_0))
        nrmse_0 = physics.nrmse(B1_0, B2_0).item()
        b_ref = b_ref + 1e-30

    if mode == "anchor":
        for i, given in ((1, Q1_target), (2, Q2_target)):
            if i in frozen and given is not None:
                raise ValueError(f"texture {i} is frozen; its charge cannot be targeted")
        if target_mode:
            Q1_target = None
        elif 1 in frozen:
            Q1_target = Q1_0
        else:
            Q1_target = _snap(Q1_0, charge_grid) if Q1_target is None else float(Q1_target)
        if 2 in frozen:
            Q2_target = Q2_0
        else:
            Q2_target = _snap(Q2_0, charge_grid) if Q2_target is None else float(Q2_target)
        targets = {"Q1": Q1_target, "Q2": Q2_target, "q": None}
    else:
        q = abs(_snap(Q1_0 - Q2_0, charge_grid)) if q is None else float(q)
        targets = {"Q1": None, "Q2": None, "q": q}

    theta2, phi2 = _to_angles(m2_init, 2 not in frozen)
    if not target_mode:
        theta1, phi1 = _to_angles(m1_init, 1 not in frozen)
    one = dict(dtype=DTYPE, device=device)
    if settings.ms_mode == "shared":
        s1 = s2 = torch.tensor(1.0, **one, requires_grad=not frozen)
        ms_params = [s1] if not frozen else []
    else:
        vary1, vary2 = MS_MODES[settings.ms_mode]
        vary1 = vary1 and 1 not in frozen
        vary2 = vary2 and 2 not in frozen
        s1 = torch.tensor(1.0, **one, requires_grad=vary1)
        s2 = torch.tensor(1.0, **one, requires_grad=vary2)
        ms_params = [s for s, v in ((s1, vary1), (s2, vary2)) if v]
    params = ([] if 1 in frozen else [theta1, phi1]) + ([] if 2 in frozen else [theta2, phi2]) + ms_params
    opt = torch.optim.Adam(params, lr=settings.lr)
    lo, hi = settings.ms_bounds

    if 1 in frozen and not target_mode:
        with torch.no_grad():
            M1_const = s1 * _from_angles(theta1, phi1)
            B1_const = field(M1_const)

    beta = settings.beta0
    hist = {k: [] for k in ("loss", "nrmse", "abs_dB", "Q1", "Q2", "Ms1_ratio", "Ms2_ratio", "beta")}
    stopped_early, n_steps = False, 0
    zero = torch.zeros((), **one)

    for it in range(settings.n_iter):
        opt.zero_grad()
        m2 = _from_angles(theta2, phi2)
        M2 = s2 * m2
        B2 = field(M2)
        if target_mode:
            m1, B1, Q1 = None, Bt, None
        elif 1 in frozen:
            m1, B1 = None, B1_const
            Q1 = torch.tensor(Q1_target, **one) if mode == "anchor" else torch.tensor(Q1_0, **one)
        else:
            m1 = _from_angles(theta1, phi1)
            M1 = s1 * m1
            B1 = field(M1)
            Q1 = physics.topological_charge(m1)
        Q2 = physics.topological_charge(m2)

        nr = physics.nrmse(B1, B2)
        abs_db = torch.linalg.norm(B1 - B2, dim=-1).mean()
        field_err = nr if settings.metric == "nrmse" else abs_db / b_ref

        if mode == "anchor":
            e_topo = zero
            if 1 not in frozen:
                e_topo = e_topo + (Q1 - Q1_target) ** 2
            if 2 not in frozen:
                e_topo = e_topo + (Q2 - Q2_target) ** 2
        else:
            e_topo = (torch.abs(Q1 - Q2) - q) ** 2
        e_smooth = zero
        for m, s, i in ((m1, s1, 1), (m2, s2, 2)):
            if i not in frozen:
                e_smooth = e_smooth + physics.smoothness(s * m if settings.smoothness_on == "M" else m)
        loss = field_err ** 2 + beta * e_topo + settings.gamma * e_smooth

        q1 = float("nan") if Q1 is None else Q1.item()
        q2 = Q2.item()
        for k, v in (("loss", loss.item()), ("nrmse", nr.item()), ("abs_dB", abs_db.item()),
                     ("Q1", q1), ("Q2", q2), ("Ms1_ratio", float("nan") if target_mode else s1.item()),
                     ("Ms2_ratio", s2.item()), ("beta", beta)):
            hist[k].append(v)

        if mode == "anchor":
            violation = max(abs(q1 - Q1_target) if Q1_target is not None else 0.0, abs(q2 - Q2_target))
        else:
            violation = abs(abs(q1 - q2) - q)
        if settings.tol is not None and nr.item() <= settings.tol and violation <= settings.charge_tol:
            stopped_early = True
            break

        loss.backward()
        opt.step()
        n_steps += 1
        with torch.no_grad():
            for s in ms_params:
                s.clamp_(lo, hi)

        if (it + 1) % settings.beta_update_every == 0 and violation > settings.charge_tol:
            beta = min(beta * settings.beta_growth, settings.beta_max)

        if settings.log_every and (it + 1) % settings.log_every == 0:
            print(f"it {it + 1:5d} | NRMSE {nr.item() * 100:6.3f}% | "
                  f"Q1 {q1:+.4f} Q2 {q2:+.4f} | Ms {s1.item():.3f}/{s2.item():.3f} | beta {beta:.2e}")

    # Final state (the history lags it by one step unless the run stopped early).
    with torch.no_grad():
        m2 = _from_angles(theta2, phi2)
        B2 = field(s2 * m2)
        Q2f = physics.topological_charge(m2).item()
        if target_mode:
            m1_out = np.full(m2.shape, np.nan)
            B1, Q1f, Ms1 = Bt, float("nan"), float("nan")
        else:
            m1 = _from_angles(theta1, phi1)
            B1 = field(s1 * m1)
            Q1f = physics.topological_charge(m1).item()
            m1_out, Ms1 = m1.cpu().numpy(), s1.item() * Ms0
        final_nrmse = physics.nrmse(B1, B2).item()

    return PairResult(
        geometry=geometry, settings=settings,
        m1=m1_out, m2=m2.cpu().numpy(),
        Ms1=Ms1, Ms2=s2.item() * Ms0,
        B1=(B1 * Ms0).cpu().numpy(), B2=(B2 * Ms0).cpu().numpy(),
        Q1=Q1f, Q2=Q2f, nrmse=final_nrmse,
        initial={"Q1": Q1_0, "Q2": Q2_0, "nrmse": nrmse_0},
        targets=targets, history=hist, labels=tuple(labels),
        seed=seed, n_iter_run=n_steps, stopped_early=stopped_early, target_field=target_mode,
    )


TextureArg = Union[str, Tuple[str, dict], torch.Tensor]


def _resolve_texture(tex: TextureArg, geometry: Geometry):
    if isinstance(tex, torch.Tensor):
        return tex, "custom"
    if isinstance(tex, str):
        name, params = parse_texture(tex)
    else:
        name, params = tex
    return make_texture(name, geometry.grid, sample_size_nm=geometry.sample_size_nm, **params), name


def generate_pair(texture1: TextureArg = "skyrmion", texture2: TextureArg = "meron",
                  geometry: Optional[Geometry] = None, *,
                  perturbation: float = 0.0, seed: Optional[int] = None,
                  **kwargs) -> PairResult:
    """Build two textures and optimize them into an adversarial pair.

    ``texture1``/``texture2`` can be a name ("skyrmion"), a spec string
    ("meron:polarity=-1", "skyrmion:radius_nm=8"), a (name, params) tuple, or
    a (nx, ny, 3) tensor. ``perturbation`` adds Gaussian noise (std, in units
    of |m|) before optimizing. Remaining keyword arguments go to
    :func:`optimize_pair` (``settings``, ``mode``, ``Q1_target``,
    ``Q2_target``, ``q``, ``charge_grid``, ``device``).
    """
    geometry = geometry or Geometry()
    m1, name1 = _resolve_texture(texture1, geometry)
    m2, name2 = _resolve_texture(texture2, geometry)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    m1, m2 = perturb(m1, perturbation, gen), perturb(m2, perturbation, gen)
    kwargs.setdefault("labels", (name1, name2))
    return optimize_pair(m1, m2, geometry, seed=seed, **kwargs)


def load_field(path: Union[str, Path]) -> np.ndarray:
    """Load a field map (tesla) from ``.npy`` or from a ``pair.npz`` (its B1)."""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as z:
            return z["B1"] if "B1" in z.files else z[z.files[0]]
    return np.load(p)


def fit_target_field(B_target, texture: TextureArg = "skyrmion", geometry: Optional[Geometry] = None, *,
                     Q_target: Optional[float] = None, charge_grid: Optional[float] = None,
                     perturbation: float = 0.0, seed: Optional[int] = None,
                     settings: Optional[OptimizerSettings] = None, device: str = "cpu") -> PairResult:
    """Match one texture to a fixed field instead of a co-evolving partner.

    ``B_target`` is an array in tesla (or a path to ``.npy``/``pair.npz``) on
    the measurement window of ``geometry``: shape ``field_grid + (3,)``, or
    ``field_grid`` for a Bz-only map, in which case only Bz is compared.
    ``Q_target`` anchors the fitted texture's charge (default: its initial
    charge, snapped to ``charge_grid``). The result stores the target as
    texture 1 (``B1``) and the fitted texture as texture 2.
    """
    geometry = geometry or Geometry()
    if isinstance(B_target, (str, Path)):
        B_target = load_field(B_target)
    m, name = _resolve_texture(texture, geometry)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    m = perturb(m, perturbation, gen)
    return optimize_pair(None, m, geometry, settings=settings, mode="anchor", Q2_target=Q_target,
                         charge_grid=charge_grid, labels=("target", name), device=device,
                         B_target=B_target, seed=seed)
