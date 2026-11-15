"""Monte Carlo ensembles of adversarial pairs and statistical validation.

Each sample draws two distinct texture classes (or a fixed first class),
randomizes their polarity / in-plane direction, adds Gaussian noise, and
optimizes the pair. Samples are seeded individually (``seed``, index), so an
ensemble is reproducible and independent of the number of worker processes.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .geometry import Geometry
from .optimize import OptimizerSettings, optimize_pair
from .textures import canonical_name, make_texture, perturb, random_params

DEFAULT_CLASSES = ("skyrmion", "meron", "fractional", "uniform")


@dataclass
class MonteCarloConfig:
    """Ensemble definition.

    texture_classes : classes to draw from (two distinct ones per sample)
    first_texture   : if set, texture 1 is always this class
    perturbation    : std of the Gaussian noise added to each initial texture
    mode            : "anchor" or "difference" (see ``optimize_pair``)
    charge_grid     : initial charges are snapped to multiples of this to
                      define the targets (0.25 covers +-1, +-0.5, +-0.25, 0)
    n_workers       : processes to use (1 = run serially in this process)
    """

    n_samples: int = 100
    texture_classes: Tuple[str, ...] = DEFAULT_CLASSES
    first_texture: Optional[str] = None
    perturbation: float = 0.01
    mode: str = "anchor"
    charge_grid: float = 0.25
    seed: int = 0
    n_workers: int = 1

    def __post_init__(self):
        self.texture_classes = tuple(canonical_name(t) for t in self.texture_classes)
        if self.first_texture is not None:
            self.first_texture = canonical_name(self.first_texture)
        pool = set(self.texture_classes) | ({self.first_texture} if self.first_texture else set())
        if len(pool) < 2:
            raise ValueError("need at least two distinct texture classes")


@dataclass
class MonteCarloResult:
    geometry: Geometry
    config: MonteCarloConfig
    settings: OptimizerSettings
    records: List[dict]

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records], dtype=float)

    def summary(self) -> dict:
        """Ensemble statistics used for statistical validation."""
        nrmse_f, nrmse_i = self.column("nrmse"), self.column("initial_nrmse")
        out = {
            "n_samples": len(self.records),
            "final_nrmse_mean": float(nrmse_f.mean()),
            "final_nrmse_std": float(nrmse_f.std()),
            "final_nrmse_median": float(np.median(nrmse_f)),
            "final_nrmse_max": float(nrmse_f.max()),
            "initial_nrmse_mean": float(nrmse_i.mean()),
            "Ms1_ratio_mean": float(self.column("Ms1_ratio").mean()),
            "Ms1_ratio_std": float(self.column("Ms1_ratio").std()),
            "Ms2_ratio_mean": float(self.column("Ms2_ratio").mean()),
            "Ms2_ratio_std": float(self.column("Ms2_ratio").std()),
            "fraction_nrmse_below_2pct": float((nrmse_f < 0.02).mean()),
        }
        if self.config.mode == "anchor":
            err = np.maximum(np.abs(self.column("Q1") - self.column("Q1_target")),
                             np.abs(self.column("Q2") - self.column("Q2_target")))
        else:
            err = np.abs(np.abs(self.column("Q1") - self.column("Q2")) - self.column("q_target"))
        out["charge_error_max"] = float(err.max())
        out["charge_error_mean"] = float(err.mean())
        out["fraction_charge_within_tol"] = float((err <= self.settings.charge_tol).mean())
        out["fraction_trivial_pairs"] = float(
            np.mean(np.abs(self.column("Q1_initial") - self.column("Q2_initial")) < 1e-3))
        return out

    def by_pair_type(self) -> Dict[str, dict]:
        """Mean/max final NRMSE grouped by (texture1, texture2) class."""
        groups: Dict[str, List[float]] = {}
        for r in self.records:
            groups.setdefault(f"{r['texture1']}-{r['texture2']}", []).append(r["nrmse"])
        return {k: {"n": len(v), "nrmse_mean": float(np.mean(v)), "nrmse_max": float(np.max(v))}
                for k, v in sorted(groups.items())}

    def save(self, directory: Union[str, Path]) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "samples.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.records[0]))
            w.writeheader()
            w.writerows(self.records)
        meta = {"summary": self.summary(), "by_pair_type": self.by_pair_type(),
                "geometry": self.geometry.to_dict(), "config": dataclasses.asdict(self.config),
                "settings": dataclasses.asdict(self.settings)}
        (d / "summary.json").write_text(json.dumps(meta, indent=2))
        return d


def _draw_pair(cfg: MonteCarloConfig, rng: np.random.Generator):
    t1 = cfg.first_texture or str(rng.choice(cfg.texture_classes))
    t2 = str(rng.choice([t for t in cfg.texture_classes if t != t1]))
    return t1, random_params(t1, rng), t2, random_params(t2, rng)


def run_sample(index: int, geometry: Geometry, cfg: MonteCarloConfig,
               settings: OptimizerSettings, device: str = "cpu") -> dict:
    """Run Monte Carlo sample ``index`` (deterministic given cfg.seed)."""
    rng = np.random.default_rng([cfg.seed, index])
    gen = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
    t1, p1, t2, p2 = _draw_pair(cfg, rng)
    m1 = perturb(make_texture(t1, geometry.grid, **p1), cfg.perturbation, gen)
    m2 = perturb(make_texture(t2, geometry.grid, **p2), cfg.perturbation, gen)
    res = optimize_pair(m1, m2, geometry, settings=settings, mode=cfg.mode,
                        charge_grid=cfg.charge_grid, labels=(t1, t2), device=device)
    return {
        "index": index, "texture1": t1, "texture2": t2,
        "params1": json.dumps(p1), "params2": json.dumps(p2),
        "Q1_initial": res.initial["Q1"], "Q2_initial": res.initial["Q2"],
        "initial_nrmse": res.initial["nrmse"],
        "Q1_target": res.targets["Q1"] if res.targets["Q1"] is not None else float("nan"),
        "Q2_target": res.targets["Q2"] if res.targets["Q2"] is not None else float("nan"),
        "q_target": res.targets["q"] if res.targets["q"] is not None else float("nan"),
        "Q1": res.Q1, "Q2": res.Q2, "nrmse": res.nrmse,
        "Ms1_ratio": res.Ms1 / geometry.Ms, "Ms2_ratio": res.Ms2 / geometry.Ms,
        "mean_abs_dB_T": res.mean_field_difference_T,
        "max_abs_dB_T": res.max_field_difference_T,
        "height_nm": geometry.height_nm,
    }


def _worker(args):
    torch.set_num_threads(1)
    return run_sample(*args)


def run_monte_carlo(geometry: Optional[Geometry] = None, config: Optional[MonteCarloConfig] = None,
                    settings: Optional[OptimizerSettings] = None, *, device: str = "cpu",
                    progress: bool = True) -> MonteCarloResult:
    """Optimize ``config.n_samples`` random pairs and collect per-sample records."""
    geometry = geometry or Geometry()
    config = config or MonteCarloConfig()
    settings = settings or OptimizerSettings()
    tasks = [(i, geometry, config, settings, device) for i in range(config.n_samples)]
    records: List[dict] = []

    def report(n):
        if progress and (n % max(1, config.n_samples // 10) == 0 or n == config.n_samples):
            nr = np.mean([r["nrmse"] for r in records]) * 100
            print(f"  {n}/{config.n_samples} samples | mean final NRMSE {nr:.2f}%", flush=True)

    if config.n_workers > 1:
        # autograd cannot run in fork()ed children once the parent has used it
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=config.n_workers, mp_context=ctx) as ex:
            for rec in ex.map(_worker, tasks):
                records.append(rec)
                report(len(records))
    else:
        for t in tasks:
            records.append(run_sample(*t))
            report(len(records))
    return MonteCarloResult(geometry=geometry, config=config, settings=settings, records=records)


def standoff_sweep(heights_nm: Sequence[float], geometry: Optional[Geometry] = None,
                   config: Optional[MonteCarloConfig] = None,
                   settings: Optional[OptimizerSettings] = None, **kwargs) -> Dict[float, MonteCarloResult]:
    """Run the same seeded ensemble at each standoff height (Appendix 1).

    Because samples are seeded by index, every height sees the same initial
    pairs, so differences between heights are not sampling noise.
    """
    geometry = geometry or Geometry()
    out = {}
    for h in heights_nm:
        if kwargs.get("progress", True):
            print(f"standoff {h:g} nm")
        out[float(h)] = run_monte_carlo(geometry.replace(height_nm=float(h)), config, settings, **kwargs)
    return out
