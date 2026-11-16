"""Post-hoc analysis of adversarial pairs.

* Cartesian multipole moments of the magnetic sources (Appendix 2.1).
* Wavevector localization of the field agreement (Appendix 2.2).
* Sensitivity studies: grid resolution, standoff height, loss weights.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import shift as nd_shift

from .geometry import Geometry
from .optimize import OptimizerSettings, PairResult, TextureArg, generate_pair


# --------------------------------------------------------------------------- #
#  Multipole moments                                                          #
# --------------------------------------------------------------------------- #
def source_density(result: PairResult, which: int, source: str = "mz") -> np.ndarray:
    """Source distribution of texture ``which`` (1 or 2), in units of the
    initial Ms.

    source="mz"     : surface-charge source s * m_z (manuscript convention)
    source="volume" : volume charge -s * div(m_par), per nm
    """
    if which == 1 and result.target_field:
        raise ValueError("texture 1 of a target-field fit is a field, not a magnetization")
    m = result.m1 if which == 1 else result.m2
    s = (result.Ms1 if which == 1 else result.Ms2) / result.geometry.Ms
    if source == "mz":
        return s * m[..., 2]
    if source == "volume":
        g = result.geometry
        div = np.gradient(m[..., 0], g.dx_nm, axis=0) + np.gradient(m[..., 1], g.dy_nm, axis=1)
        return -s * div
    raise ValueError("source must be 'mz' or 'volume'")


def center_on_centroid(density: np.ndarray) -> np.ndarray:
    """Shift so the centroid of |density| sits at the grid center (periodic)."""
    w = np.abs(density)
    total = w.sum()
    if total == 0:
        return density.copy()
    cx = (w * np.arange(density.shape[0])[:, None]).sum() / total
    cy = (w * np.arange(density.shape[1])[None, :]).sum() / total
    return nd_shift(density, [density.shape[0] // 2 - cx, density.shape[1] // 2 - cy], mode="wrap")


def cartesian_moments(density: np.ndarray, order_max: int = 8) -> Dict[Tuple[int, int], float]:
    """M_pq = sum x^p y^q density dA on coordinates normalized to [-1/2, 1/2]."""
    nx, ny = density.shape
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dA = (x[1] - x[0]) * (y[1] - y[0])
    return {(p, q): float(np.sum(X ** p * Y ** q * density) * dA)
            for p in range(order_max + 1) for q in range(order_max + 1 - p)}


def multipole_comparison(result: PairResult, order_max: int = 8, source: str = "mz",
                         center: bool = True, even_only: bool = True) -> List[dict]:
    """Compare the moments of the two textures order by order.

    Returns rows with keys p, q, l, M1, M2, rel_diff where
    rel_diff = |M1 - M2| / (0.5 (|M1| + |M2|)).
    """
    d1, d2 = source_density(result, 1, source), source_density(result, 2, source)
    if center:
        d1, d2 = center_on_centroid(d1), center_on_centroid(d2)
    m1, m2 = cartesian_moments(d1, order_max), cartesian_moments(d2, order_max)
    rows = []
    for l in range(order_max + 1):
        if even_only and l % 2:
            continue
        for p in range(l + 1):
            q = l - p
            if even_only and (p % 2 or q % 2):
                continue
            a, b = m1[(p, q)], m2[(p, q)]
            rows.append({"p": p, "q": q, "l": l, "M1": a, "M2": b,
                         "rel_diff": abs(a - b) / (0.5 * (abs(a) + abs(b)) + 1e-30)})
    return rows


def format_moment_table(rows: List[dict]) -> str:
    lines = [f"{'(p,q)':<8}{'l':<4}{'M1':>13}{'M2':>13}{'|dM|/avg':>11}", "-" * 49]
    last_l = None
    for r in rows:
        if last_l is not None and r["l"] != last_l:
            lines.append("")
        last_l = r["l"]
        lines.append(f"({r['p']},{r['q']}){'':<3}{r['l']:<4}{r['M1']:>13.4e}{r['M2']:>13.4e}{r['rel_diff']:>11.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Wavevector localization of the field agreement                             #
# --------------------------------------------------------------------------- #
def kspace_localization(result: PairResult, n_bins: int = 64, power_fraction: float = 0.90) -> dict:
    """Radially binned field power and field difference versus in-plane k.

    Returns k bin centres (rad/nm), per-shell powers P1, P2, difference power
    D, per-shell relative difference rel = sqrt(D / mean(P1, P2)), the band
    [k_lo, k_hi] holding ``power_fraction`` of the field power, and the
    fraction of field power / residual energy inside that band. ``l2_real``
    and ``l2_kspace`` must agree (Parseval check).
    """
    g = result.geometry
    B1, B2 = result.B1, result.B2
    wx, wy = B1.shape[:2]           # measurement window (may differ from the sample grid)
    F1 = np.fft.fftshift(np.fft.fft2(B1, axes=(0, 1)), axes=(0, 1))
    F2 = np.fft.fftshift(np.fft.fft2(B2, axes=(0, 1)), axes=(0, 1))
    p1 = (np.abs(F1) ** 2).sum(-1)
    p2 = (np.abs(F2) ** 2).sum(-1)
    dd = (np.abs(F1 - F2) ** 2).sum(-1)

    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(wx, d=g.dx_nm))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(wy, d=g.dy_nm))
    K = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

    n_bins = max(1, min(n_bins, min(wx, wy) // 2))
    edges = np.linspace(0.0, K.max() / np.sqrt(2.0), n_bins + 1)
    kc = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(K.ravel(), edges) - 1, 0, n_bins - 1)
    P1 = np.bincount(idx, weights=p1.ravel(), minlength=n_bins)
    P2 = np.bincount(idx, weights=p2.ravel(), minlength=n_bins)
    D = np.bincount(idx, weights=dd.ravel(), minlength=n_bins)
    Pavg = 0.5 * (P1 + P2)

    def l2(num, e1, e2):
        return np.sqrt(num) / (0.5 * (np.sqrt(e1) + np.sqrt(e2)) + 1e-300)

    cum = np.cumsum(Pavg) / Pavg.sum()
    k_hi = kc[min(np.searchsorted(cum, power_fraction), n_bins - 1)]
    band = kc <= k_hi
    rel = np.sqrt(D / (Pavg + 1e-300))
    l2_real = l2(np.sum((B1 - B2) ** 2), np.sum(B1 ** 2), np.sum(B2 ** 2))
    return {
        "k": kc, "P1": P1, "P2": P2, "D": D, "rel": rel,
        # rel rescaled to the NRMSE convention used in the loss
        "rel_nrmse_convention": rel * (result.nrmse / max(l2_real, 1e-300)),
        "band": (float(kc[0]), float(k_hi)),
        "frac_power_in_band": float(Pavg[band].sum() / Pavg.sum()),
        "frac_residual_in_band": float(D[band].sum() / max(D.sum(), 1e-300)),
        "rel_in_band_mean": float(np.mean(rel[band])),
        "nrmse": result.nrmse,
        "l2_real": float(l2_real),
        "l2_kspace": float(l2(D.sum(), P1.sum(), P2.sum())),
    }


# --------------------------------------------------------------------------- #
#  Sensitivity studies                                                        #
# --------------------------------------------------------------------------- #
def _row(label: str, value, res: PairResult) -> dict:
    return {label: value, "nrmse": res.nrmse, "Q1": res.Q1, "Q2": res.Q2,
            "charge_constraint_met": res.charge_constraint_met,
            "Ms1_ratio": res.Ms1 / res.geometry.Ms, "Ms2_ratio": res.Ms2 / res.geometry.Ms,
            "Ms2_over_Ms1": res.Ms_ratio,
            "mean_abs_dB_T": res.mean_field_difference_T, "n_iter_run": res.n_iter_run}


def grid_sensitivity(grids: Sequence[Tuple[int, int]], texture1: TextureArg = "skyrmion",
                     texture2: TextureArg = "meron", geometry: Optional[Geometry] = None,
                     **pair_kwargs) -> List[dict]:
    """Optimize the same pair on several grids at fixed physical sample size."""
    geometry = geometry or Geometry()
    rows = []
    for grid in grids:
        res = generate_pair(texture1, texture2, geometry.replace(grid=tuple(grid)), **pair_kwargs)
        rows.append(_row("grid", f"{grid[0]}x{grid[1]}", res))
    return rows


def standoff_sensitivity(heights_nm: Sequence[float], texture1: TextureArg = "skyrmion",
                         texture2: TextureArg = "meron", geometry: Optional[Geometry] = None,
                         **pair_kwargs) -> List[dict]:
    """Optimize one pair at several standoff heights."""
    geometry = geometry or Geometry()
    return [_row("height_nm", float(h), generate_pair(texture1, texture2,
                                                       geometry.replace(height_nm=float(h)), **pair_kwargs))
            for h in heights_nm]


def weight_sensitivity(name: str, values: Sequence[float], texture1: TextureArg = "skyrmion",
                       texture2: TextureArg = "meron", geometry: Optional[Geometry] = None,
                       settings: Optional[OptimizerSettings] = None, **pair_kwargs) -> List[dict]:
    """Sweep one ``OptimizerSettings`` field (e.g. "gamma", "lr", "beta0")."""
    settings = settings or OptimizerSettings()
    if not hasattr(settings, name):
        raise ValueError(f"OptimizerSettings has no field {name!r}")
    return [_row(name, v, generate_pair(texture1, texture2, geometry,
                                        settings=settings.replace(**{name: v}), **pair_kwargs))
            for v in values]


def write_rows_csv(rows: List[dict], path) -> None:
    """Write the rows of a sensitivity sweep to CSV."""
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def format_rows(rows: List[dict]) -> str:
    """Plain-text table for the sensitivity helpers."""
    if not rows:
        return ""
    keys = list(rows[0])
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    widths = [max(len(k), *(len(fmt(r[k])) for r in rows)) for k in keys]
    head = "  ".join(k.rjust(w) for k, w in zip(keys, widths))
    body = ["  ".join(fmt(r[k]).rjust(w) for k, w in zip(keys, widths)) for r in rows]
    return "\n".join([head, "-" * len(head), *body])
