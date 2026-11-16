"""Figures. Every function returns a matplotlib Figure and never calls show()."""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .montecarlo import MonteCarloResult
from .optimize import PairResult


def _extent(result: PairResult):
    Lx, Ly = result.geometry.sample_size_nm
    return [0, Ly, 0, Lx]


def _field_extent(result: PairResult):
    x0, x1, y0, y1 = result.geometry.field_extent_nm
    return [y0, y1, x0, x1]


def _outline_sample(a, result: PairResult):
    """Dashed outline of the sample footprint on a field panel."""
    from matplotlib.patches import Rectangle
    Lx, Ly = result.geometry.sample_size_nm
    a.add_patch(Rectangle((0, 0), Ly, Lx, fill=False, ls="--", lw=1, ec="w", alpha=0.8))


def plot_history(result: PairResult):
    h = result.history
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(np.array(h["nrmse"]) * 100)
    ax[0].set(xlabel="iteration", ylabel="NRMSE (%)", title="Field match")
    ax[1].plot(h["Q1"], label=f"Q1 ({result.labels[0]})")
    ax[1].plot(h["Q2"], label=f"Q2 ({result.labels[1]})")
    ax[1].set(xlabel="iteration", ylabel="topological charge", title="Charge trajectory")
    ax[1].legend()
    if not result.target_field:
        ax[2].plot(h["Ms1_ratio"], label="Ms1")
    ax[2].plot(h["Ms2_ratio"], label="Ms2")
    ax[2].set(xlabel="iteration", ylabel="Ms / Ms0", title="Saturation magnetization")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pair(result: PairResult):
    """m_z (A/m) and |B| (mT) of both textures, plus |B1 - B2|."""
    ext = _extent(result)
    Mz1, Mz2 = result.M1[..., 2], result.M2[..., 2]
    b1 = np.linalg.norm(result.B1, axis=-1) * 1e3
    b2 = np.linalg.norm(result.B2, axis=-1) * 1e3
    db = np.linalg.norm(result.B1 - result.B2, axis=-1) * 1e3
    mlim = np.nanmax(np.abs([Mz1, Mz2]))
    bmin, bmax = min(b1.min(), b2.min()), max(b1.max(), b2.max())

    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    # (axis, data, cmap, vmin, vmax, colorbar label, title, is a field panel)
    panels = [
        (ax[0, 0], Mz1, "RdBu", -mlim, mlim, "$M_z$ (A/m)", f"$M_z$ {result.labels[0]}, Q = {result.Q1:+.3f}", False),
        (ax[0, 1], Mz2, "RdBu", -mlim, mlim, "$M_z$ (A/m)", f"$M_z$ {result.labels[1]}, Q = {result.Q2:+.3f}", False),
        (ax[1, 0], b1, "viridis", bmin, bmax, "|B| (mT)", f"|B| {result.labels[0]}", True),
        (ax[1, 1], b2, "viridis", bmin, bmax, "|B| (mT)", f"|B| {result.labels[1]}", True),
        (ax[1, 2], db, "magma", 0, db.max(), "|B1 - B2| (mT)",
         f"|B1 - B2|, NRMSE {result.nrmse * 100:.2f}%, peak {db.max():.3g} mT", True),
    ]
    fext = _field_extent(result)
    windowed = result.geometry.field_window_nm is not None
    for a, data, cmap, vmin, vmax, clabel, title, is_field in panels:
        im = a.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=fext if is_field else ext)
        if is_field and windowed:
            _outline_sample(a, result)
        a.set(title=title, xlabel="y (nm)", ylabel="x (nm)")
        fig.colorbar(im, ax=a, label=clabel)
    if result.target_field:
        ax[0, 0].set_title("target field (no magnetization)")
    _quiver(ax[0, 2], result)
    win = ""
    if windowed:
        wx, wy = result.geometry.field_window_size_nm
        win = f"field window {wx:g} x {wy:g} nm (sample dashed), "
    fig.suptitle(f"h = {result.geometry.height_nm:g} nm, {win}"
                 f"Ms = {result.Ms1:.3g} / {result.Ms2:.3g} A/m", fontsize=13)
    fig.tight_layout()
    return fig


def _quiver(a, result: PairResult, step: Optional[int] = None):
    """Overlay of the in-plane components of both textures."""
    g = result.geometry
    step = step or max(1, g.nx // 16)
    x = (np.arange(g.nx) + 0.5) * g.dx_nm
    y = (np.arange(g.ny) + 0.5) * g.dy_nm
    Y, X = np.meshgrid(y[::step], x[::step])
    for m, color, lab in ((result.m1, "C0", result.labels[0]), (result.m2, "C3", result.labels[1])):
        a.quiver(Y, X, m[::step, ::step, 1], m[::step, ::step, 0], color=color, label=lab,
                 scale=1.2 * g.nx / step, width=0.004, alpha=0.8)
    a.set(title="in-plane m (both textures)", xlabel="y (nm)", ylabel="x (nm)", aspect="equal")
    a.legend(loc="upper right", fontsize=8)


def plot_mc_summary(mc: MonteCarloResult):
    """Charge retention, NRMSE before/after and final Ms distributions."""
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    if mc.config.mode == "anchor":
        e1 = mc.column("Q1") - mc.column("Q1_target")
        e2 = mc.column("Q2") - mc.column("Q2_target")
        for a, e, c, i in ((ax[0, 0], e1, "C0", 1), (ax[0, 1], e2, "C3", 2)):
            a.hist(e, bins=30, color=c, alpha=0.75, edgecolor="k")
            a.axvline(0, color="k", ls="--", lw=1)
            a.set(title=f"Texture {i} charge retention (max |dQ| = {np.abs(e).max():.2e})",
                  xlabel=f"Q{i} - Q{i}*", ylabel="count")
    else:
        dq = np.abs(mc.column("Q1") - mc.column("Q2"))
        ax[0, 0].hist(np.abs(mc.column("Q1_initial") - mc.column("Q2_initial")), bins=20, alpha=0.75, edgecolor="k")
        ax[0, 0].set(title="Initial |Q1 - Q2|", xlabel="|dQ|", ylabel="count")
        ax[0, 1].hist(dq - mc.column("q_target"), bins=30, alpha=0.75, edgecolor="k")
        ax[0, 1].set(title="Final |Q1 - Q2| - q", xlabel="error", ylabel="count")
    ax[0, 2].hist(mc.column("initial_nrmse") * 100, bins=20, alpha=0.75, edgecolor="k")
    ax[0, 2].set(title="Initial NRMSE", xlabel="NRMSE (%)", ylabel="count")
    ax[1, 0].hist(mc.column("nrmse") * 100, bins=20, alpha=0.75, edgecolor="k")
    ax[1, 0].set(title=f"Final NRMSE (mean {mc.column('nrmse').mean() * 100:.2f}%)",
                 xlabel="NRMSE (%)", ylabel="count")
    s1, s2 = mc.column("Ms1_ratio"), mc.column("Ms2_ratio")
    ax[1, 1].hist([s1, s2], bins=15, label=["texture 1", "texture 2"], color=["C0", "C3"],
                  alpha=0.75, edgecolor="k")
    ax[1, 1].set(title="Final Ms", xlabel="Ms / Ms0", ylabel="count")
    ax[1, 1].legend()
    ax[1, 2].bar(["texture 1", "texture 2"], [s1.mean(), s2.mean()], yerr=[s1.std(), s2.std()],
                 capsize=5, color=["C0", "C3"], alpha=0.75, edgecolor="k")
    ax[1, 2].set(title="Mean final Ms +- std", ylabel="Ms / Ms0")
    fig.tight_layout()
    return fig


def plot_standoff_sweep(sweep: Dict[float, MonteCarloResult]):
    h = np.array(sorted(sweep))
    db = [sweep[k].column("mean_abs_dB_T") * 1e3 for k in h]
    nr = [sweep[k].column("nrmse") * 100 for k in h]
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for a, data, c, lab in ((ax[0], db, "purple", "mean |B1 - B2| (mT)"), (ax[1], nr, "green", "NRMSE (%)")):
        mean = np.array([d.mean() for d in data])
        std = np.array([d.std() for d in data])
        a.plot(h, mean, "-o", color=c)
        a.fill_between(h, mean - std, mean + std, color=c, alpha=0.2)
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[1].set_xlabel("standoff height (nm)")
    ax[0].set_title("Monte Carlo ensemble vs standoff (mean +- std)")
    fig.tight_layout()
    return fig


def plot_kspace(loc: dict):
    k, show = loc["k"], slice(None)
    k_lo, k_hi = loc["band"]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 8), sharex=True)
    ax[0].semilogy(k[show], np.sqrt(loc["P1"][show]), lw=2, label="texture 1 $|\\tilde B(k)|$")
    ax[0].semilogy(k[show], np.sqrt(loc["P2"][show]), lw=2, label="texture 2 $|\\tilde B(k)|$")
    ax[0].axvspan(k_lo, k_hi, color="0.85", label="90% field-power band")
    ax[0].set(ylabel="field amplitude per shell")
    ax[0].legend(fontsize=9)
    ax[1].semilogy(k[show], loc["rel_nrmse_convention"][show], lw=2, color="C3",
                   label="per-shell field difference (NRMSE convention)")
    ax[1].axhline(loc["nrmse"], ls="--", color="0.4", label=f"NRMSE = {loc['nrmse'] * 100:.2f}%")
    ax[1].axvspan(k_lo, k_hi, color="0.85")
    ax[1].set(xlabel="in-plane wavevector k (rad/nm)", ylabel="relative difference per shell")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_multipoles(rows: List[dict]):
    labels = [f"({r['p']},{r['q']})" for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [abs(r["M1"]) for r in rows], 0.4, label="texture 1")
    ax[0].bar(x + 0.2, [abs(r["M2"]) for r in rows], 0.4, label="texture 2")
    ax[0].set_yscale("log")
    ax[0].set(ylabel="|M_pq|", title="Cartesian moments")
    ax[0].legend()
    ax[1].bar(x, [r["rel_diff"] for r in rows], color="C2")
    ax[1].set(ylabel="|dM| / mean", title="Relative difference")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=60)
    fig.tight_layout()
    return fig
