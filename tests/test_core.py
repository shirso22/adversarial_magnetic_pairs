"""Tests. Run with ``pytest`` or ``python tests/test_core.py``."""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversarial_pairs import (Geometry, MonteCarloConfig, OptimizerSettings, PairResult, analysis,
                               generate_pair, make_texture, parse_texture, run_monte_carlo, stray_field,
                               topological_charge)
from adversarial_pairs.physics import MU_0


def reference_B(M, dx, dy, thickness, height):
    """Verbatim kernel from the manuscript notebook (compute_B_field, float64).
    It has the pre-1.1.0 sign error; see test_fix_is_pi_rotation_of_manuscript_kernel."""
    dtype = torch.float64
    nx, ny = M.shape[:2]
    nxp, nyp = 2 * nx, 2 * ny
    px, py = nx // 2, ny // 2
    Mp = torch.zeros((nxp, nyp, 3), dtype=dtype)
    Mp[px:px + nx, py:py + ny] = M
    Mk = torch.fft.rfftn(Mp, s=(nxp, nyp), dim=(0, 1), norm="forward")
    c = Mk.dtype
    kx = (2 * torch.pi * torch.fft.fftfreq(nxp, d=dx)).to(dtype)
    ky = (2 * torch.pi * torch.fft.rfftfreq(nyp, d=dy)).to(dtype)
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    k = torch.sqrt(kx ** 2 + ky ** 2)
    zero = k == 0
    ks = torch.sqrt(k ** 2 + 1e-30)
    F = 0.5 * MU_0 * (-torch.expm1(-ks * thickness)) * torch.exp(-ks * height)
    inv_k = torch.zeros_like(ks, dtype=c)
    inv_k2 = torch.zeros_like(ks, dtype=c)
    inv_k[~zero] = (1 / ks[~zero]).to(c)
    inv_k2[~zero] = (1 / ks[~zero] ** 2).to(c)
    phi = F.to(c) * (Mk[..., 2] * inv_k + 1j * (kx.to(c) * Mk[..., 0] + ky.to(c) * Mk[..., 1]) * inv_k2)
    Bk = torch.zeros_like(Mk)
    Bk[..., 0] = -1j * kx.to(c) * phi
    Bk[..., 1] = -1j * ky.to(c) * phi
    Bk[..., 2] = -ks.to(c) * phi
    Bk[zero, :] = 0
    B = torch.fft.irfftn(Bk, s=(nxp, nyp), dim=(0, 1), norm="forward")
    return B[px:px + nx, py:py + ny]


def dipole_sum_B(M, g):
    """Brute-force field of point dipoles (one per cell, at the film mid-plane)
    on the measurement window of ``g``. Independent of the FFT kernel."""
    M = np.asarray(M)
    nx, ny = g.grid
    xs, ys = (np.arange(nx) + 0.5) * g.dx_m, (np.arange(ny) + 0.5) * g.dy_m
    (ox, oy), (wx, wy) = g.field_offset_cells, g.field_grid
    xo, yo = (np.arange(wx) + ox + 0.5) * g.dx_m, (np.arange(wy) + oy + 0.5) * g.dy_m
    S = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    O = np.stack(np.meshgrid(xo, yo, indexing="ij"), -1).reshape(-1, 2)
    r = np.zeros((len(O), len(S), 3))
    r[..., :2] = O[:, None] - S[None]
    r[..., 2] = g.height_m + g.thickness_m / 2
    m = M.reshape(-1, 3) * g.dx_m * g.dy_m * g.thickness_m
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    mr = (m[None] * r).sum(-1, keepdims=True)
    B = MU_0 / (4 * math.pi) * (3 * mr * r / rn ** 5 - m[None] / rn ** 3)
    return B.sum(1).reshape(wx, wy, 3)


def test_stray_field_matches_dipole_sum():
    """Every component and source channel has the right sign and magnitude,
    including the field outside the sample (window larger than the sample)."""
    g = Geometry(grid=(24, 24), sample_size_nm=(48, 48), thickness_nm=1, height_nm=10,
                 field_window_nm=(96, 96), fft_pad_factor=8)
    x = torch.linspace(-1, 1, 24, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    w = torch.exp(-(X ** 2 + Y ** 2) / 0.15)
    z = torch.zeros_like(w)
    for M in (torch.stack([z, z, w], -1), torch.stack([w, z, z], -1), torch.stack([z, -w, z], -1)):
        B = stray_field(M, g).numpy()
        Bd = dipole_sum_B(M, g)
        err = np.abs(B - Bd).max() / np.abs(Bd).max()
        assert err < 0.01, err


def test_fix_is_pi_rotation_of_manuscript_kernel():
    """Fixed kernel = notebook kernel applied to the texture rotated by pi about
    z, with Bz negated. Hence charges and NRMSE in the manuscript are unchanged."""
    g = Geometry(grid=(32, 24), sample_size_nm=(64, 48), thickness_nm=1, height_nm=20)
    M = torch.randn(32, 24, 3, dtype=torch.float64)
    P = torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)
    B = stray_field(M, g)
    Bref = reference_B(P * M, g.dx_m, g.dy_m, g.thickness_m, g.height_m) * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    assert (B - Bref).abs().max() < 1e-6 * Bref.abs().max()


def test_field_window():
    M = torch.randn(32, 32, 3, dtype=torch.float64)
    base = Geometry()
    B0 = stray_field(M, base)
    # explicit window equal to the sample is identical to the default
    assert torch.equal(stray_field(M, base.replace(field_window_nm=(64, 64))), B0)
    # a smaller window is a centered crop of the default field
    small = base.replace(field_window_nm=(32, 40))
    assert small.field_grid == (16, 20) and small.field_offset_cells == (8, 6)
    assert torch.allclose(stray_field(M, small), B0[8:24, 6:26], rtol=0, atol=1e-12 * B0.abs().max())
    # a larger window contains the sample-window field in its center (same FFT domain)
    p4 = base.replace(fft_pad_factor=4)
    big = p4.replace(field_window_nm=(128, 96))
    assert big.field_grid == (64, 48) and big.field_extent_nm == (-32.0, 96.0, -16.0, 80.0)
    assert big.fft_grid == p4.fft_grid == (128, 128)
    Bc = stray_field(M, p4)
    assert torch.allclose(stray_field(M, big)[16:48, 8:40], Bc, rtol=0, atol=1e-12 * Bc.abs().max())
    # the FFT domain grows to fit a window larger than the padding allows
    assert base.replace(field_window_nm=(192, 64)).fft_grid == (128, 64)
    # sizes round to whole cells with the grid's parity, so the window stays centered
    assert base.replace(field_window_nm=(70, 61)).field_grid == (36, 30)
    try:
        base.replace(field_window_nm=(1, 64))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_fft_padding_converges_to_isolated_film():
    """Default padding (2x) is the manuscript setting; it carries periodic-image
    error that larger padding removes."""
    g = Geometry()
    M = make_texture("skyrmion", g.grid)
    Bd = dipole_sum_B(M, g)
    errs = [float(np.abs(stray_field(M, g.replace(fft_pad_factor=f)).numpy() - Bd).max() / np.abs(Bd).max())
            for f in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.005, errs


def test_window_pair_save_load():
    g = Geometry(grid=(16, 16), sample_size_nm=(32, 32), field_window_nm=(64, 48), fft_pad_factor=3)
    res = generate_pair("skyrmion", "meron", g, settings=OptimizerSettings(n_iter=30))
    assert res.B1.shape == (32, 24, 3)
    loc = analysis.kspace_localization(res)
    assert abs(loc["l2_real"] - loc["l2_kspace"]) < 1e-9 * max(1.0, loc["l2_real"])
    with tempfile.TemporaryDirectory() as d:
        res.save(d)
        back = PairResult.load(d)
    assert back.geometry == g and np.allclose(back.B1, res.B1)


def test_canonical_charges():
    grid = (32, 32)
    expected = {("skyrmion", 1): -1.0, ("skyrmion", -1): 1.0, ("meron", 1): -0.5,
                ("meron", -1): 0.5, ("fractional", 1): -0.25}
    for (name, pol), q in expected.items():
        Q = topological_charge(make_texture(name, grid, polarity=pol)).item()
        assert abs(Q - q) < 1e-3, (name, pol, Q)
    assert abs(topological_charge(make_texture("uniform", grid, phi_deg=37)).item()) < 1e-12
    assert abs(topological_charge(make_texture("antiskyrmion", grid)).item() - 1.0) < 1e-3


def test_divergence_free_swirl_is_invisible():
    """Sec. 3.3: an azimuthal in-plane field vanishing at the edge sources no
    stray field; a radial one of the same magnitude does."""
    n = 64
    x = torch.linspace(-1, 1, n, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r, ph = torch.sqrt(X ** 2 + Y ** 2), torch.atan2(Y, X)
    amp = torch.where(r < 0.8, torch.sin(math.pi * r / 0.8) ** 2, torch.zeros_like(r))
    zeros = torch.zeros_like(r)
    swirl = torch.stack([-amp * torch.sin(ph), amp * torch.cos(ph), zeros], -1)
    radial = torch.stack([amp * torch.cos(ph), amp * torch.sin(ph), zeros], -1)
    g = Geometry(grid=(n, n), sample_size_nm=(128, 128), height_nm=20)
    b_swirl = stray_field(swirl, g).norm(dim=-1).max()
    b_radial = stray_field(radial, g).norm(dim=-1).max()
    assert b_swirl < 1e-2 * b_radial, (b_swirl, b_radial)


def test_parse_texture():
    assert parse_texture("vortex:polarity=-1,radius=0.25") == ("meron", {"polarity": -1, "radius": 0.25})
    name, p = parse_texture("skyrmion:helicity=pi/2")
    assert name == "skyrmion" and abs(p["helicity"] - math.pi / 2) < 1e-15


def test_skyrmion_meron_reproduces_manuscript():
    """Sec. 5 test case: Q -> (-1, -0.5), NRMSE ~1.1%, Ms2 ~1.5 Ms0."""
    res = generate_pair("skyrmion", "meron", Geometry())
    assert abs(res.Q1 + 1.0) < 2e-3 and abs(res.Q2 + 0.5) < 2e-3
    assert 0.005 < res.nrmse < 0.02, res.nrmse
    assert res.initial["nrmse"] > 0.5
    assert abs(res.Ms1 / 1e6 - 1.0) < 1e-9 and 1.3 < res.Ms2 / 1e6 < 1.7
    assert np.allclose(np.linalg.norm(res.m1, axis=-1), 1.0)


def test_fixed_ms_mode_keeps_ms_and_charges():
    """Equal, fixed Ms still yields an adversarial pair if beta starts high."""
    s = OptimizerSettings(n_iter=450, ms_mode="fixed", beta0=100)
    res = generate_pair("skyrmion", "meron", Geometry(), settings=s)
    assert res.Ms1 == res.Ms2 == 1e6
    assert res.charge_constraint_met, res
    assert res.nrmse < 0.03


def test_constraint_violation_is_flagged():
    s = OptimizerSettings(n_iter=450, ms_mode="fixed", beta0=1)
    res = generate_pair("skyrmion", "meron", Geometry(), settings=s)
    assert not res.charge_constraint_met and "WARNING" in str(res)


def test_difference_mode_and_save_load():
    s = OptimizerSettings(n_iter=100)
    res = generate_pair("meron", "uniform:phi_deg=30", Geometry(grid=(16, 16), sample_size_nm=(32, 32)),
                        settings=s, mode="difference", q=0.5)
    assert res.targets["q"] == 0.5
    with tempfile.TemporaryDirectory() as d:
        res.save(d)
        back = PairResult.load(d)
    assert np.allclose(back.B1, res.B1) and back.labels == res.labels and back.Q2 == res.Q2


def test_monte_carlo_deterministic_across_workers():
    g = Geometry(grid=(16, 16), sample_size_nm=(32, 32))
    s = OptimizerSettings(n_iter=30)
    a = run_monte_carlo(g, MonteCarloConfig(n_samples=4, seed=3, n_workers=1), s, progress=False)
    b = run_monte_carlo(g, MonteCarloConfig(n_samples=4, seed=3, n_workers=2), s, progress=False)
    assert [r["texture1"] for r in a.records] == [r["texture1"] for r in b.records]
    assert np.allclose(a.column("nrmse"), b.column("nrmse"))
    summ = a.summary()
    assert summ["n_samples"] == 4 and "charge_error_max" in summ


def test_analysis_runs():
    res = generate_pair("skyrmion", "meron", Geometry(), settings=OptimizerSettings(n_iter=50))
    rows = analysis.multipole_comparison(res)
    assert rows[0]["l"] == 0 and len(rows) == 1 + 2 + 3 + 4 + 5
    vol = analysis.multipole_comparison(res, source="volume")
    assert len(vol) == len(rows)
    loc = analysis.kspace_localization(res)
    assert abs(loc["l2_real"] - loc["l2_kspace"]) < 1e-9 * max(1.0, loc["l2_real"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
