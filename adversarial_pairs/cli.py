"""Command-line interface: ``python -m adversarial_pairs <command> --help``.

Commands
  pair        optimize one adversarial pair (+ optional multipole / k-space analysis)
  montecarlo  random ensemble with statistical validation
  standoff    Monte Carlo ensemble at several standoff heights
  grid        grid-resolution sensitivity of one pair
  analyze     multipole + k-space analysis of a saved pair
  textures    list available texture classes
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from .geometry import Geometry
from .optimize import MS_MODES, OptimizerSettings


def _floats(text):
    return [float(v) for v in text.split(",")]


def _pair_of(cast):
    def parse(text):
        vals = [cast(v) for v in text.replace("x", ",").split(",")]
        if len(vals) == 1:
            vals *= 2
        if len(vals) != 2:
            raise argparse.ArgumentTypeError(f"expected one or two values, got {text!r}")
        return tuple(vals)
    return parse


def _add_geometry(p):
    g = p.add_argument_group("geometry")
    g.add_argument("--grid", type=_pair_of(int), default=(32, 32), help="cells, e.g. 32 or 64x48 (default 32)")
    g.add_argument("--size", type=_pair_of(float), default=(64.0, 64.0), help="sample size in nm (default 64)")
    g.add_argument("--thickness", type=float, default=1.0, help="film thickness in nm (default 1)")
    g.add_argument("--height", type=float, default=20.0, help="standoff height in nm (default 20)")
    g.add_argument("--Ms", type=float, default=1e6, help="initial saturation magnetization, A/m (default 1e6)")


def _add_settings(p):
    s = p.add_argument_group("optimizer")
    s.add_argument("--iters", type=int, default=450, help="iterations (default 450)")
    s.add_argument("--lr", type=float, default=1e-2)
    s.add_argument("--gamma", type=float, default=1e-3, help="smoothness weight")
    s.add_argument("--beta0", type=float, default=1.0, help="initial topology weight")
    s.add_argument("--ms-mode", choices=sorted(MS_MODES), default="free",
                   help="which saturation magnetizations may vary (default free)")
    s.add_argument("--ms-bounds", type=_floats, default=[1.0, 5.0], help="bounds on Ms/Ms0 (default 1,5)")
    s.add_argument("--smoothness-on", choices=["M", "m"], default="M")
    s.add_argument("--charge-tol", type=float, default=1e-2)
    s.add_argument("--device", default="cpu")


def _add_charge(p, mc=False):
    c = p.add_argument_group("topology constraint")
    c.add_argument("--mode", choices=["anchor", "difference"], default="anchor")
    c.add_argument("--charge-grid", type=float, default=0.25 if mc else None,
                   help="snap initial charges to multiples of this to define targets")
    if not mc:
        c.add_argument("--Q1", type=float, default=None, help="target charge of texture 1 (anchor mode)")
        c.add_argument("--Q2", type=float, default=None, help="target charge of texture 2 (anchor mode)")
        c.add_argument("--q", type=float, default=None, help="target |Q1 - Q2| (difference mode)")


def _add_mc(p):
    m = p.add_argument_group("Monte Carlo")
    m.add_argument("--samples", type=int, default=100)
    m.add_argument("--classes", default="skyrmion,meron,fractional,uniform",
                   help="comma-separated texture classes to draw from")
    m.add_argument("--first", default=None, help="fix texture 1 to this class")
    m.add_argument("--perturbation", type=float, default=0.01)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--workers", type=int, default=1, help="parallel processes")


def _geometry(a):
    return Geometry(grid=a.grid, sample_size_nm=a.size, thickness_nm=a.thickness, height_nm=a.height, Ms=a.Ms)


def _settings(a):
    return OptimizerSettings(n_iter=a.iters, lr=a.lr, gamma=a.gamma, beta0=a.beta0, ms_mode=a.ms_mode,
                             ms_bounds=tuple(a.ms_bounds), smoothness_on=a.smoothness_on,
                             charge_tol=a.charge_tol, log_every=getattr(a, "log_every", 0))


def _mc_config(a):
    from .montecarlo import MonteCarloConfig
    return MonteCarloConfig(n_samples=a.samples, texture_classes=tuple(a.classes.split(",")),
                            first_texture=a.first, perturbation=a.perturbation, mode=a.mode,
                            charge_grid=a.charge_grid, seed=a.seed, n_workers=a.workers)


def _save_fig(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _analyze(res, out: Path, order: int, source: str, plots: bool):
    from . import analysis
    rows = analysis.multipole_comparison(res, order_max=order, source=source)
    table = analysis.format_moment_table(rows)
    (out / f"multipoles_{source}.txt").write_text(table + "\n")
    print(f"\nMultipole moments ({source} source):\n{table}")
    loc = analysis.kspace_localization(res)
    keys = ("band", "frac_power_in_band", "frac_residual_in_band", "rel_in_band_mean", "nrmse",
            "l2_real", "l2_kspace")
    (out / "kspace.json").write_text(json.dumps({k: loc[k] for k in keys}, indent=2))
    print(f"\n90% field-power band: k = {loc['band'][0]:.3f}-{loc['band'][1]:.3f} rad/nm; "
          f"residual energy in band {loc['frac_residual_in_band'] * 100:.1f}%; "
          f"Parseval check {loc['l2_real'] * 100:.3f}% vs {loc['l2_kspace'] * 100:.3f}%")
    if plots:
        from . import plotting
        _save_fig(plotting.plot_multipoles(rows), out / f"multipoles_{source}.png")
        _save_fig(plotting.plot_kspace(loc), out / "kspace.png")


def cmd_pair(a):
    from .optimize import generate_pair
    out = Path(a.out)
    res = generate_pair(a.texture1, a.texture2, _geometry(a), perturbation=a.perturbation, seed=a.seed,
                        settings=_settings(a), mode=a.mode, Q1_target=a.Q1, Q2_target=a.Q2, q=a.q,
                        charge_grid=a.charge_grid, device=a.device)
    res.save(out)
    print(res)
    if not a.no_plots:
        from . import plotting
        _save_fig(plotting.plot_history(res), out / "history.png")
        _save_fig(plotting.plot_pair(res), out / "pair.png")
    if a.analyze:
        _analyze(res, out, a.order, a.source, not a.no_plots)
    print(f"\nresults written to {out}/")


def cmd_analyze(a):
    from .optimize import PairResult
    res = PairResult.load(a.pair)
    print(res)
    _analyze(res, Path(a.pair), a.order, a.source, not a.no_plots)


def cmd_montecarlo(a):
    from .montecarlo import run_monte_carlo
    out = Path(a.out)
    mc = run_monte_carlo(_geometry(a), _mc_config(a), _settings(a), device=a.device)
    mc.save(out)
    print(json.dumps(mc.summary(), indent=2))
    if not a.no_plots:
        from . import plotting
        _save_fig(plotting.plot_mc_summary(mc), out / "mc_summary.png")
    print(f"\nresults written to {out}/")


def cmd_standoff(a):
    from .montecarlo import standoff_sweep
    out = Path(a.out)
    sweep = standoff_sweep(a.heights, _geometry(a), _mc_config(a), _settings(a), device=a.device)
    rows = []
    for h, mc in sorted(sweep.items()):
        mc.save(out / f"h_{h:g}nm")
        s = mc.summary()
        rows.append({"height_nm": h, "nrmse_mean": s["final_nrmse_mean"], "nrmse_std": s["final_nrmse_std"],
                     "mean_abs_dB_T": float(mc.column("mean_abs_dB_T").mean())})
    (out / "standoff_summary.json").write_text(json.dumps(rows, indent=2))
    from .analysis import format_rows
    print(format_rows(rows))
    if not a.no_plots:
        from . import plotting
        _save_fig(plotting.plot_standoff_sweep(sweep), out / "standoff.png")


def cmd_grid(a):
    from .analysis import format_rows, grid_sensitivity
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = grid_sensitivity(a.grids, a.texture1, a.texture2, _geometry(a), settings=_settings(a),
                            mode=a.mode, Q1_target=a.Q1, Q2_target=a.Q2, q=a.q,
                            charge_grid=a.charge_grid, device=a.device)
    (out / "grid_sensitivity.json").write_text(json.dumps(rows, indent=2))
    print(format_rows(rows))


def cmd_textures(a):
    from .textures import ALIASES, TEXTURES
    for name, fn in TEXTURES.items():
        sig = inspect.signature(fn)
        params = ", ".join(f"{k}={v.default}" for k, v in sig.parameters.items()
                           if v.default is not inspect.Parameter.empty)
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<13} {doc}\n{'':<13} params: {params or '(see skyrmion)'}")
    print("aliases: " + ", ".join(f"{k} -> {v}" for k, v in ALIASES.items()))
    print('\nUse as e.g. --texture1 "skyrmion:polarity=-1,radius=0.25,helicity=0"')


def build_parser():
    p = argparse.ArgumentParser(prog="adversarial_pairs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common_out(sp, default):
        sp.add_argument("--out", default=default, help=f"output directory (default {default})")
        sp.add_argument("--no-plots", action="store_true")

    def analysis_opts(sp):
        sp.add_argument("--order", type=int, default=8, help="max multipole order")
        sp.add_argument("--source", choices=["mz", "volume"], default="mz",
                        help="moments of s*m_z (manuscript) or of the volume charge -div(m_par)")

    sp = sub.add_parser("pair", help="optimize one adversarial pair")
    sp.add_argument("--texture1", default="skyrmion", help='e.g. "skyrmion" or "skyrmion:polarity=-1"')
    sp.add_argument("--texture2", default="meron")
    sp.add_argument("--perturbation", type=float, default=0.0, help="initial Gaussian noise std")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--analyze", action="store_true", help="also run multipole + k-space analysis")
    sp.add_argument("--log-every", type=int, default=50)
    analysis_opts(sp)
    _add_geometry(sp); _add_settings(sp); _add_charge(sp)
    common_out(sp, "results/pair")
    sp.set_defaults(func=cmd_pair)

    sp = sub.add_parser("analyze", help="multipole + k-space analysis of a saved pair")
    sp.add_argument("pair", help="directory written by the 'pair' command")
    analysis_opts(sp)
    sp.add_argument("--no-plots", action="store_true")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("montecarlo", help="random ensemble + statistical validation")
    _add_mc(sp); _add_geometry(sp); _add_settings(sp); _add_charge(sp, mc=True)
    common_out(sp, "results/montecarlo")
    sp.set_defaults(func=cmd_montecarlo)

    sp = sub.add_parser("standoff", help="Monte Carlo ensemble at several standoff heights")
    sp.add_argument("--heights", type=_floats, default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                    help="comma-separated heights in nm")
    _add_mc(sp); _add_geometry(sp); _add_settings(sp); _add_charge(sp, mc=True)
    common_out(sp, "results/standoff")
    sp.set_defaults(func=cmd_standoff)

    sp = sub.add_parser("grid", help="grid-resolution sensitivity at fixed sample size")
    sp.add_argument("--grids", type=lambda t: [_pair_of(int)(g) for g in t.split(",")],
                    default=[(32, 32), (64, 64), (128, 128)], help="e.g. 32,64,128 or 32x32,64x64")
    sp.add_argument("--texture1", default="skyrmion")
    sp.add_argument("--texture2", default="meron")
    _add_geometry(sp); _add_settings(sp); _add_charge(sp)
    sp.set_defaults(no_plots=True)
    sp.add_argument("--out", default="results/grid")
    sp.set_defaults(func=cmd_grid)

    sp = sub.add_parser("textures", help="list texture classes and their parameters")
    sp.set_defaults(func=cmd_textures)
    return p


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
