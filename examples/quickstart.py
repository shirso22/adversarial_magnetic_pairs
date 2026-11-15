"""End-to-end example: one pair, its analysis, and a small ensemble.

    python examples/quickstart.py
"""
import matplotlib

matplotlib.use("Agg")

from adversarial_pairs import Geometry, MonteCarloConfig, OptimizerSettings, analysis, generate_pair, run_monte_carlo
from adversarial_pairs import plotting

if __name__ == "__main__":       # required for multiprocessing (spawn)
    geom = Geometry(grid=(32, 32), sample_size_nm=(64, 64), thickness_nm=1, height_nm=20, Ms=1e6)
    settings = OptimizerSettings(n_iter=450, ms_mode="free", log_every=150)

    # 1. Skyrmion vs meron (Sec. 5 test case)
    res = generate_pair("skyrmion", "meron", geom, settings=settings)
    print(res)
    res.save("results/quickstart_pair")
    plotting.plot_pair(res).savefig("results/quickstart_pair/pair.png", dpi=150)

    # 2. Where do the textures differ, and why can't the field tell?
    print(analysis.format_moment_table(analysis.multipole_comparison(res)))
    loc = analysis.kspace_localization(res)
    print(f"90% of field power below k = {loc['band'][1]:.3f} rad/nm")

    # 3. Same pair with both saturation magnetizations held fixed. Without the
    #    Ms freedom the topology term needs a stronger start (beta0) to hold Q.
    fixed = generate_pair("skyrmion", "meron", geom,
                          settings=settings.replace(ms_mode="fixed", beta0=100, log_every=0))
    print("fixed Ms:", fixed)

    # 4. Small Monte Carlo ensemble
    mc = run_monte_carlo(geom, MonteCarloConfig(n_samples=20, n_workers=4), settings.replace(log_every=0))
    print(mc.summary())
    mc.save("results/quickstart_mc")
