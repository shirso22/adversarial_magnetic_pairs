# adversarial_magnetic_pairs

Generate **adversarial magnetization pairs**: two thin-film textures with
different topological charge (for example a skyrmion and a meron) whose stray
fields are nearly indistinguishable at a realistic magnetometer standoff.

This is the code behind *Topological ambiguity in stray-field magnetometry*
(S. Chattopadhyay and A. Singha). Both textures are optimized together with a
physics-informed loss. Neither field is fixed in advance:

```
L = NRMSE(B1, B2)^2  +  beta * E_topo  +  gamma * E_smooth
```

- **Field term.** NRMSE of the two stray fields, computed with an FFT thin-film
  kernel (zero-padded, float64).
- **Topology term.** Either anchors each charge to a target,
  `(Q1 - Q1*)^2 + (Q2 - Q2*)^2`, or targets a charge difference,
  `(|Q1 - Q2| - q)^2`. Charges use the Berg–Lüscher solid-angle method. `beta`
  is set by penalty continuation: it grows until the charge constraint is met.
- **Smoothness term.** Mean squared nearest-neighbour difference.

Each texture is parameterized by polar angles per cell, so |m| = 1 everywhere.
Each texture also has its own saturation magnetization, which can be held fixed
or optimized.

## Install

Requires Python 3.9 or newer, plus PyTorch, NumPy, SciPy and Matplotlib.

```bash
pip install -e .            # from the repository root
python tests/test_core.py   # or: pytest
```

## Walkthrough notebook

[`examples/walkthrough.ipynb`](examples/walkthrough.ipynb) goes through the full workflow step by step:
geometry and settings, textures, optimizing one pair (free and fixed Ms), multipole and k-space
analysis, Monte Carlo validation, standoff sweeps, and grid/weight sensitivity. It runs in about a
minute on a laptop CPU with the default (reduced) ensemble sizes.

## Command line

```bash
# One pair: skyrmion vs meron, 20 nm standoff, 64 nm sample on a 32x32 grid
python -m adversarial_pairs pair --texture1 skyrmion --texture2 meron \
       --height 20 --size 64 --grid 32 --Ms 1e6 --analyze --out results/sk_meron

# Hold both saturation magnetizations at the initial value
python -m adversarial_pairs pair --texture1 "skyrmion:helicity=0" \
       --texture2 "fractional:charge=0.25" --ms-mode fixed --out results/fixed_ms

# Monte Carlo ensemble with statistical validation (8 processes)
python -m adversarial_pairs montecarlo --samples 100 --workers 8 --out results/mc

# The same seeded ensemble at several standoff heights
python -m adversarial_pairs standoff --heights 5,10,20,30,50 --samples 100 --workers 8

# Grid-resolution sensitivity at fixed physical sample size
python -m adversarial_pairs grid --grids 32,64,128 --iters 3000

# Multipole and k-space analysis of a saved pair
python -m adversarial_pairs analyze results/sk_meron --source volume

python -m adversarial_pairs textures   # list texture classes and parameters
python -m adversarial_pairs pair --help
```

### Main options

| Option | Meaning | Default |
|---|---|---|
| `--texture1`, `--texture2` | `name` or `name:key=value,...` (see `textures`) | skyrmion, meron |
| `--height` | standoff height (nm) | 20 |
| `--size` | sample size (nm), `64` or `64x48` | 64 |
| `--grid` | cells, `32` or `64x48` | 32 |
| `--thickness` | film thickness (nm) | 1 |
| `--Ms` | initial saturation magnetization (A/m) | 1e6 |
| `--ms-mode` | `fixed` (both held), `free` (both vary), `first`, `second` | free |
| `--ms-bounds` | bounds on Ms/Ms0 for the textures that vary | 1,5 |
| `--mode` | `anchor` (each charge to its target) or `difference` (\|Q1−Q2\| = q) | anchor |
| `--Q1`, `--Q2`, `--q` | explicit charge targets | initial charges |
| `--charge-grid` | snap initial charges to multiples of this to set the targets | none (pair), 0.25 (MC) |
| `--iters`, `--lr`, `--gamma`, `--beta0` | optimizer settings | 450, 1e-2, 1e-3, 1 |
| `--smoothness-on` | apply smoothness to `M` = s·m or to the unit field `m` | M |
| `--workers` | parallel processes for Monte Carlo | 1 |

Textures: `skyrmion`, `antiskyrmion`, `meron` (alias `vortex`), `fractional`,
`uniform`. Parameters include `polarity`, `radius` (as a fraction of the
half-width), `helicity` (`pi/2` for Bloch, `0` for Néel), `winding` and
`center`. `fractional` also takes `charge`, and `uniform` takes `theta_deg`
and `phi_deg`.

## Python API

```python
from adversarial_pairs import Geometry, OptimizerSettings, generate_pair, analysis
from adversarial_pairs import MonteCarloConfig, run_monte_carlo, standoff_sweep
from adversarial_pairs import plotting

geom = Geometry(grid=(32, 32), sample_size_nm=(64, 64), thickness_nm=1,
                height_nm=20, Ms=1e6)
settings = OptimizerSettings(n_iter=450, ms_mode="free")

res = generate_pair("skyrmion", "meron:polarity=1", geom, settings=settings)
print(res)                      # Q1, Q2, NRMSE, final Ms
res.save("results/pair")        # pair.npz + summary.json
plotting.plot_pair(res)

rows = analysis.multipole_comparison(res, order_max=8, source="mz")
print(analysis.format_moment_table(rows))
loc = analysis.kspace_localization(res)

mc = run_monte_carlo(geom, MonteCarloConfig(n_samples=100, n_workers=8), settings)
print(mc.summary())
print(mc.by_pair_type())
plotting.plot_mc_summary(mc)
```

Always check `res.charge_constraint_met` (also printed as a warning). If β
starts too low, the optimizer can lower the field mismatch by unwinding a
texture. With `ms_mode="fixed"` this happens at the default `beta0=1`, so use
`beta0=100` or more iterations.

When using `n_workers > 1` from a script, put the call under
`if __name__ == "__main__":` (worker processes are spawned).

Custom textures: call `register_texture("name", fn)` with
`fn(nx, ny, **params)` returning an `(nx, ny, 3)` tensor, or pass a tensor
directly to `generate_pair` or `optimize_pair`. See `examples/quickstart.py`
for a full run.

## Outputs

| Command | Files |
|---|---|
| `pair` | `pair.npz` (m1, m2, B1, B2 in T, Ms, history), `summary.json`, `history.png`, `pair.png`, plus `multipoles_*.txt/png` and `kspace.json/png` with `--analyze` |
| `montecarlo` | `samples.csv` (one row per sample), `summary.json` (ensemble statistics and per-pair-type breakdown), `mc_summary.png` |
| `standoff` | one Monte Carlo folder per height, `standoff_summary.json`, `standoff.png` |

Monte Carlo samples are seeded by `(seed, index)`. An ensemble is therefore
reproducible regardless of `--workers`, and `standoff` runs the same initial
pairs at every height.

## Layout

```
adversarial_pairs/
  geometry.py    sample/measurement geometry (nm, A/m)
  physics.py     stray-field operator, Berg–Lüscher charge, NRMSE, smoothness
  textures.py    texture generators, registry, spec parsing, perturbation
  optimize.py    OptimizerSettings, optimize_pair, generate_pair, PairResult
  montecarlo.py  ensembles, statistical summary, standoff sweep
  analysis.py    multipole moments, k-space localization, sensitivity sweeps
  plotting.py    figures
  cli.py         command-line interface
examples/        walkthrough notebook and quickstart script
tests/           physics and regression tests
legacy/          original notebook from the first submission
```

## Notes on conventions

- `PairResult.B1` and `B2` are in tesla for the given `Ms`. `Ms1` and `Ms2` are in A/m.
- Multipole moments are reported in units of the initial Ms, using normalized
  coordinates on [−1/2, 1/2].
- The NRMSE is the per-component RMS of B1 − B2 divided by the mean field
  magnitude, averaged over the two textures.
