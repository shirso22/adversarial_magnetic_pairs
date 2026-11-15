"""Adversarial magnetization pairs: topologically distinct textures with
near-identical stray fields.

Quick start::

    from adversarial_pairs import Geometry, OptimizerSettings, generate_pair
    res = generate_pair("skyrmion", "meron", Geometry(height_nm=20))
    print(res)
"""
from .geometry import Geometry
from .optimize import MS_MODES, OptimizerSettings, PairResult, generate_pair, optimize_pair
from .physics import (StrayFieldOperator, nrmse, smoothness, stray_field, topological_charge,
                      topological_charge_finite_difference)
from .textures import TEXTURES, make_texture, parse_texture, perturb, register_texture
from .montecarlo import MonteCarloConfig, MonteCarloResult, run_monte_carlo, standoff_sweep
from . import analysis

__version__ = "1.0.0"

__all__ = [
    "Geometry", "OptimizerSettings", "PairResult", "MS_MODES", "generate_pair", "optimize_pair",
    "StrayFieldOperator", "stray_field", "topological_charge", "topological_charge_finite_difference",
    "nrmse", "smoothness", "TEXTURES", "make_texture", "parse_texture", "perturb", "register_texture",
    "MonteCarloConfig", "MonteCarloResult", "run_monte_carlo", "standoff_sweep", "analysis",
]
