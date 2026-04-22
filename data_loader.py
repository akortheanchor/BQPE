"""
bqpe/src/data_loader.py
========================
Load, validate, and cross-validate digitised experimental data
against BQPE theoretical predictions.

Supports all three hardware platforms:
  - NV-centre    (Santagati et al. 2022)
  - Photonic     (Valeri et al. 2020)
  - Trapped-ion  (Pogorelov et al. 2021)

Usage
-----
    from data_loader import load_platform_data, cross_validate_all

    df = load_platform_data('nv_centre')
    results = cross_validate_all()
"""

from __future__ import annotations

import os
import json
import csv
from pathlib import Path
from typing import Optional
import numpy as np


# Default data directory (relative to this file)
_HERE = Path(__file__).parent
_DATA_DIR = _HERE.parent / "data" / "experimental"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_platform_data(
    platform: str,
    data_dir: Optional[str] = None,
) -> list[dict]:
    """
    Load digitised experimental data for one hardware platform.

    Parameters
    ----------
    platform : str
        One of 'nv_centre', 'photonic', 'trapped_ion'.
    data_dir : str, optional
        Override the default data directory path.

    Returns
    -------
    list of dicts, each with keys:
        'platform', 'N_shots', 'MSE_exp', 'MSE_err',
        'CRLB_corrected', 'efficiency_ratio'
    """
    _file_map = {
        'nv_centre':   'santagati2022_nv_centre.csv',
        'photonic':    'valeri2020_photonic.csv',
        'trapped_ion': 'pogorelov2021_trapped_ion.csv',
    }
    if platform not in _file_map:
        raise ValueError(
            f"Unknown platform '{platform}'. "
            f"Choose from: {list(_file_map.keys())}"
        )

    base = Path(data_dir) if data_dir else _DATA_DIR
    fpath = base / _file_map[platform]

    if not fpath.exists():
        raise FileNotFoundError(
            f"Data file not found: {fpath}\n"
            f"Ensure the data/experimental/ directory is populated."
        )

    rows = []
    with open(fpath, newline='') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
        # re-open cleanly for csv
    with open(fpath, newline='') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append({
            'platform':          row['platform'].strip(),
            'N_shots':           int(row['N_shots']),
            'MSE_exp':           float(row['MSE_exp']),
            'MSE_err':           float(row['MSE_err']),
            'CRLB_corrected':    float(row['CRLB_corrected']),
            'efficiency_ratio':  float(row['efficiency_ratio']),
        })
    return rows


def load_all_platforms(data_dir: Optional[str] = None) -> dict[str, list[dict]]:
    """
    Load digitised experimental data for all three platforms.

    Returns
    -------
    dict mapping platform name -> list of data rows.
    """
    platforms = ['nv_centre', 'photonic', 'trapped_ion']
    return {p: load_platform_data(p, data_dir) for p in platforms}


def load_platform_parameters(
    json_path: Optional[str] = None,
) -> dict:
    """
    Load hardware platform parameters from JSON file.

    Returns
    -------
    dict: full JSON content of platform_parameters.json
    """
    if json_path is None:
        json_path = str(_DATA_DIR / 'platform_parameters.json')
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cross-validation: theory vs experiment
# ---------------------------------------------------------------------------

def cross_validate_platform(
    platform_name: str,
    data_dir: Optional[str] = None,
    json_path: Optional[str] = None,
) -> dict:
    """
    Cross-validate BQPE theoretical prediction against experimental data.

    For each data point (N, MSE_exp), computes:
      - BQPE prediction: CRLB_hw = 1 / (N * (pg_eff)^2)
      - Discrepancy (%): |MSE_exp - CRLB_hw| / MSE_exp * 100
      - Efficiency ratio: MSE_exp / CRLB_hw

    Parameters
    ----------
    platform_name : str
        One of 'nv_centre', 'photonic', 'trapped_ion'.
    data_dir : str, optional
        Path to data directory.
    json_path : str, optional
        Path to platform_parameters.json.

    Returns
    -------
    dict with keys:
        'platform'       : str
        'pg_eff'         : float
        'points'         : list of per-shot-count dicts
        'mean_discrepancy_pct' : float
        'mean_efficiency_ratio': float
        'max_N_result'   : dict (result at largest N)
    """
    # Load experimental data
    data = load_platform_data(platform_name, data_dir)

    # Load platform parameters
    params_all = load_platform_parameters(json_path)
    pentry = params_all[platform_name]['parameters']
    p      = pentry['p']
    gamma  = pentry['gamma']
    eta_ro = pentry['eta_ro']
    eta    = pentry.get('eta_detection', 1.0)

    # Effective signal amplitude
    pg_eff = p * gamma * eta * (1.0 - 2.0 * eta_ro)

    points = []
    for row in data:
        N       = row['N_shots']
        mse_exp = row['MSE_exp']
        mse_err = row['MSE_err']

        crlb_hw     = 1.0 / (N * pg_eff ** 2)
        discrepancy = abs(mse_exp - crlb_hw) / mse_exp * 100.0
        eff_ratio   = mse_exp / crlb_hw

        points.append({
            'N_shots':          N,
            'MSE_exp':          mse_exp,
            'MSE_err':          mse_err,
            'CRLB_hw':          crlb_hw,
            'discrepancy_pct':  discrepancy,
            'efficiency_ratio': eff_ratio,
        })

    mean_disc  = float(np.mean([pt['discrepancy_pct'] for pt in points]))
    mean_eff   = float(np.mean([pt['efficiency_ratio'] for pt in points]))
    max_N_pt   = max(points, key=lambda x: x['N_shots'])

    return {
        'platform':                platform_name,
        'pg_eff':                  pg_eff,
        'points':                  points,
        'mean_discrepancy_pct':    mean_disc,
        'mean_efficiency_ratio':   mean_eff,
        'max_N_result':            max_N_pt,
    }


def cross_validate_all(
    data_dir: Optional[str] = None,
    json_path: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run cross-validation for all three hardware platforms.

    Parameters
    ----------
    verbose : bool
        Print summary table if True.

    Returns
    -------
    dict mapping platform name -> cross-validation result dict.
    """
    platforms = ['nv_centre', 'photonic', 'trapped_ion']
    results = {}

    for pname in platforms:
        try:
            results[pname] = cross_validate_platform(pname, data_dir, json_path)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

    if verbose and results:
        print("\n" + "=" * 72)
        print(f"{'Platform':<22} {'(pg)_eff':>8} "
              f"{'Max N':>7} {'MSE_exp':>12} {'CRLB_hw':>12} "
              f"{'Disc%':>7} {'Eff.ratio':>10}")
        print("-" * 72)
        for pname, res in results.items():
            pt = res['max_N_result']
            print(
                f"{pname:<22} {res['pg_eff']:>8.4f} "
                f"{pt['N_shots']:>7d} {pt['MSE_exp']:>12.2e} "
                f"{pt['CRLB_hw']:>12.2e} "
                f"{pt['discrepancy_pct']:>7.1f} "
                f"{pt['efficiency_ratio']:>10.3f}"
            )
        print("=" * 72)

    return results


# ---------------------------------------------------------------------------
# Synthetic data generation (for testing)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    platform_name: str,
    N_trials: int = 1000,
    seed: int = 2024,
    data_dir: Optional[str] = None,
) -> list[dict]:
    """
    Generate synthetic experimental data by running simulated BQPE trials
    on a given hardware platform.

    Useful for testing the cross-validation pipeline without the CSV files.

    Parameters
    ----------
    platform_name : str
        One of the built-in platform names.
    N_trials : int
        Monte Carlo trials per shot count.
    seed : int
        Random seed.

    Returns
    -------
    list of dicts with the same schema as load_platform_data().
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hardware_noise import PLATFORMS, run_hardware_monte_carlo
    import math

    platform = PLATFORMS[platform_name]

    shot_ranges = {
        'nv_centre':   [50, 100, 150, 200, 250, 300],
        'photonic':    [30, 60,  90,  120, 150, 200],
        'trapped_ion': [40, 80,  120, 160, 200, 280],
        'ideal':       [10, 20,  50,  100, 200, 500],
    }
    shots = shot_ranges.get(platform_name, [50, 100, 200, 300])

    rows = []
    for N in shots:
        res = run_hardware_monte_carlo(
            N, math.pi / 3, platform, N_trials=N_trials, seed=seed
        )
        rows.append({
            'platform':         platform_name,
            'N_shots':          N,
            'MSE_exp':          res['mse'],
            'MSE_err':          res['mse'] * 0.08,   # 8% CI estimate
            'CRLB_corrected':   res['crlb_hw'],
            'efficiency_ratio': res['efficiency_ratio'],
        })
    return rows
