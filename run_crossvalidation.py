#!/usr/bin/env python3
"""
bqpe/scripts/run_crossvalidation.py
=====================================
Command-line script: cross-validate BQPE theoretical predictions
against all three published experimental datasets and print a
detailed report.

Usage
-----
    python scripts/run_crossvalidation.py
    python scripts/run_crossvalidation.py --datadir data/experimental
    python scripts/run_crossvalidation.py --save results/crossval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from hardware_noise import PLATFORMS, run_hardware_monte_carlo
from data_loader import (
    load_all_platforms, load_platform_parameters,
    cross_validate_all, generate_synthetic_data,
)


def print_header(title: str, width: int = 72):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_full_report(
    data_dir: str | None = None,
    json_path: str | None = None,
    use_synthetic: bool = False,
    N_trials: int = 500,
    seed: int = 2024,
    theta_true: float = math.pi / 3,
    save_path: str | None = None,
):
    """Run the full cross-validation report."""

    print_header("BQPE Experimental Cross-Validation Report")
    print(f"  theta_true = pi/3 = {theta_true:.6f} rad")
    print(f"  N_trials   = {N_trials}")
    print(f"  seed       = {seed}")
    print(f"  data_dir   = {data_dir or 'default (data/experimental/)'}")
    print(f"  synthetic  = {use_synthetic}")

    # ── Platform parameters ─────────────────────────────────────────────────
    print_header("Platform Parameters (from published papers)")
    fmt = "{:<22} {:>6} {:>6} {:>9} {:>10} {:>8} {:>10}"
    print(fmt.format("Platform", "p", "γ", "η_ro", "σ_φ(rad)",
                     "η_det", "(pγ)_eff"))
    print("-" * 72)
    for pname, pl in PLATFORMS.items():
        if pname == 'ideal':
            continue
        print(fmt.format(
            pl.name[:22], f"{pl.p:.3f}", f"{pl.gamma:.3f}",
            f"{pl.eta_ro:.4f}", f"{pl.sigma_phi:.4f}",
            f"{pl.eta_detection:.3f}", f"{pl.pg_eff:.4f}",
        ))

    # ── Simulation results ──────────────────────────────────────────────────
    print_header("Simulation Results (Monte Carlo, true adaptive protocol)")
    shot_map = {
        'nv_centre':   [50, 100, 200, 300],
        'photonic':    [30, 60,  100, 200],
        'trapped_ion': [40, 80,  160, 280],
    }

    sim_results = {}
    for pname in ['nv_centre', 'photonic', 'trapped_ion']:
        pl   = PLATFORMS[pname]
        data = []
        for N in shot_map[pname]:
            res = run_hardware_monte_carlo(N, theta_true, pl, N_trials, seed)
            data.append({
                'N': N, 'mse': res['mse'], 'bias': res['bias'],
                'crlb_hw': res['crlb_hw'],
                'ratio': res['efficiency_ratio'],
            })
        sim_results[pname] = data

        print(f"\n  {pl.name}")
        print(f"  {'N':>5} {'MSE':>12} {'Bias':>12} "
              f"{'CRLB_hw':>12} {'MSE/CRLB':>10}")
        print("  " + "-" * 55)
        for d in data:
            print(f"  {d['N']:>5} {d['mse']:>12.4e} {d['bias']:>12.4e} "
                  f"{d['crlb_hw']:>12.4e} {d['ratio']:>10.3f}")

    # ── Experimental cross-validation ───────────────────────────────────────
    print_header("Cross-Validation: Experiment vs BQPE Prediction")

    if use_synthetic:
        print("  (Using synthetic data — real CSV files not available)")

    try:
        cv = cross_validate_all(data_dir=data_dir, json_path=json_path,
                                verbose=False)
    except Exception as e:
        print(f"  Warning: could not load CSV data ({e})")
        print("  Generating synthetic data for demonstration...")
        cv = {}
        for pname in ['nv_centre', 'photonic', 'trapped_ion']:
            rows = generate_synthetic_data(pname, N_trials=N_trials, seed=seed)
            pl   = PLATFORMS[pname]
            pts  = []
            for r in rows:
                crlb_hw = pl.crlb_hw(r['N_shots'])
                pts.append({
                    'N_shots': r['N_shots'],
                    'MSE_exp': r['MSE_exp'],
                    'MSE_err': r['MSE_err'],
                    'CRLB_hw': crlb_hw,
                    'discrepancy_pct': abs(r['MSE_exp'] - crlb_hw) / r['MSE_exp'] * 100,
                    'efficiency_ratio': r['efficiency_ratio'],
                })
            cv[pname] = {
                'pg_eff': pl.pg_eff,
                'points': pts,
                'mean_discrepancy_pct': float(np.mean([p['discrepancy_pct'] for p in pts])),
                'mean_efficiency_ratio': float(np.mean([p['efficiency_ratio'] for p in pts])),
                'max_N_result': max(pts, key=lambda x: x['N_shots']),
            }

    print(f"\n  {'Platform':<22} {'N':>6} {'MSE_exp':>12} "
          f"{'CRLB_hw':>12} {'Disc%':>8} {'Ratio':>8}")
    print("  " + "-" * 72)
    for pname, res in cv.items():
        pt = res['max_N_result']
        pl = PLATFORMS[pname]
        print(f"  {pl.name[:22]:<22} {pt['N_shots']:>6} "
              f"{pt['MSE_exp']:>12.3e} {pt['CRLB_hw']:>12.3e} "
              f"{pt['discrepancy_pct']:>8.1f} {pt['efficiency_ratio']:>8.3f}")
    print()
    print("  Mean discrepancy across platforms:")
    for pname, res in cv.items():
        print(f"    {PLATFORMS[pname].name[:30]:<30}: "
              f"{res['mean_discrepancy_pct']:.1f}%  "
              f"(mean ratio {res['mean_efficiency_ratio']:.3f})")

    # ── Summary statement ────────────────────────────────────────────────────
    print_header("Summary")
    all_ratios = [
        cv[p]['max_N_result']['efficiency_ratio']
        for p in cv if 'max_N_result' in cv[p]
    ]
    if all_ratios:
        print(f"  Efficiency ratios at max N: "
              f"{min(all_ratios):.3f} -- {max(all_ratios):.3f}")
        print(f"  All within 15% of CRLB: "
              f"{'YES' if max(all_ratios) < 1.15 else 'NO (> 1.15 for some)'}")
        mean_ratio = float(np.mean(all_ratios))
        print(f"  Mean efficiency ratio: {mean_ratio:.3f}")
    print()
    print("  Corrected CRLB (Theorem 1): Var(theta_hat) >= 1/(N * p^2 * gamma^2 * nbar^2)")
    print("  Sample complexity (Theorem 3): C1 = 1/8, C2 = 12")
    print()

    # ── Optional save ────────────────────────────────────────────────────────
    output = {
        'theta_true': theta_true,
        'N_trials': N_trials,
        'seed': seed,
        'simulation': sim_results,
        'cross_validation': {
            p: {k: v for k, v in res.items() if k != 'points'}
            for p, res in cv.items()
        },
    }
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to: {save_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description='BQPE experimental cross-validation report.')
    parser.add_argument('--datadir', default=None,
                        help='Path to data/experimental/ directory.')
    parser.add_argument('--jsonparams', default=None,
                        help='Path to platform_parameters.json.')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data if CSV files not found.')
    parser.add_argument('--trials', type=int, default=500,
                        help='Monte Carlo trials per configuration.')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--save', default=None,
                        help='Save JSON results to this path.')
    args = parser.parse_args()

    run_full_report(
        data_dir=args.datadir,
        json_path=args.jsonparams,
        use_synthetic=args.synthetic,
        N_trials=args.trials,
        seed=args.seed,
        save_path=args.save,
    )


if __name__ == '__main__':
    main()
