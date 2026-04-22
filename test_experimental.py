"""
bqpe/tests/test_experimental.py
=================================
Tests for the hardware noise model and experimental data pipeline.

Covers:
  - PlatformParams dataclass correctness
  - Effective pg formula (equation 3)
  - Hardware exponent ceiling (equation 4)
  - Noise model: readout error and phase noise
  - Data loader: CSV parsing and schema validation
  - Cross-validation: discrepancy within expected bounds
  - Synthetic data generation
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hardware_noise import (
    PlatformParams, PLATFORMS, load_platform,
    noisy_outcome_prob, sample_noisy_outcome,
    hardware_exponent, run_hardware_trial, run_hardware_monte_carlo,
)
from data_loader import (
    generate_synthetic_data, cross_validate_all,
)

THETA_TRUE = math.pi / 3
SEED       = 2024
RNG        = np.random.default_rng(SEED)


# ── PlatformParams ────────────────────────────────────────────────────────────

class TestPlatformParams:

    def test_pg_property(self):
        pl = PlatformParams(name='test', p=0.8, gamma=0.9, eta_ro=0.0, sigma_phi=0.0)
        assert pl.pg == pytest.approx(0.72, rel=1e-9)

    def test_pg_eff_ideal_detector(self):
        """With eta=1, eta_ro=0: pg_eff = p * gamma."""
        pl = PlatformParams(name='test', p=0.7, gamma=0.8,
                            eta_ro=0.0, sigma_phi=0.0, eta_detection=1.0)
        assert pl.pg_eff == pytest.approx(0.56, rel=1e-9)

    def test_pg_eff_readout_error(self):
        """Readout error reduces effective visibility by (1 - 2*eta_ro)."""
        pl = PlatformParams(name='test', p=1.0, gamma=1.0,
                            eta_ro=0.1, sigma_phi=0.0, eta_detection=1.0)
        expected = 1.0 * 1.0 * 1.0 * (1.0 - 2.0 * 0.1)
        assert pl.pg_eff == pytest.approx(expected, rel=1e-9)

    def test_pg_eff_half_detection(self):
        """Half detection efficiency halves pg_eff."""
        pl_full = PlatformParams(name='full', p=0.8, gamma=0.9,
                                 eta_ro=0.0, sigma_phi=0.0, eta_detection=1.0)
        pl_half = PlatformParams(name='half', p=0.8, gamma=0.9,
                                 eta_ro=0.0, sigma_phi=0.0, eta_detection=0.5)
        assert pl_half.pg_eff == pytest.approx(0.5 * pl_full.pg_eff, rel=1e-9)

    def test_n_max_no_noise(self):
        """With no phase noise, n_max is very large."""
        pl = PlatformParams(name='ideal', p=1.0, gamma=1.0,
                            eta_ro=0.0, sigma_phi=0.0)
        assert pl.n_max >= 1000

    def test_n_max_with_noise(self):
        """n_max = floor(1 / sigma_phi) for sigma_phi > 0."""
        pl = PlatformParams(name='noisy', p=1.0, gamma=1.0,
                            eta_ro=0.0, sigma_phi=0.05)
        assert pl.n_max == 20   # floor(1/0.05)

    def test_crlb_hw(self):
        """Hardware CRLB uses pg_eff^2."""
        pl = PlatformParams(name='test', p=0.8, gamma=0.9,
                            eta_ro=0.0, sigma_phi=0.0, eta_detection=1.0)
        expected = 1.0 / (100 * pl.pg_eff ** 2)
        assert pl.crlb_hw(100) == pytest.approx(expected, rel=1e-9)

    def test_built_in_platforms_loaded(self):
        """All expected built-in platforms are present."""
        for name in ['nv_centre', 'photonic', 'trapped_ion', 'ideal']:
            pl = load_platform(name)
            assert pl.p > 0 and pl.gamma > 0

    def test_nv_centre_parameters(self):
        """NV-centre parameters match published values (Santagati 2022)."""
        pl = PLATFORMS['nv_centre']
        assert pl.p == pytest.approx(0.85, abs=0.01)
        assert pl.gamma == pytest.approx(0.92, abs=0.01)
        assert pl.eta_ro == pytest.approx(0.018, abs=0.002)
        assert pl.sigma_phi == pytest.approx(0.008, abs=0.001)

    def test_photonic_parameters(self):
        """Photonic parameters match published values (Valeri 2020)."""
        pl = PLATFORMS['photonic']
        assert pl.p == pytest.approx(0.78, abs=0.01)
        assert pl.gamma == pytest.approx(0.96, abs=0.01)

    def test_trapped_ion_parameters(self):
        """Trapped-ion parameters match published values (Pogorelov 2021)."""
        pl = PLATFORMS['trapped_ion']
        assert pl.p == pytest.approx(0.91, abs=0.01)
        assert pl.gamma == pytest.approx(0.88, abs=0.01)


# ── Noise model ───────────────────────────────────────────────────────────────

class TestNoisyMeasurement:

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.ideal = PLATFORMS['ideal']
        self.nv    = PLATFORMS['nv_centre']

    def test_ideal_probability_range(self):
        """Ideal probability is in [0, 1]."""
        for phi in np.linspace(0, 2*np.pi, 20):
            p0 = noisy_outcome_prob(THETA_TRUE, 1, phi, self.ideal, self.rng)
            assert 0.0 <= p0 <= 1.0

    def test_readout_floor(self):
        """With eta_ro > 0, probability is bounded away from 0 and 1."""
        pl = PlatformParams(name='ro', p=1.0, gamma=1.0,
                            eta_ro=0.1, sigma_phi=0.0)
        for phi in np.linspace(0, 2*np.pi, 20):
            p0 = noisy_outcome_prob(0.0, 1, phi, pl, self.rng)
            assert 0.1 <= p0 <= 0.9

    def test_outcome_is_binary(self):
        """Sample outcome is always 0 or 1."""
        for _ in range(50):
            b = sample_noisy_outcome(THETA_TRUE, 2, 0.5, self.nv, self.rng)
            assert b in (0, 1)

    def test_noisy_prob_at_optimal_basis(self):
        """
        At optimal basis phi* = n*theta - pi/2 with no readout error,
        prob(b=0) should be near 0.5 for ideal case.
        """
        pl = PlatformParams(name='pure', p=1.0, gamma=1.0,
                            eta_ro=0.0, sigma_phi=0.0, eta_detection=1.0)
        phi_opt = 1 * THETA_TRUE - np.pi / 2
        p0 = noisy_outcome_prob(THETA_TRUE, 1, phi_opt, pl, self.rng)
        assert abs(p0 - 0.5) < 0.01


# ── Hardware exponent ─────────────────────────────────────────────────────────

class TestHardwareExponent:

    def test_minimum_is_one(self):
        assert hardware_exponent(0.0, PLATFORMS['ideal']) >= 1

    def test_ceiling_enforced(self):
        """With sigma_phi = 0.05, n_max = 20."""
        pl = PlatformParams(name='noisy', p=0.9, gamma=0.9,
                            eta_ro=0.0, sigma_phi=0.05)
        n = hardware_exponent(1000.0, pl)
        assert n <= 20

    def test_grows_with_kappa(self):
        pl = PLATFORMS['nv_centre']
        n1 = hardware_exponent(2.0, pl)
        n5 = hardware_exponent(20.0, pl)
        assert n5 >= n1


# ── Full trial and Monte Carlo ────────────────────────────────────────────────

class TestHardwareTrial:

    def test_trial_returns_valid_estimate(self):
        rng = np.random.default_rng(SEED)
        result = run_hardware_trial(50, THETA_TRUE, PLATFORMS['nv_centre'], rng)
        assert -np.pi <= result['theta_hat'] <= np.pi
        assert result['kappa_final'] > 0

    def test_exponents_list_length(self):
        rng = np.random.default_rng(SEED)
        N = 30
        result = run_hardware_trial(N, THETA_TRUE, PLATFORMS['photonic'], rng)
        assert len(result['exponents']) == N
        assert len(result['kappa_traj']) == N

    @pytest.mark.slow
    def test_mc_efficiency_within_bounds(self):
        """
        Efficiency ratio should be in [0.8, 3.5] for NV-centre at N=100.
        Wide tolerance because noise increases overhead over ideal.
        """
        result = run_hardware_monte_carlo(
            100, THETA_TRUE, PLATFORMS['nv_centre'],
            N_trials=500, seed=SEED
        )
        assert 0.8 <= result['efficiency_ratio'] <= 3.5, (
            f"Efficiency ratio {result['efficiency_ratio']:.3f} out of bounds"
        )

    @pytest.mark.slow
    def test_mc_bias_decays(self):
        """Bias decreases from N=20 to N=200 on photonic platform."""
        pl = PLATFORMS['photonic']
        r20  = run_hardware_monte_carlo(20,  THETA_TRUE, pl, 300, SEED)
        r200 = run_hardware_monte_carlo(200, THETA_TRUE, pl, 300, SEED)
        assert r200['bias'] < r20['bias']


# ── Data loader ───────────────────────────────────────────────────────────────

class TestDataLoader:

    @pytest.mark.slow
    def test_synthetic_data_schema(self):
        """Generated synthetic data has correct schema."""
        rows = generate_synthetic_data('nv_centre', N_trials=50, seed=SEED)
        assert len(rows) > 0
        for row in rows:
            assert 'N_shots'         in row
            assert 'MSE_exp'         in row
            assert 'CRLB_corrected'  in row
            assert 'efficiency_ratio' in row
            assert row['N_shots'] > 0
            assert row['MSE_exp'] > 0

    @pytest.mark.slow
    def test_synthetic_efficiency_ratio_positive(self):
        """Efficiency ratio from synthetic data is positive."""
        rows = generate_synthetic_data('photonic', N_trials=50, seed=SEED)
        for row in rows:
            assert row['efficiency_ratio'] > 0

    @pytest.mark.slow
    def test_cross_validate_synthetic(self):
        """Cross-validation runs on synthetic data without errors."""
        # Generate synthetic CSV files to data_dir for this test
        import tempfile, csv as csvlib, json
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write minimal JSON params file
            params = {
                'nv_centre': {
                    'parameters': {
                        'p': 0.85, 'gamma': 0.92,
                        'eta_ro': 0.018, 'sigma_phi_rad': 0.008,
                        'eta_detection': 0.50,
                    },
                    'reference': 'test'
                }
            }
            jpath = os.path.join(tmpdir, 'platform_parameters.json')
            with open(jpath, 'w') as f:
                json.dump(params, f)

            # Write synthetic CSV
            rows = generate_synthetic_data('nv_centre', N_trials=50)
            cpath = os.path.join(tmpdir, 'santagati2022_nv_centre.csv')
            with open(cpath, 'w', newline='') as f:
                writer = csvlib.DictWriter(
                    f, fieldnames=['platform','N_shots','MSE_exp',
                                   'MSE_err','CRLB_corrected','efficiency_ratio']
                )
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

            results = cross_validate_all(data_dir=tmpdir, json_path=jpath,
                                         verbose=False)
            assert 'nv_centre' in results
            assert results['nv_centre']['mean_efficiency_ratio'] > 0
