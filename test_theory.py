"""
bqpe/tests/test_theory.py
==========================
Unit tests verifying all theoretical claims of the BQPE manuscript.
Run: python -m pytest tests/ -v -m "not slow"

Authors: Akoramurthy B., Surendiran B., Xiaochun Cheng  |  License: MIT
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pytest
from adaptive_protocol import (
    C1_FANO, C2_MART, HardwareParams,
    vonmises_var, vonmises_std, vonmises_update,
    crlb, crlb_noisy, effective_pg,
    optimal_exponent, optimal_basis, fisher_information,
    sample_complexity_lower, sample_complexity_upper, minimum_shots,
    run_adaptive_trial, run_simplified_trial, run_monte_carlo, scan_shot_range,
    pg_overhead, ALL_PLATFORMS, PLATFORM_NV, PLATFORM_PHOTONIC, PLATFORM_TRAPPED_ION,
)

SEED = 2024; THETA = np.pi/3; RTOL = 1e-9


# ── Theorem 1: Corrected CRLB ────────────────────────────────────────────────
class TestCRLB:
    def test_eigenstate_N1(self):              assert crlb(1,1.0,1.0) == pytest.approx(1.0, rel=RTOL)
    def test_scaling_N(self):                  assert crlb(100,0.7,0.8) == pytest.approx(crlb(1,0.7,0.8)/100, rel=RTOL)
    def test_scaling_p(self):                  assert crlb(10,0.5,1.0) == pytest.approx(4*crlb(10,1.0,1.0), rel=RTOL)
    def test_scaling_gamma(self):              assert crlb(10,1.0,0.5) == pytest.approx(4*crlb(10,1.0,1.0), rel=RTOL)
    def test_nbar2_scaling(self):              assert crlb(10,0.7,0.9,1.0) == pytest.approx(4*crlb(10,0.7,0.9,4.0), rel=RTOL)
    def test_array_input(self):
        result = crlb(np.array([10,100,1000]), 0.7, 0.9)
        assert result.shape == (3,)
    def test_corrected_vs_original(self):
        p,g = 0.5, 1.0                        # Original: (1-pg)/(N*pg)=1; Corrected: 1/(N*0.25)=4
        assert crlb(1,p,g) == pytest.approx(4*(1-p*g)/(1*p*g), rel=RTOL)
    def test_noisy_larger_than_ideal(self):
        hw = HardwareParams(p=0.85, gamma=0.92, eta=0.5)
        assert crlb_noisy(100, hw) > crlb(100, 0.85, 0.92)


# ── Equation 3: MCP-DLD effective signal ────────────────────────────────────
class TestEffectivePG:
    def test_ideal(self):                      assert effective_pg(0.7,0.9) == pytest.approx(0.63, rel=RTOL)
    def test_half_eta(self):                   assert effective_pg(0.8,1.0,0.5) == pytest.approx(0.5*effective_pg(0.8,1.0,1.0), rel=RTOL)
    def test_dead_time(self):                  assert effective_pg(1.0,1.0,1.0,1e3,1e-8) == pytest.approx(1-1e3*1e-8, rel=RTOL)
    def test_hw_pg_eff(self):                  assert PLATFORM_NV.pg_eff == pytest.approx(0.85*0.92, rel=RTOL)
    def test_n_max_from_sigma(self):           assert HardwareParams(p=0.8,gamma=0.9,sigma_phi=0.05).n_max == 20
    def test_n_max_ideal(self):                assert HardwareParams(p=0.8,gamma=0.9).n_max == int(1e9)


# ── Theorem 3: Sample complexity C1=1/8, C2=12 ──────────────────────────────
class TestSampleComplexity:
    def test_c1(self):                         assert C1_FANO == pytest.approx(1/8, rel=RTOL)
    def test_c2(self):                         assert C2_MART == pytest.approx(12.0, rel=RTOL)
    def test_c2_factorisation(self):           assert C2_MART == pytest.approx(3*4, rel=RTOL)
    def test_lower_formula(self):
        lb = sample_complexity_lower(0.05,0.05,1,1)
        assert lb == pytest.approx((1/8)*np.log(20)/0.05**2, rel=RTOL)
    def test_upper_formula(self):
        ub = sample_complexity_upper(0.05,0.05,1,1)
        assert ub == pytest.approx(12*np.log(20)*np.log(np.log(20))/0.05**2, rel=RTOL)
    def test_lower_leq_upper(self):
        for p in [0.3,0.5,0.7,1.0]:
            for g in [0.5,0.8,1.0]:
                assert sample_complexity_lower(0.01,0.05,p,g) <= sample_complexity_upper(0.01,0.05,p,g)
    def test_pg_squared_scaling(self):
        lb1 = sample_complexity_lower(0.05,0.05,1.0,1.0)
        lb2 = sample_complexity_lower(0.05,0.05,0.5,1.0)
        assert lb2 == pytest.approx(4*lb1, rel=RTOL)
    def test_log_delta_scaling(self):
        ratio = np.log(1/0.01)/np.log(1/0.05)
        assert sample_complexity_lower(0.05,0.01,0.7,0.9) == pytest.approx(ratio*sample_complexity_lower(0.05,0.05,0.7,0.9), rel=RTOL)
    def test_minimum_shots_dict(self):
        r = minimum_shots(0.05, 0.05, 0.7, 0.9)
        assert r['lower'] < r['upper'] and r['overhead'] > 1.0


# ── Von Mises posterior update ───────────────────────────────────────────────
class TestVonMisesUpdate:
    def test_kappa_positive_from_zero(self):
        k,_ = vonmises_update(0.0,0.0,1,-np.pi/2,0,0.8)
        assert k > 0
    def test_natural_parameter_identity(self):
        kappa,mu,n,phi,b,pg = 2.5,1.2,3,0.4,1,0.75
        sign = 2*b-1
        Rc = kappa*np.cos(mu)+pg*n*np.cos(n*mu-phi)*sign
        Rs = kappa*np.sin(mu)+pg*n*np.sin(n*mu-phi)*sign
        kg,mg = vonmises_update(kappa,mu,n,phi,b,pg)
        assert kg == pytest.approx(np.hypot(Rc,Rs), rel=RTOL)
        assert mg == pytest.approx(np.arctan2(Rs,Rc), rel=RTOL)
    def test_var_monotone_decreasing(self):
        vs = [vonmises_var(k) for k in [0.1,1,5,20,100]]
        assert all(vs[i]>vs[i+1] for i in range(len(vs)-1))
    def test_var_in_01(self):
        assert all(0<=vonmises_var(k)<=1 for k in [0.01,1,5,50])
    def test_std_formula(self):
        assert vonmises_std(4.0) == pytest.approx(np.sqrt(vonmises_var(4.0)), rel=RTOL)
    def test_concentration_grows_consistent(self):
        kappa,mu = 0.0,THETA; pg=0.9
        for _ in range(30):
            n=max(1,int(kappa/(2*pg)))+1; phi=n*mu-np.pi/2
            kappa,mu = vonmises_update(kappa,mu,n,phi,0,pg)
        assert kappa > 10.0


# ── Optimal exponent and basis ───────────────────────────────────────────────
class TestOptimalProtocol:
    def test_exponent_at_least_one(self):      assert optimal_exponent(0.0,0.8) >= 1
    def test_exponent_grows_kappa(self):       assert optimal_exponent(10.0,0.8) >= optimal_exponent(1.0,0.8)
    def test_noise_ceiling(self):              assert optimal_exponent(1000.0,0.9,0.05) <= 20
    def test_basis_formula(self):              assert optimal_basis(5,1.2) == pytest.approx(5*1.2-np.pi/2, rel=RTOL)
    def test_fisher_at_optimal_equals_pg2_n2(self):
        pg,n,theta = 0.65,4,THETA
        F = fisher_information(n, n*theta-np.pi/2, theta, pg)
        assert F == pytest.approx((pg*n)**2, rel=1e-5)
    def test_fisher_non_optimal_less(self):
        pg,n,theta = 0.7,3,THETA
        F_opt = fisher_information(n, n*theta-np.pi/2, theta, pg)
        F_bad = fisher_information(n, n*theta, theta, pg)
        assert F_opt > F_bad


# ── Optimal operating point pγ=0.5 ──────────────────────────────────────────
class TestOptimalPoint:
    def test_bernoulli_max_at_half(self):
        pg = np.linspace(0.01,0.99,10000)
        assert pg[np.argmax(pg*(1-pg))] == pytest.approx(0.5, abs=0.01)
    def test_overhead_at_unity(self):          assert pg_overhead(1.0,1.0) == pytest.approx(1.0, rel=RTOL)
    def test_overhead_monotone(self):          assert pg_overhead(0.5,1.0) > pg_overhead(0.8,1.0)


# ── Hardware platforms ───────────────────────────────────────────────────────
class TestHardwarePlatforms:
    def test_all_platforms_valid(self):
        for name,hw in ALL_PLATFORMS.items():
            assert 0<hw.p<=1 and 0<hw.gamma<=1 and hw.pg_eff>0, name
    def test_validation_errors(self):
        with pytest.raises(ValueError): HardwareParams(p=1.5)
        with pytest.raises(ValueError): HardwareParams(gamma=-0.1)
        with pytest.raises(ValueError): HardwareParams(readout_err=0.6)
    def test_nv_params(self):
        assert PLATFORM_NV.p == 0.85 and PLATFORM_NV.gamma == 0.92
    def test_noise_trial_completes(self):
        rng = np.random.default_rng(SEED)
        hw = PLATFORM_NV
        r = run_adaptive_trial(20,hw.p,hw.gamma,THETA,rng,eta=hw.eta,
                               sigma_phi=hw.sigma_phi,readout_err=hw.readout_err,
                               return_full=True)
        assert len(r.exponents)==20 and -np.pi<=r.theta_hat<=np.pi


# ── TrialResult ──────────────────────────────────────────────────────────────
class TestTrialResult:
    def test_full_traj_lengths(self):
        rng = np.random.default_rng(SEED)
        r = run_adaptive_trial(25, 0.7, 0.9, THETA, rng, return_full=True)
        assert all(len(x)==25 for x in [r.exponents,r.kappa_trajectory,r.outcomes])
    def test_fast_mode_empty(self):
        rng = np.random.default_rng(SEED)
        r = run_adaptive_trial(25, 0.7, 0.9, THETA, rng, return_full=False)
        assert len(r.exponents)==0 and isinstance(r.theta_hat, float)
    def test_kappa_positive(self):
        rng = np.random.default_rng(SEED)
        r = run_adaptive_trial(10, 0.8, 0.9, THETA, rng)
        assert r.kappa_final > 0
    def test_posterior_var_01(self):
        rng = np.random.default_rng(SEED)
        r = run_adaptive_trial(10, 0.8, 0.9, THETA, rng)
        assert 0 <= r.posterior_var <= 1


# ── MCResult and scan ────────────────────────────────────────────────────────
class TestMCResult:
    def test_fields(self):
        r = run_monte_carlo(10,0.8,1.0,THETA,N_trials=30,seed=SEED,bootstrap=False)
        assert r.protocol=='adaptive' and r.N_shots==10 and len(r.estimates)==30
    def test_invalid_protocol(self):
        with pytest.raises(ValueError):
            run_monte_carlo(10,0.8,1.0,THETA,N_trials=5,protocol='bad')
    def test_bootstrap_order(self):
        r = run_monte_carlo(20,0.8,0.9,THETA,N_trials=50,seed=SEED,bootstrap=True)
        assert r.ci95[0] <= r.ci95[1]
    def test_scan_length(self):
        results = scan_shot_range(0.7,0.9,THETA,shot_range=[5,10,20],N_trials=20,seed=SEED)
        assert len(results)==3
    def test_mse_decreases_N(self):
        results = scan_shot_range(1.0,1.0,THETA,shot_range=[5,50,500],N_trials=200,seed=SEED)
        mses = [r.mse for r in results]
        assert mses[0]>mses[1]>mses[2]

    @pytest.mark.slow
    def test_efficiency_near_unity_N500(self):
        r = run_monte_carlo(500,1.0,1.0,THETA,N_trials=2000,seed=SEED)
        assert r.efficiency_ratio == pytest.approx(1.0, rel=0.20)

    @pytest.mark.slow
    def test_adaptive_beats_simplified(self):
        r_a = run_monte_carlo(100,0.7,0.9,THETA,N_trials=1000,seed=SEED,protocol='adaptive')
        r_s = run_monte_carlo(100,0.7,0.9,THETA,N_trials=1000,seed=SEED,protocol='simplified')
        assert r_a.efficiency_ratio < r_s.efficiency_ratio
