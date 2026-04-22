"""
bqpe/src/adaptive_protocol.py
==============================
Complete implementation of Adaptive Bayesian Quantum Phase Estimation
(BQPE) and all supporting mathematical infrastructure.

Reference
---------
Akoramurthy B., Surendiran B., Cheng X. (2026)
"Bayesian Quantum Phase Estimation Beyond the Eigenstate Approximation"
Frontiers in Quantum Science and Technology.

Algorithm 1 (TRUE adaptive protocol):
    n_t* = floor(kappa_{t-1} / (2*p*gamma)) + 1
    phi_t* = n_t * mu_{t-1} - pi/2

Corrected CRLB (Theorem 1):
    Var(theta_hat) >= 1 / (N * p^2 * gamma^2 * nbar^2)

Sample complexity (Theorem 3):
    C1 = 1/8  (Fano's inequality)
    C2 = 12   (von Mises martingale, 3x4)

Authors:  Akoramurthy B., Surendiran B., Xiaochun Cheng
License:  MIT  |  Version: 2.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.special import iv as _bessel

FloatArray = NDArray[np.float64]

# ── Constants ──────────────────────────────────────────────────────────────
C1_FANO: float = 1.0 / 8.0   # Fano lower bound constant
C2_MART: float = 12.0         # Martingale upper bound constant  (3x4)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class HardwareParams:
    """
    Hardware-specific noise parameters for MCP-DLD detector model.

    Parameters
    ----------
    p : float           Eigenstate overlap |<psi|phi>|^2 in (0,1].
    gamma : float       State purity Tr(rho^2) in (0,1].
    eta : float         MCP detection efficiency (default 1.0).
                        Standard: ~0.5; CsI-coated for He*: ~0.9.
    R : float           Mean atom arrival rate per shot (Hz).
    tau_d : float       Multi-hit dead-time (seconds).
    sigma_phi : float   Per-shot phase noise std-dev (radians).
    readout_err : float Bit-flip readout error probability.

    Notes
    -----
    Effective signal: (p*gamma)_eff = p*gamma*eta*(1 - R*tau_d)
    Exponent ceiling: n_max = floor(1/sigma_phi)
    """
    p:           float = 1.0
    gamma:       float = 1.0
    eta:         float = 1.0
    R:           float = 0.0
    tau_d:       float = 0.0
    sigma_phi:   float = 0.0
    readout_err: float = 0.0

    def __post_init__(self):
        for name, val, lo, hi in [
            ('p',           self.p,           1e-6, 1.0),
            ('gamma',       self.gamma,       1e-6, 1.0),
            ('eta',         self.eta,         0.0,  1.0),
            ('readout_err', self.readout_err, 0.0,  0.5),
        ]:
            if not (lo <= val <= hi):
                raise ValueError(f"{name}={val} must be in [{lo},{hi}]")

    @property
    def pg_ideal(self) -> float:
        return self.p * self.gamma

    @property
    def pg_eff(self) -> float:
        return self.p * self.gamma * self.eta * (1.0 - self.R * self.tau_d)

    @property
    def n_max(self) -> int:
        return int(1e9) if self.sigma_phi <= 0 else max(1, int(1.0 / self.sigma_phi))


@dataclass
class TrialResult:
    """Output of a single adaptive BQPE trial."""
    theta_hat:        float
    kappa_final:      float
    mu_trajectory:    list
    kappa_trajectory: list
    exponents:        list
    bases:            list
    outcomes:         list
    posterior_var:    float
    N_shots:          int


@dataclass
class MCResult:
    """Output of a Monte Carlo BQPE experiment."""
    mse:              float
    bias:             float
    variance:         float
    crlb_val:         float
    efficiency_ratio: float
    estimates:        FloatArray
    ci95:             tuple
    N_shots:          int
    p:                float
    gamma:            float
    N_trials:         int
    protocol:         str

    def __str__(self) -> str:
        return (f"MCResult N={self.N_shots} p={self.p} γ={self.gamma} | "
                f"MSE={self.mse:.4e} CRLB={self.crlb_val:.4e} "
                f"Ratio={self.efficiency_ratio:.3f}")


# ===========================================================================
# Von Mises posterior utilities
# ===========================================================================

def vonmises_var(kappa: float | FloatArray) -> float | FloatArray:
    """
    Circular variance of von Mises(mu, kappa).
    Var = 1 - I_1(kappa)/I_0(kappa)
    """
    k = np.maximum(np.asarray(kappa, float), 1e-12)
    return 1.0 - _bessel(1, k) / _bessel(0, k)


def vonmises_std(kappa: float | FloatArray) -> float | FloatArray:
    """Circular standard deviation of von Mises distribution."""
    return np.sqrt(vonmises_var(kappa))


def vonmises_update(
    kappa: float,
    mu: float,
    n: int,
    phi: float,
    b: int,
    pg: float,
) -> tuple[float, float]:
    """
    Von Mises natural-parameter posterior update (Berry & Wiseman 2000).

    R_cos = kappa*cos(mu) + pg*n*cos(n*mu - phi)*(2b-1)
    R_sin = kappa*sin(mu) + pg*n*sin(n*mu - phi)*(2b-1)
    kappa_new = sqrt(R_cos^2 + R_sin^2)
    mu_new    = atan2(R_sin, R_cos)

    Error: O(kappa^{-2}), negligible for kappa >= 5.
    """
    sign = 2 * int(b) - 1
    Rc = kappa * np.cos(mu) + pg * n * np.cos(n * mu - phi) * sign
    Rs = kappa * np.sin(mu) + pg * n * np.sin(n * mu - phi) * sign
    return float(np.hypot(Rc, Rs)), float(np.arctan2(Rs, Rc))


# ===========================================================================
# CRLB and sample complexity (Theorems 1, 3)
# ===========================================================================

def crlb(
    N: int | FloatArray,
    p: float,
    gamma: float,
    nbar2: float = 1.0,
) -> float | FloatArray:
    """
    Corrected Cramér–Rao lower bound (Theorem 1).
    CRLB = 1 / (N * p^2 * gamma^2 * nbar^2)

    At optimal basis phi* = n*theta* - pi/2, cos = 0 exactly,
    giving F_C = (pg*n)^2 with no approximation.

    Original erroneous formula (1-pg)/(N*pg*nbar^2) is WRONG;
    error is 4x at pg=0.5, diverges at pg->1.
    """
    return 1.0 / (np.asarray(N, float) * (p * gamma) ** 2 * nbar2)


def crlb_noisy(N: int, hw: HardwareParams, nbar_eff2: float = 1.0) -> float:
    """
    Noise-corrected CRLB incorporating MCP-DLD hardware model.
    CRLB_hw = 1 / (N * pg_eff^2 * nbar_eff^2)
    """
    return 1.0 / (N * hw.pg_eff ** 2 * nbar_eff2)


def effective_pg(
    p: float, gamma: float,
    eta: float = 1.0, R: float = 0.0, tau_d: float = 0.0,
) -> float:
    """
    Effective signal amplitude with MCP-DLD hardware corrections.
    (pg)_eff = p * gamma * eta * (1 - R * tau_d)
    """
    return p * gamma * eta * (1.0 - R * tau_d)


def sample_complexity_lower(
    eps: float | FloatArray, delta: float, p: float, gamma: float
) -> float | FloatArray:
    """
    Lower bound N* >= (1/8)*log(1/delta) / (eps^2*(pg)^2).
    C1=1/8 via Fano's inequality + two-point minimax bound.
    """
    eps = np.asarray(eps, float)
    return C1_FANO * np.log(1.0 / delta) / (eps ** 2 * (p * gamma) ** 2)


def sample_complexity_upper(
    eps: float | FloatArray, delta: float, p: float, gamma: float
) -> float | FloatArray:
    """
    Upper bound N* <= 12*log(1/delta)*loglog(1/eps) / (eps^2*(pg)^2).
    C2=12 = 3x4 from martingale stopping time analysis.
    eps must satisfy eps <= 1/e.
    """
    eps = np.clip(np.asarray(eps, float), 1e-10, 1.0 / np.e - 1e-12)
    loglog = np.log(np.log(1.0 / eps) + 1e-15)
    return C2_MART * np.log(1.0 / delta) * loglog / (eps ** 2 * (p * gamma) ** 2)


def minimum_shots(eps: float, delta: float, p: float, gamma: float) -> dict:
    """Return lower bound, upper bound and overhead factor for given parameters."""
    return {
        'lower':   float(sample_complexity_lower(eps, delta, p, gamma)),
        'upper':   float(sample_complexity_upper(eps, delta, p, gamma)),
        'overhead': float(1.0 / (p * gamma) ** 2),
        'pg':      float(p * gamma),
    }


# ===========================================================================
# Optimal adaptive exponent and basis
# ===========================================================================

def optimal_exponent(kappa: float, pg: float, sigma_phi: float = 0.0) -> int:
    """
    Optimal measurement exponent (Algorithm 1, line 6).

    n_t* = floor(kappa / (2*pg)) + 1

    At kappa=0 (uniform prior): n_t* = 1  (single exponent, maximum prior uncertainty).
    As kappa grows: n_t* grows proportionally, concentrating measurement
    resolution to match posterior sharpness.

    With hardware noise ceiling (equation 4):
        n_t* = min(n_t*, floor(1/sigma_phi))

    NOTE: The formula does NOT include a max(1, ...) wrapper before +1.
    floor(0/(2*pg)) + 1 = 0 + 1 = 1  correctly gives n=1 at kappa=0.
    Adding max(1,...) would incorrectly give n=2 at kappa=0, causing
    phase ambiguity and oscillation at the start of every trial.
    """
    n_star = int(kappa / (2.0 * max(pg, 1e-10))) + 1
    if sigma_phi > 0:
        n_star = min(n_star, max(1, int(1.0 / sigma_phi)))
    return max(1, n_star)  # safety floor only


def optimal_basis(n: int, mu: float) -> float:
    """
    Optimal measurement basis (Algorithm 1, line 7).
    phi* = n*mu - pi/2
    At this angle, F_C = (pg*n)^2 exactly (no approximation).
    """
    return n * mu - np.pi / 2.0


def fisher_information(n: int, phi: float, theta: float, pg: float) -> float:
    """
    Classical Fisher information per shot.
    F_C = (pg*n)^2*sin^2(nθ-φ) / [1 - (pg)^2*cos^2(nθ-φ)]
    At optimal basis: F_C = (pg*n)^2 exactly.
    """
    ct = pg * np.cos(n * theta - phi)
    denom = 1.0 - ct ** 2
    return float((pg * n) ** 2 * np.sin(n * theta - phi) ** 2 / max(denom, 1e-14))


# ===========================================================================
# Single-trial adaptive protocol
# ===========================================================================

def run_adaptive_trial(
    N_shots: int,
    p: float,
    gamma: float,
    theta_true: float,
    rng: np.random.Generator,
    eta: float = 1.0,
    sigma_phi: float = 0.0,
    readout_err: float = 0.0,
    R: float = 0.0,
    tau_d: float = 0.0,
    return_full: bool = False,
) -> TrialResult:
    """
    Run one complete adaptive BQPE trial (Algorithm 1 of manuscript).

    TRUE adaptive protocol (not the simplified n_t=t schedule):
        n_t = optimal_exponent(kappa, pg)
        phi_t = n_t * mu - pi/2
        b_t ~ Bernoulli(...)
        (kappa, mu) updated via vonmises_update(...)

    Hardware noise model:
        Phase noise:  theta_obs ~ N(theta_true, sigma_phi^2) per shot
        Readout:      bit-flip with prob readout_err
        Detection:    enters via pg_eff = p*gamma*eta*(1-R*tau_d)

    Parameters
    ----------
    N_shots : int       Total shots.
    p, gamma : float    Overlap and purity.
    theta_true : float  True phase (radians).
    rng : Generator     Seeded RNG.
    eta : float         MCP efficiency (default 1.0).
    sigma_phi : float   Phase noise std-dev (default 0.0).
    readout_err : float Bit-flip prob (default 0.0).
    R, tau_d : float    MCP arrival rate and dead-time.
    return_full : bool  If True, store full trajectory (slower).

    Returns
    -------
    TrialResult
    """
    hw = HardwareParams(p=p, gamma=gamma, eta=eta, R=R, tau_d=tau_d,
                        sigma_phi=sigma_phi, readout_err=readout_err)
    pg = hw.pg_eff
    kappa, mu = 0.0, 0.0
    mt, kt, ex, ba, oc = [], [], [], [], []

    for _ in range(N_shots):
        n_t   = optimal_exponent(kappa, pg, sigma_phi)
        phi_t = optimal_basis(n_t, mu)

        theta_obs = (theta_true + rng.normal(0.0, sigma_phi)
                     if sigma_phi > 0 else theta_true)
        prob0 = float(np.clip(
            0.5 * (1.0 + pg * np.cos(n_t * theta_obs - phi_t))
            * (1.0 - readout_err) +
            0.5 * (1.0 - pg * np.cos(n_t * theta_obs - phi_t))
            * readout_err, 0.0, 1.0))
        b_t = int(rng.random() > prob0)
        kappa, mu = vonmises_update(kappa, mu, n_t, phi_t, b_t, pg)

        if return_full:
            mt.append(mu); kt.append(kappa)
            ex.append(n_t); ba.append(phi_t); oc.append(b_t)

    return TrialResult(
        theta_hat=float(mu), kappa_final=float(kappa),
        mu_trajectory=mt, kappa_trajectory=kt,
        exponents=ex, bases=ba, outcomes=oc,
        posterior_var=float(vonmises_var(kappa)),
        N_shots=N_shots,
    )


def run_simplified_trial(
    N_shots: int, p: float, gamma: float,
    theta_true: float, rng: np.random.Generator,
) -> float:
    """
    ERRONEOUS prior-draft protocol n_t = t (NOT CRLB-achieving).
    Provided for Figure 7B' comparison ONLY. Do not use for validation.
    """
    pg = p * gamma
    kappa, mu = 0.0, 0.0
    for t in range(1, N_shots + 1):
        phi_t = t * mu - np.pi / 2.0
        prob0 = float(np.clip(0.5 * (1.0 + pg * np.cos(t * theta_true - phi_t)), 0.0, 1.0))
        b_t = int(rng.random() > prob0)
        kappa, mu = vonmises_update(kappa, mu, t, phi_t, b_t, pg)
    return float(mu)


# ===========================================================================
# Monte Carlo engine
# ===========================================================================

def _bootstrap_ci(data: FloatArray, n_boot: int = 1000,
                  rng: Optional[np.random.Generator] = None,
                  alpha: float = 0.05) -> tuple:
    rng = rng or np.random.default_rng(0)
    n = len(data)
    boot = np.array([np.mean(rng.choice(data, n, replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def run_monte_carlo(
    N_shots: int, p: float, gamma: float, theta_true: float,
    N_trials: int = 2000, seed: int = 2024,
    protocol: str = 'adaptive', bootstrap: bool = True,
    **hw_kwargs,
) -> MCResult:
    """
    Monte Carlo estimation of BQPE performance.

    Parameters
    ----------
    N_shots : int       Shots per trial.
    p, gamma : float    Overlap and purity.
    theta_true : float  True phase.
    N_trials : int      Independent trials (default 2000).
    seed : int          RNG seed (default 2024 — manuscript seed).
    protocol : str      'adaptive' (Algorithm 1) or 'simplified' (n_t=t).
    bootstrap : bool    Compute 95% CI.
    **hw_kwargs         Hardware noise params (eta, sigma_phi, etc.).

    Returns
    -------
    MCResult
    """
    if protocol not in ('adaptive', 'simplified'):
        raise ValueError(f"protocol must be 'adaptive' or 'simplified'")
    rng = np.random.default_rng(seed)
    ests = np.empty(N_trials)
    for i in range(N_trials):
        if protocol == 'adaptive':
            ests[i] = run_adaptive_trial(N_shots, p, gamma, theta_true, rng,
                                         return_full=False, **hw_kwargs).theta_hat
        else:
            ests[i] = run_simplified_trial(N_shots, p, gamma, theta_true, rng)

    mse = float(np.mean((ests - theta_true) ** 2))
    cv  = crlb(N_shots, p, gamma)
    eff = mse / cv
    ci  = _bootstrap_ci((ests - theta_true) ** 2 / cv, rng=rng) if bootstrap else (0., 0.)
    return MCResult(mse=mse, bias=float(abs(np.mean(ests) - theta_true)),
                    variance=float(np.var(ests, ddof=1)), crlb_val=float(cv),
                    efficiency_ratio=eff, estimates=ests, ci95=ci,
                    N_shots=N_shots, p=p, gamma=gamma, N_trials=N_trials, protocol=protocol)


def scan_shot_range(p: float, gamma: float, theta_true: float,
                    shot_range: list | None = None, N_trials: int = 2000,
                    seed: int = 2024, protocol: str = 'adaptive',
                    **hw_kwargs) -> list[MCResult]:
    """Sweep over shot counts; returns list[MCResult]."""
    if shot_range is None:
        shot_range = [2, 5, 10, 20, 50, 100, 200, 500]
    return [run_monte_carlo(ns, p, gamma, theta_true, N_trials=N_trials,
                            seed=seed + i, protocol=protocol, **hw_kwargs)
            for i, ns in enumerate(shot_range)]


# ===========================================================================
# Pre-configured hardware platforms (Section 3.8)
# ===========================================================================

PLATFORM_NV           = HardwareParams(p=0.85, gamma=0.92, eta=1.0, sigma_phi=0.008, readout_err=0.018)
PLATFORM_PHOTONIC     = HardwareParams(p=0.78, gamma=0.96, eta=1.0, sigma_phi=0.005, readout_err=0.022)
PLATFORM_TRAPPED_ION  = HardwareParams(p=0.91, gamma=0.88, eta=1.0, sigma_phi=0.003, readout_err=0.009)
PLATFORM_MCP_STANDARD = HardwareParams(p=0.85, gamma=0.92, eta=0.50, sigma_phi=0.008, readout_err=0.018, R=1e3, tau_d=1e-8)
PLATFORM_MCP_CSI      = HardwareParams(p=0.85, gamma=0.92, eta=0.90, sigma_phi=0.008, readout_err=0.018, R=1e3, tau_d=1e-8)

ALL_PLATFORMS = {
    'nv_centre':    PLATFORM_NV,
    'photonic':     PLATFORM_PHOTONIC,
    'trapped_ion':  PLATFORM_TRAPPED_ION,
    'mcp_standard': PLATFORM_MCP_STANDARD,
    'mcp_csi':      PLATFORM_MCP_CSI,
}
