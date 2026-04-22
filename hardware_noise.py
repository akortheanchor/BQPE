"""
bqpe/src/hardware_noise.py
===========================
Hardware-faithful noise model for BQPE experimental validation.

Implements the noise-corrected likelihood model from Section 3.1
and equations (3)-(4) of the manuscript:

    p_noisy(b=0 | theta, n, phi) =
        eta_ro + (1 - 2*eta_ro) * 0.5 * [1 + pg * cos(n*theta_noisy - phi)]

    theta_noisy ~ N(theta*, sigma_phi^2)

    (pg)_eff = p * gamma * eta * (1 - R * tau_d)

    n_t^eff = min(n_t*, floor(1/sigma_phi))  [noise ceiling]

All three hardware platforms (NV-centre, photonic, trapped-ion)
are parametrised here from published experimental data.

References
----------
Santagati et al., npj Quantum Inf 8:64 (2022)  [NV-centre]
Valeri et al.,    npj Quantum Inf 6:92 (2020)  [Photonic]
Pogorelov et al., PRX Quantum 2:020343 (2021)  [Trapped-ion]
Berry & Wiseman,  Phys Rev Lett 85:5098 (2000) [Von Mises update]
"""

from __future__ import annotations

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Platform parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlatformParams:
    """
    Complete hardware parameter set for one experimental platform.

    Attributes
    ----------
    name : str
        Human-readable platform name.
    p : float
        Eigenstate overlap |<psi|phi>|^2 in (0, 1].
    gamma : float
        State purity Tr(rho^2) in (0, 1].
    eta_ro : float
        Single-shot readout bit-flip error probability.
    sigma_phi : float
        Per-shot phase noise standard deviation (radians).
    eta_detection : float
        Detector quantum efficiency (e.g. MCP or APD).
    R : float
        Mean atom/photon arrival rate per shot (for dead-time correction).
    tau_d : float
        Multi-hit dead-time (seconds).
    reference : str
        Citation for the source paper.
    doi : str
        DOI of the source paper.
    """
    name: str
    p: float
    gamma: float
    eta_ro: float
    sigma_phi: float
    eta_detection: float = 1.0
    R: float = 0.0
    tau_d: float = 0.0
    reference: str = ""
    doi: str = ""

    @property
    def pg(self) -> float:
        """Raw signal amplitude p * gamma."""
        return self.p * self.gamma

    @property
    def pg_eff(self) -> float:
        """
        Effective signal amplitude after all hardware corrections.

        (pg)_eff = p * gamma * eta * (1 - 2*eta_ro) * (1 - R*tau_d)

        The (1 - 2*eta_ro) factor accounts for symmetric bit-flip
        readout error reducing the visibility of the interference fringe.
        """
        return (
            self.p
            * self.gamma
            * self.eta_detection
            * (1.0 - 2.0 * self.eta_ro)
            * (1.0 - self.R * self.tau_d)
        )

    @property
    def n_max(self) -> int:
        """
        Maximum useful measurement exponent before phase noise dominates.

        n_max = floor(1 / sigma_phi)   [from equation (4) of manuscript]

        For sigma_phi = 0, returns a large number (no ceiling).
        """
        if self.sigma_phi <= 0:
            return 10_000
        return max(1, int(1.0 / self.sigma_phi))

    def crlb_hw(self, N: int, nbar2: float = 1.0) -> float:
        """
        Hardware-corrected CRLB using effective signal amplitude.

        CRLB_hw = 1 / (N * (pg_eff)^2 * nbar^2)
        """
        return 1.0 / (N * self.pg_eff ** 2 * nbar2)

    def __str__(self) -> str:
        return (
            f"{self.name}: p={self.p}, gamma={self.gamma}, "
            f"eta_ro={self.eta_ro}, sigma_phi={self.sigma_phi} rad, "
            f"(pg)_eff={self.pg_eff:.4f}"
        )


# ---------------------------------------------------------------------------
# Built-in platform definitions (from published experimental papers)
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, PlatformParams] = {

    "nv_centre": PlatformParams(
        name="NV-centre (Santagati et al. 2022)",
        p=0.85,
        gamma=0.92,
        eta_ro=0.018,
        sigma_phi=0.008,
        eta_detection=0.50,
        R=0.0,
        tau_d=0.0,
        reference="Santagati et al., npj Quantum Inf 8:64 (2022)",
        doi="10.1038/s41534-022-00547-x",
    ),

    "photonic": PlatformParams(
        name="Photonic (Valeri et al. 2020)",
        p=0.78,
        gamma=0.96,
        eta_ro=0.022,
        sigma_phi=0.005,
        eta_detection=0.85,
        R=0.0,
        tau_d=0.0,
        reference="Valeri et al., npj Quantum Inf 6:92 (2020)",
        doi="10.1038/s41534-020-00326-6",
    ),

    "trapped_ion": PlatformParams(
        name="Trapped-ion (Pogorelov et al. 2021)",
        p=0.91,
        gamma=0.88,
        eta_ro=0.009,
        sigma_phi=0.003,
        eta_detection=0.95,
        R=0.0,
        tau_d=0.0,
        reference="Pogorelov et al., PRX Quantum 2:020343 (2021)",
        doi="10.1103/PRXQuantum.2.020343",
    ),

    "ideal": PlatformParams(
        name="Ideal (no noise)",
        p=1.0,
        gamma=1.0,
        eta_ro=0.0,
        sigma_phi=0.0,
        eta_detection=1.0,
        reference="Theoretical ideal",
    ),
}


def load_platform(name: str) -> PlatformParams:
    """
    Load a built-in platform by name.

    Parameters
    ----------
    name : str
        One of 'nv_centre', 'photonic', 'trapped_ion', 'ideal'.

    Returns
    -------
    PlatformParams

    Raises
    ------
    KeyError if name not found.
    """
    if name not in PLATFORMS:
        raise KeyError(
            f"Unknown platform '{name}'. "
            f"Available: {list(PLATFORMS.keys())}"
        )
    return PLATFORMS[name]


def load_platform_from_json(json_path: str, platform_key: str) -> PlatformParams:
    """
    Load platform parameters from the JSON file in data/experimental/.

    Parameters
    ----------
    json_path : str
        Path to platform_parameters.json.
    platform_key : str
        Key in the JSON file ('nv_centre', 'photonic', 'trapped_ion').
    """
    with open(json_path) as f:
        data = json.load(f)
    entry = data[platform_key]
    params = entry["parameters"]
    return PlatformParams(
        name=platform_key,
        p=params["p"],
        gamma=params["gamma"],
        eta_ro=params["eta_ro"],
        sigma_phi=params["sigma_phi_rad"],
        eta_detection=params.get("eta_detection", 1.0),
        R=params.get("R", 0.0),
        tau_d=params.get("tau_d", 0.0),
        reference=entry.get("reference", ""),
        doi=entry.get("doi", ""),
    )


# ---------------------------------------------------------------------------
# Noisy measurement model
# ---------------------------------------------------------------------------

def noisy_outcome_prob(
    theta_true: float,
    n: int,
    phi: float,
    platform: PlatformParams,
    rng: np.random.Generator,
) -> float:
    """
    Compute the noisy probability of outcome b=0.

    Implements equation (3) of the manuscript:
        p_noisy(b=0) = eta_ro + (1 - 2*eta_ro) * 0.5 *
                       [1 + pg * cos(n * theta_noisy - phi)]

    where theta_noisy ~ N(theta*, sigma_phi^2).

    Parameters
    ----------
    theta_true : float
        True phase value (radians).
    n : int
        Measurement exponent.
    phi : float
        Measurement basis angle (radians).
    platform : PlatformParams
        Hardware noise parameters.
    rng : np.random.Generator
        Random number generator (for phase noise sampling).

    Returns
    -------
    float
        Probability of observing outcome b=0.
    """
    # Phase noise: theta observed with Gaussian jitter
    if platform.sigma_phi > 0:
        theta_obs = theta_true + rng.normal(0.0, platform.sigma_phi)
    else:
        theta_obs = theta_true

    # Ideal likelihood
    pg = platform.p * platform.gamma
    p_ideal = 0.5 * (1.0 + pg * np.cos(n * theta_obs - phi))

    # Readout error (symmetric bit-flip)
    p_noisy = (
        platform.eta_ro
        + (1.0 - 2.0 * platform.eta_ro) * p_ideal
    )

    return float(np.clip(p_noisy, 0.0, 1.0))


def sample_noisy_outcome(
    theta_true: float,
    n: int,
    phi: float,
    platform: PlatformParams,
    rng: np.random.Generator,
) -> int:
    """
    Draw one noisy binary measurement outcome b ∈ {0, 1}.

    Parameters
    ----------
    (same as noisy_outcome_prob)

    Returns
    -------
    int : 0 or 1
    """
    p0 = noisy_outcome_prob(theta_true, n, phi, platform, rng)
    return int(rng.random() > p0)


# ---------------------------------------------------------------------------
# Noise-aware adaptive exponent (with hardware ceiling)
# ---------------------------------------------------------------------------

def hardware_exponent(
    kappa: float,
    platform: PlatformParams,
) -> int:
    """
    Optimal measurement exponent with hardware phase-noise ceiling.

    Implements Algorithm 1 + equation (4) of the manuscript:
        n_t* = min(floor(kappa / (2 * pg)) + 1, floor(1/sigma_phi))

    Parameters
    ----------
    kappa : float
        Current posterior concentration.
    platform : PlatformParams
        Hardware noise parameters.

    Returns
    -------
    int
        Effective optimal exponent.
    """
    pg = platform.pg_eff
    n_adaptive = max(1, int(kappa / (2.0 * pg))) + 1
    return min(n_adaptive, platform.n_max)


# ---------------------------------------------------------------------------
# Full hardware-aware trial
# ---------------------------------------------------------------------------

def run_hardware_trial(
    N_shots: int,
    theta_true: float,
    platform: PlatformParams,
    rng: np.random.Generator,
) -> dict:
    """
    Run one BQPE trial with full hardware noise model.

    Uses:
      - Noisy outcome probability (readout error + phase noise)
      - Hardware-capped exponent (noise ceiling from sigma_phi)
      - Effective signal amplitude (pg_eff) for posterior update

    Parameters
    ----------
    N_shots : int
        Number of measurement rounds.
    theta_true : float
        True phase value (radians).
    platform : PlatformParams
        Hardware noise parameters.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    dict with keys:
        'theta_hat'     : MAP estimate
        'kappa_final'   : final concentration
        'exponents'     : list of exponents used
        'kappa_traj'    : concentration trajectory
        'posterior_var' : final posterior variance
    """
    from adaptive_protocol import vonmises_update, vonmises_var

    pg = platform.pg_eff
    kappa, mu = 0.0, 0.0
    exponents, kappa_traj = [], []

    for _ in range(N_shots):
        n_t   = hardware_exponent(kappa, platform)
        phi_t = n_t * mu - np.pi / 2.0
        b_t   = sample_noisy_outcome(theta_true, n_t, phi_t, platform, rng)
        kappa, mu = vonmises_update(kappa, mu, n_t, phi_t, b_t, pg)
        exponents.append(n_t)
        kappa_traj.append(kappa)

    return {
        'theta_hat':    float(mu),
        'kappa_final':  float(kappa),
        'exponents':    exponents,
        'kappa_traj':   kappa_traj,
        'posterior_var': vonmises_var(kappa),
    }


# ---------------------------------------------------------------------------
# Monte Carlo for hardware platforms
# ---------------------------------------------------------------------------

def run_hardware_monte_carlo(
    N_shots: int,
    theta_true: float,
    platform: PlatformParams,
    N_trials: int = 2000,
    seed: int = 2024,
) -> dict:
    """
    Monte Carlo estimation of BQPE performance on a hardware platform.

    Parameters
    ----------
    N_shots : int
        Shots per trial.
    theta_true : float
        True phase (radians).
    platform : PlatformParams
        Hardware noise parameters.
    N_trials : int
        Number of independent trials (default 2000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'mse', 'bias', 'variance', 'crlb_hw', 'efficiency_ratio',
        'estimates', 'N_shots', 'platform_name'
    """
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_trials)

    for i in range(N_trials):
        result = run_hardware_trial(N_shots, theta_true, platform, rng)
        estimates[i] = result['theta_hat']

    mse_val   = float(np.mean((estimates - theta_true) ** 2))
    bias_val  = float(np.abs(np.mean(estimates) - theta_true))
    var_val   = float(np.var(estimates, ddof=1))
    crlb_hw   = platform.crlb_hw(N_shots)
    eff_ratio = mse_val / crlb_hw if crlb_hw > 0 else float('inf')

    return {
        'mse':              mse_val,
        'bias':             bias_val,
        'variance':         var_val,
        'crlb_hw':          crlb_hw,
        'efficiency_ratio': eff_ratio,
        'estimates':        estimates,
        'N_shots':          N_shots,
        'platform_name':    platform.name,
    }
