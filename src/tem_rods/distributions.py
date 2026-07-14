"""
Population statistics for nanoparticle size / shape distributions.

Follows Aviles & Lear (ACS Nanosci. Au 2025): nanoparticle sizes are typically
log-normal; report geometric mean from the fit with standard error in the mean,
and do not conflate sample SD with uncertainty of the mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LognormalFit:
    """Log-normal parameters for positive continuous measurements (e.g. size)."""

    n: int
    # Real-space geometric mean = exp(μ) where μ = mean(ln x)
    geometric_mean: float
    # σ of ln(x); real-space multiplicative spread
    sigma_ln: float
    # Approximate SE of the geometric mean via delta method
    geometric_mean_se: float
    arithmetic_mean: float
    arithmetic_std: float

    @property
    def summary(self) -> str:
        return (
            f"{self.geometric_mean:.2f} ± {self.geometric_mean_se:.2f} "
            f"(geom.; n={self.n}; arith. {self.arithmetic_mean:.2f}±{self.arithmetic_std:.2f})"
        )


def fit_lognormal(values: list[float] | np.ndarray, *, min_n: int = 3) -> LognormalFit | None:
    """
    MLE log-normal fit on positive values.

    Returns None if too few positive samples.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = int(arr.size)
    if n < min_n:
        return None
    log_x = np.log(arr)
    mu = float(np.mean(log_x))
    sigma = float(np.std(log_x, ddof=1)) if n > 1 else 0.0
    geom = float(np.exp(mu))
    se_mu = sigma / math.sqrt(n) if n > 0 else 0.0
    geom_se = float(geom * se_mu)
    return LognormalFit(
        n=n,
        geometric_mean=geom,
        sigma_ln=sigma,
        geometric_mean_se=geom_se,
        arithmetic_mean=float(np.mean(arr)),
        arithmetic_std=float(np.std(arr, ddof=1)) if n > 1 else 0.0,
    )


def sample_size_note(n: int, *, recommended: int = 200) -> str | None:
    """Warn when fewer particles than the common ~200 rule of thumb."""
    if n <= 0:
        return None
    if n < recommended:
        return (
            f"Only {n} particles approved (rule of thumb ≥ {recommended} for "
            "stable population estimates). Consider analyzing more images."
        )
    return None
