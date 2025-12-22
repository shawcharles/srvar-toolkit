from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .results import FitResult, ForecastResult


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("matplotlib is required; install with 'srvar-toolkit[plot]'") from e
    return plt


def _time_index_to_array(index: Any) -> np.ndarray:
    try:
        if hasattr(index, "to_numpy"):
            return np.asarray(index.to_numpy())
        return np.asarray(index)
    except Exception:
        return np.arange(len(index), dtype=int)


def _is_datetime_like(x: np.ndarray) -> bool:
    return bool(np.issubdtype(np.asarray(x).dtype, np.datetime64))


def plot_shadow_rate(
    fit: FitResult,
    *,
    var: str,
    bands: tuple[float, float] = (0.1, 0.9),
    ax: Any | None = None,
    overlays: dict[str, Iterable[float]] | None = None,
    show_observed: bool = True,
) -> tuple[Any, Any]:
    """Plot observed vs. latent shadow-rate series.

    Parameters
    ----------
    fit:
        Output from :func:`srvar.api.fit`.
    var:
        Variable name to plot.
    bands:
        Quantile band (low, high) for uncertainty visualization when latent draws
        are available.
    ax:
        Optional Matplotlib axis to draw into.
    overlays:
        Optional additional named series to overlay (e.g., benchmark rates).
    show_observed:
        Whether to plot the observed (censored) series.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    plt = _require_matplotlib()

    idx = fit.dataset.variables.index(var)
    x = _time_index_to_array(fit.dataset.time_index)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if show_observed:
        ax.plot(x, fit.dataset.values[:, idx], color="C0", lw=1.5, label="Observed")

    if fit.latent_draws is not None:
        y = fit.latent_draws[:, :, idx]
        qlo, qhi = bands
        lo = np.quantile(y, q=qlo, axis=0)
        med = np.quantile(y, q=0.5, axis=0)
        hi = np.quantile(y, q=qhi, axis=0)
        ax.fill_between(x, lo, hi, color="C3", alpha=0.2, lw=0)
        ax.plot(x, med, color="C3", lw=2.0, label="Shadow (median)")
    elif fit.latent_dataset is not None:
        ax.plot(x, fit.latent_dataset.values[:, idx], color="C3", lw=2.0, label="Shadow")

    if overlays is not None:
        for name, series in overlays.items():
            ax.plot(x, np.asarray(list(series), dtype=float), lw=1.5, ls="--", label=name)

    ax.set_title(f"{var}: observed vs shadow")
    ax.legend(loc="best")
    if _is_datetime_like(np.asarray(x)):
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_forecast_fanchart(
    fc: ForecastResult,
    *,
    var: str,
    bands: tuple[float, float] = (0.1, 0.9),
    ax: Any | None = None,
    use_latent: bool = False,
) -> tuple[Any, Any]:
    """Plot a forecast fan chart from predictive simulations.

    Parameters
    ----------
    fc:
        Output from :func:`srvar.api.forecast`.
    var:
        Variable name to plot.
    bands:
        Quantile band (low, high) for the fan.
    ax:
        Optional Matplotlib axis.
    use_latent:
        If True and latent draws exist (ELB model), use latent draws instead of
        floored observed draws.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    plt = _require_matplotlib()

    idx = fc.variables.index(var)
    sims = fc.latent_draws if (use_latent and fc.latent_draws is not None) else fc.draws

    x = np.arange(1, sims.shape[1] + 1, dtype=int)
    y = sims[:, :, idx]

    qlo, qhi = bands
    lo = np.quantile(y, q=qlo, axis=0)
    med = np.quantile(y, q=0.5, axis=0)
    hi = np.quantile(y, q=qhi, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.fill_between(x, lo, hi, color="C0", alpha=0.2, lw=0)
    ax.plot(x, med, color="C0", lw=2.0, label="Median")

    ax.set_title(f"Forecast fan chart: {var}")
    ax.set_xlabel("Horizon")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_volatility(
    fit: FitResult,
    *,
    var: str,
    bands: tuple[float, float] = (0.1, 0.9),
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot stochastic volatility (posterior std dev) for a given series.

    Parameters
    ----------
    fit:
        Output from :func:`srvar.api.fit` with volatility enabled.
    var:
        Variable name.
    bands:
        Quantile band (low, high) used to summarize posterior uncertainty.
    ax:
        Optional Matplotlib axis.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if fit.h_draws is None:
        raise ValueError("fit.h_draws is required (fit must be run with volatility enabled)")

    plt = _require_matplotlib()

    idx = fit.dataset.variables.index(var)

    p = fit.model.p
    x = _time_index_to_array(fit.dataset.time_index)[p:]

    h = fit.h_draws[:, :, idx]
    sd = np.exp(0.5 * h)

    qlo, qhi = bands
    lo = np.quantile(sd, q=qlo, axis=0)
    med = np.quantile(sd, q=0.5, axis=0)
    hi = np.quantile(sd, q=qhi, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.fill_between(x, lo, hi, color="C2", alpha=0.2, lw=0)
    ax.plot(x, med, color="C2", lw=2.0, label="Median")

    ax.set_title(f"Stochastic volatility (std dev): {var}")
    ax.legend(loc="best")
    if _is_datetime_like(np.asarray(x)):
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_ssvs_inclusion(
    fit: FitResult,
    *,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot posterior inclusion probabilities for SSVS.

    Parameters
    ----------
    fit:
        Output from :func:`srvar.api.fit` with ``prior.family='ssvs'``.
    ax:
        Optional Matplotlib axis.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if fit.gamma_draws is None:
        raise ValueError("fit.gamma_draws is required (fit must be run with prior.family='ssvs')")

    plt = _require_matplotlib()

    probs = fit.gamma_draws.mean(axis=0)
    k = probs.shape[0]

    if fit.model.include_intercept:
        names = ["intercept"]
        base = 1
    else:
        names = []
        base = 0

    n = fit.dataset.N
    for lag in range(1, fit.model.p + 1):
        for j in range(n):
            names.append(f"lag{lag}:{fit.dataset.variables[j]}")

    if len(names) != k:
        names = [f"x{i}" for i in range(k)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * k)))
    else:
        fig = ax.figure

    y = np.arange(k, dtype=int)
    ax.barh(y, probs, color="C1", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()

    ax.set_title("SSVS inclusion probabilities")
    fig.tight_layout()
    return fig, ax


def plot_trace(
    draws: np.ndarray,
    *,
    ax: Any | None = None,
    label: str | None = None,
) -> tuple[Any, Any]:
    """Plot a simple MCMC trace for a 1D parameter draw sequence."""
    plt = _require_matplotlib()

    x = np.asarray(draws, dtype=float)
    if x.ndim != 1:
        raise ValueError("draws must be 1D")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    ax.plot(np.arange(x.size, dtype=int), x, lw=1.0)
    if label is not None:
        ax.set_title(label)
    fig.tight_layout()
    return fig, ax
