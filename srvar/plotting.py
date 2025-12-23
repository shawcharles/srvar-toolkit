from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .results import FitResult, ForecastResult
from .metrics import crps_draws
from .theme import (
    DEFAULT_THEME,
    Theme,
    get_alpha,
    get_color,
    get_figsize,
    get_linewidth,
    srvar_style,
)


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
    theme: Theme | None = None,
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
    theme:
        Optional theme for styling. If None, uses DEFAULT_THEME.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
        plt = _require_matplotlib()

        idx = fit.dataset.variables.index(var)
        x = _time_index_to_array(fit.dataset.time_index)

        if ax is None:
            fig, ax = plt.subplots(figsize=get_figsize("wide", theme))
        else:
            fig = ax.figure

        if show_observed:
            ax.plot(
                x,
                fit.dataset.values[:, idx],
                color=get_color("observed", theme),
                lw=get_linewidth("data", theme),
                label="Observed",
            )

        if fit.latent_draws is not None:
            y = fit.latent_draws[:, :, idx]
            qlo, qhi = bands
            lo = np.quantile(y, q=qlo, axis=0)
            med = np.quantile(y, q=0.5, axis=0)
            hi = np.quantile(y, q=qhi, axis=0)
            ax.fill_between(
                x, lo, hi, color=get_color("shadow", theme), alpha=get_alpha("band", theme), lw=0
            )
            ax.plot(
                x,
                med,
                color=get_color("shadow", theme),
                lw=get_linewidth("median", theme),
                label="Shadow (median)",
            )
        elif fit.latent_dataset is not None:
            ax.plot(
                x,
                fit.latent_dataset.values[:, idx],
                color=get_color("shadow", theme),
                lw=get_linewidth("median", theme),
                label="Shadow",
            )

        if overlays is not None:
            for name, series in overlays.items():
                ax.plot(
                    x,
                    np.asarray(list(series), dtype=float),
                    lw=get_linewidth("data", theme),
                    ls="--",
                    label=name,
                )

        ax.set_title(f"{var}: observed vs shadow")
        ax.legend(loc="best")
        if _is_datetime_like(np.asarray(x)):
            fig.autofmt_xdate()
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax


def plot_forecast_coverage(
    forecasts: list[ForecastResult],
    y_true: np.ndarray,
    *,
    intervals: list[float] = [0.5, 0.8, 0.9],
    horizons: list[int] | None = None,
    var: str | None = None,
    ax: Any | None = None,
    use_latent: bool = False,
    theme: Theme | None = None,
) -> tuple[Any, Any]:
    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
        plt = _require_matplotlib()

        if len(forecasts) < 1:
            raise ValueError("forecasts must be non-empty")

        yt = np.asarray(y_true, dtype=float)
        if yt.ndim != 3:
            raise ValueError("y_true must have shape (K, H, N)")

        hmax = int(np.asarray(forecasts[0].draws).shape[1])
        n = int(np.asarray(forecasts[0].draws).shape[2])
        k = int(yt.shape[0])
        if yt.shape[1] != hmax or yt.shape[2] != n:
            raise ValueError("y_true shape must match forecasts: (K, H, N)")
        if len(forecasts) != k:
            raise ValueError("len(forecasts) must equal y_true.shape[0]")

        if horizons is None:
            h_list = list(range(1, hmax + 1))
        else:
            if (not isinstance(horizons, list)) or (len(horizons) < 1):
                raise ValueError("horizons must be a non-empty list[int]")
            h_list = [int(h) for h in horizons]
            if any(h < 1 or h > hmax for h in h_list):
                raise ValueError("horizons contains values out of range")
            if len(set(h_list)) != len(h_list):
                raise ValueError("horizons must not contain duplicates")
            h_list = sorted(h_list)
        h_idx = [h - 1 for h in h_list]

        if var is None:
            var_idx = None
            title = "Forecast coverage (avg across variables)"
        else:
            if var not in forecasts[0].variables:
                raise ValueError("var not in forecasts[0].variables")
            var_idx = int(forecasts[0].variables.index(var))
            title = f"Forecast coverage: {var}"

        intervals_f = [float(c) for c in intervals]
        for c in intervals_f:
            if not np.isfinite(c) or not (0.0 < c < 1.0):
                raise ValueError("intervals must be in (0, 1)")

        cov_by_int: dict[float, np.ndarray] = {}
        for c in intervals_f:
            qlo = 0.5 - 0.5 * c
            qhi = 0.5 + 0.5 * c

            hits = np.empty((k, len(h_idx)), dtype=float)
            for i, fc in enumerate(forecasts):
                sims = fc.latent_draws if (use_latent and fc.latent_draws is not None) else fc.draws
                sims = np.asarray(sims, dtype=float)
                if var_idx is None:
                    lo = np.quantile(sims, q=qlo, axis=0)[h_idx, :]
                    hi = np.quantile(sims, q=qhi, axis=0)[h_idx, :]
                    yti = yt[i][h_idx, :]
                    ok = (yti >= lo) & (yti <= hi)
                    hits[i] = np.mean(ok, axis=1)
                else:
                    y = yt[i, :, var_idx][h_idx]
                    lo = np.quantile(sims[:, :, var_idx], q=qlo, axis=0)[h_idx]
                    hi = np.quantile(sims[:, :, var_idx], q=qhi, axis=0)[h_idx]
                    hits[i] = ((y >= lo) & (y <= hi)).astype(float)

            cov_by_int[c] = hits.mean(axis=0)

        x = np.asarray(h_list, dtype=int)
        if ax is None:
            fig, ax = plt.subplots(figsize=get_figsize("wide", theme))
        else:
            fig = ax.figure

        for _j, c in enumerate(sorted(cov_by_int.keys())):
            ax.plot(x, cov_by_int[c], lw=get_linewidth("median", theme), label=f"{int(round(100*c))}%")

        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax


def plot_pit_histogram(
    forecasts: list[ForecastResult],
    y_true: np.ndarray,
    *,
    var: str,
    horizon: int,
    bins: int = 10,
    ax: Any | None = None,
    use_latent: bool = False,
    theme: Theme | None = None,
) -> tuple[Any, Any]:
    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
        plt = _require_matplotlib()

        if len(forecasts) < 1:
            raise ValueError("forecasts must be non-empty")
        yt = np.asarray(y_true, dtype=float)
        if yt.ndim != 3:
            raise ValueError("y_true must have shape (K, H, N)")

        if var not in forecasts[0].variables:
            raise ValueError("var not in forecasts[0].variables")
        var_idx = int(forecasts[0].variables.index(var))

        hmax = int(np.asarray(forecasts[0].draws).shape[1])
        if horizon < 1 or horizon > hmax:
            raise ValueError("horizon out of range")
        h_idx = int(horizon - 1)

        k = int(yt.shape[0])
        if len(forecasts) != k:
            raise ValueError("len(forecasts) must equal y_true.shape[0]")
        if yt.shape[1] != hmax:
            raise ValueError("y_true horizon dimension must match forecasts")

        u = np.empty(k, dtype=float)
        for i, fc in enumerate(forecasts):
            sims = fc.latent_draws if (use_latent and fc.latent_draws is not None) else fc.draws
            sims = np.asarray(sims, dtype=float)
            y = float(yt[i, h_idx, var_idx])
            u[i] = float(np.mean(sims[:, h_idx, var_idx] <= y))

        if ax is None:
            fig, ax = plt.subplots(figsize=get_figsize("single", theme))
        else:
            fig = ax.figure

        ax.hist(
            u,
            bins=int(bins),
            range=(0.0, 1.0),
            density=True,
            color=get_color("pit", theme),
            alpha=get_alpha("bar", theme),
        )
        ax.axhline(
            1.0,
            color=get_color("reference", theme),
            lw=get_linewidth("reference", theme),
            ls="--",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT histogram: {var}, h={horizon}")
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax


def plot_crps_by_horizon(
    forecasts: list[ForecastResult],
    y_true: np.ndarray,
    *,
    horizons: list[int] | None = None,
    var: str | None = None,
    ax: Any | None = None,
    use_latent: bool = False,
    theme: Theme | None = None,
) -> tuple[Any, Any]:
    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
        plt = _require_matplotlib()

        if len(forecasts) < 1:
            raise ValueError("forecasts must be non-empty")
        yt = np.asarray(y_true, dtype=float)
        if yt.ndim != 3:
            raise ValueError("y_true must have shape (K, H, N)")

        hmax = int(np.asarray(forecasts[0].draws).shape[1])
        n = int(np.asarray(forecasts[0].draws).shape[2])
        k = int(yt.shape[0])
        if len(forecasts) != k or yt.shape[1] != hmax or yt.shape[2] != n:
            raise ValueError("y_true shape must match forecasts")

        if var is None:
            var_idx = None
            title = "CRPS by horizon (avg across variables)"
        else:
            if var not in forecasts[0].variables:
                raise ValueError("var not in forecasts[0].variables")
            var_idx = int(forecasts[0].variables.index(var))
            title = f"CRPS by horizon: {var}"

        if horizons is None:
            h_list = list(range(1, hmax + 1))
        else:
            if (not isinstance(horizons, list)) or (len(horizons) < 1):
                raise ValueError("horizons must be a non-empty list[int]")
            h_list = [int(h) for h in horizons]
            if any(h < 1 or h > hmax for h in h_list):
                raise ValueError("horizons contains values out of range")
            if len(set(h_list)) != len(h_list):
                raise ValueError("horizons must not contain duplicates")
            h_list = sorted(h_list)
        h_idx = [h - 1 for h in h_list]

        crps_h = np.zeros(len(h_idx), dtype=float)
        counts_h = np.zeros(len(h_idx), dtype=int)

        for i, fc in enumerate(forecasts):
            sims = fc.latent_draws if (use_latent and fc.latent_draws is not None) else fc.draws
            sims = np.asarray(sims, dtype=float)
            for hh, h in enumerate(h_idx):
                if var_idx is None:
                    for j in range(n):
                        y = float(yt[i, h, j])
                        crps_h[hh] += crps_draws(y, sims[:, h, j])
                        counts_h[hh] += 1
                else:
                    y = float(yt[i, h, var_idx])
                    crps_h[hh] += crps_draws(y, sims[:, h, var_idx])
                    counts_h[hh] += 1

        crps_h = np.where(counts_h > 0, crps_h / np.maximum(counts_h, 1), np.nan)

        x = np.asarray(h_list, dtype=int)
        if ax is None:
            fig, ax = plt.subplots(figsize=get_figsize("wide", theme))
        else:
            fig = ax.figure

        ax.plot(x, crps_h, lw=get_linewidth("median", theme), color=get_color("crps", theme))
        ax.set_xlabel("Horizon")
        ax.set_ylabel("CRPS")
        ax.set_title(title)
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax


def plot_forecast_fanchart(
    fc: ForecastResult,
    *,
    var: str,
    bands: tuple[float, float] = (0.1, 0.9),
    ax: Any | None = None,
    use_latent: bool = False,
    theme: Theme | None = None,
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
    theme:
        Optional theme for styling. If None, uses DEFAULT_THEME.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
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
            fig, ax = plt.subplots(figsize=get_figsize("single", theme))
        else:
            fig = ax.figure

        ax.fill_between(
            x, lo, hi, color=get_color("forecast", theme), alpha=get_alpha("band", theme), lw=0
        )
        ax.plot(
            x, med, color=get_color("forecast", theme), lw=get_linewidth("median", theme), label="Median"
        )

        ax.set_title(f"Forecast fan chart: {var}")
        ax.set_xlabel("Horizon")
        ax.legend(loc="best")
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax


def plot_volatility(
    fit: FitResult,
    *,
    var: str,
    bands: tuple[float, float] = (0.1, 0.9),
    ax: Any | None = None,
    theme: Theme | None = None,
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
    theme:
        Optional theme for styling. If None, uses DEFAULT_THEME.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if fit.h_draws is None:
        raise ValueError("fit.h_draws is required (fit must be run with volatility enabled)")

    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
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
            fig, ax = plt.subplots(figsize=get_figsize("wide", theme))
        else:
            fig = ax.figure

        ax.fill_between(
            x, lo, hi, color=get_color("volatility", theme), alpha=get_alpha("band", theme), lw=0
        )
        ax.plot(
            x, med, color=get_color("volatility", theme), lw=get_linewidth("median", theme), label="Median"
        )

        ax.set_title(f"Stochastic volatility (std dev): {var}")
        ax.legend(loc="best")
        if _is_datetime_like(np.asarray(x)):
            fig.autofmt_xdate()
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax


def plot_ssvs_inclusion(
    fit: FitResult,
    *,
    ax: Any | None = None,
    theme: Theme | None = None,
) -> tuple[Any, Any]:
    """Plot posterior inclusion probabilities for SSVS.

    Parameters
    ----------
    fit:
        Output from :func:`srvar.api.fit` with ``prior.family='ssvs'``.
    ax:
        Optional Matplotlib axis.
    theme:
        Optional theme for styling. If None, uses DEFAULT_THEME.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if fit.gamma_draws is None:
        raise ValueError("fit.gamma_draws is required (fit must be run with prior.family='ssvs')")

    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
        plt = _require_matplotlib()

        probs = fit.gamma_draws.mean(axis=0)
        k = probs.shape[0]

        if fit.model.include_intercept:
            names = ["intercept"]
        else:
            names = []

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
        ax.barh(y, probs, color=get_color("inclusion", theme), alpha=get_alpha("bar", theme))
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0.0, 1.0)
        ax.invert_yaxis()

        ax.set_title("SSVS inclusion probabilities")
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax


def plot_trace(
    draws: np.ndarray,
    *,
    ax: Any | None = None,
    label: str | None = None,
    theme: Theme | None = None,
) -> tuple[Any, Any]:
    """Plot a simple MCMC trace for a 1D parameter draw sequence.

    Parameters
    ----------
    draws:
        1D array of MCMC draws.
    ax:
        Optional Matplotlib axis.
    label:
        Optional title for the plot.
    theme:
        Optional theme for styling. If None, uses DEFAULT_THEME.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if theme is None:
        theme = DEFAULT_THEME

    with srvar_style(theme):
        plt = _require_matplotlib()

        x = np.asarray(draws, dtype=float)
        if x.ndim != 1:
            raise ValueError("draws must be 1D")

        if ax is None:
            fig, ax = plt.subplots(figsize=get_figsize("single", theme))
        else:
            fig = ax.figure

        ax.plot(np.arange(x.size, dtype=int), x, lw=get_linewidth("data", theme))
        if label is not None:
            ax.set_title(label)
        fig.tight_layout(pad=theme.layout.tight_layout_pad)
        return fig, ax
