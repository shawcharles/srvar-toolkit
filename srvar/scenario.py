from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .bvar import sample_posterior_niw
from .data.dataset import Dataset
from .elb import apply_elb_floor
from .linalg import solve_psd
from .results import FitResult, ForecastResult
from .var import is_stationary


def _parse_horizons(horizons: list[int]) -> list[int]:
    if not isinstance(horizons, list) or len(horizons) < 1:
        raise ValueError("horizons must be a non-empty list[int]")
    out: list[int] = []
    for h in horizons:
        if not isinstance(h, (int, np.integer)) or isinstance(h, bool):
            raise ValueError("horizons must contain only integers")
        hi = int(h)
        if hi < 1:
            raise ValueError("horizons must contain only positive integers")
        out.append(hi)
    return out


def _parse_quantiles(quantile_levels: list[float] | None) -> list[float]:
    if quantile_levels is None:
        return [0.1, 0.5, 0.9]
    if not isinstance(quantile_levels, list) or len(quantile_levels) < 1:
        raise ValueError("quantile_levels must be a non-empty list")

    qs: list[float] = []
    for q in quantile_levels:
        qf = float(q)
        if not np.isfinite(qf) or not (0.0 < qf < 1.0):
            raise ValueError("quantile_levels must be finite and in (0, 1)")
        qs.append(qf)
    return qs


def _parse_stationarity(
    *,
    stationarity: str,
    stationarity_tol: float,
    stationarity_max_draws: int | None,
) -> tuple[str, float, int | None]:
    if not isinstance(stationarity, str) or not stationarity:
        raise ValueError("stationarity must be a non-empty string")
    stationarity_l = stationarity.lower()
    if stationarity_l not in {"allow", "reject"}:
        raise ValueError("stationarity must be one of: allow, reject")

    tol = float(stationarity_tol)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("stationarity_tol must be finite and >= 0")

    max_draws: int | None
    if stationarity_max_draws is None:
        max_draws = None
    else:
        if not isinstance(stationarity_max_draws, (int, np.integer)) or isinstance(
            stationarity_max_draws, bool
        ):
            raise ValueError("stationarity_max_draws must be an integer when provided")
        if int(stationarity_max_draws) < 1:
            raise ValueError("stationarity_max_draws must be >= 1")
        max_draws = int(stationarity_max_draws)

    return stationarity_l, tol, max_draws


def _parse_constraints(
    constraints: Mapping[str, Mapping[int, float]] | None,
    *,
    variables: list[str],
    max_h: int,
) -> list[tuple[int, int, float]]:
    if constraints is None:
        return []
    if not isinstance(constraints, Mapping):
        raise ValueError("constraints must be a mapping of variable -> {horizon: value}")

    out: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for var, by_h in constraints.items():
        if not isinstance(var, str) or not var:
            raise ValueError("constraints keys must be non-empty strings (variable names)")
        if var not in variables:
            raise ValueError(f"constraints contains unknown variable: {var}")
        if not isinstance(by_h, Mapping):
            raise ValueError(f"constraints[{var!r}] must be a mapping of horizon -> value")

        var_idx = int(variables.index(var))
        for h_raw, v_raw in by_h.items():
            if not isinstance(h_raw, (int, np.integer)) or isinstance(h_raw, bool):
                raise ValueError("constraint horizons must be integers")
            h = int(h_raw)
            if h < 1:
                raise ValueError("constraint horizons must be >= 1")
            if h > int(max_h):
                raise ValueError("constraint horizon exceeds max(horizons)")
            h_idx = h - 1

            key = (h_idx, var_idx)
            if key in seen:
                raise ValueError("duplicate constraint for the same (variable, horizon)")
            seen.add(key)

            val = float(v_raw)
            if not np.isfinite(val):
                raise ValueError("constraint values must be finite")

            out.append((h_idx, var_idx, val))

    return sorted(out, key=lambda t: (t[0], t[1]))


def _ar_matrices_from_beta(
    *, beta: np.ndarray, n: int, p: int, include_intercept: bool
) -> list[np.ndarray]:
    b = np.asarray(beta, dtype=float)
    k_expected = (1 if include_intercept else 0) + int(n) * int(p)
    if b.shape != (k_expected, int(n)):
        raise ValueError("beta has wrong shape for VAR(p)")

    rows = b[1:, :] if include_intercept else b
    mats: list[np.ndarray] = []
    for lag in range(int(p)):
        block = rows[lag * int(n) : (lag + 1) * int(n), :]
        mats.append(block.T.copy())
    return mats


def _ma_matrices(*, a_mats: list[np.ndarray], horizon: int) -> np.ndarray:
    n = int(a_mats[0].shape[0])
    p = int(len(a_mats))
    hmax = int(horizon)

    phi = np.zeros((hmax + 1, n, n), dtype=float)
    phi[0] = np.eye(n, dtype=float)

    for h in range(1, hmax + 1):
        acc = np.zeros((n, n), dtype=float)
        for lag in range(1, min(p, h) + 1):
            acc += a_mats[lag - 1] @ phi[h - lag]
        phi[h] = acc

    return phi


def _mean_path(
    *,
    y_last: np.ndarray,
    beta: np.ndarray,
    horizon: int,
    include_intercept: bool,
    p: int,
) -> np.ndarray:
    lags = np.asarray(y_last, dtype=float).copy()
    hmax = int(horizon)
    n = int(lags.shape[1])
    path = np.empty((hmax, n), dtype=float)

    for h_step in range(hmax):
        x_parts = []
        if include_intercept:
            x_parts.append(np.array([1.0], dtype=float))
        for lag in range(1, int(p) + 1):
            x_parts.append(lags[-lag, :])
        x = np.concatenate(x_parts)

        mean = x @ beta
        path[h_step] = mean
        lags = np.vstack([lags[1:, :], mean]) if int(p) > 1 else mean.reshape(1, -1)

    return path


def _simulate_with_innovations(
    *,
    y_last: np.ndarray,
    beta: np.ndarray,
    innovations: np.ndarray,
    include_intercept: bool,
    p: int,
) -> np.ndarray:
    lags = np.asarray(y_last, dtype=float).copy()
    eps = np.asarray(innovations, dtype=float)
    if eps.ndim != 2:
        raise ValueError("innovations must have shape (H, N)")

    hmax, n = int(eps.shape[0]), int(eps.shape[1])
    path = np.empty((hmax, n), dtype=float)

    for h_step in range(hmax):
        x_parts = []
        if include_intercept:
            x_parts.append(np.array([1.0], dtype=float))
        for lag in range(1, int(p) + 1):
            x_parts.append(lags[-lag, :])
        x = np.concatenate(x_parts)

        mean = x @ beta
        y_next = mean + eps[h_step, :]
        path[h_step] = y_next
        lags = np.vstack([lags[1:, :], y_next]) if int(p) > 1 else y_next.reshape(1, -1)

    return path


def conditional_forecast(
    fit: FitResult,
    horizons: list[int],
    *,
    constraints: Mapping[str, Mapping[int, float]] | None,
    draws: int = 1000,
    quantile_levels: list[float] | None = None,
    stationarity: str = "allow",
    stationarity_tol: float = 1e-10,
    stationarity_max_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> ForecastResult:
    """Generate conditional/scenario forecasts under hard equality constraints.

    Parameters
    ----------
    fit:
        Result from :func:`srvar.api.fit`.
    horizons:
        List of requested horizons (steps ahead). Internally, simulation runs out to
        `H = max(horizons)`.
    constraints:
        Mapping of `variable -> {horizon: value}`. Horizons are **1-indexed steps ahead**
        (e.g. `1` means `t+1`).
    draws:
        Number of conditional predictive paths to generate.
    quantile_levels:
        Quantiles to compute from the simulated draws. Defaults to `[0.1, 0.5, 0.9]`.
    stationarity, stationarity_tol, stationarity_max_draws:
        Same semantics as :func:`srvar.api.forecast`. These apply to the **parameter draws**
        used to generate conditional predictive paths.

    Notes
    -----
    - This implementation currently supports **linear Gaussian VARs with time-invariant
      covariance** (i.e., homoskedastic models). Stochastic volatility models are not yet
      supported for conditional forecasting.
    - When ELB is enabled, constraints are applied to the **latent (unfloored)** process
      used for simulation. Returned `ForecastResult.draws` are observed (floored) draws and
      `ForecastResult.latent_draws` contains the latent conditional draws.
    """
    if rng is None:
        rng = np.random.default_rng()

    horizons = _parse_horizons(horizons)
    q_levels = _parse_quantiles(quantile_levels)

    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool):
        raise ValueError("draws must be an integer")
    if int(draws) < 1:
        raise ValueError("draws must be >= 1")
    draws = int(draws)

    st_policy, st_tol, st_max = _parse_stationarity(
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
    )

    hmax = int(max(horizons))
    p = int(fit.model.p)

    if fit.model.volatility is not None and fit.model.volatility.enabled:
        raise ValueError(
            "conditional_forecast is not yet supported for stochastic volatility models"
        )

    base_dataset: Dataset = fit.latent_dataset if fit.latent_dataset is not None else fit.dataset
    if base_dataset.T < p:
        raise ValueError("dataset is too short for requested lag order p")
    y_last = base_dataset.values[-p:, :]

    constraints_list = _parse_constraints(
        constraints,
        variables=list(base_dataset.variables),
        max_h=hmax,
    )
    if len(constraints_list) < 1:
        from .api import forecast

        return forecast(
            fit,
            horizons=horizons,
            draws=draws,
            quantile_levels=q_levels,
            stationarity=st_policy,
            stationarity_tol=st_tol,
            stationarity_max_draws=st_max,
            rng=rng,
        )

    # Select or sample parameter draws for each predictive path.
    if fit.beta_draws is not None and fit.sigma_draws is not None:
        avail = int(fit.beta_draws.shape[0])
        if avail < 1:
            raise ValueError("fit.beta_draws is empty")

        if st_policy == "reject":
            stable = [
                is_stationary(
                    fit.beta_draws[i],
                    n=fit.dataset.N,
                    p=p,
                    include_intercept=fit.model.include_intercept,
                    tol=st_tol,
                )
                for i in range(avail)
            ]
            stable_idx = np.flatnonzero(np.asarray(stable, dtype=bool))
            if stable_idx.size < 1:
                raise ValueError("no stationary coefficient draws available in fit.beta_draws")
            sel = rng.integers(0, stable_idx.size, size=draws)
            idx = stable_idx[sel]
        else:
            idx = rng.integers(0, avail, size=draws)

        beta_draws = np.asarray(fit.beta_draws[idx], dtype=float)
        sigma_draws = np.asarray(fit.sigma_draws[idx], dtype=float)

    else:
        if fit.posterior is None:
            raise ValueError("fit has no stored draws and no posterior parameters")

        if st_policy == "allow":
            beta_draws, sigma_draws = sample_posterior_niw(
                mn=fit.posterior.mn,
                vn=fit.posterior.vn,
                sn=fit.posterior.sn,
                nun=fit.posterior.nun,
                draws=draws,
                rng=rng,
            )
        else:
            max_total = int(50 * draws) if st_max is None else int(st_max)
            accepted_beta: list[np.ndarray] = []
            accepted_sigma: list[np.ndarray] = []
            attempted = 0
            while len(accepted_beta) < draws and attempted < max_total:
                need = int(draws - len(accepted_beta))
                batch = min(max(10, 2 * need), max_total - attempted)
                beta_b, sigma_b = sample_posterior_niw(
                    mn=fit.posterior.mn,
                    vn=fit.posterior.vn,
                    sn=fit.posterior.sn,
                    nun=fit.posterior.nun,
                    draws=batch,
                    rng=rng,
                )
                attempted += batch
                for i in range(int(batch)):
                    if is_stationary(
                        beta_b[i],
                        n=fit.dataset.N,
                        p=p,
                        include_intercept=fit.model.include_intercept,
                        tol=st_tol,
                    ):
                        accepted_beta.append(beta_b[i])
                        accepted_sigma.append(sigma_b[i])
                        if len(accepted_beta) == draws:
                            break

            if len(accepted_beta) < draws:
                raise ValueError(
                    f"stationarity='reject' could not generate {draws} stationary draws within "
                    f"{max_total} candidate draws; try increasing stationarity_max_draws or relaxing priors"
                )
            beta_draws = np.stack(accepted_beta)
            sigma_draws = np.stack(accepted_sigma)

    n = int(base_dataset.N)
    sims = np.empty((draws, hmax, n), dtype=float)

    for d in range(draws):
        beta = beta_draws[d]
        sigma = sigma_draws[d]

        a_mats = _ar_matrices_from_beta(
            beta=beta,
            n=n,
            p=p,
            include_intercept=fit.model.include_intercept,
        )
        phi = _ma_matrices(a_mats=a_mats, horizon=hmax - 1)  # (hmax, N, N) for offsets 0..hmax-1
        mu = _mean_path(
            y_last=y_last,
            beta=beta,
            horizon=hmax,
            include_intercept=fit.model.include_intercept,
            p=p,
        )

        m = int(len(constraints_list))
        c = np.zeros((m, hmax * n), dtype=float)
        dvec = np.zeros(m, dtype=float)

        for r, (h_idx, var_idx, value) in enumerate(constraints_list):
            h = int(h_idx + 1)
            for j in range(h):
                c[r, j * n : (j + 1) * n] = phi[h - 1 - j, var_idx, :]
            dvec[r] = float(value - mu[h_idx, var_idx])

        k = np.zeros((hmax * n, m), dtype=float)
        for j in range(hmax):
            cj = c[:, j * n : (j + 1) * n]  # (m, N)
            k[j * n : (j + 1) * n, :] = np.asarray(sigma, dtype=float) @ cj.T

        s = c @ k

        eps = rng.multivariate_normal(mean=np.zeros(n, dtype=float), cov=sigma, size=hmax)
        e = np.asarray(eps, dtype=float).reshape(hmax * n)

        rvec = dvec - c @ e
        alpha = solve_psd(s, rvec)
        e_adj = e + k @ alpha
        eps_adj = e_adj.reshape(hmax, n)

        sims[d] = _simulate_with_innovations(
            y_last=y_last,
            beta=beta,
            innovations=eps_adj,
            include_intercept=fit.model.include_intercept,
            p=p,
        )

        for h_idx, var_idx, value in constraints_list:
            if not np.isclose(sims[d, h_idx, var_idx], value):
                raise RuntimeError(
                    "internal error: conditional forecast draw does not satisfy constraints"
                )

    mean = sims.mean(axis=0)
    quantiles = {q: np.quantile(sims, q=q, axis=0) for q in q_levels}

    latent_sims: np.ndarray | None = None
    if fit.model.elb is not None and fit.model.elb.enabled:
        latent_sims = sims.copy()
        applies_to_idx = [fit.dataset.variables.index(v) for v in fit.model.elb.applies_to]
        sims = apply_elb_floor(sims, bound=fit.model.elb.bound, indices=applies_to_idx)
        mean = sims.mean(axis=0)
        quantiles = {q: np.quantile(sims, q=q, axis=0) for q in q_levels}

    return ForecastResult(
        variables=list(fit.dataset.variables),
        horizons=list(horizons),
        draws=sims,
        latent_draws=latent_sims,
        mean=mean,
        quantiles=quantiles,
    )
