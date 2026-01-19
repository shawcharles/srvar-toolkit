from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..bvar import sample_posterior_niw
from ..linalg import cholesky_jitter
from ..results import FitResult, HistoricalDecompositionResult
from ..var import design_matrix, is_stationary
from .irf import _ar_matrices_from_beta, _parse_ordering, _parse_stationarity, _select_draw_indices


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


def _permute_beta(
    *,
    beta: np.ndarray,
    order_idx: np.ndarray,
    n: int,
    p: int,
    include_intercept: bool,
) -> np.ndarray:
    b = np.asarray(beta, dtype=float)
    k_expected = (1 if include_intercept else 0) + int(n) * int(p)
    if b.shape != (k_expected, int(n)):
        raise ValueError("beta has wrong shape for VAR(p)")

    idx = np.asarray(order_idx, dtype=int).reshape(-1)
    if idx.shape != (int(n),):
        raise ValueError("order_idx has wrong shape")

    blocks: list[np.ndarray] = []
    base = 0
    if include_intercept:
        c = b[0, :][idx]
        blocks.append(c.reshape(1, -1))
        base = 1

    for lag in range(int(p)):
        block = b[base + lag * int(n) : base + (lag + 1) * int(n), :]
        blocks.append(block[np.ix_(idx, idx)])

    return np.vstack(blocks)


def historical_decomposition_cholesky(
    fit: FitResult,
    *,
    draws: int | None = 200,
    ordering: Sequence[str] | None = None,
    shock_scale: Literal["one_sd", "unit"] = "one_sd",
    quantile_levels: list[float] | None = None,
    stationarity: str = "allow",
    stationarity_tol: float = 1e-10,
    stationarity_max_draws: int | None = None,
    use_latent: bool | None = None,
    rng: np.random.Generator | None = None,
) -> HistoricalDecompositionResult:
    """Historical decomposition from Cholesky-identified structural shocks.

    This decomposes the observed (or latent) dataset into:
    - a shock-free baseline path conditional on the initial lags
    - additive contributions from each structural shock at each date

    Notes
    -----
    - For ELB models, the VAR is defined on the latent shadow series. If `use_latent` is not
      specified, this function defaults to using `fit.latent_dataset` when available.
    - For volatility models, shocks are identified using the time-varying covariance state
      `Sigma_t` implied by `h_draws` (and `q_draws` for triangular covariance).
    """
    if rng is None:
        rng = np.random.default_rng()

    if draws is not None:
        if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool):
            raise ValueError("draws must be an integer when provided")
        if int(draws) < 1:
            raise ValueError("draws must be >= 1")
        draws = int(draws)

    q_list = _parse_quantiles(quantile_levels)
    vars_out, order_idx = _parse_ordering(list(fit.dataset.variables), ordering)
    st_policy, st_tol, st_max = _parse_stationarity(
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
    )

    elb_enabled = bool(fit.model.elb is not None and getattr(fit.model.elb, "enabled", False))
    if use_latent is None:
        use_latent = elb_enabled
    if use_latent:
        if fit.latent_dataset is None:
            raise ValueError("use_latent=True requires fit.latent_dataset")
        dataset = fit.latent_dataset
    else:
        dataset = fit.dataset

    y_raw = np.asarray(dataset.values, dtype=float)
    if y_raw.ndim != 2:
        raise ValueError("dataset.values must be 2D")

    p = int(fit.model.p)
    t, n = y_raw.shape
    if t <= p:
        raise ValueError("dataset must have T > p for historical decomposition")
    if n != int(fit.dataset.N):
        raise ValueError("dataset.values has inconsistent number of variables")

    idx = np.asarray(order_idx, dtype=int)
    y = y_raw[:, idx]

    volatility_enabled = bool(
        fit.model.volatility is not None and getattr(fit.model.volatility, "enabled", False)
    )
    if volatility_enabled and shock_scale == "unit":
        raise ValueError("shock_scale='unit' is not supported for volatility models")

    metadata: dict[str, Any] = {
        "ordering": list(vars_out),
        "shock_scale": str(shock_scale),
        "stationarity": st_policy,
        "stationarity_tol": float(st_tol),
        "use_latent": bool(use_latent),
        "p": int(p),
    }

    if fit.beta_draws is not None:
        idx_draws = _select_draw_indices(
            fit=fit, draws=draws, stationarity=st_policy, stationarity_tol=st_tol, rng=rng
        )
        beta_draws = np.asarray(fit.beta_draws[idx_draws], dtype=float)
        sigma_draws = (
            np.asarray(fit.sigma_draws[idx_draws], dtype=float)
            if fit.sigma_draws is not None
            else None
        )
        h_draws = (
            np.asarray(fit.h_draws[idx_draws], dtype=float) if fit.h_draws is not None else None
        )
        q_draws = (
            np.asarray(fit.q_draws[idx_draws], dtype=float) if fit.q_draws is not None else None
        )
        lambda_draws = (
            np.asarray(fit.lambda_draws[idx_draws], dtype=float)
            if fit.lambda_draws is not None
            else None
        )
        h_factor_draws = (
            np.asarray(fit.h_factor_draws[idx_draws], dtype=float)
            if fit.h_factor_draws is not None
            else None
        )
        metadata["draw_source"] = "stored"
    else:
        if fit.posterior is None:
            raise ValueError("fit has no stored beta_draws and no posterior parameters")
        if volatility_enabled:
            raise ValueError(
                "volatility models require stored beta_draws for historical decomposition"
            )

        d_out = int(1000 if draws is None else draws)
        metadata["draw_source"] = "posterior"

        if st_policy == "allow":
            beta_draws, sigma_draws = sample_posterior_niw(
                mn=fit.posterior.mn,
                vn=fit.posterior.vn,
                sn=fit.posterior.sn,
                nun=fit.posterior.nun,
                draws=d_out,
                rng=rng,
            )
        else:
            max_total = int(50 * d_out) if st_max is None else int(st_max)
            accepted_beta: list[np.ndarray] = []
            accepted_sigma: list[np.ndarray] = []
            attempted = 0
            while len(accepted_beta) < d_out and attempted < max_total:
                need = int(d_out - len(accepted_beta))
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
                        n=n,
                        p=p,
                        include_intercept=fit.model.include_intercept,
                        tol=st_tol,
                    ):
                        accepted_beta.append(beta_b[i])
                        accepted_sigma.append(sigma_b[i])
                        if len(accepted_beta) == d_out:
                            break
            if len(accepted_beta) < d_out:
                raise ValueError(
                    f"stationarity='reject' could not generate {d_out} stationary draws within "
                    f"{max_total} candidate draws; try increasing stationarity_max_draws"
                )
            beta_draws = np.stack(accepted_beta)
            sigma_draws = np.stack(accepted_sigma)

        h_draws = None
        q_draws = None
        lambda_draws = None
        h_factor_draws = None

    d_used = int(beta_draws.shape[0])
    t_eff = int(t - p)

    baseline_draws = np.empty((d_used, t_eff, n), dtype=float)
    shock_draws = np.empty((d_used, t_eff, n), dtype=float)
    contrib_draws = np.empty((d_used, t_eff, n, n), dtype=float)

    max_abs_err = 0.0

    for d in range(d_used):
        beta_perm = _permute_beta(
            beta=beta_draws[d],
            order_idx=idx,
            n=n,
            p=p,
            include_intercept=fit.model.include_intercept,
        )
        a_mats = _ar_matrices_from_beta(
            beta=beta_perm, n=n, p=p, include_intercept=fit.model.include_intercept
        )
        c = beta_perm[0, :] if fit.model.include_intercept else np.zeros(n, dtype=float)

        x, y_tgt = design_matrix(y, p, include_intercept=fit.model.include_intercept)
        resid = y_tgt - x @ beta_perm  # (T-p, N)

        if not volatility_enabled:
            if sigma_draws is None:
                raise ValueError(
                    "missing covariance draws required for homoskedastic decomposition"
                )
            sigma = np.asarray(sigma_draws[d], dtype=float)[np.ix_(idx, idx)]
            impact0 = cholesky_jitter(sigma)
            if shock_scale == "unit":
                diag = np.diag(impact0)
                if np.any(diag <= 0):
                    raise ValueError(
                        "cannot use shock_scale='unit' when Cholesky diagonal is non-positive"
                    )
                impact0 = impact0 / diag.reshape(1, -1)
            eps = np.linalg.solve(impact0, resid.T).T
            impacts_t: np.ndarray | None = impact0[None, :, :]
        else:
            if h_draws is None:
                raise ValueError("missing volatility state draws required for decomposition")
            h = np.asarray(h_draws[d], dtype=float)
            if h.shape == (t, n):
                h_eff = h[p:, :]
            elif h.shape == (t_eff, n):
                h_eff = h
            else:
                raise ValueError("h_draws must have shape (T, N) or (T - p, N)")

            cov_mode = getattr(fit.model.volatility, "covariance", "diagonal")
            if cov_mode == "triangular":
                if q_draws is None:
                    raise ValueError("triangular volatility decomposition requires fit.q_draws")
                q = np.asarray(q_draws[d], dtype=float)
                if q.shape != (n, n):
                    raise ValueError("q_draws must have shape (N, N)")
                import scipy.linalg

                inv_q = scipy.linalg.solve_triangular(
                    q, np.eye(n, dtype=float), lower=False, check_finite=False
                )

                eps = np.empty_like(resid, dtype=float)
                impacts = np.empty((t_eff, n, n), dtype=float)
                for tt in range(t_eff):
                    ht = h_eff[tt, :]
                    d_sqrt = np.exp(0.5 * ht)
                    a = inv_q * d_sqrt.reshape(1, -1)
                    sigma_t = (a @ a.T)[np.ix_(idx, idx)]
                    l_t = cholesky_jitter(sigma_t)
                    impacts[tt] = l_t
                    eps[tt] = np.linalg.solve(l_t, resid[tt])
                impacts_t = impacts
            elif cov_mode == "factor":
                if lambda_draws is None or h_factor_draws is None:
                    raise ValueError(
                        "factor SV decomposition requires fit.lambda_draws and fit.h_factor_draws"
                    )
                lam = np.asarray(lambda_draws[d], dtype=float)
                if lam.ndim != 2:
                    raise ValueError("lambda_draws must have shape (D, N, k)")
                if lam.shape[0] != n:
                    raise ValueError("lambda_draws has wrong N dimension")
                k = int(lam.shape[1])

                hf = np.asarray(h_factor_draws[d], dtype=float)
                if hf.shape == (t, k):
                    hf_eff = hf[p:, :]
                elif hf.shape == (t_eff, k):
                    hf_eff = hf
                else:
                    raise ValueError("h_factor_draws must have shape (T, k) or (T - p, k)")

                lam_perm = lam[idx, :]  # permute to match y/resid ordering
                h_eta_perm = h_eff[:, idx]

                eps = np.empty_like(resid, dtype=float)
                impacts = np.empty((t_eff, n, n), dtype=float)
                for tt in range(t_eff):
                    sigma_t = (
                        lam_perm @ np.diag(np.exp(hf_eff[tt, :])) @ lam_perm.T
                        + np.diag(np.exp(h_eta_perm[tt, :]))
                    )
                    sigma_t = 0.5 * (sigma_t + sigma_t.T)
                    l_t = cholesky_jitter(sigma_t)
                    impacts[tt] = l_t
                    eps[tt] = np.linalg.solve(l_t, resid[tt])
                impacts_t = impacts
            else:
                scale = np.exp(0.5 * h_eff[:, idx])
                eps = resid / scale
                impacts = np.zeros((t_eff, n, n), dtype=float)
                for i in range(n):
                    impacts[:, i, i] = scale[:, i]
                impacts_t = impacts

        shock_draws[d] = eps

        baseline = np.empty((t, n), dtype=float)
        baseline[:p, :] = y[:p, :]
        for tt in range(p, t):
            acc = c.copy()
            for lag in range(1, p + 1):
                acc += a_mats[lag - 1] @ baseline[tt - lag]
            baseline[tt] = acc
        baseline_draws[d] = baseline[p:, :]

        contrib = np.zeros((t, n, n), dtype=float)
        for tt in range(p, t):
            acc = np.zeros((n, n), dtype=float)
            for lag in range(1, p + 1):
                acc += a_mats[lag - 1] @ contrib[tt - lag]
            if impacts_t is None:
                raise RuntimeError("internal error: impacts_t is None")
            if impacts_t.shape[0] == 1:
                imp = impacts_t[0]
            else:
                imp = impacts_t[tt - p]
            innov = imp * eps[tt - p].reshape(1, -1)
            contrib[tt] = acc + innov
        contrib_draws[d] = contrib[p:, :, :]

        recon = baseline[p:, :] + contrib[p:, :, :].sum(axis=2)
        err = float(np.max(np.abs(recon - y[p:, :])))
        max_abs_err = max(max_abs_err, err)

    mean = contrib_draws.mean(axis=0)
    quants = {q: np.quantile(contrib_draws, q=q, axis=0) for q in q_list}
    metadata["reconstruction_max_abs_error"] = float(max_abs_err)

    return HistoricalDecompositionResult(
        variables=list(vars_out),
        shocks=list(vars_out),
        time_index=dataset.time_index[p:],
        baseline_draws=baseline_draws,
        shock_draws=shock_draws,
        contributions_draws=contrib_draws,
        mean=mean,
        quantiles=quants,
        identification="cholesky",
        metadata=metadata,
    )
