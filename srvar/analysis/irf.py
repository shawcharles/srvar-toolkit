from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..bvar import sample_posterior_niw
from ..identification.sign_restrictions import parse_sign_restrictions, sample_sign_restricted_irf
from ..linalg import cholesky_jitter
from ..results import FitResult, IRFResult
from ..var import is_stationary


def _parse_horizons(horizons: int | Sequence[int]) -> list[int]:
    if isinstance(horizons, (int, np.integer)) and not isinstance(horizons, bool):
        hmax = int(horizons)
        if hmax < 0:
            raise ValueError("horizons must be >= 0")
        return list(range(hmax + 1))

    if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be an int or a sequence of ints")

    hs: list[int] = []
    for h in horizons:
        if not isinstance(h, (int, np.integer)) or isinstance(h, bool):
            raise ValueError("horizons must contain only integers")
        hi = int(h)
        if hi < 0:
            raise ValueError("horizons must contain only non-negative integers")
        hs.append(hi)

    if len(hs) < 1:
        raise ValueError("horizons must be non-empty")
    if len(set(hs)) != len(hs):
        raise ValueError("horizons must not contain duplicates")
    return sorted(hs)


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


def _parse_ordering(
    variables: list[str], ordering: Sequence[str] | None
) -> tuple[list[str], np.ndarray]:
    if ordering is None:
        return list(variables), np.arange(len(variables), dtype=int)
    if not isinstance(ordering, Sequence) or isinstance(ordering, (str, bytes)):
        raise ValueError("ordering must be a sequence of variable names")

    out = list(ordering)
    if len(out) != len(variables):
        raise ValueError("ordering must have length N (number of variables)")
    if len(set(out)) != len(out):
        raise ValueError("ordering must not contain duplicates")

    idx = np.empty(len(out), dtype=int)
    for i, name in enumerate(out):
        if name not in variables:
            raise ValueError(f"ordering contains unknown variable: {name}")
        idx[i] = int(variables.index(name))

    return out, idx


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


def _reduced_form_irf(*, a_mats: list[np.ndarray], horizons: list[int]) -> np.ndarray:
    n = int(a_mats[0].shape[0])
    p = int(len(a_mats))
    hmax = int(max(horizons))

    phi = np.zeros((hmax + 1, n, n), dtype=float)
    phi[0] = np.eye(n, dtype=float)

    for h in range(1, hmax + 1):
        acc = np.zeros((n, n), dtype=float)
        for lag in range(1, min(p, h) + 1):
            acc += a_mats[lag - 1] @ phi[h - lag]
        phi[h] = acc

    return phi[np.asarray(horizons, dtype=int)]


def _sigma0_from_components(
    *,
    sigma_draws: np.ndarray | None,
    h_draws: np.ndarray | None,
    q_draws: np.ndarray | None,
    draw_index: int,
) -> np.ndarray:
    if sigma_draws is not None:
        return np.asarray(sigma_draws[draw_index], dtype=float)

    if h_draws is None:
        raise ValueError("missing volatility state draws required to build covariance")
    h_last = np.asarray(h_draws[draw_index, -1, :], dtype=float).reshape(-1)
    if q_draws is None:
        return np.diag(np.exp(h_last))

    import scipy.linalg

    q = np.asarray(q_draws[draw_index], dtype=float)
    n = int(h_last.shape[0])
    inv_q = scipy.linalg.solve_triangular(
        q, np.eye(n, dtype=float), lower=False, check_finite=False
    )
    d_sqrt = np.exp(0.5 * h_last)
    a = inv_q * d_sqrt.reshape(1, -1)
    return a @ a.T


def _select_draw_indices(
    *,
    fit: FitResult,
    draws: int | None,
    stationarity: str,
    stationarity_tol: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if fit.beta_draws is None:
        raise ValueError("fit.beta_draws is required for selection")

    d_avail = int(fit.beta_draws.shape[0])
    if d_avail < 1:
        raise ValueError("fit.beta_draws is empty")

    if stationarity == "reject":
        stable = [
            is_stationary(
                fit.beta_draws[i],
                n=fit.dataset.N,
                p=fit.model.p,
                include_intercept=fit.model.include_intercept,
                tol=stationarity_tol,
            )
            for i in range(d_avail)
        ]
        stable_idx = np.flatnonzero(np.asarray(stable, dtype=bool))
        if stable_idx.size < 1:
            raise ValueError("no stationary coefficient draws available in fit.beta_draws")
        if draws is None:
            return stable_idx
        sel = rng.integers(0, stable_idx.size, size=int(draws))
        return stable_idx[sel]

    if draws is None:
        return np.arange(d_avail, dtype=int)
    return rng.integers(0, d_avail, size=int(draws))


def irf_reduced_form(
    fit: FitResult,
    *,
    horizons: int | Sequence[int] = 24,
    draws: int | None = None,
    ordering: Sequence[str] | None = None,
    quantile_levels: list[float] | None = None,
    stationarity: str = "allow",
    stationarity_tol: float = 1e-10,
    stationarity_max_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> IRFResult:
    """Compute reduced-form VAR impulse responses from posterior coefficient draws.

    Notes
    -----
    - Reduced-form IRFs do not require a covariance matrix. The impact response at horizon 0 is
      the identity matrix.
    - When `ordering` is provided, coefficient matrices are permuted into that ordering prior to
      computing IRFs.
    - When `stationarity="reject"`, unstable VAR draws are filtered/rejection-sampled.
    """
    if rng is None:
        rng = np.random.default_rng()

    h_list = _parse_horizons(horizons)
    q_list = _parse_quantiles(quantile_levels)
    vars_out, order_idx = _parse_ordering(list(fit.dataset.variables), ordering)
    st_policy, st_tol, st_max = _parse_stationarity(
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
    )

    if draws is not None:
        if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool):
            raise ValueError("draws must be an integer when provided")
        if int(draws) < 1:
            raise ValueError("draws must be >= 1")
        draws = int(draws)

    beta_draws: np.ndarray
    metadata: dict[str, Any] = {"stationarity": st_policy, "stationarity_tol": float(st_tol)}
    if fit.beta_draws is not None:
        idx = _select_draw_indices(
            fit=fit, draws=draws, stationarity=st_policy, stationarity_tol=st_tol, rng=rng
        )
        beta_draws = np.asarray(fit.beta_draws[idx], dtype=float)
        metadata["draw_source"] = "stored"
    else:
        if fit.posterior is None:
            raise ValueError("fit has no stored beta_draws and no posterior parameters")
        if fit.model.volatility is not None and fit.model.volatility.enabled:
            raise ValueError("reduced-form IRFs require stored beta_draws for volatility models")

        d_out = int(1000 if draws is None else draws)
        metadata["draw_source"] = "posterior"

        if st_policy == "allow":
            beta_draws, _sigma = sample_posterior_niw(
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
            attempted = 0
            while len(accepted_beta) < d_out and attempted < max_total:
                need = int(d_out - len(accepted_beta))
                batch = min(max(10, 2 * need), max_total - attempted)
                beta_b, _sigma_b = sample_posterior_niw(
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
                        p=fit.model.p,
                        include_intercept=fit.model.include_intercept,
                        tol=st_tol,
                    ):
                        accepted_beta.append(beta_b[i])
                        if len(accepted_beta) == d_out:
                            break
            if len(accepted_beta) < d_out:
                raise ValueError(
                    f"stationarity='reject' could not generate {d_out} stationary draws within "
                    f"{max_total} candidate draws; try increasing stationarity_max_draws or relaxing priors"
                )
            beta_draws = np.stack(accepted_beta)

    d_used = int(beta_draws.shape[0])
    n = int(fit.dataset.N)
    p = int(fit.model.p)
    irf_draws = np.empty((d_used, len(h_list), n, n), dtype=float)
    for d in range(d_used):
        a_mats = _ar_matrices_from_beta(
            beta=beta_draws[d], n=n, p=p, include_intercept=fit.model.include_intercept
        )
        a_mats = [a[np.ix_(order_idx, order_idx)] for a in a_mats]
        irf_draws[d] = _reduced_form_irf(a_mats=a_mats, horizons=h_list)

    mean = irf_draws.mean(axis=0)
    quants = {q: np.quantile(irf_draws, q=q, axis=0) for q in q_list}
    metadata.update({"ordering": list(vars_out)})

    return IRFResult(
        variables=list(vars_out),
        shocks=list(vars_out),
        horizons=list(h_list),
        draws=irf_draws,
        mean=mean,
        quantiles=quants,
        identification="reduced_form",
        metadata=metadata,
    )


def irf_cholesky(
    fit: FitResult,
    *,
    horizons: int | Sequence[int] = 24,
    draws: int | None = None,
    ordering: Sequence[str] | None = None,
    shock_scale: Literal["one_sd", "unit"] = "one_sd",
    quantile_levels: list[float] | None = None,
    stationarity: str = "allow",
    stationarity_tol: float = 1e-10,
    stationarity_max_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> IRFResult:
    """Compute Cholesky-identified structural IRFs.

    This function computes reduced-form IRFs for each posterior draw and then applies a Cholesky
    factorization of the contemporaneous covariance matrix to obtain orthogonal shocks.

    Volatility models
    -----------------
    For stochastic volatility models, the Cholesky factor is computed from the *last* latent
    covariance state in each draw (e.g. `diag(exp(h_T))`, or its triangular-covariance analogue).

    Shock scaling
    -------------
    - `"one_sd"`: a 1-s.d. structural shock (`Var(u_j)=1`), impact matrix is the Cholesky factor.
    - `"unit"`: normalize each shock so its own-variable impact at horizon 0 equals 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    h_list = _parse_horizons(horizons)
    q_list = _parse_quantiles(quantile_levels)
    vars_out, order_idx = _parse_ordering(list(fit.dataset.variables), ordering)
    st_policy, st_tol, st_max = _parse_stationarity(
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
    )

    if draws is not None:
        if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool):
            raise ValueError("draws must be an integer when provided")
        if int(draws) < 1:
            raise ValueError("draws must be >= 1")
        draws = int(draws)

    metadata: dict[str, Any] = {
        "stationarity": st_policy,
        "stationarity_tol": float(st_tol),
        "shock_scale": str(shock_scale),
    }

    beta_draws: np.ndarray
    sigma_draws: np.ndarray | None = None
    if fit.beta_draws is not None:
        idx = _select_draw_indices(
            fit=fit, draws=draws, stationarity=st_policy, stationarity_tol=st_tol, rng=rng
        )
        beta_draws = np.asarray(fit.beta_draws[idx], dtype=float)
        sigma_draws = (
            np.asarray(fit.sigma_draws[idx], dtype=float) if fit.sigma_draws is not None else None
        )
        h_draws = np.asarray(fit.h_draws[idx], dtype=float) if fit.h_draws is not None else None
        q_draws = np.asarray(fit.q_draws[idx], dtype=float) if fit.q_draws is not None else None
        metadata["draw_source"] = "stored"
    else:
        if fit.posterior is None:
            raise ValueError("fit has no stored beta_draws and no posterior parameters")
        if fit.model.volatility is not None and fit.model.volatility.enabled:
            raise ValueError("Cholesky IRFs require stored beta_draws for volatility models")

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
                        n=fit.dataset.N,
                        p=fit.model.p,
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
                    f"{max_total} candidate draws; try increasing stationarity_max_draws or relaxing priors"
                )
            beta_draws = np.stack(accepted_beta)
            sigma_draws = np.stack(accepted_sigma)

        h_draws = None
        q_draws = None

    d_used = int(beta_draws.shape[0])
    n = int(fit.dataset.N)
    p = int(fit.model.p)
    irf_draws = np.empty((d_used, len(h_list), n, n), dtype=float)

    for d in range(d_used):
        a_mats = _ar_matrices_from_beta(
            beta=beta_draws[d], n=n, p=p, include_intercept=fit.model.include_intercept
        )
        a_mats = [a[np.ix_(order_idx, order_idx)] for a in a_mats]

        red = _reduced_form_irf(a_mats=a_mats, horizons=h_list)  # (H, N, N)

        sigma0 = _sigma0_from_components(
            sigma_draws=sigma_draws,
            h_draws=h_draws,
            q_draws=q_draws,
            draw_index=d,
        )

        sigma0 = np.asarray(sigma0, dtype=float)[np.ix_(order_idx, order_idx)]
        impact = cholesky_jitter(sigma0)

        if shock_scale == "unit":
            diag = np.diag(impact)
            if np.any(diag <= 0):
                raise ValueError(
                    "cannot use shock_scale='unit' when Cholesky diagonal is non-positive"
                )
            impact = impact / diag.reshape(1, -1)

        irf_draws[d] = red @ impact

    mean = irf_draws.mean(axis=0)
    quants = {q: np.quantile(irf_draws, q=q, axis=0) for q in q_list}
    metadata.update({"ordering": list(vars_out)})

    return IRFResult(
        variables=list(vars_out),
        shocks=list(vars_out),
        horizons=list(h_list),
        draws=irf_draws,
        mean=mean,
        quantiles=quants,
        identification="cholesky",
        metadata=metadata,
    )


def irf_sign_restricted(
    fit: FitResult,
    *,
    horizons: int | Sequence[int] = 24,
    restrictions: dict[str, Any],
    max_attempts: int = 5000,
    min_accept: int = 1000,
    draws: int | None = None,
    ordering: Sequence[str] | None = None,
    quantile_levels: list[float] | None = None,
    stationarity: str = "allow",
    stationarity_tol: float = 1e-10,
    stationarity_max_draws: int | None = None,
    sign_tol: float = 0.0,
    zero_tol: float = 0.0,
    rng: np.random.Generator | None = None,
) -> IRFResult:
    """Compute sign-restricted structural IRFs using random orthonormal rotations.

    This uses the standard algorithm:
    1) For each posterior draw, compute reduced-form IRFs and a Cholesky factor `P` of the
       contemporaneous covariance.
    2) Draw random orthonormal rotations `Q` (Haar measure) and set the candidate impact matrix
       `B = P @ Q`.
    3) Accept the first rotation that satisfies the sign restrictions (within tolerances).

    Restriction schema
    ------------------
    `restrictions` is a nested dict:

    - top-level keys are shock names (in dict insertion order; shocks not listed are unrestricted)
    - second-level keys are variable names
    - leaf specs are either:
      - `{horizon: sign, ...}` where `sign` is "+", "-", or "0"
      - `{"sign": "+", "horizons": [0, 1, 2], "cumulative": false}`

    Horizon conventions
    -------------------
    Restriction horizons are IRF horizons (0 = impact response). Output `horizons` follow the same
    conventions as `irf_cholesky` and may be a full 0..H grid (when `horizons` is an int) or an
    explicit list.

    Notes
    -----
    - `"+"` / `"-"` require `sign * response >= sign_tol`.
    - `"0"` requires `abs(response) <= zero_tol` and can be infeasible in 1D.
    """
    if rng is None:
        rng = np.random.default_rng()

    h_list = _parse_horizons(horizons)
    q_list = _parse_quantiles(quantile_levels)
    vars_out, order_idx = _parse_ordering(list(fit.dataset.variables), ordering)
    st_policy, st_tol, st_max = _parse_stationarity(
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
    )

    if draws is not None:
        if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool):
            raise ValueError("draws must be an integer when provided")
        if int(draws) < 1:
            raise ValueError("draws must be >= 1")
        draws_out = int(draws)
    else:
        if not isinstance(min_accept, (int, np.integer)) or isinstance(min_accept, bool):
            raise ValueError("min_accept must be an integer when draws is not provided")
        if int(min_accept) < 1:
            raise ValueError("min_accept must be >= 1")
        draws_out = int(min_accept)

    if not isinstance(max_attempts, (int, np.integer)) or isinstance(max_attempts, bool):
        raise ValueError("max_attempts must be an integer")
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be >= 1")
    max_attempts = int(max_attempts)

    n = int(fit.dataset.N)
    p = int(fit.model.p)
    if n < 1:
        raise ValueError("fit.dataset.N must be >= 1")

    shock_names, parsed, has_cum = parse_sign_restrictions(
        restrictions, variables=vars_out, n_shocks=n
    )
    max_restr_h = 0
    if parsed:
        max_restr_h = int(max(int(h) for r in parsed for h in r.horizons.reshape(-1)))
    hmax_internal = int(max(max(h_list), max_restr_h))
    internal_h = list(range(hmax_internal + 1))

    metadata: dict[str, Any] = {
        "stationarity": st_policy,
        "stationarity_tol": float(st_tol),
        "max_attempts": int(max_attempts),
        "draws": int(draws_out),
        "sign_tol": float(sign_tol),
        "zero_tol": float(zero_tol),
        "ordering": list(vars_out),
        "restrictions": [
            {
                "shock": shock_names[int(r.shock)],
                "variable": vars_out[int(r.response)],
                "horizons": [int(h) for h in r.horizons.reshape(-1)],
                "sign": int(r.sign),
                "cumulative": bool(r.cumulative),
            }
            for r in parsed
        ],
    }

    max_total_draws = int(50 * draws_out if st_max is None else st_max)
    metadata["max_draw_attempts"] = int(max_total_draws)

    irf_draws = np.empty((draws_out, len(h_list), n, n), dtype=float)
    rotation_attempts: list[int] = []
    total_rotations = 0
    attempted_draws = 0
    failed_draws = 0
    last_err: Exception | None = None

    def _compute_red(beta: np.ndarray) -> np.ndarray:
        a_mats = _ar_matrices_from_beta(
            beta=beta, n=n, p=p, include_intercept=fit.model.include_intercept
        )
        a_mats = [a[np.ix_(order_idx, order_idx)] for a in a_mats]
        return _reduced_form_irf(a_mats=a_mats, horizons=internal_h)

    if fit.beta_draws is not None:
        idx_pool = _select_draw_indices(
            fit=fit, draws=None, stationarity=st_policy, stationarity_tol=st_tol, rng=rng
        )
        beta_pool = np.asarray(fit.beta_draws[idx_pool], dtype=float)
        sigma_pool = (
            np.asarray(fit.sigma_draws[idx_pool], dtype=float)
            if fit.sigma_draws is not None
            else None
        )
        h_pool = np.asarray(fit.h_draws[idx_pool], dtype=float) if fit.h_draws is not None else None
        q_pool = np.asarray(fit.q_draws[idx_pool], dtype=float) if fit.q_draws is not None else None
        metadata["draw_source"] = "stored"

        accepted = 0
        while accepted < draws_out and attempted_draws < max_total_draws:
            attempted_draws += 1
            pool_i = int(rng.integers(0, int(beta_pool.shape[0])))
            red = _compute_red(beta_pool[pool_i])
            sigma0 = _sigma0_from_components(
                sigma_draws=sigma_pool,
                h_draws=h_pool,
                q_draws=q_pool,
                draw_index=pool_i,
            )
            sigma0 = np.asarray(sigma0, dtype=float)[np.ix_(order_idx, order_idx)]
            impact = cholesky_jitter(sigma0)

            try:
                theta, _impact_rot, n_rot = sample_sign_restricted_irf(
                    reduced_irf=red,
                    impact=impact,
                    restrictions=parsed,
                    max_attempts=max_attempts,
                    rng=rng,
                    sign_tol=sign_tol,
                    zero_tol=zero_tol,
                    has_cumulative=has_cum,
                )
            except ValueError as e:
                failed_draws += 1
                last_err = e
                continue

            total_rotations += int(n_rot)
            rotation_attempts.append(int(n_rot))
            irf_draws[accepted] = theta[np.asarray(h_list, dtype=int)]
            accepted += 1

    else:
        if fit.posterior is None:
            raise ValueError("fit has no stored beta_draws and no posterior parameters")
        if fit.model.volatility is not None and fit.model.volatility.enabled:
            raise ValueError("sign-restricted IRFs require stored beta_draws for volatility models")

        metadata["draw_source"] = "posterior"
        accepted = 0
        stationarity_rejects = 0
        while accepted < draws_out and attempted_draws < max_total_draws:
            need = int(draws_out - accepted)
            batch = min(max(10, 2 * need), max_total_draws - attempted_draws)
            beta_b, sigma_b = sample_posterior_niw(
                mn=fit.posterior.mn,
                vn=fit.posterior.vn,
                sn=fit.posterior.sn,
                nun=fit.posterior.nun,
                draws=int(batch),
                rng=rng,
            )
            for i in range(int(batch)):
                attempted_draws += 1
                if st_policy == "reject":
                    if not is_stationary(
                        beta_b[i],
                        n=n,
                        p=p,
                        include_intercept=fit.model.include_intercept,
                        tol=st_tol,
                    ):
                        stationarity_rejects += 1
                        continue

                red = _compute_red(beta_b[i])
                sigma0 = np.asarray(sigma_b[i], dtype=float)[np.ix_(order_idx, order_idx)]
                impact = cholesky_jitter(sigma0)

                try:
                    theta, _impact_rot, n_rot = sample_sign_restricted_irf(
                        reduced_irf=red,
                        impact=impact,
                        restrictions=parsed,
                        max_attempts=max_attempts,
                        rng=rng,
                        sign_tol=sign_tol,
                        zero_tol=zero_tol,
                        has_cumulative=has_cum,
                    )
                except ValueError as e:
                    failed_draws += 1
                    last_err = e
                    continue

                total_rotations += int(n_rot)
                rotation_attempts.append(int(n_rot))
                irf_draws[accepted] = theta[np.asarray(h_list, dtype=int)]
                accepted += 1
                if accepted == draws_out:
                    break
        metadata["stationarity_rejects"] = int(stationarity_rejects)

    if rotation_attempts:
        metadata["rotation_attempts_mean"] = float(np.mean(rotation_attempts))
        metadata["rotation_attempts_max"] = int(np.max(rotation_attempts))
    metadata["attempted_draws"] = int(attempted_draws)
    metadata["accepted_draws"] = int(len(rotation_attempts))
    metadata["failed_draws"] = int(failed_draws)
    metadata["total_rotation_attempts"] = int(total_rotations)
    metadata["rotation_acceptance_rate"] = (
        float(len(rotation_attempts) / total_rotations) if total_rotations else float("nan")
    )

    if len(rotation_attempts) != draws_out:
        last_msg = f"; last_error={last_err}" if last_err is not None else ""
        raise ValueError(
            "could not generate the requested number of sign-restricted IRF draws: "
            f"accepted={len(rotation_attempts)}, requested={draws_out}, "
            f"attempted_draws={attempted_draws}, max_draw_attempts={max_total_draws}"
            f"{last_msg}"
        )

    mean = irf_draws.mean(axis=0)
    quants = {q: np.quantile(irf_draws, q=q, axis=0) for q in q_list}

    return IRFResult(
        variables=list(vars_out),
        shocks=list(shock_names),
        horizons=list(h_list),
        draws=irf_draws,
        mean=mean,
        quantiles=quants,
        identification="sign_restricted",
        metadata=metadata,
    )
