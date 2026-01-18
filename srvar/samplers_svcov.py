from __future__ import annotations

import numpy as np
import scipy.linalg

from .bvar import posterior_niw
from .data.dataset import Dataset
from .elb import sample_shadow_value_svcov
from .linalg import cholesky_jitter, solve_psd, symmetrize
from .results import FitResult
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .sv import (
    log_e2_star,
    sample_ar1_params,
    sample_h0,
    sample_h0_ar1,
    sample_h_ar1,
    sample_h_svrw,
    sample_sigma_eta2,
    sample_sigma_eta2_ar1,
)
from .var import design_matrix


def _sample_beta_triangular_svrw(
    *,
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    h: np.ndarray,
    m0: np.ndarray,
    v0: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Sample VAR coefficients under triangular SV covariance.

    Model:
        Y = X B + V
        Q V' = E' ,   E_t ~ N(0, diag(exp(h_t)))

    where Q is upper-triangular with ones on the diagonal. Conditional on (Q, h),
    columns of B can be sampled sequentially from last to first.
    """
    xt = np.asarray(x, dtype=float)
    yt = np.asarray(y, dtype=float)
    qt = np.asarray(q, dtype=float)
    ht = np.asarray(h, dtype=float)
    m0t = np.asarray(m0, dtype=float)
    v0t = np.asarray(v0, dtype=float)

    if xt.ndim != 2 or yt.ndim != 2:
        raise ValueError("x and y must be 2D")
    if ht.ndim != 2 or ht.shape != yt.shape:
        raise ValueError("h must have shape (T, N) matching y")

    t_eff, k = xt.shape
    _t2, n = yt.shape

    if qt.shape != (n, n):
        raise ValueError("q must have shape (N, N)")
    if m0t.shape != (k, n):
        raise ValueError("m0 must have shape (K, N)")
    if v0t.shape != (k, k):
        raise ValueError("v0 must have shape (K, K)")
    if jitter <= 0 or not np.isfinite(jitter):
        raise ValueError("jitter must be positive")

    inv_v0 = solve_psd(v0t, np.eye(k, dtype=float))
    beta = np.empty((k, n), dtype=float)

    # work backwards because y_star[:, i] depends on columns i..N-1
    for i in range(n - 1, -1, -1):
        y_star_i = yt[:, i].copy()
        for j in range(i + 1, n):
            qij = float(qt[i, j])
            if qij != 0.0:
                y_star_i += qij * yt[:, j]

        offset = np.zeros(t_eff, dtype=float)
        for j in range(i + 1, n):
            qij = float(qt[i, j])
            if qij != 0.0:
                offset += qij * (xt @ beta[:, j])
        y_tilde = y_star_i - offset

        w = np.exp(-ht[:, i])
        xtwx = xt.T @ (w[:, None] * xt)
        ktheta = symmetrize(inv_v0 + xtwx + jitter * np.eye(k, dtype=float))

        rhs = inv_v0 @ m0t[:, i] + xt.T @ (w * y_tilde)
        thetahat = solve_psd(ktheta, rhs)

        chol = cholesky_jitter(ktheta)
        z = rng.standard_normal(k)
        theta = thetahat + scipy.linalg.solve_triangular(chol.T, z, lower=False, check_finite=False)
        beta[:, i] = theta

    return beta


def _update_q_triangular(
    *,
    v: np.ndarray,
    h: np.ndarray,
    q_prior_var: float,
    rng: np.random.Generator,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Sample Q (upper-triangular, ones diagonal) given residuals and log-variances."""
    vt = np.asarray(v, dtype=float)
    ht = np.asarray(h, dtype=float)
    if vt.ndim != 2 or ht.ndim != 2 or vt.shape != ht.shape:
        raise ValueError("v and h must be 2D and have the same shape")
    if q_prior_var <= 0 or not np.isfinite(q_prior_var):
        raise ValueError("q_prior_var must be positive and finite")
    if jitter <= 0 or not np.isfinite(jitter):
        raise ValueError("jitter must be positive and finite")

    t_eff, n = vt.shape
    q = np.eye(n, dtype=float)
    prior_prec = 1.0 / float(q_prior_var)

    for i in range(n - 1):
        z = vt[:, i + 1 :]
        if z.shape[1] == 0:
            continue
        y_i = vt[:, i]

        w = np.exp(-ht[:, i])
        kq = z.T @ (w[:, None] * z)
        kq = symmetrize(kq + (prior_prec + jitter) * np.eye(kq.shape[0], dtype=float))

        rhs = -(z.T @ (w * y_i))
        mean = solve_psd(kq, rhs)

        chol = cholesky_jitter(kq)
        u = rng.standard_normal(z.shape[1])
        delta = scipy.linalg.solve_triangular(chol.T, u, lower=False, check_finite=False)
        q[i, i + 1 :] = mean + delta

    return q


def _fit_svcov(
    *,
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    rng: np.random.Generator,
) -> FitResult:
    vol = model.volatility
    if vol is None or not vol.enabled:
        raise ValueError("volatility must be enabled")
    if vol.covariance != "triangular":
        raise ValueError("volatility.covariance must be 'triangular' for _fit_svcov")
    ar1 = bool(vol.dynamics == "ar1")

    if model.steady_state is not None:
        raise ValueError("steady_state is not yet supported with triangular SV covariance")

    prior_family = prior.family.lower()
    if prior_family != "niw":
        raise ValueError("triangular SV covariance currently supports only prior.family='niw'")

    applies_to_idx: list[int] = []
    elb_t_idx: dict[int, np.ndarray] = {}

    y_lat = dataset.values.copy()
    if model.elb is not None and model.elb.enabled:
        for name in model.elb.applies_to:
            try:
                applies_to_idx.append(dataset.variables.index(name))
            except ValueError as e:
                raise ValueError(f"elb.applies_to contains unknown variable: {name}") from e

        for j in applies_to_idx:
            mask = dataset.values[:, j] <= (model.elb.bound + model.elb.tol)
            elb_t_idx[j] = np.where(mask)[0]
            y_lat[mask, j] = model.elb.bound - model.elb.init_offset

    x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)
    t_eff, n = y.shape

    niw = prior.niw
    if niw.m0.shape != (x.shape[1], n):
        raise ValueError("prior.niw.m0 has incompatible shape for dataset/model")
    if niw.v0.shape != (x.shape[1], x.shape[1]):
        raise ValueError("prior.niw.v0 has incompatible shape for dataset/model")

    mn, _vn, _sn, _nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
    beta = mn.copy()

    q = np.eye(n, dtype=float)

    resid0 = y - x @ beta
    h0 = np.log(np.var(resid0, axis=0) + 1e-12)
    h = np.tile(h0.reshape(1, -1), (t_eff, 1))
    sigma_eta2 = 0.05 * np.ones(n, dtype=float)
    gamma0 = (1.0 - float(vol.phi_prior_mean)) * h0 if ar1 else None
    phi = np.full(n, float(vol.phi_prior_mean), dtype=float) if ar1 else None

    beta_keep: list[np.ndarray] = []
    h_keep: list[np.ndarray] = []
    h0_keep: list[np.ndarray] = []
    sigma_eta2_keep: list[np.ndarray] = []
    gamma0_keep: list[np.ndarray] = []
    phi_keep: list[np.ndarray] = []
    q_keep: list[np.ndarray] = []
    y_lat_keep: list[np.ndarray] | None = (
        [] if (model.elb is not None and model.elb.enabled) else None
    )

    for it in range(sampler.draws):
        x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        beta = _sample_beta_triangular_svrw(
            x=x,
            y=y,
            q=q,
            h=h,
            m0=niw.m0,
            v0=niw.v0,
            rng=rng,
        )

        if model.elb is not None and model.elb.enabled:
            h_full = np.vstack([np.tile(h0.reshape(1, -1), (model.p, 1)), h])
            for j in applies_to_idx:
                for t in elb_t_idx[j]:
                    y_lat[t, j] = sample_shadow_value_svcov(
                        y=y_lat,
                        h=h_full,
                        q=q,
                        t=int(t),
                        j=int(j),
                        p=model.p,
                        beta=beta,
                        upper=model.elb.bound,
                        include_intercept=model.include_intercept,
                        rng=rng,
                    )

            x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        v = y - x @ beta

        q = _update_q_triangular(v=v, h=h, q_prior_var=vol.q_prior_var, rng=rng)

        eps = v @ q.T
        for i in range(n):
            y_star = log_e2_star(eps[:, i], epsilon=vol.epsilon)
            if ar1:
                if gamma0 is None or phi is None:
                    raise RuntimeError("AR(1) volatility state missing")
                h[:, i] = sample_h_ar1(
                    y_star=y_star,
                    h=h[:, i],
                    sigma_eta2=float(sigma_eta2[i]),
                    h0=float(h0[i]),
                    gamma0=float(gamma0[i]),
                    phi=float(phi[i]),
                    rng=rng,
                )
                h0[i] = sample_h0_ar1(
                    h1=float(h[0, i]),
                    sigma_eta2=float(sigma_eta2[i]),
                    gamma0=float(gamma0[i]),
                    phi=float(phi[i]),
                    prior_mean=vol.h0_prior_mean,
                    prior_var=vol.h0_prior_var,
                    rng=rng,
                )
                g0_i, phi_i = sample_ar1_params(
                    h=h[:, i],
                    h0=float(h0[i]),
                    sigma_eta2=float(sigma_eta2[i]),
                    phi_prior_mean=float(vol.phi_prior_mean),
                    phi_prior_var=float(vol.phi_prior_var),
                    gamma0_prior_mean=float(vol.gamma0_prior_mean),
                    gamma0_prior_var=float(vol.gamma0_prior_var),
                    rng=rng,
                )
                gamma0[i] = float(g0_i)
                phi[i] = float(phi_i)
                sigma_eta2[i] = sample_sigma_eta2_ar1(
                    h=h[:, i],
                    h0=float(h0[i]),
                    gamma0=float(gamma0[i]),
                    phi=float(phi[i]),
                    nu0=vol.sigma_eta_prior_nu0,
                    s0=vol.sigma_eta_prior_s0,
                    rng=rng,
                )
            else:
                h[:, i] = sample_h_svrw(
                    y_star=y_star,
                    h=h[:, i],
                    sigma_eta2=float(sigma_eta2[i]),
                    h0=float(h0[i]),
                    rng=rng,
                )
                h0[i] = sample_h0(
                    h1=float(h[0, i]),
                    sigma_eta2=float(sigma_eta2[i]),
                    prior_mean=vol.h0_prior_mean,
                    prior_var=vol.h0_prior_var,
                    rng=rng,
                )
                sigma_eta2[i] = sample_sigma_eta2(
                    h=h[:, i],
                    h0=float(h0[i]),
                    nu0=vol.sigma_eta_prior_nu0,
                    s0=vol.sigma_eta_prior_s0,
                    rng=rng,
                )

        if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
            beta_keep.append(beta.copy())
            h_keep.append(h.copy())
            h0_keep.append(h0.copy())
            sigma_eta2_keep.append(sigma_eta2.copy())
            if ar1 and gamma0 is not None and phi is not None:
                gamma0_keep.append(gamma0.copy())
                phi_keep.append(phi.copy())
            q_keep.append(q.copy())
            if y_lat_keep is not None:
                y_lat_keep.append(y_lat.copy())

    latent_dataset = None
    if model.elb is not None and model.elb.enabled:
        latent_dataset = Dataset.from_arrays(
            values=y_lat, variables=dataset.variables, time_index=dataset.time_index
        )

    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        latent_dataset=latent_dataset,
        latent_draws=np.stack(y_lat_keep) if y_lat_keep else None,
        beta_draws=np.stack(beta_keep) if beta_keep else None,
        sigma_draws=None,
        q_draws=np.stack(q_keep) if q_keep else None,
        h_draws=np.stack(h_keep) if h_keep else None,
        h0_draws=np.stack(h0_keep) if h0_keep else None,
        sigma_eta2_draws=np.stack(sigma_eta2_keep) if sigma_eta2_keep else None,
        sv_gamma0_draws=np.stack(gamma0_keep) if gamma0_keep else None,
        sv_phi_draws=np.stack(phi_keep) if phi_keep else None,
    )
