import numpy as np
import scipy.stats

from srvar.elb import apply_elb_floor, sample_shadow_value, truncnorm_rvs_upper


def test_truncnorm_upper_respects_bound() -> None:
    rng = np.random.default_rng(0)
    samples = [truncnorm_rvs_upper(mean=0.0, sd=1.0, upper=-0.5, rng=rng) for _ in range(200)]
    assert max(samples) <= -0.5 + 1e-12


def test_apply_elb_floor_works_for_forecast_draws() -> None:
    sims = np.array(
        [
            [[-1.0, 1.0], [0.1, 2.0]],
            [[-0.2, 3.0], [-0.5, 4.0]],
        ],
        dtype=float,
    )
    floored = apply_elb_floor(sims, bound=0.0, indices=[0])
    assert floored.shape == sims.shape
    assert np.all(floored[:, :, 0] >= 0.0)


def test_sample_shadow_value_matches_truncated_normal_when_no_future_terms() -> None:
    rng = np.random.default_rng(1234)

    p = 1
    include_intercept = True

    # Set t as the last observation so there are no future likelihood terms.
    t = 1
    j = 0

    y = np.array(
        [
            [0.2, -0.1, 0.4],
            [0.0, 0.3, -0.2],
        ],
        dtype=float,
    )

    beta = np.array(
        [
            [0.10, -0.05, 0.20],
            [0.30, 0.00, -0.10],
            [0.00, 0.20, 0.05],
            [-0.15, 0.10, 0.25],
        ],
        dtype=float,
    )

    sigma = np.array(
        [
            [0.30, 0.05, 0.02],
            [0.05, 0.25, 0.01],
            [0.02, 0.01, 0.20],
        ],
        dtype=float,
    )

    x_t = np.concatenate([np.array([1.0], dtype=float), y[t - 1]])
    mu_t = x_t @ beta

    others = [1, 2]
    sigma_12 = sigma[j, others]
    sigma_22 = sigma[np.ix_(others, others)]
    inv_sigma_22 = np.linalg.inv(sigma_22)
    mean_cond = float(mu_t[j] + sigma_12 @ inv_sigma_22 @ (y[t, others] - mu_t[others]))
    var_cond = float(sigma[j, j] - sigma_12 @ inv_sigma_22 @ sigma_12.T)
    sd_cond = float(np.sqrt(var_cond))

    upper = mean_cond - 0.5 * sd_cond

    b = (upper - mean_cond) / sd_cond
    theory_mean = float(scipy.stats.truncnorm.mean(a=-np.inf, b=b, loc=mean_cond, scale=sd_cond))
    theory_var = float(scipy.stats.truncnorm.var(a=-np.inf, b=b, loc=mean_cond, scale=sd_cond))

    draws = np.array(
        [
            sample_shadow_value(
                y=y,
                t=t,
                j=j,
                p=p,
                beta=beta,
                sigma=sigma,
                upper=upper,
                include_intercept=include_intercept,
                rng=rng,
            )
            for _ in range(5000)
        ],
        dtype=float,
    )

    assert np.all(np.isfinite(draws))
    assert float(np.max(draws)) <= upper + 1e-10

    emp_mean = float(draws.mean())
    emp_var = float(draws.var())

    assert abs(emp_mean - theory_mean) < 0.02
    assert abs(emp_var - theory_var) < 0.02
