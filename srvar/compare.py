from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .api import fit, forecast
from .artifacts import load_run_dir, save_fit_npz, save_forecast_npz
from .config import _prepare_from_config, build_backtest_config, build_prior, load_config
from .data.dataset import Dataset
from .results import FitResult, ForecastResult
from .spec import MinnesotaCanonicalSpec, NIWPrior, PriorSpec
from .var import companion_matrix


@dataclass(frozen=True, slots=True)
class MinnesotaBacktestComparison:
    out_root: Path
    baseline_config: Path
    candidate_config: Path
    baseline_out_dir: Path
    candidate_out_dir: Path
    comparison_csv: Path
    summary_json: Path


@dataclass(frozen=True, slots=True)
class MinnesotaOriginDiagnostic:
    out_root: Path
    baseline_config: Path
    candidate_config: Path
    baseline_out_dir: Path
    candidate_out_dir: Path
    metadata_json: Path
    state_csv: Path
    forecast_csv: Path
    beta_csv: Path


@dataclass(frozen=True, slots=True)
class MinnesotaPriorScaleDiagnostic:
    out_root: Path
    baseline_config: Path
    candidate_config: Path
    metadata_json: Path
    summary_csv: Path


@dataclass(frozen=True, slots=True)
class MinnesotaTemperedOriginExperiment:
    out_root: Path
    metadata_json: Path
    forecast_csv: Path
    state_csv: Path
    beta_csv: Path
    baseline_dir: Path
    canonical_dir: Path
    tempered_dir: Path


def _require_pyyaml() -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for Minnesota backtest comparison. "
            "Install with 'srvar-toolkit[cli]'."
        ) from exc
    return yaml


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    yaml = _require_pyyaml()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return raw


def _write_yaml_mapping(path: str | Path, data: dict[str, Any]) -> None:
    yaml = _require_pyyaml()
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _build_minnesota_variant_config(
    cfg: dict[str, Any],
    *,
    method: Literal[
        "minnesota",
        "minnesota_legacy",
        "minnesota_canonical",
        "minnesota_tempered",
    ],
    save_forecasts: bool | None = None,
) -> dict[str, Any]:
    variant = copy.deepcopy(cfg)
    prior = variant.get("prior")
    if not isinstance(prior, dict):
        raise ValueError("config must define prior as a mapping")
    family = str(prior.get("family", "")).lower()
    if family != "niw":
        raise ValueError("Minnesota comparison requires prior.family='niw'")
    if "backtest" not in variant:
        raise ValueError("Minnesota comparison requires a backtest config")
    prior["method"] = str(method)
    prior.setdefault("minnesota", {})
    if save_forecasts is not None:
        output = variant.setdefault("output", {})
        if not isinstance(output, dict):
            raise ValueError("config output must be a mapping when present")
        output["save_forecasts"] = bool(save_forecasts)
    return variant


def _regressor_names(*, variables: list[str], p: int, include_intercept: bool) -> list[str]:
    names: list[str] = []
    if include_intercept:
        names.append("const")
    for lag in range(1, int(p) + 1):
        for variable in variables:
            names.append(f"{variable}_lag{lag}")
    return names


def _resolve_backtest_origins(ds_full: Dataset, bt: dict[str, Any]) -> tuple[list[int], int, int]:
    horizons = list(bt["horizons"])
    max_h = int(max(horizons))
    if ds_full.T <= max_h:
        raise ValueError("dataset is too short for requested backtest horizons")

    first_origin_end = int(bt["min_obs"]) - 1
    last_origin_end = ds_full.T - max_h - 1
    if last_origin_end < first_origin_end:
        raise ValueError("backtest settings imply zero feasible forecast origins")

    origin_start = bt.get("origin_start")
    origin_end = bt.get("origin_end")
    if origin_start is not None or origin_end is not None:
        if not isinstance(ds_full.time_index, pd.DatetimeIndex):
            raise ValueError("backtest.origin_start/end requires a datetime index on the dataset")

        if origin_start is not None:
            ts = pd.to_datetime(origin_start)
            try:
                start_i = int(ds_full.time_index.get_loc(ts))
            except KeyError as exc:
                raise ValueError(
                    f"backtest.origin_start not found in dataset index: {origin_start}"
                ) from exc
            first_origin_end = max(first_origin_end, start_i)

        if origin_end is not None:
            ts = pd.to_datetime(origin_end)
            try:
                end_i = int(ds_full.time_index.get_loc(ts))
            except KeyError as exc:
                raise ValueError(
                    f"backtest.origin_end not found in dataset index: {origin_end}"
                ) from exc
            last_origin_end = min(last_origin_end, end_i)

        if last_origin_end < first_origin_end:
            raise ValueError("backtest.origin_start/end implies zero feasible forecast origins")

    origins = list(range(first_origin_end, last_origin_end + 1, int(bt["step"])))
    if not origins:
        raise ValueError("backtest settings imply zero scheduled forecast origins")
    return origins, first_origin_end, last_origin_end


def _resolve_origin_end_index(
    ds_full: Dataset,
    bt: dict[str, Any],
    *,
    origin_index: int | None = None,
    origin_date: str | None = None,
) -> int:
    origins, _first_origin_end, _last_origin_end = _resolve_backtest_origins(ds_full, bt)
    allowed = set(int(i) for i in origins)

    if origin_index is not None and origin_date is not None:
        raise ValueError("specify at most one of origin_index and origin_date")

    if origin_date is not None:
        if not isinstance(ds_full.time_index, pd.DatetimeIndex):
            raise ValueError("origin_date requires a datetime index on the dataset")
        ts = pd.to_datetime(origin_date)
        try:
            resolved = int(ds_full.time_index.get_loc(ts))
        except KeyError as exc:
            raise ValueError(f"origin_date not found in dataset index: {origin_date}") from exc
    elif origin_index is not None:
        resolved = int(origin_index)
    else:
        resolved = int(origins[-1])

    if resolved not in allowed:
        raise ValueError(
            "requested origin is not a scheduled forecast origin under the backtest config"
        )
    return resolved


def _build_train_dataset(
    ds_full: Dataset,
    *,
    mode: str,
    window: int | None,
    origin_end_i: int,
) -> tuple[int, int, Dataset]:
    if str(mode) == "expanding":
        train_start = 0
    else:
        if window is None:
            raise ValueError("rolling backtest mode requires a finite window")
        train_start = max(0, int(origin_end_i - int(window) + 1))
    train_end_excl = int(origin_end_i + 1)
    train_values = ds_full.values[train_start:train_end_excl, :]
    train_index = ds_full.time_index[train_start:train_end_excl]
    train_ds = Dataset.from_arrays(
        values=train_values,
        variables=ds_full.variables,
        time_index=train_index,
    )
    return train_start, train_end_excl, train_ds


def _select_variables(ds_full: Dataset, variables: list[str] | None) -> list[str]:
    if variables is None:
        return list(ds_full.variables)
    selected = [str(variable) for variable in variables]
    missing = [variable for variable in selected if variable not in ds_full.variables]
    if missing:
        raise ValueError(f"variables not found in dataset: {missing}")
    return selected


def _select_horizons(bt: dict[str, Any], horizons: list[int] | None) -> list[int]:
    available = [int(h) for h in bt["horizons"]]
    if horizons is None:
        return available
    selected = [int(h) for h in horizons]
    invalid = [h for h in selected if h not in available]
    if invalid:
        raise ValueError(f"horizons not present in backtest config: {invalid}")
    return selected


def _parse_variable_regressor_cases(cases: list[str] | None) -> list[tuple[str, str]] | None:
    if cases is None:
        return None
    parsed: list[tuple[str, str]] = []
    for case in cases:
        raw = str(case).strip()
        variable, sep, regressor = raw.partition(":")
        if sep != ":" or not variable or not regressor:
            raise ValueError(
                f"invalid case {case!r}; expected format VARIABLE:REGRESSOR, "
                "for example EXUSUK:HOUST_lag1"
            )
        parsed.append((variable, regressor))
    if not parsed:
        raise ValueError("case filter produced zero requested cases")
    return parsed


def _select_regressors(
    fit_res: FitResult,
    *,
    regressors: list[str] | None = None,
) -> list[str]:
    available = _regressor_names(
        variables=list(fit_res.dataset.variables),
        p=int(fit_res.model.p),
        include_intercept=bool(fit_res.model.include_intercept),
    )
    if regressors is None:
        return available
    selected = [str(regressor) for regressor in regressors]
    missing = [regressor for regressor in selected if regressor not in available]
    if missing:
        raise ValueError(f"regressors not found in fit result: {missing}")
    return selected


def _parse_regressor_name(regressor: str) -> tuple[str | None, int | None]:
    if regressor == "const":
        return None, None
    predictor, sep, lag_s = str(regressor).rpartition("_lag")
    if sep != "_lag" or not predictor or not lag_s:
        raise ValueError(f"unable to parse regressor name: {regressor}")
    return predictor, int(lag_s)


def _prior_variance_matrix(
    prior: Any,
    *,
    variables: list[str],
    p: int,
    include_intercept: bool,
) -> np.ndarray:
    regressor_names = _regressor_names(
        variables=variables,
        p=p,
        include_intercept=include_intercept,
    )
    k = len(regressor_names)
    n = len(variables)
    if prior.minnesota_canonical is not None:
        inv_v0 = np.asarray(prior.minnesota_canonical.inv_v0_vec, dtype=float).reshape(
            (k, n),
            order="F",
        )
        return 1.0 / inv_v0

    diag = np.diag(np.asarray(prior.niw.v0, dtype=float))
    if diag.shape != (k,):
        raise ValueError("prior.niw.v0 has incompatible shape for the requested model structure")
    return np.repeat(diag.reshape(-1, 1), repeats=n, axis=1)


def _state_frame(fit_res: FitResult, *, variables: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = int(fit_res.dataset.N)
    p = int(fit_res.model.p)
    base = 1 if fit_res.model.include_intercept else 0

    for variable in variables:
        idx = int(fit_res.dataset.variables.index(variable))
        row: dict[str, object] = {
            "variable": variable,
            "last_observed": float(fit_res.dataset.values[-1, idx]),
        }
        if fit_res.latent_dataset is not None:
            row["last_latent_observed"] = float(fit_res.latent_dataset.values[-1, idx])

        if fit_res.h_draws is not None:
            last_h = np.asarray(fit_res.h_draws[:, -1, idx], dtype=float)
            last_cond_sd = np.exp(0.5 * last_h)
            row["last_log_vol_mean"] = float(np.mean(last_h))
            row["last_log_vol_sd"] = float(np.std(last_h, ddof=0))
            row["last_cond_sd_mean"] = float(np.mean(last_cond_sd))
            row["last_cond_sd_sd"] = float(np.std(last_cond_sd, ddof=0))

        if fit_res.sigma_eta2_draws is not None:
            sigma_eta2 = np.asarray(fit_res.sigma_eta2_draws[:, idx], dtype=float)
            row["sigma_eta2_mean"] = float(np.mean(sigma_eta2))
            row["sigma_eta2_sd"] = float(np.std(sigma_eta2, ddof=0))

        if fit_res.sv_gamma0_draws is not None:
            gamma0 = np.asarray(fit_res.sv_gamma0_draws[:, idx], dtype=float)
            row["sv_gamma0_mean"] = float(np.mean(gamma0))
            row["sv_gamma0_sd"] = float(np.std(gamma0, ddof=0))

        if fit_res.sv_phi_draws is not None:
            phi = np.asarray(fit_res.sv_phi_draws[:, idx], dtype=float)
            row["sv_phi_mean"] = float(np.mean(phi))
            row["sv_phi_sd"] = float(np.std(phi, ddof=0))

        if fit_res.beta_draws is not None:
            beta = np.asarray(fit_res.beta_draws[:, :, idx], dtype=float)
            if fit_res.model.include_intercept:
                intercept = beta[:, 0]
                row["const_mean"] = float(np.mean(intercept))
                row["const_sd"] = float(np.std(intercept, ddof=0))
            own_indices = [base + (lag - 1) * n + idx for lag in range(1, p + 1)]
            own_lag_sum = np.sum(beta[:, own_indices], axis=1)
            row["own_lag1_mean"] = float(np.mean(beta[:, own_indices[0]]))
            row["own_lag1_sd"] = float(np.std(beta[:, own_indices[0]], ddof=0))
            row["own_lag_sum_mean"] = float(np.mean(own_lag_sum))
            row["own_lag_sum_sd"] = float(np.std(own_lag_sum, ddof=0))

        rows.append(row)

    return pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)


def _beta_frame(fit_res: FitResult, *, variables: list[str]) -> pd.DataFrame:
    if fit_res.beta_draws is None:
        raise ValueError("fit result does not contain beta_draws")

    regressor_names = _regressor_names(
        variables=list(fit_res.dataset.variables),
        p=int(fit_res.model.p),
        include_intercept=bool(fit_res.model.include_intercept),
    )
    rows: list[dict[str, object]] = []
    for variable in variables:
        idx = int(fit_res.dataset.variables.index(variable))
        beta = np.asarray(fit_res.beta_draws[:, :, idx], dtype=float)
        beta_mean = np.mean(beta, axis=0)
        beta_sd = np.std(beta, axis=0, ddof=0)
        for regressor, mean_value, sd_value in zip(
            regressor_names, beta_mean, beta_sd, strict=True
        ):
            rows.append(
                {
                    "variable": variable,
                    "regressor": str(regressor),
                    "beta_mean": float(mean_value),
                    "beta_sd": float(sd_value),
                }
            )
    return pd.DataFrame(rows).sort_values(["variable", "regressor"]).reset_index(drop=True)


def build_fit_coefficient_detail(
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    *,
    variables: list[str] | None = None,
    regressors: list[str] | None = None,
    cases: list[str] | None = None,
    allow_legacy_pickle: bool = False,
) -> pd.DataFrame:
    """Build a long-format coefficient-draw table for paired fit runs.

    Parameters
    ----------
    baseline_run_dir, candidate_run_dir:
        Run directories containing `config.yml` and `fit_result.npz`.
    variables, regressors:
        Optional broad filters applied before case filtering.
    cases:
        Optional `VARIABLE:REGRESSOR` filters. When provided, only those exact
        equation/regressor pairs are returned.
    allow_legacy_pickle:
        Set only for trusted pre-migration artifacts. This may execute pickle code.
    """
    fit_b = load_run_dir(baseline_run_dir, allow_legacy_pickle=allow_legacy_pickle)
    fit_c = load_run_dir(candidate_run_dir, allow_legacy_pickle=allow_legacy_pickle)

    if fit_b.dataset.variables != fit_c.dataset.variables:
        raise ValueError("fit result variable lists do not match")
    if (
        fit_b.model.p != fit_c.model.p
        or fit_b.model.include_intercept != fit_c.model.include_intercept
    ):
        raise ValueError("fit result model structures do not match")
    if fit_b.beta_draws is None or fit_c.beta_draws is None:
        raise ValueError("both fit results must contain beta_draws")

    selected_variables = _select_variables(fit_b.dataset, variables)
    selected_regressors = _select_regressors(fit_b, regressors=regressors)
    case_filter = _parse_variable_regressor_cases(cases)

    regressor_names = _regressor_names(
        variables=list(fit_b.dataset.variables),
        p=int(fit_b.model.p),
        include_intercept=bool(fit_b.model.include_intercept),
    )
    regressor_to_idx = {name: idx for idx, name in enumerate(regressor_names)}

    rows: list[dict[str, object]] = []
    for method_name, fit_res in (("baseline", fit_b), ("candidate", fit_c)):
        beta_draws = np.asarray(fit_res.beta_draws, dtype=float)
        for variable in selected_variables:
            v_idx = int(fit_res.dataset.variables.index(variable))
            for regressor in selected_regressors:
                if case_filter is not None and (variable, regressor) not in case_filter:
                    continue
                r_idx = int(regressor_to_idx[regressor])
                draws = beta_draws[:, r_idx, v_idx]
                for draw_idx, value in enumerate(draws):
                    rows.append(
                        {
                            "method": method_name,
                            "draw": int(draw_idx),
                            "variable": variable,
                            "regressor": regressor,
                            "value": float(value),
                        }
                    )

    if not rows:
        raise ValueError("no coefficient draw rows produced")
    return (
        pd.DataFrame(rows)
        .sort_values(["variable", "regressor", "method", "draw"])
        .reset_index(drop=True)
    )


def summarize_fit_coefficient_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired coefficient draws from :func:`build_fit_coefficient_detail`."""
    required = {"method", "draw", "variable", "regressor", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"coefficient detail frame is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    grouped = df.groupby(["variable", "regressor"], sort=True)
    for (variable, regressor), block in grouped:
        parts: dict[str, pd.Series] = {}
        for method_name in ("baseline", "candidate"):
            values = block.loc[block["method"] == method_name, "value"]
            if values.empty:
                raise ValueError(
                    f"coefficient detail frame is missing {method_name!r} rows for "
                    f"{variable}:{regressor}"
                )
            parts[method_name] = pd.to_numeric(values, errors="raise")

        base = parts["baseline"].to_numpy(dtype=float)
        cand = parts["candidate"].to_numpy(dtype=float)
        base_q10, base_q50, base_q90 = np.quantile(base, [0.10, 0.50, 0.90])
        cand_q10, cand_q50, cand_q90 = np.quantile(cand, [0.10, 0.50, 0.90])
        rows.append(
            {
                "variable": variable,
                "regressor": regressor,
                "baseline_mean": float(np.mean(base)),
                "baseline_sd": float(np.std(base, ddof=0)),
                "baseline_q10": float(base_q10),
                "baseline_q50": float(base_q50),
                "baseline_q90": float(base_q90),
                "baseline_prob_positive": float(np.mean(base > 0.0)),
                "candidate_mean": float(np.mean(cand)),
                "candidate_sd": float(np.std(cand, ddof=0)),
                "candidate_q10": float(cand_q10),
                "candidate_q50": float(cand_q50),
                "candidate_q90": float(cand_q90),
                "candidate_prob_positive": float(np.mean(cand > 0.0)),
                "mean_diff": float(np.mean(cand) - np.mean(base)),
                "sd_diff": float(np.std(cand, ddof=0) - np.std(base, ddof=0)),
                "q50_diff": float(cand_q50 - base_q50),
                "prob_positive_diff": float(np.mean(cand > 0.0) - np.mean(base > 0.0)),
                "mean_sign_flip": bool(np.sign(np.mean(base)) != np.sign(np.mean(cand))),
                "median_sign_flip": bool(np.sign(base_q50) != np.sign(cand_q50)),
                "q80_disjoint": bool((base_q90 < cand_q10) or (cand_q90 < base_q10)),
            }
        )

    return pd.DataFrame(rows).sort_values(["variable", "regressor"]).reset_index(drop=True)


def run_minnesota_prior_scale_diagnostic(
    config_path: str | Path,
    *,
    out_root: str | Path,
    origin_index: int | None = None,
    origin_date: str | None = None,
    baseline_method: Literal["minnesota", "minnesota_legacy"] = "minnesota_legacy",
    candidate_method: Literal["minnesota_canonical", "minnesota_tempered"] = "minnesota_canonical",
    variables: list[str] | None = None,
    regressors: list[str] | None = None,
    cases: list[str] | None = None,
) -> MinnesotaPriorScaleDiagnostic:
    """Diagnose legacy-vs-canonical Minnesota prior scales at one scheduled backtest origin."""
    cfg_path = Path(config_path)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(cfg_path)
    ds_full, model, _prior0, _sampler, _rng, _fc_cfg = _prepare_from_config(
        base_cfg, build_full_prior=False
    )
    bt = build_backtest_config(base_cfg, model=model)
    selected_variables = _select_variables(ds_full, variables)
    case_filter = _parse_variable_regressor_cases(cases)
    origin_end_i = _resolve_origin_end_index(
        ds_full,
        bt,
        origin_index=origin_index,
        origin_date=origin_date,
    )
    train_start, train_end_excl, train_ds = _build_train_dataset(
        ds_full,
        mode=str(bt["mode"]),
        window=bt["window"],
        origin_end_i=origin_end_i,
    )

    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg_path = configs_dir / "baseline.yml"
    candidate_cfg_path = configs_dir / "candidate.yml"
    baseline_cfg = _build_minnesota_variant_config(base_cfg, method=baseline_method)
    candidate_cfg = _build_minnesota_variant_config(base_cfg, method=candidate_method)
    _write_yaml_mapping(baseline_cfg_path, baseline_cfg)
    _write_yaml_mapping(candidate_cfg_path, candidate_cfg)

    prior_b = build_prior(baseline_cfg, dataset=train_ds, model=model)
    prior_c = build_prior(candidate_cfg, dataset=train_ds, model=model)
    regressor_names = _select_regressors(
        FitResult(
            dataset=train_ds,
            model=model,
            prior=prior_b,
            sampler=_prepare_from_config(baseline_cfg, build_full_prior=False)[3],
            posterior=None,
        ),
        regressors=regressors,
    )
    if case_filter is not None:
        missing_cases = [
            f"{variable}:{regressor}"
            for variable, regressor in case_filter
            if variable not in selected_variables or regressor not in regressor_names
        ]
        if missing_cases:
            raise ValueError(f"cases not found in selected variables/regressors: {missing_cases}")

    var_b = _prior_variance_matrix(
        prior_b,
        variables=list(train_ds.variables),
        p=int(model.p),
        include_intercept=bool(model.include_intercept),
    )
    var_c = _prior_variance_matrix(
        prior_c,
        variables=list(train_ds.variables),
        p=int(model.p),
        include_intercept=bool(model.include_intercept),
    )

    sigma2_source = (
        np.asarray(prior_c.minnesota_canonical.sigma2, dtype=float)
        if prior_c.minnesota_canonical is not None
        else np.diag(np.asarray(prior_c.niw.s0, dtype=float))
    )
    sigma2_map = {
        variable: float(value)
        for variable, value in zip(train_ds.variables, sigma2_source, strict=True)
    }
    reg_to_idx = {
        regressor: idx
        for idx, regressor in enumerate(
            _regressor_names(
                variables=list(train_ds.variables),
                p=int(model.p),
                include_intercept=bool(model.include_intercept),
            )
        )
    }

    prior_cfg = base_cfg.get("prior", {})
    minnesota_cfg = prior_cfg.get("minnesota", {}) if isinstance(prior_cfg, dict) else {}
    lambda2 = float(minnesota_cfg.get("lambda2", 0.5))
    n = int(train_ds.N)
    if n == 1:
        cross_weight = 1.0
    else:
        cross_weight = float((1.0 + (n - 1) * (lambda2**2)) / n)

    rows: list[dict[str, object]] = []
    for variable in selected_variables:
        eq_idx = int(train_ds.variables.index(variable))
        sigma2_eq = sigma2_map[variable]
        for regressor in regressor_names:
            if case_filter is not None and (variable, regressor) not in case_filter:
                continue
            reg_idx = int(reg_to_idx[regressor])
            predictor, lag = _parse_regressor_name(regressor)
            is_intercept = predictor is None
            is_own = predictor == variable if predictor is not None else False
            sigma2_pred = np.nan if predictor is None else float(sigma2_map[predictor])
            baseline_variance = float(var_b[reg_idx, eq_idx])
            candidate_variance = float(var_c[reg_idx, eq_idx])
            if is_intercept:
                theoretical_ratio = float(sigma2_eq)
            elif is_own:
                theoretical_ratio = float(sigma2_eq / cross_weight)
            else:
                theoretical_ratio = float((lambda2**2) * sigma2_eq / cross_weight)

            rows.append(
                {
                    "variable": variable,
                    "regressor": regressor,
                    "lag": None if lag is None else int(lag),
                    "predictor_variable": predictor,
                    "is_intercept": bool(is_intercept),
                    "is_own_lag": bool(is_own),
                    "sigma2_equation": sigma2_eq,
                    "sigma2_predictor": sigma2_pred,
                    "sigma2_eq_over_pred": (
                        np.nan if predictor is None else float(sigma2_eq / sigma2_pred)
                    ),
                    "baseline_variance": baseline_variance,
                    "candidate_variance": candidate_variance,
                    "variance_ratio": float(candidate_variance / baseline_variance),
                    "log_variance_ratio": float(np.log(candidate_variance / baseline_variance)),
                    "baseline_precision": float(1.0 / baseline_variance),
                    "candidate_precision": float(1.0 / candidate_variance),
                    "precision_ratio": float(
                        (1.0 / candidate_variance) / (1.0 / baseline_variance)
                    ),
                    "theoretical_variance_ratio": theoretical_ratio,
                    "ratio_minus_theoretical": float(
                        (candidate_variance / baseline_variance) - theoretical_ratio
                    ),
                    "cross_weight": cross_weight,
                    "lambda2": lambda2,
                }
            )

    if not rows:
        raise ValueError("no prior scale rows produced")

    summary = pd.DataFrame(rows).sort_values(["variable", "regressor"]).reset_index(drop=True)
    summary_csv = root / "prior_scale_comparison.csv"
    summary.to_csv(summary_csv, index=False)

    metadata = {
        "config_path": str(cfg_path),
        "baseline_method": str(baseline_method),
        "candidate_method": str(candidate_method),
        "origin_index": int(origin_end_i),
        "origin_date": str(ds_full.time_index[origin_end_i]),
        "train_start_index": int(train_start),
        "train_start_date": str(ds_full.time_index[train_start]),
        "train_end_index": int(train_end_excl - 1),
        "train_end_date": str(ds_full.time_index[train_end_excl - 1]),
        "train_T": int(train_ds.T),
        "variables": list(selected_variables),
        "regressors": list(regressor_names),
        "cross_weight": cross_weight,
        "lambda2": lambda2,
    }
    metadata_json = root / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return MinnesotaPriorScaleDiagnostic(
        out_root=root,
        baseline_config=baseline_cfg_path,
        candidate_config=candidate_cfg_path,
        metadata_json=metadata_json,
        summary_csv=summary_csv,
    )


def _forecast_frame(
    ds_full: Dataset,
    fc_res: ForecastResult,
    *,
    origin_end_i: int,
    variables: list[str],
    horizons: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    draws = np.asarray(fc_res.draws, dtype=float)
    for variable in variables:
        v_idx = int(fc_res.variables.index(variable))
        for horizon in horizons:
            target_i = int(origin_end_i + horizon)
            if target_i >= ds_full.T:
                continue
            h_idx = int(horizon) - 1
            forecast_draws = draws[:, h_idx, v_idx]
            realized = float(ds_full.values[target_i, v_idx])
            rows.append(
                {
                    "variable": variable,
                    "horizon": int(horizon),
                    "target_index": target_i,
                    "target_date": str(ds_full.time_index[target_i]),
                    "realized": realized,
                    "forecast_mean": float(fc_res.mean[h_idx, v_idx]),
                    "forecast_std": float(np.std(forecast_draws, ddof=0)),
                }
            )
    if not rows:
        raise ValueError("no forecast comparison rows produced")
    return pd.DataFrame(rows).sort_values(["variable", "horizon"]).reset_index(drop=True)


def _prefix_frame(df: pd.DataFrame, *, prefix: str, key_cols: Iterable[str]) -> pd.DataFrame:
    keys = set(key_cols)
    rename = {col: f"{prefix}{col}" for col in df.columns if col not in keys}
    return df.rename(columns=rename)


def _add_prefixed_diff_columns(
    df: pd.DataFrame,
    *,
    baseline_prefix: str = "baseline_",
    candidate_prefix: str = "candidate_",
) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if not col.startswith(baseline_prefix):
            continue
        stem = col[len(baseline_prefix) :]
        candidate_col = f"{candidate_prefix}{stem}"
        if candidate_col not in out.columns:
            continue
        baseline_series = out[col]
        candidate_series = out[candidate_col]
        if not (
            pd.api.types.is_numeric_dtype(baseline_series)
            and pd.api.types.is_numeric_dtype(candidate_series)
        ):
            continue
        out[f"{stem}_diff"] = candidate_series - baseline_series
    return out


def _merge_method_frames(
    frames: dict[str, pd.DataFrame],
    *,
    key_cols: Iterable[str],
) -> pd.DataFrame:
    keys = list(key_cols)
    items = list(frames.items())
    if not items:
        raise ValueError("frames must be non-empty")
    merged = _prefix_frame(items[0][1], prefix=f"{items[0][0]}_", key_cols=keys)
    for method_name, frame in items[1:]:
        merged = merged.merge(
            _prefix_frame(frame, prefix=f"{method_name}_", key_cols=keys),
            on=keys,
            how="inner",
        )
    return merged


def _add_method_diff_columns(df: pd.DataFrame, *, left: str, right: str) -> pd.DataFrame:
    out = df.copy()
    left_prefix = f"{left}_"
    right_prefix = f"{right}_"
    for col in list(out.columns):
        if not col.startswith(left_prefix):
            continue
        stem = col[len(left_prefix) :]
        right_col = f"{right_prefix}{stem}"
        if right_col not in out.columns:
            continue
        if not (
            pd.api.types.is_numeric_dtype(out[col])
            and pd.api.types.is_numeric_dtype(out[right_col])
        ):
            continue
        out[f"{right}_minus_{left}_{stem}"] = out[right_col] - out[col]
    return out


def _companion_radius_summary(fit_res: FitResult) -> dict[str, float] | None:
    if fit_res.beta_draws is None:
        return None
    radii = []
    for beta in np.asarray(fit_res.beta_draws, dtype=float):
        companion = companion_matrix(
            beta,
            n=int(fit_res.dataset.N),
            p=int(fit_res.model.p),
            include_intercept=bool(fit_res.model.include_intercept),
        )
        eigvals = np.linalg.eigvals(companion)
        radii.append(float(np.max(np.abs(eigvals))))

    radius_arr = np.asarray(radii, dtype=float)
    return {
        "companion_radius_mean": float(np.mean(radius_arr)),
        "companion_radius_p90": float(np.quantile(radius_arr, 0.90)),
        "companion_radius_max": float(np.max(radius_arr)),
        "nonstationary_share": float(np.mean(radius_arr >= 1.0)),
    }


def make_tempered_canonical_prior(
    *,
    legacy: PriorSpec,
    canonical: PriorSpec,
    alpha: float,
) -> PriorSpec:
    """Blend legacy and canonical Minnesota coefficient variances in log space.

    Parameters
    ----------
    legacy:
        Legacy Minnesota prior with shared NIW `v0` row variances.
    canonical:
        Canonical Minnesota prior with equation-wise `inv_v0_vec`.
    alpha:
        Blend weight in `[0, 1]`. `0` reproduces the legacy variance map and `1`
        reproduces the canonical variance map.
    """
    alpha_f = float(alpha)
    if not np.isfinite(alpha_f) or not (0.0 <= alpha_f <= 1.0):
        raise ValueError("alpha must be finite and in [0, 1]")
    if canonical.minnesota_canonical is None:
        raise ValueError("canonical prior must define minnesota_canonical")

    m0 = np.asarray(canonical.niw.m0, dtype=float)
    k, n = m0.shape
    legacy_diag = np.diag(np.asarray(legacy.niw.v0, dtype=float))
    if legacy_diag.shape != (k,):
        raise ValueError("legacy prior niw.v0 has incompatible shape")
    legacy_var = np.repeat(legacy_diag.reshape(-1, 1), repeats=n, axis=1)

    canonical_var = 1.0 / np.asarray(canonical.minnesota_canonical.inv_v0_vec, dtype=float).reshape(
        (k, n),
        order="F",
    )
    if canonical_var.shape != (k, n):
        raise ValueError("canonical prior inv_v0_vec has incompatible shape")

    tempered_var = legacy_var * np.power(canonical_var / legacy_var, alpha_f)
    v0_summary = np.diag(np.mean(tempered_var, axis=1))
    tempered_canonical = MinnesotaCanonicalSpec(
        sigma2=np.asarray(canonical.minnesota_canonical.sigma2, dtype=float).copy(),
        inv_v0_vec=1.0 / tempered_var.reshape(-1, order="F"),
        mode="tempered",
        tempered_alpha=alpha_f,
    )
    niw = NIWPrior(
        m0=m0.copy(),
        v0=v0_summary,
        s0=np.asarray(canonical.niw.s0, dtype=float).copy(),
        nu0=float(canonical.niw.nu0),
    )
    return PriorSpec(
        family="niw",
        niw=niw,
        minnesota_canonical=tempered_canonical,
    )


def _summarize_metrics_comparison(df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for col in df.columns:
        if not col.endswith("_baseline"):
            continue
        metric = col[: -len("_baseline")]
        entry: dict[str, Any] = {
            "baseline_mean": float(df[col].mean()),
            "candidate_mean": float(df[f"{metric}_candidate"].mean()),
        }
        rel_col = f"{metric}_rel"
        diff_col = f"{metric}_diff"
        if rel_col in df.columns:
            entry["relative_mean"] = float(df[rel_col].mean())
        if diff_col in df.columns:
            entry["diff_mean"] = float(df[diff_col].mean())
        metrics[metric] = entry
    return {"rows": int(df.shape[0]), "metrics": metrics}


def compare_metrics_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key_cols: Iterable[str] = ("variable", "horizon"),
    mode: Literal["ratio", "diff", "both"] = "ratio",
) -> pd.DataFrame:
    keys = list(key_cols)
    for k in keys:
        if k not in baseline.columns:
            raise ValueError(f"baseline is missing required column: {k}")
        if k not in candidate.columns:
            raise ValueError(f"candidate is missing required column: {k}")

    metric_cols = sorted(set(baseline.columns).intersection(candidate.columns) - set(keys))
    if not metric_cols:
        raise ValueError("no common metric columns found to compare")

    base = baseline.loc[:, keys + metric_cols].copy()
    cand = candidate.loc[:, keys + metric_cols].copy()

    merged = cand.merge(base, on=keys, how="inner", suffixes=("_candidate", "_baseline"))
    if merged.empty:
        raise ValueError("no overlapping rows after merging on key columns")

    out_cols: list[str] = keys.copy()
    for m in metric_cols:
        b = f"{m}_baseline"
        c = f"{m}_candidate"
        out_cols.extend([b, c])

        if mode in {"ratio", "both"}:
            merged[f"{m}_rel"] = merged[c] / merged[b]
            out_cols.append(f"{m}_rel")
        if mode in {"diff", "both"}:
            merged[f"{m}_diff"] = merged[c] - merged[b]
            out_cols.append(f"{m}_diff")

    return merged.loc[:, out_cols]


def compare_metrics_csv(
    baseline_csv: str | Path,
    candidate_csv: str | Path,
    *,
    mode: Literal["ratio", "diff", "both"] = "ratio",
) -> pd.DataFrame:
    base = pd.read_csv(Path(baseline_csv))
    cand = pd.read_csv(Path(candidate_csv))

    for df in (base, cand):
        for col in df.columns:
            if col in {"variable"}:
                continue
            if col in {"horizon"}:
                df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                continue

    return compare_metrics_frames(base, cand, mode=mode)


def run_minnesota_backtest_comparison(
    config_path: str | Path,
    *,
    out_root: str | Path,
    baseline_method: Literal["minnesota", "minnesota_legacy"] = "minnesota_legacy",
    candidate_method: Literal["minnesota_canonical", "minnesota_tempered"] = "minnesota_canonical",
    mode: Literal["ratio", "diff", "both"] = "both",
    save_forecasts: bool | None = None,
) -> MinnesotaBacktestComparison:
    """Run paired Minnesota backtests and write a metrics comparison bundle.

    Parameters
    ----------
    config_path:
        Base backtest YAML config. The config must define ``prior.family: niw``.
    out_root:
        Output directory where variant configs, backtest outputs, and comparison files
        will be written.
    baseline_method:
        Baseline Minnesota method. Defaults to the explicit legacy compatibility path.
    candidate_method:
        Candidate Minnesota method. Defaults to the explicit canonical path.
    mode:
        Comparison mode forwarded to :func:`compare_metrics_csv`.
    save_forecasts:
        Optional output override applied to both variant configs. When provided, this forces
        ``output.save_forecasts`` to the given boolean in the generated configs.
    """
    from .backtest import backtest_from_config

    cfg_path = Path(config_path)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml_mapping(cfg_path)
    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg_path = configs_dir / "baseline.yml"
    candidate_cfg_path = configs_dir / "candidate.yml"
    baseline_out = root / "baseline"
    candidate_out = root / "candidate"

    _write_yaml_mapping(
        baseline_cfg_path,
        _build_minnesota_variant_config(
            base_cfg, method=baseline_method, save_forecasts=save_forecasts
        ),
    )
    _write_yaml_mapping(
        candidate_cfg_path,
        _build_minnesota_variant_config(
            base_cfg, method=candidate_method, save_forecasts=save_forecasts
        ),
    )

    backtest_from_config(baseline_cfg_path, out_dir=baseline_out)
    backtest_from_config(candidate_cfg_path, out_dir=candidate_out)

    comparison_df = compare_metrics_csv(
        baseline_out / "metrics.csv",
        candidate_out / "metrics.csv",
        mode=mode,
    )
    comparison_csv = root / "metrics_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    summary = {
        "baseline_method": str(baseline_method),
        "candidate_method": str(candidate_method),
        "mode": str(mode),
        "baseline_metrics_csv": str(baseline_out / "metrics.csv"),
        "candidate_metrics_csv": str(candidate_out / "metrics.csv"),
        **_summarize_metrics_comparison(comparison_df),
    }
    summary_json = root / "comparison_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return MinnesotaBacktestComparison(
        out_root=root,
        baseline_config=baseline_cfg_path,
        candidate_config=candidate_cfg_path,
        baseline_out_dir=baseline_out,
        candidate_out_dir=candidate_out,
        comparison_csv=comparison_csv,
        summary_json=summary_json,
    )


def run_minnesota_origin_diagnostic(
    config_path: str | Path,
    *,
    out_root: str | Path,
    origin_index: int | None = None,
    origin_date: str | None = None,
    baseline_method: Literal["minnesota", "minnesota_legacy"] = "minnesota_legacy",
    candidate_method: Literal["minnesota_canonical", "minnesota_tempered"] = "minnesota_canonical",
    variables: list[str] | None = None,
    horizons: list[int] | None = None,
) -> MinnesotaOriginDiagnostic:
    """Run a paired single-origin Minnesota diagnostic with saved fit and forecast artifacts.

    The training slice matches the backtest origin logic from the supplied config. This writes
    baseline and candidate fit/forecast artifacts plus compact CSV summaries of fit-state,
    forecast-center, and coefficient-mean deltas.
    """
    cfg_path = Path(config_path)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(cfg_path)
    ds_full, model, _prior0, sampler, _rng, _fc_cfg = _prepare_from_config(
        base_cfg, build_full_prior=False
    )
    vol = model.volatility
    if candidate_method == "minnesota_canonical" and (
        vol is not None and vol.enabled and vol.covariance in {"triangular", "factor"}
    ):
        raise ValueError(
            "minnesota_canonical currently supports only homoskedastic models "
            "and diagonal stochastic volatility"
        )
    if candidate_method == "minnesota_tempered" and (
        vol is None or not vol.enabled or vol.covariance != "diagonal"
    ):
        raise ValueError(
            "tempered Minnesota origin experiments currently require diagonal stochastic volatility"
        )
    bt = build_backtest_config(base_cfg, model=model)

    selected_variables = _select_variables(ds_full, variables)
    selected_horizons = _select_horizons(bt, horizons)
    origin_end_i = _resolve_origin_end_index(
        ds_full,
        bt,
        origin_index=origin_index,
        origin_date=origin_date,
    )
    train_start, train_end_excl, train_ds = _build_train_dataset(
        ds_full,
        mode=str(bt["mode"]),
        window=bt["window"],
        origin_end_i=origin_end_i,
    )

    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg_path = configs_dir / "baseline.yml"
    candidate_cfg_path = configs_dir / "candidate.yml"
    baseline_out = root / "baseline"
    candidate_out = root / "candidate"
    baseline_out.mkdir(parents=True, exist_ok=True)
    candidate_out.mkdir(parents=True, exist_ok=True)

    baseline_cfg = _build_minnesota_variant_config(base_cfg, method=baseline_method)
    candidate_cfg = _build_minnesota_variant_config(base_cfg, method=candidate_method)
    _write_yaml_mapping(baseline_cfg_path, baseline_cfg)
    _write_yaml_mapping(candidate_cfg_path, candidate_cfg)
    (baseline_out / "config.yml").write_text(
        baseline_cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (candidate_out / "config.yml").write_text(
        candidate_cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    _sampler_b, rng_b = _prepare_from_config(baseline_cfg, build_full_prior=False)[3:5]
    _sampler_c, rng_c = _prepare_from_config(candidate_cfg, build_full_prior=False)[3:5]
    prior_b = build_prior(baseline_cfg, dataset=train_ds, model=model)
    prior_c = build_prior(candidate_cfg, dataset=train_ds, model=model)

    fit_b = fit(train_ds, model, prior_b, sampler, rng=rng_b)
    fit_c = fit(train_ds, model, prior_c, sampler, rng=rng_c)

    pred_draws = int(bt["draws"])
    q_levels = list(bt["quantile_levels"])
    stationarity = str(bt.get("stationarity", "allow"))
    stationarity_tol = float(bt.get("stationarity_tol", 1e-10))
    stationarity_max_draws = bt.get("stationarity_max_draws", None)
    fc_b = forecast(
        fit_b,
        horizons=selected_horizons,
        draws=pred_draws,
        quantile_levels=q_levels,
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
        rng=rng_b,
    )
    fc_c = forecast(
        fit_c,
        horizons=selected_horizons,
        draws=pred_draws,
        quantile_levels=q_levels,
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
        rng=rng_c,
    )

    save_fit_npz(baseline_out / "fit_result.npz", fit_b)
    save_fit_npz(candidate_out / "fit_result.npz", fit_c)
    save_forecast_npz(baseline_out / "forecast_result.npz", fc_b)
    save_forecast_npz(candidate_out / "forecast_result.npz", fc_c)

    state_b = _state_frame(fit_b, variables=selected_variables)
    state_c = _state_frame(fit_c, variables=selected_variables)
    state_comparison = _prefix_frame(state_b, prefix="baseline_", key_cols=["variable"]).merge(
        _prefix_frame(state_c, prefix="candidate_", key_cols=["variable"]),
        on=["variable"],
        how="inner",
    )
    state_comparison = _add_prefixed_diff_columns(state_comparison)

    beta_b = _beta_frame(fit_b, variables=selected_variables)
    beta_c = _beta_frame(fit_c, variables=selected_variables)
    beta_comparison = _prefix_frame(
        beta_b, prefix="baseline_", key_cols=["variable", "regressor"]
    ).merge(
        _prefix_frame(beta_c, prefix="candidate_", key_cols=["variable", "regressor"]),
        on=["variable", "regressor"],
        how="inner",
    )
    beta_comparison = _add_prefixed_diff_columns(beta_comparison)

    forecast_b = _forecast_frame(
        ds_full,
        fc_b,
        origin_end_i=origin_end_i,
        variables=selected_variables,
        horizons=selected_horizons,
    )
    forecast_c = _forecast_frame(
        ds_full,
        fc_c,
        origin_end_i=origin_end_i,
        variables=selected_variables,
        horizons=selected_horizons,
    )
    forecast_comparison = _prefix_frame(
        forecast_b,
        prefix="baseline_",
        key_cols=["variable", "horizon", "target_index", "target_date", "realized"],
    ).merge(
        _prefix_frame(
            forecast_c,
            prefix="candidate_",
            key_cols=["variable", "horizon", "target_index", "target_date", "realized"],
        ),
        on=["variable", "horizon", "target_index", "target_date", "realized"],
        how="inner",
    )
    forecast_comparison["baseline_error"] = (
        forecast_comparison["baseline_forecast_mean"] - forecast_comparison["realized"]
    )
    forecast_comparison["candidate_error"] = (
        forecast_comparison["candidate_forecast_mean"] - forecast_comparison["realized"]
    )
    forecast_comparison["baseline_abs_error"] = np.abs(forecast_comparison["baseline_error"])
    forecast_comparison["candidate_abs_error"] = np.abs(forecast_comparison["candidate_error"])
    forecast_comparison["forecast_mean_diff"] = (
        forecast_comparison["candidate_forecast_mean"]
        - forecast_comparison["baseline_forecast_mean"]
    )
    forecast_comparison["forecast_std_diff"] = (
        forecast_comparison["candidate_forecast_std"] - forecast_comparison["baseline_forecast_std"]
    )
    forecast_comparison["abs_error_diff"] = (
        forecast_comparison["candidate_abs_error"] - forecast_comparison["baseline_abs_error"]
    )

    state_csv = root / "state_comparison.csv"
    beta_csv = root / "beta_mean_comparison.csv"
    forecast_csv = root / "forecast_comparison.csv"
    state_comparison.to_csv(state_csv, index=False)
    beta_comparison.to_csv(beta_csv, index=False)
    forecast_comparison.to_csv(forecast_csv, index=False)

    metadata = {
        "config_path": str(cfg_path),
        "baseline_method": str(baseline_method),
        "candidate_method": str(candidate_method),
        "origin_index": int(origin_end_i),
        "origin_date": str(ds_full.time_index[origin_end_i]),
        "train_start_index": int(train_start),
        "train_start_date": str(ds_full.time_index[train_start]),
        "train_end_index": int(train_end_excl - 1),
        "train_end_date": str(ds_full.time_index[train_end_excl - 1]),
        "train_T": int(train_ds.T),
        "backtest_mode": str(bt["mode"]),
        "backtest_window": None if bt["window"] is None else int(bt["window"]),
        "forecast_horizons": list(selected_horizons),
        "variables": list(selected_variables),
        "sampler_draws": int(sampler.draws),
        "sampler_burn_in": int(sampler.burn_in),
        "sampler_thin": int(sampler.thin),
        "forecast_draws": pred_draws,
        "baseline_stability": _companion_radius_summary(fit_b),
        "candidate_stability": _companion_radius_summary(fit_c),
    }
    metadata_json = root / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return MinnesotaOriginDiagnostic(
        out_root=root,
        baseline_config=baseline_cfg_path,
        candidate_config=candidate_cfg_path,
        baseline_out_dir=baseline_out,
        candidate_out_dir=candidate_out,
        metadata_json=metadata_json,
        state_csv=state_csv,
        forecast_csv=forecast_csv,
        beta_csv=beta_csv,
    )


def run_tempered_minnesota_origin_experiment(
    config_path: str | Path,
    *,
    out_root: str | Path,
    alpha: float,
    origin_index: int | None = None,
    origin_date: str | None = None,
    variables: list[str] | None = None,
    horizons: list[int] | None = None,
) -> MinnesotaTemperedOriginExperiment:
    """Run a three-way legacy/canonical/tempered Minnesota origin experiment."""
    cfg_path = Path(config_path)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(cfg_path)
    ds_full, model, _prior0, sampler, _rng, _fc_cfg = _prepare_from_config(
        base_cfg, build_full_prior=False
    )
    bt = build_backtest_config(base_cfg, model=model)

    selected_variables = _select_variables(ds_full, variables)
    selected_horizons = _select_horizons(bt, horizons)
    origin_end_i = _resolve_origin_end_index(
        ds_full,
        bt,
        origin_index=origin_index,
        origin_date=origin_date,
    )
    train_start, train_end_excl, train_ds = _build_train_dataset(
        ds_full,
        mode=str(bt["mode"]),
        window=bt["window"],
        origin_end_i=origin_end_i,
    )

    baseline_cfg = _build_minnesota_variant_config(base_cfg, method="minnesota_legacy")
    canonical_cfg = _build_minnesota_variant_config(base_cfg, method="minnesota_canonical")
    prior_b = build_prior(baseline_cfg, dataset=train_ds, model=model)
    prior_c = build_prior(canonical_cfg, dataset=train_ds, model=model)
    prior_t = make_tempered_canonical_prior(legacy=prior_b, canonical=prior_c, alpha=alpha)

    sampler_cfg = base_cfg.get("sampler", {})
    if isinstance(sampler_cfg, dict):
        seed_raw = sampler_cfg.get("seed", 0)
    else:
        seed_raw = 0
    base_seed = 0 if seed_raw is None else int(seed_raw)

    def _fit_and_forecast_bundle(
        method_name: str,
        prior_i: PriorSpec,
        *,
        seed: int,
    ) -> tuple[Path, FitResult, ForecastResult]:
        out_dir = root / method_name
        out_dir.mkdir(parents=True, exist_ok=True)
        fit_rng = np.random.default_rng(seed)
        fc_rng = np.random.default_rng(seed + 1)
        fit_res = fit(train_ds, model, prior_i, sampler, rng=fit_rng)
        fc_res = forecast(
            fit_res,
            horizons=selected_horizons,
            draws=int(bt["draws"]),
            quantile_levels=list(bt["quantile_levels"]),
            stationarity=str(bt.get("stationarity", "allow")),
            stationarity_tol=float(bt.get("stationarity_tol", 1e-10)),
            stationarity_max_draws=bt.get("stationarity_max_draws", None),
            rng=fc_rng,
        )
        save_fit_npz(out_dir / "fit_result.npz", fit_res)
        save_forecast_npz(out_dir / "forecast_result.npz", fc_res)
        return out_dir, fit_res, fc_res

    baseline_dir, fit_b, fc_b = _fit_and_forecast_bundle("baseline", prior_b, seed=base_seed)
    canonical_dir, fit_c, fc_c = _fit_and_forecast_bundle("canonical", prior_c, seed=base_seed)
    tempered_dir, fit_t, fc_t = _fit_and_forecast_bundle("tempered", prior_t, seed=base_seed)

    forecast_frames = {
        "baseline": _forecast_frame(
            ds_full,
            fc_b,
            origin_end_i=origin_end_i,
            variables=selected_variables,
            horizons=selected_horizons,
        ),
        "canonical": _forecast_frame(
            ds_full,
            fc_c,
            origin_end_i=origin_end_i,
            variables=selected_variables,
            horizons=selected_horizons,
        ),
        "tempered": _forecast_frame(
            ds_full,
            fc_t,
            origin_end_i=origin_end_i,
            variables=selected_variables,
            horizons=selected_horizons,
        ),
    }
    forecast_comparison = _merge_method_frames(
        forecast_frames,
        key_cols=["variable", "horizon", "target_index", "target_date", "realized"],
    )
    for method_name in ("baseline", "canonical", "tempered"):
        forecast_comparison[f"{method_name}_error"] = (
            forecast_comparison[f"{method_name}_forecast_mean"] - forecast_comparison["realized"]
        )
        forecast_comparison[f"{method_name}_abs_error"] = np.abs(
            forecast_comparison[f"{method_name}_error"]
        )
    for left, right in (
        ("baseline", "canonical"),
        ("baseline", "tempered"),
        ("canonical", "tempered"),
    ):
        forecast_comparison = _add_method_diff_columns(forecast_comparison, left=left, right=right)

    state_frames = {
        "baseline": _state_frame(fit_b, variables=selected_variables),
        "canonical": _state_frame(fit_c, variables=selected_variables),
        "tempered": _state_frame(fit_t, variables=selected_variables),
    }
    state_comparison = _merge_method_frames(state_frames, key_cols=["variable"])
    for left, right in (
        ("baseline", "canonical"),
        ("baseline", "tempered"),
        ("canonical", "tempered"),
    ):
        state_comparison = _add_method_diff_columns(state_comparison, left=left, right=right)

    beta_frames = {
        "baseline": _beta_frame(fit_b, variables=selected_variables),
        "canonical": _beta_frame(fit_c, variables=selected_variables),
        "tempered": _beta_frame(fit_t, variables=selected_variables),
    }
    beta_comparison = _merge_method_frames(beta_frames, key_cols=["variable", "regressor"])
    for left, right in (
        ("baseline", "canonical"),
        ("baseline", "tempered"),
        ("canonical", "tempered"),
    ):
        beta_comparison = _add_method_diff_columns(beta_comparison, left=left, right=right)

    forecast_csv = root / "forecast_comparison.csv"
    state_csv = root / "state_comparison.csv"
    beta_csv = root / "beta_comparison.csv"
    forecast_comparison.to_csv(forecast_csv, index=False)
    state_comparison.to_csv(state_csv, index=False)
    beta_comparison.to_csv(beta_csv, index=False)

    rows_summary: dict[str, dict[str, float]] = {}
    for method_name in ("baseline", "canonical", "tempered"):
        rows_summary[method_name] = {
            "mean_abs_error": float(np.mean(forecast_comparison[f"{method_name}_abs_error"])),
            "mean_forecast_std": float(np.mean(forecast_comparison[f"{method_name}_forecast_std"])),
        }

    metadata = {
        "config_path": str(cfg_path),
        "alpha": float(alpha),
        "origin_index": int(origin_end_i),
        "origin_date": str(ds_full.time_index[origin_end_i]),
        "train_start_index": int(train_start),
        "train_start_date": str(ds_full.time_index[train_start]),
        "train_end_index": int(train_end_excl - 1),
        "train_end_date": str(ds_full.time_index[train_end_excl - 1]),
        "train_T": int(train_ds.T),
        "variables": list(selected_variables),
        "forecast_horizons": list(selected_horizons),
        "sampler_seed_used": int(base_seed),
        "baseline_stability": _companion_radius_summary(fit_b),
        "canonical_stability": _companion_radius_summary(fit_c),
        "tempered_stability": _companion_radius_summary(fit_t),
        "forecast_summary": rows_summary,
    }
    metadata_json = root / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return MinnesotaTemperedOriginExperiment(
        out_root=root,
        metadata_json=metadata_json,
        forecast_csv=forecast_csv,
        state_csv=state_csv,
        beta_csv=beta_csv,
        baseline_dir=baseline_dir,
        canonical_dir=canonical_dir,
        tempered_dir=tempered_dir,
    )
