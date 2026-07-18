from __future__ import annotations

import errno
import math
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from numpy.lib.npyio import NpzFile

from .data.dataset import Dataset
from .results import FitResult, ForecastResult, PosteriorNIW

_FORMAT_VERSION = 1
_FORMAT_VERSION_KEY = "format_version"
_ARTIFACT_KIND_KEY = "artifact_kind"
_REAL_NUMERIC_KINDS = {"i", "u", "f"}
_INDICATOR_NUMERIC_KINDS = _REAL_NUMERIC_KINDS | {"b"}
_FIT_REQUIRED_FIELDS = frozenset(
    {_FORMAT_VERSION_KEY, _ARTIFACT_KIND_KEY, "variables", "time_index", "values"}
)
_FIT_OPTIONAL_FIELDS = frozenset(
    {
        "beta_draws",
        "sigma_draws",
        "q_draws",
        "latent_values",
        "latent_draws",
        "posterior_mn",
        "posterior_vn",
        "posterior_sn",
        "posterior_nun",
        "h_draws",
        "h0_draws",
        "sigma_eta2_draws",
        "sv_gamma0_draws",
        "sv_phi_draws",
        "lambda_draws",
        "factor_draws",
        "h_factor_draws",
        "h0_factor_draws",
        "sigma_eta2_factor_draws",
        "gamma_draws",
        "mu_draws",
        "mu_gamma_draws",
    }
)
_FORECAST_REQUIRED_FIELDS = frozenset(
    {_FORMAT_VERSION_KEY, _ARTIFACT_KIND_KEY, "variables", "horizons", "draws", "mean"}
)
_FORECAST_OPTIONAL_FIELDS = frozenset({"latent_draws"})
_QUANTILE_KEY_RE = re.compile(r"^q_([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


@dataclass(frozen=True, slots=True)
class ArtifactLoadLimits:
    """Metadata limits applied before an NPZ artifact is opened by NumPy."""

    max_archive_bytes: int = 512 * 1024 * 1024
    max_member_count: int = 128
    max_member_uncompressed_bytes: int = 512 * 1024 * 1024
    max_total_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_expansion_ratio: float = 100.0

    def __post_init__(self) -> None:
        for field_name in (
            "max_archive_bytes",
            "max_member_count",
            "max_member_uncompressed_bytes",
            "max_total_uncompressed_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        ratio = self.max_expansion_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, Real)
            or not math.isfinite(float(ratio))
            or ratio <= 0
        ):
            raise ValueError("max_expansion_ratio must be a positive finite real number")


def _add_optional(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _save_safe_npz(path: Path, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if np.asarray(value).dtype.hasobject:
            raise TypeError(f"cannot save safe NPZ artifact: field {key!r} has object dtype")
    np.savez_compressed(path, **payload)


def _artifact_format_error(path: Path, detail: str) -> ValueError:
    return ValueError(f"invalid srvar artifact {path}: {detail}")


def _resolve_load_limits(limits: ArtifactLoadLimits | None) -> ArtifactLoadLimits:
    if limits is None:
        return ArtifactLoadLimits()
    if not isinstance(limits, ArtifactLoadLimits):
        raise TypeError("limits must be an ArtifactLoadLimits instance or None")
    return limits


@contextmanager
def _open_regular_artifact(path: Path) -> Iterator[tuple[Any, os.stat_result]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK

    fd = os.open(os.fspath(path), flags)
    try:
        file_status = os.fstat(fd)
        if stat.S_ISDIR(file_status.st_mode):
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), os.fspath(path))
        if not stat.S_ISREG(file_status.st_mode):
            raise _artifact_format_error(path, "expected a regular file")

        with os.fdopen(fd, "rb") as handle:
            fd = -1
            yield handle, file_status
    finally:
        if fd != -1:
            os.close(fd)


def _preflight_npz_metadata(
    handle: Any, path: Path, *, archive_bytes: int, limits: ArtifactLoadLimits
) -> None:
    if archive_bytes > limits.max_archive_bytes:
        raise _artifact_format_error(
            path,
            f"archive size {archive_bytes} exceeds max_archive_bytes {limits.max_archive_bytes}",
        )

    try:
        with ZipFile(handle) as archive:
            members = archive.infolist()
            member_count = len(members)
            if member_count > limits.max_member_count:
                raise _artifact_format_error(
                    path,
                    f"member count {member_count} exceeds max_member_count {limits.max_member_count}",
                )

            total_uncompressed = 0
            for member in members:
                member_size = member.file_size
                if member_size > limits.max_member_uncompressed_bytes:
                    raise _artifact_format_error(
                        path,
                        "member uncompressed size "
                        f"{member_size} exceeds max_member_uncompressed_bytes "
                        f"{limits.max_member_uncompressed_bytes}",
                    )

                total_uncompressed += member_size
                if total_uncompressed > limits.max_total_uncompressed_bytes:
                    raise _artifact_format_error(
                        path,
                        "total uncompressed size "
                        f"{total_uncompressed} exceeds max_total_uncompressed_bytes "
                        f"{limits.max_total_uncompressed_bytes}",
                    )

                if member_size:
                    expansion_ratio = (
                        math.inf
                        if member.compress_size == 0
                        else member_size / member.compress_size
                    )
                    if expansion_ratio > limits.max_expansion_ratio:
                        raise _artifact_format_error(
                            path,
                            "expansion ratio "
                            f"{expansion_ratio:g} exceeds max_expansion_ratio "
                            f"{limits.max_expansion_ratio:g}",
                        )
    except (BadZipFile, EOFError) as exc:
        raise _artifact_format_error(path, "could not parse a readable NPZ archive") from exc
    finally:
        handle.seek(0)


def _load_npz_from_handle(handle: Any, path: Path, *, allow_pickle: bool) -> NpzFile | Any:
    try:
        return np.load(handle, allow_pickle=allow_pickle)
    except OSError:
        raise
    except (BadZipFile, EOFError, ValueError) as exc:
        raise _artifact_format_error(path, "could not parse a readable NPZ archive") from exc


def _load_npz_array(npz: NpzFile, key: str, path: Path) -> np.ndarray:
    try:
        return npz[key]
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise _artifact_format_error(
                path, f"field {key!r} has object dtype and cannot be loaded safely"
            ) from exc
        raise


def _validate_marker(npz: NpzFile, path: Path, *, expected_kind: str) -> None:
    format_version = _load_npz_array(npz, _FORMAT_VERSION_KEY, path)
    if format_version.shape != () or format_version.dtype.kind not in {"i", "u"}:
        raise _artifact_format_error(
            path, "format_version must be a zero-dimensional signed or unsigned integer"
        )
    if format_version.item() != _FORMAT_VERSION:
        raise _artifact_format_error(path, "unsupported format_version")

    artifact_kind = _load_npz_array(npz, _ARTIFACT_KIND_KEY, path)
    if artifact_kind.shape != () or artifact_kind.dtype.kind != "U":
        raise _artifact_format_error(
            path, "artifact_kind must be a zero-dimensional Unicode string"
        )
    if artifact_kind.item() != expected_kind:
        raise _artifact_format_error(path, f"artifact_kind must be {expected_kind!r}")


def _schema_error(path: Path, field: str, detail: str) -> ValueError:
    return _artifact_format_error(path, f"field {field!r} {detail}")


def _require_dtype_rank(
    arr: np.ndarray,
    path: Path,
    field: str,
    *,
    dtype_kinds: set[str],
    ndim: int,
) -> None:
    if arr.dtype.kind not in dtype_kinds:
        raise _schema_error(path, field, "has an unsupported dtype")
    if arr.ndim != ndim:
        raise _schema_error(path, field, f"must be {ndim}-dimensional")


def _require_shape(arr: np.ndarray, path: Path, field: str, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise _schema_error(path, field, "has incompatible shape")


def _validate_unique_time_index(time_index: np.ndarray, path: Path) -> None:
    if np.unique(time_index).size != time_index.size:
        raise _schema_error(path, "time_index", "must not contain duplicate timestamps")


def _validate_v1_field_set(
    npz: NpzFile,
    path: Path,
    *,
    required_fields: frozenset[str],
    optional_fields: frozenset[str],
) -> set[str]:
    fields = set(npz.files)
    if required_fields - fields:
        raise _artifact_format_error(path, "missing required v1 field")
    if fields - required_fields - optional_fields:
        raise _artifact_format_error(path, "unknown v1 field")
    return fields


def _load_v1_arrays(npz: NpzFile, path: Path, fields: set[str]) -> dict[str, np.ndarray]:
    return {
        field: _load_npz_array(npz, field, path)
        for field in fields
        if field not in {_FORMAT_VERSION_KEY, _ARTIFACT_KIND_KEY}
    }


def _validate_fit_v1_payload(npz: NpzFile, path: Path) -> dict[str, np.ndarray]:
    fields = _validate_v1_field_set(
        npz,
        path,
        required_fields=_FIT_REQUIRED_FIELDS,
        optional_fields=_FIT_OPTIONAL_FIELDS,
    )
    arrays = _load_v1_arrays(npz, path, fields)

    variables = arrays["variables"]
    _require_dtype_rank(variables, path, "variables", dtype_kinds={"U"}, ndim=1)
    n = variables.shape[0]
    time_index = arrays["time_index"]
    _require_dtype_rank(time_index, path, "time_index", dtype_kinds={"M"}, ndim=1)
    _validate_unique_time_index(time_index, path)
    t = time_index.shape[0]
    values = arrays["values"]
    _require_dtype_rank(values, path, "values", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
    _require_shape(values, path, "values", (t, n))

    def optional(name: str) -> np.ndarray | None:
        return arrays.get(name)

    latent_values = optional("latent_values")
    if latent_values is not None:
        _require_dtype_rank(
            latent_values, path, "latent_values", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2
        )
        _require_shape(latent_values, path, "latent_values", (t, n))

    draw_count: int | None = None
    coefficient_width: int | None = None
    factor_width: int | None = None
    state_time: int | None = None

    def require_draw_count(arr: np.ndarray, field: str) -> None:
        nonlocal draw_count
        if draw_count is None:
            draw_count = arr.shape[0]
        elif arr.shape[0] != draw_count:
            raise _schema_error(path, field, "has an incompatible draw count")

    def require_coefficient_width(width: int, field: str) -> None:
        nonlocal coefficient_width
        if coefficient_width is None:
            coefficient_width = width
        elif width != coefficient_width:
            raise _schema_error(path, field, "has an incompatible coefficient width")

    def require_factor_width(width: int, field: str) -> None:
        nonlocal factor_width
        if factor_width is None:
            factor_width = width
        elif width != factor_width:
            raise _schema_error(path, field, "has an incompatible factor width")

    def require_state_time(length: int, field: str) -> None:
        nonlocal state_time
        if state_time is None:
            state_time = length
        elif length != state_time:
            raise _schema_error(path, field, "has an incompatible state-time length")

    latent_draws = optional("latent_draws")
    if latent_draws is not None:
        _require_dtype_rank(
            latent_draws, path, "latent_draws", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3
        )
        require_draw_count(latent_draws, "latent_draws")
        _require_shape(latent_draws, path, "latent_draws", (latent_draws.shape[0], t, n))

    beta_draws = optional("beta_draws")
    if beta_draws is not None:
        _require_dtype_rank(beta_draws, path, "beta_draws", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3)
        require_draw_count(beta_draws, "beta_draws")
        if beta_draws.shape[2] != n:
            raise _schema_error(path, "beta_draws", "has an incompatible variable dimension")
        require_coefficient_width(beta_draws.shape[1], "beta_draws")

    for field in ("sigma_draws", "q_draws"):
        arr = optional(field)
        if arr is None:
            continue
        _require_dtype_rank(arr, path, field, dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3)
        require_draw_count(arr, field)
        _require_shape(arr, path, field, (arr.shape[0], n, n))

    posterior_fields = ("posterior_mn", "posterior_vn", "posterior_sn", "posterior_nun")
    present_posterior_fields = [field for field in posterior_fields if optional(field) is not None]
    if present_posterior_fields and len(present_posterior_fields) != len(posterior_fields):
        raise _artifact_format_error(path, "partial posterior_* block")
    if present_posterior_fields:
        mn = arrays["posterior_mn"]
        vn = arrays["posterior_vn"]
        sn = arrays["posterior_sn"]
        nun = arrays["posterior_nun"]
        _require_dtype_rank(mn, path, "posterior_mn", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
        if mn.shape[1] != n:
            raise _schema_error(path, "posterior_mn", "has an incompatible variable dimension")
        require_coefficient_width(mn.shape[0], "posterior_mn")
        _require_dtype_rank(vn, path, "posterior_vn", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
        _require_shape(vn, path, "posterior_vn", (mn.shape[0], mn.shape[0]))
        _require_dtype_rank(sn, path, "posterior_sn", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
        _require_shape(sn, path, "posterior_sn", (n, n))
        _require_dtype_rank(nun, path, "posterior_nun", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=0)

    h_draws = optional("h_draws")
    if h_draws is not None:
        _require_dtype_rank(h_draws, path, "h_draws", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3)
        require_draw_count(h_draws, "h_draws")
        if h_draws.shape[2] != n:
            raise _schema_error(path, "h_draws", "has an incompatible variable dimension")
        require_state_time(h_draws.shape[1], "h_draws")

    for field in (
        "h0_draws",
        "sigma_eta2_draws",
        "sv_gamma0_draws",
        "sv_phi_draws",
        "mu_draws",
        "mu_gamma_draws",
    ):
        arr = optional(field)
        if arr is None:
            continue
        _require_dtype_rank(arr, path, field, dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
        require_draw_count(arr, field)
        _require_shape(arr, path, field, (arr.shape[0], n))

    lambda_draws = optional("lambda_draws")
    if lambda_draws is not None:
        _require_dtype_rank(
            lambda_draws, path, "lambda_draws", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3
        )
        require_draw_count(lambda_draws, "lambda_draws")
        if lambda_draws.shape[1] != n:
            raise _schema_error(path, "lambda_draws", "has an incompatible variable dimension")
        require_factor_width(lambda_draws.shape[2], "lambda_draws")

    for field in ("factor_draws", "h_factor_draws"):
        arr = optional(field)
        if arr is None:
            continue
        _require_dtype_rank(arr, path, field, dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3)
        require_draw_count(arr, field)
        require_state_time(arr.shape[1], field)
        require_factor_width(arr.shape[2], field)

    for field in ("h0_factor_draws", "sigma_eta2_factor_draws"):
        arr = optional(field)
        if arr is None:
            continue
        _require_dtype_rank(arr, path, field, dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
        require_draw_count(arr, field)
        require_factor_width(arr.shape[1], field)

    gamma_draws = optional("gamma_draws")
    if gamma_draws is not None:
        _require_dtype_rank(
            gamma_draws,
            path,
            "gamma_draws",
            dtype_kinds=_INDICATOR_NUMERIC_KINDS,
            ndim=2,
        )
        require_draw_count(gamma_draws, "gamma_draws")
        require_coefficient_width(gamma_draws.shape[1], "gamma_draws")

    return arrays


def _validate_forecast_v1_payload(
    npz: NpzFile, path: Path
) -> tuple[dict[str, np.ndarray], list[tuple[float, str]]]:
    fields = set(npz.files)
    if _FORECAST_REQUIRED_FIELDS - fields:
        raise _artifact_format_error(path, "missing required v1 field")
    quantiles: list[tuple[float, str]] = []
    quantile_levels: set[float] = set()
    for field in npz.files:
        if field in _FORECAST_REQUIRED_FIELDS | _FORECAST_OPTIONAL_FIELDS:
            continue
        match = _QUANTILE_KEY_RE.fullmatch(field)
        if match is None:
            detail = (
                "malformed forecast quantile field"
                if field.startswith("q_")
                else "unknown v1 field"
            )
            raise _artifact_format_error(path, detail)
        suffix = match.group(1)
        quantile = float(suffix)
        if not math.isfinite(quantile) or not 0 < quantile < 1 or suffix != str(quantile):
            raise _artifact_format_error(path, "malformed forecast quantile field")
        if quantile in quantile_levels:
            raise _artifact_format_error(path, "duplicate forecast quantile field")
        quantile_levels.add(quantile)
        quantiles.append((quantile, field))

    arrays = _load_v1_arrays(npz, path, fields)
    variables = arrays["variables"]
    _require_dtype_rank(variables, path, "variables", dtype_kinds={"U"}, ndim=1)
    n = variables.shape[0]
    horizons = arrays["horizons"]
    _require_dtype_rank(horizons, path, "horizons", dtype_kinds={"i", "u"}, ndim=1)
    h = horizons.shape[0]
    draws = arrays["draws"]
    _require_dtype_rank(draws, path, "draws", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3)
    d = draws.shape[0]
    _require_shape(draws, path, "draws", (d, h, n))
    mean = arrays["mean"]
    _require_dtype_rank(mean, path, "mean", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
    _require_shape(mean, path, "mean", (h, n))

    latent_draws = arrays.get("latent_draws")
    if latent_draws is not None:
        _require_dtype_rank(
            latent_draws, path, "latent_draws", dtype_kinds=_REAL_NUMERIC_KINDS, ndim=3
        )
        _require_shape(latent_draws, path, "latent_draws", (d, h, n))
    for _, field in quantiles:
        arr = arrays[field]
        _require_dtype_rank(arr, path, field, dtype_kinds=_REAL_NUMERIC_KINDS, ndim=2)
        _require_shape(arr, path, field, (h, n))

    return arrays, quantiles


@contextmanager
def _open_artifact_npz(
    path: Path,
    *,
    expected_kind: str,
    allow_legacy_pickle: bool,
    limits: ArtifactLoadLimits | None,
) -> Iterator[tuple[NpzFile, bool]]:
    resolved_limits = _resolve_load_limits(limits)
    with _open_regular_artifact(path) as (handle, file_status):
        _preflight_npz_metadata(
            handle,
            path,
            archive_bytes=file_status.st_size,
            limits=resolved_limits,
        )
        strict_loaded = _load_npz_from_handle(handle, path, allow_pickle=False)
        if not isinstance(strict_loaded, NpzFile):
            raise _artifact_format_error(path, "expected an NPZ archive")

        marker_keys = {_FORMAT_VERSION_KEY, _ARTIFACT_KIND_KEY}
        present_markers = marker_keys.intersection(strict_loaded.files)
        if not present_markers:
            strict_loaded.close()
            if not allow_legacy_pickle:
                raise _artifact_format_error(
                    path,
                    "legacy pickle-backed format is not loaded by default for security; "
                    "load only a trusted source with allow_legacy_pickle=True",
                )
            handle.seek(0)
            legacy_loaded = _load_npz_from_handle(handle, path, allow_pickle=True)
            if not isinstance(legacy_loaded, NpzFile):
                raise _artifact_format_error(path, "expected an NPZ archive")
            try:
                yield legacy_loaded, False
            finally:
                legacy_loaded.close()
            return

        try:
            if len(strict_loaded.files) != len(set(strict_loaded.files)):
                raise _artifact_format_error(path, "duplicate v1 field")
            if present_markers != marker_keys:
                raise _artifact_format_error(path, "incomplete format markers")
            _validate_marker(strict_loaded, path, expected_kind=expected_kind)
            yield strict_loaded, True
        finally:
            strict_loaded.close()


def save_fit_npz(path: str | Path, fit_res: FitResult) -> None:
    p = Path(path)
    payload: dict[str, Any] = {
        _FORMAT_VERSION_KEY: np.asarray(_FORMAT_VERSION, dtype=np.int64),
        _ARTIFACT_KIND_KEY: np.asarray("fit", dtype=str),
        "variables": np.asarray(fit_res.dataset.variables, dtype=str),
        "time_index": np.asarray(fit_res.dataset.time_index.to_numpy(), dtype="datetime64[ns]"),
        "values": fit_res.dataset.values,
    }
    _add_optional(payload, "beta_draws", fit_res.beta_draws)
    _add_optional(payload, "sigma_draws", fit_res.sigma_draws)
    _add_optional(payload, "q_draws", fit_res.q_draws)
    _add_optional(payload, "latent_draws", fit_res.latent_draws)
    _add_optional(
        payload, "latent_values", fit_res.latent_dataset.values if fit_res.latent_dataset else None
    )
    if fit_res.posterior is not None:
        _add_optional(payload, "posterior_mn", fit_res.posterior.mn)
        _add_optional(payload, "posterior_vn", fit_res.posterior.vn)
        _add_optional(payload, "posterior_sn", fit_res.posterior.sn)
        _add_optional(payload, "posterior_nun", fit_res.posterior.nun)
    for key in (
        "h_draws",
        "h0_draws",
        "sigma_eta2_draws",
        "sv_gamma0_draws",
        "sv_phi_draws",
        "lambda_draws",
        "factor_draws",
        "h_factor_draws",
        "h0_factor_draws",
        "sigma_eta2_factor_draws",
        "gamma_draws",
        "mu_draws",
        "mu_gamma_draws",
    ):
        _add_optional(payload, key, getattr(fit_res, key))
    _save_safe_npz(p, payload)


def save_forecast_npz(path: str | Path, fc: ForecastResult) -> None:
    p = Path(path)
    payload: dict[str, Any] = {
        _FORMAT_VERSION_KEY: np.asarray(_FORMAT_VERSION, dtype=np.int64),
        _ARTIFACT_KIND_KEY: np.asarray("forecast", dtype=str),
        "variables": np.asarray(fc.variables, dtype=str),
        "horizons": np.asarray(fc.horizons, dtype=int),
        "draws": fc.draws,
        "mean": fc.mean,
    }
    _add_optional(payload, "latent_draws", fc.latent_draws)
    payload.update({f"q_{q}": arr for q, arr in fc.quantiles.items()})
    _save_safe_npz(p, payload)


def _optional_npz_array(npz: NpzFile, key: str, path: Path) -> np.ndarray | None:
    if key not in npz:
        return None
    arr = _load_npz_array(npz, key, path)
    if (
        isinstance(arr, np.ndarray)
        and arr.shape == ()
        and arr.dtype == object
        and arr.item() is None
    ):
        return None
    return arr


@dataclass(frozen=True, slots=True)
class FitNPZ:
    dataset: Dataset
    posterior: PosteriorNIW | None
    beta_draws: np.ndarray | None
    sigma_draws: np.ndarray | None
    q_draws: np.ndarray | None
    latent_dataset: Dataset | None
    latent_draws: np.ndarray | None
    h_draws: np.ndarray | None
    h0_draws: np.ndarray | None
    sigma_eta2_draws: np.ndarray | None
    sv_gamma0_draws: np.ndarray | None
    sv_phi_draws: np.ndarray | None
    lambda_draws: np.ndarray | None
    factor_draws: np.ndarray | None
    h_factor_draws: np.ndarray | None
    h0_factor_draws: np.ndarray | None
    sigma_eta2_factor_draws: np.ndarray | None
    gamma_draws: np.ndarray | None
    mu_draws: np.ndarray | None
    mu_gamma_draws: np.ndarray | None


def _fit_npz_from_v1_payload(payload: dict[str, np.ndarray]) -> FitNPZ:
    variables = [str(value) for value in payload["variables"].tolist()]
    time_index = pd.DatetimeIndex(pd.to_datetime(payload["time_index"]))
    values = np.asarray(payload["values"], dtype=float)
    ds = Dataset.from_arrays(values=values, variables=variables, time_index=time_index)

    latent_values = payload.get("latent_values")
    latent_dataset = None
    if latent_values is not None:
        latent_dataset = Dataset.from_arrays(
            values=np.asarray(latent_values, dtype=float),
            variables=variables,
            time_index=time_index,
        )

    posterior = None
    if "posterior_mn" in payload:
        posterior = PosteriorNIW(
            mn=np.asarray(payload["posterior_mn"], dtype=float),
            vn=np.asarray(payload["posterior_vn"], dtype=float),
            sn=np.asarray(payload["posterior_sn"], dtype=float),
            nun=float(np.asarray(payload["posterior_nun"], dtype=float)),
        )

    return FitNPZ(
        dataset=ds,
        posterior=posterior,
        beta_draws=payload.get("beta_draws"),
        sigma_draws=payload.get("sigma_draws"),
        q_draws=payload.get("q_draws"),
        latent_dataset=latent_dataset,
        latent_draws=payload.get("latent_draws"),
        h_draws=payload.get("h_draws"),
        h0_draws=payload.get("h0_draws"),
        sigma_eta2_draws=payload.get("sigma_eta2_draws"),
        sv_gamma0_draws=payload.get("sv_gamma0_draws"),
        sv_phi_draws=payload.get("sv_phi_draws"),
        lambda_draws=payload.get("lambda_draws"),
        factor_draws=payload.get("factor_draws"),
        h_factor_draws=payload.get("h_factor_draws"),
        h0_factor_draws=payload.get("h0_factor_draws"),
        sigma_eta2_factor_draws=payload.get("sigma_eta2_factor_draws"),
        gamma_draws=payload.get("gamma_draws"),
        mu_draws=payload.get("mu_draws"),
        mu_gamma_draws=payload.get("mu_gamma_draws"),
    )


def load_fit_npz(
    path: str | Path,
    *,
    allow_legacy_pickle: bool = False,
    limits: ArtifactLoadLimits | None = None,
) -> FitNPZ:
    """Load a fit artifact.

    Parameters
    ----------
    path:
        Path to a fit NPZ artifact.
    allow_legacy_pickle:
        Set only for pre-migration artifacts from a trusted source. This may execute pickle code.
    limits:
        Metadata limits applied before NumPy reads the archive. ``None`` uses the defaults.
    """
    p = Path(path)
    with _open_artifact_npz(
        p,
        expected_kind="fit",
        allow_legacy_pickle=allow_legacy_pickle,
        limits=limits,
    ) as (npz, is_v1):
        if is_v1:
            return _fit_npz_from_v1_payload(_validate_fit_v1_payload(npz, p))

        variables = [
            str(v) for v in np.asarray(_load_npz_array(npz, "variables", p), dtype=str).tolist()
        ]
        time_index = pd.DatetimeIndex(pd.to_datetime(_load_npz_array(npz, "time_index", p)))
        values = np.asarray(_load_npz_array(npz, "values", p), dtype=float)
        ds = Dataset.from_arrays(values=values, variables=variables, time_index=time_index)

        latent_values = _optional_npz_array(npz, "latent_values", p)
        latent_dataset = None
        if latent_values is not None:
            latent_dataset = Dataset.from_arrays(
                values=np.asarray(latent_values, dtype=float),
                variables=variables,
                time_index=time_index,
            )

        mn = _optional_npz_array(npz, "posterior_mn", p)
        vn = _optional_npz_array(npz, "posterior_vn", p)
        sn = _optional_npz_array(npz, "posterior_sn", p)
        nun = _optional_npz_array(npz, "posterior_nun", p)
        posterior_parts = (mn, vn, sn, nun)
        if any(p is not None for p in posterior_parts) and not all(
            p is not None for p in posterior_parts
        ):
            raise ValueError(
                "fit_result.npz contains a partial posterior_* block; expected all of "
                "posterior_mn/posterior_vn/posterior_sn/posterior_nun or none"
            )
        posterior = None
        if all(p is not None for p in posterior_parts):
            assert mn is not None and vn is not None and sn is not None and nun is not None
            posterior = PosteriorNIW(
                mn=np.asarray(mn, dtype=float),
                vn=np.asarray(vn, dtype=float),
                sn=np.asarray(sn, dtype=float),
                nun=float(np.asarray(nun, dtype=float).reshape(())),
            )

        return FitNPZ(
            dataset=ds,
            posterior=posterior,
            beta_draws=_optional_npz_array(npz, "beta_draws", p),
            sigma_draws=_optional_npz_array(npz, "sigma_draws", p),
            q_draws=_optional_npz_array(npz, "q_draws", p),
            latent_dataset=latent_dataset,
            latent_draws=_optional_npz_array(npz, "latent_draws", p),
            h_draws=_optional_npz_array(npz, "h_draws", p),
            h0_draws=_optional_npz_array(npz, "h0_draws", p),
            sigma_eta2_draws=_optional_npz_array(npz, "sigma_eta2_draws", p),
            sv_gamma0_draws=_optional_npz_array(npz, "sv_gamma0_draws", p),
            sv_phi_draws=_optional_npz_array(npz, "sv_phi_draws", p),
            lambda_draws=_optional_npz_array(npz, "lambda_draws", p),
            factor_draws=_optional_npz_array(npz, "factor_draws", p),
            h_factor_draws=_optional_npz_array(npz, "h_factor_draws", p),
            h0_factor_draws=_optional_npz_array(npz, "h0_factor_draws", p),
            sigma_eta2_factor_draws=_optional_npz_array(npz, "sigma_eta2_factor_draws", p),
            gamma_draws=_optional_npz_array(npz, "gamma_draws", p),
            mu_draws=_optional_npz_array(npz, "mu_draws", p),
            mu_gamma_draws=_optional_npz_array(npz, "mu_gamma_draws", p),
        )


def load_forecast_npz(
    path: str | Path,
    *,
    allow_legacy_pickle: bool = False,
    limits: ArtifactLoadLimits | None = None,
) -> ForecastResult:
    """Load a forecast artifact.

    Parameters
    ----------
    path:
        Path to a forecast NPZ artifact.
    allow_legacy_pickle:
        Set only for pre-migration artifacts from a trusted source. This may execute pickle code.
    limits:
        Metadata limits applied before NumPy reads the archive. ``None`` uses the defaults.
    """
    p = Path(path)
    with _open_artifact_npz(
        p,
        expected_kind="forecast",
        allow_legacy_pickle=allow_legacy_pickle,
        limits=limits,
    ) as (npz, is_v1):
        if is_v1:
            payload, quantile_fields = _validate_forecast_v1_payload(npz, p)
            return ForecastResult(
                variables=[str(value) for value in payload["variables"].tolist()],
                horizons=[int(horizon) for horizon in payload["horizons"].tolist()],
                draws=np.asarray(payload["draws"], dtype=float),
                mean=np.asarray(payload["mean"], dtype=float),
                quantiles={
                    quantile: np.asarray(payload[field], dtype=float)
                    for quantile, field in quantile_fields
                },
                latent_draws=(
                    None
                    if "latent_draws" not in payload
                    else np.asarray(payload["latent_draws"], dtype=float)
                ),
            )

        variables = [
            str(v) for v in np.asarray(_load_npz_array(npz, "variables", p), dtype=str).tolist()
        ]
        horizons = [
            int(h) for h in np.asarray(_load_npz_array(npz, "horizons", p), dtype=int).tolist()
        ]
        draws = np.asarray(_load_npz_array(npz, "draws", p), dtype=float)
        mean = np.asarray(_load_npz_array(npz, "mean", p), dtype=float)
        latent_draws = _optional_npz_array(npz, "latent_draws", p)
        latent = None if latent_draws is None else np.asarray(latent_draws, dtype=float)

        quantiles: dict[float, np.ndarray] = {}
        for key in npz.files:
            if not key.startswith("q_"):
                continue
            try:
                q = float(key[2:])
            except ValueError:
                continue
            quantiles[q] = np.asarray(_load_npz_array(npz, key, p), dtype=float)

        return ForecastResult(
            variables=variables,
            horizons=horizons,
            draws=draws,
            mean=mean,
            quantiles=quantiles,
            latent_draws=latent,
        )


def load_run_dir(
    out_dir: str | Path,
    *,
    config_filename: str = "config.yml",
    fit_filename: str = "fit_result.npz",
    allow_legacy_pickle: bool = False,
    limits: ArtifactLoadLimits | None = None,
) -> FitResult:
    """Load a :class:`~srvar.results.FitResult` from a `srvar run` output directory.

    This function:

    1) Loads the stored draws/state from ``fit_result.npz``.
    2) Reconstructs ``ModelSpec``, ``PriorSpec``, and ``SamplerConfig`` from the saved
       ``config.yml`` (without re-loading the original CSV).

    Notes
    -----
    - The returned object is suitable for downstream analysis (IRFs/FEVD/HD) and forecasting.
    - If the saved config and saved dataset are inconsistent (e.g., variable list changed),
      config parsing may fail.
    - Set ``allow_legacy_pickle=True`` only for a trusted pre-migration artifact; it may execute
      pickle code.
    - ``limits`` controls metadata limits for the stored fit artifact. ``None`` uses the defaults.
    """
    out = Path(out_dir)
    cfg_path = out / str(config_filename)
    fit_path = out / str(fit_filename)

    if not cfg_path.exists():
        raise FileNotFoundError(f"run directory is missing config file: {cfg_path}")
    if not fit_path.exists():
        raise FileNotFoundError(f"run directory is missing fit artifact: {fit_path}")

    from .config import build_model, build_prior, build_sampler, load_config

    cfg = load_config(cfg_path)
    resolved_limits = _resolve_load_limits(limits)
    fit_npz = load_fit_npz(
        fit_path,
        allow_legacy_pickle=allow_legacy_pickle,
        limits=resolved_limits,
    )

    ds = fit_npz.dataset
    model = build_model(cfg, dataset=ds)
    prior = build_prior(cfg, dataset=ds, model=model)
    sampler, _rng = build_sampler(cfg)

    return FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=fit_npz.posterior,
        latent_dataset=fit_npz.latent_dataset,
        latent_draws=fit_npz.latent_draws,
        beta_draws=fit_npz.beta_draws,
        sigma_draws=fit_npz.sigma_draws,
        q_draws=fit_npz.q_draws,
        h_draws=fit_npz.h_draws,
        h0_draws=fit_npz.h0_draws,
        sigma_eta2_draws=fit_npz.sigma_eta2_draws,
        sv_gamma0_draws=fit_npz.sv_gamma0_draws,
        sv_phi_draws=fit_npz.sv_phi_draws,
        lambda_draws=fit_npz.lambda_draws,
        factor_draws=fit_npz.factor_draws,
        h_factor_draws=fit_npz.h_factor_draws,
        h0_factor_draws=fit_npz.h0_factor_draws,
        sigma_eta2_factor_draws=fit_npz.sigma_eta2_factor_draws,
        gamma_draws=fit_npz.gamma_draws,
        mu_draws=fit_npz.mu_draws,
        mu_gamma_draws=fit_npz.mu_gamma_draws,
    )
