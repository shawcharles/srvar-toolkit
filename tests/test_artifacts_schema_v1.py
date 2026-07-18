from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

import srvar.artifacts as artifacts
from srvar.artifacts import load_fit_npz, load_forecast_npz


def _fit_payload() -> dict[str, np.ndarray]:
    t, n, d, k, f, state_time = 5, 2, 3, 4, 1, 3
    return {
        "format_version": np.asarray(1, dtype=np.int64),
        "artifact_kind": np.asarray("fit", dtype=str),
        "variables": np.asarray(["y1", "y2"], dtype=str),
        "time_index": np.datetime64("2000-01-01") + np.arange(t).astype("timedelta64[D]"),
        "values": np.ones((t, n)),
        "latent_values": np.ones((t, n)),
        "latent_draws": np.ones((d, t, n)),
        "beta_draws": np.ones((d, k, n)),
        "sigma_draws": np.ones((d, n, n)),
        "q_draws": np.ones((d, n, n)),
        "posterior_mn": np.ones((k, n)),
        "posterior_vn": np.ones((k, k)),
        "posterior_sn": np.ones((n, n)),
        "posterior_nun": np.asarray(4.0),
        "h_draws": np.ones((d, state_time, n)),
        "h0_draws": np.ones((d, n)),
        "sigma_eta2_draws": np.ones((d, n)),
        "sv_gamma0_draws": np.ones((d, n)),
        "sv_phi_draws": np.ones((d, n)),
        "lambda_draws": np.ones((d, n, f)),
        "factor_draws": np.ones((d, state_time, f)),
        "h_factor_draws": np.ones((d, state_time, f)),
        "h0_factor_draws": np.ones((d, f)),
        "sigma_eta2_factor_draws": np.ones((d, f)),
        "gamma_draws": np.ones((d, k), dtype=bool),
        "mu_draws": np.ones((d, n)),
        "mu_gamma_draws": np.ones((d, n)),
    }


def _forecast_payload() -> dict[str, np.ndarray]:
    d, h, n = 2, 3, 2
    return {
        "format_version": np.asarray(1, dtype=np.int64),
        "artifact_kind": np.asarray("forecast", dtype=str),
        "variables": np.asarray(["y1", "y2"], dtype=str),
        "horizons": np.asarray([1, 2, 3], dtype=np.int64),
        "draws": np.ones((d, h, n)),
        "mean": np.ones((h, n)),
        "latent_draws": np.ones((d, h, n)),
        "q_0.5": np.ones((h, n)),
        "q_1e-05": np.ones((h, n)),
    }


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _reject_fit_before_construction(
    monkeypatch: pytest.MonkeyPatch, path: Path, match: str
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("fit domain construction must not run")

    monkeypatch.setattr(artifacts.Dataset, "from_arrays", fail)
    monkeypatch.setattr(artifacts, "PosteriorNIW", fail)
    monkeypatch.setattr(artifacts, "FitNPZ", fail)
    with pytest.raises(ValueError, match=match):
        load_fit_npz(path)


def _reject_forecast_before_construction(
    monkeypatch: pytest.MonkeyPatch, path: Path, match: str
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("forecast domain construction must not run")

    monkeypatch.setattr(artifacts, "ForecastResult", fail)
    with pytest.raises(ValueError, match=match):
        load_forecast_npz(path)


@pytest.mark.parametrize(
    ("loader", "expected_kind", "marker_payload", "sentinel", "match"),
    [
        (
            load_fit_npz,
            "fit",
            {
                "format_version": np.asarray(8675309, dtype=np.int64),
                "artifact_kind": np.asarray("fit", dtype=str),
            },
            "8675309",
            "unsupported format_version",
        ),
        (
            load_forecast_npz,
            "forecast",
            {
                "format_version": np.asarray(1, dtype=np.int64),
                "artifact_kind": np.asarray("attacker-controlled-kind-sentinel", dtype=str),
            },
            "attacker-controlled-kind-sentinel",
            "artifact_kind must be 'forecast'",
        ),
    ],
)
def test_marker_errors_never_echo_untrusted_marker_values(
    tmp_path: Path,
    loader,
    expected_kind: str,
    marker_payload: dict[str, np.ndarray],
    sentinel: str,
    match: str,
) -> None:
    path = tmp_path / f"bad_{expected_kind}_marker.npz"
    _write_npz(path, marker_payload)

    with pytest.raises(ValueError, match=match) as exc_info:
        loader(path)

    assert sentinel not in str(exc_info.value)


def test_maximal_fit_v1_accepts_boolean_gamma_and_state_time_not_equal_dataset_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fit.npz"
    _write_npz(path, _fit_payload())

    loaded = load_fit_npz(path)

    assert loaded.dataset.T == 5
    assert loaded.h_draws is not None and loaded.h_draws.shape[1] == 3
    assert loaded.gamma_draws is not None and loaded.gamma_draws.dtype == bool


@pytest.mark.parametrize(
    "field",
    [
        "values",
        "latent_values",
        "latent_draws",
        "beta_draws",
        "sigma_draws",
        "q_draws",
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
        "mu_draws",
        "mu_gamma_draws",
    ],
)
def test_fit_real_numeric_fields_reject_boolean_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    path = tmp_path / f"boolean_{field}.npz"
    payload = _fit_payload()
    payload[field] = np.ones(payload[field].shape, dtype=bool)
    _write_npz(path, payload)

    _reject_fit_before_construction(monkeypatch, path, field)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update({"unknown": np.asarray(1.0)}), "unknown v1 field"),
        (lambda payload: payload.pop("values"), "missing required v1 field"),
        (
            lambda payload: payload.update(
                {"time_index": np.asarray(["2000", "2000"], dtype="datetime64[D]")}
            ),
            "duplicate timestamps",
        ),
        (lambda payload: payload.update({"beta_draws": np.ones((2, 4, 2))}), "draw count"),
        (
            lambda payload: payload.update({"gamma_draws": np.ones((3, 5), dtype=bool)}),
            "coefficient width",
        ),
        (lambda payload: payload.update({"h_factor_draws": np.ones((3, 4, 1))}), "state-time"),
        (lambda payload: payload.pop("posterior_sn"), "partial posterior"),
    ],
)
def test_fit_v1_schema_rejects_cross_field_contracts_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, match: str
) -> None:
    path = tmp_path / "invalid_fit.npz"
    payload = _fit_payload()
    mutate(payload)
    _write_npz(path, payload)

    _reject_fit_before_construction(monkeypatch, path, match)


def test_fit_v1_rejects_duplicate_member_names_before_field_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "duplicate_member.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in _fit_payload().items():
            archive.writestr(f"{name}.npy", _npy_bytes(value))
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("variables.npy", _npy_bytes(np.asarray(["other"], dtype=str)))

    def fail_marker_validation(*args, **kwargs):
        raise AssertionError("duplicate member validation must precede marker lookup")

    monkeypatch.setattr(artifacts, "_validate_marker", fail_marker_validation)
    _reject_fit_before_construction(monkeypatch, path, "duplicate v1 field")


def test_fit_v1_retains_missing_dataset_values(tmp_path: Path) -> None:
    path = tmp_path / "missing_values.npz"
    payload = _fit_payload()
    payload["values"][0, 0] = np.nan
    _write_npz(path, payload)

    assert np.isnan(load_fit_npz(path).dataset.values[0, 0])


def test_maximal_forecast_v1_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "forecast.npz"
    _write_npz(path, _forecast_payload())

    loaded = load_forecast_npz(path)

    assert set(loaded.quantiles) == {0.5, 1e-05}
    assert loaded.latent_draws is not None


@pytest.mark.parametrize("field", ["draws", "mean", "latent_draws", "q_0.5"])
def test_forecast_real_numeric_fields_reject_boolean_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    path = tmp_path / f"boolean_{field}.npz"
    payload = _forecast_payload()
    payload[field] = np.ones(payload[field].shape, dtype=bool)
    _write_npz(path, payload)

    _reject_forecast_before_construction(monkeypatch, path, field)


@pytest.mark.parametrize("field", ["q_.5", "q_0.50", "q_5e-01", "q_nan", "q_0", "q_1"])
def test_forecast_rejects_noncanonical_or_invalid_quantile_names_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    path = tmp_path / "bad_quantile.npz"
    payload = _forecast_payload()
    payload[field] = np.ones((3, 2))
    _write_npz(path, payload)

    _reject_forecast_before_construction(monkeypatch, path, "quantile field")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update({"unknown": np.asarray(1.0)}), "unknown v1 field"),
        (lambda payload: payload.pop("mean"), "missing required v1 field"),
        (lambda payload: payload.update({"mean": np.ones((2, 2))}), "mean"),
        (lambda payload: payload.update({"q_0.5": np.ones((2, 2))}), "q_0.5"),
    ],
)
def test_forecast_v1_schema_rejects_contracts_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, match: str
) -> None:
    path = tmp_path / "invalid_forecast.npz"
    payload = _forecast_payload()
    mutate(payload)
    _write_npz(path, payload)

    _reject_forecast_before_construction(monkeypatch, path, match)


def test_legacy_paths_do_not_call_v1_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_path = tmp_path / "legacy_fit.npz"
    forecast_path = tmp_path / "legacy_forecast.npz"
    np.savez_compressed(
        fit_path,
        variables=np.asarray(["y"], dtype=object),
        time_index=np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
        values=np.asarray([[1.0]]),
    )
    np.savez_compressed(
        forecast_path,
        variables=np.asarray(["y"], dtype=object),
        horizons=np.asarray([1]),
        draws=np.asarray([[[1.0]]]),
        mean=np.asarray([[1.0]]),
    )

    def fail(*args, **kwargs):
        raise AssertionError("v1 validator must not run for markerless legacy artifacts")

    monkeypatch.setattr(artifacts, "_validate_fit_v1_payload", fail)
    monkeypatch.setattr(artifacts, "_validate_forecast_v1_payload", fail)

    assert load_fit_npz(fit_path, allow_legacy_pickle=True).dataset.variables == ["y"]
    assert load_forecast_npz(forecast_path, allow_legacy_pickle=True).variables == ["y"]
