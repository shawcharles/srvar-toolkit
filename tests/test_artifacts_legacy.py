from pathlib import Path

import numpy as np
import pytest

from srvar.artifacts import load_fit_npz, load_forecast_npz, load_run_dir


def _write_legacy_fit(path: Path) -> None:
    np.savez_compressed(
        path,
        variables=np.asarray(["y"], dtype=object),
        time_index=np.asarray(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
        values=np.asarray([[1.0], [2.0]]),
        beta_draws=np.asarray([[[0.1], [0.2]]]),
        latent_values=None,
    )


def _write_legacy_forecast(path: Path) -> None:
    np.savez_compressed(
        path,
        variables=np.asarray(["y"], dtype=object),
        horizons=np.asarray([1], dtype=int),
        draws=np.asarray([[[1.0]], [[2.0]]]),
        mean=np.asarray([[1.5]]),
        latent_draws=None,
        **{"q_0.5": np.asarray([[1.5]])},
    )


def test_legacy_fit_requires_explicit_trusted_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "legacy_fit.npz"
    _write_legacy_fit(path)

    with pytest.raises(ValueError, match="legacy pickle-backed"):
        load_fit_npz(path)

    loaded = load_fit_npz(path, allow_legacy_pickle=True)
    assert loaded.dataset.variables == ["y"]
    np.testing.assert_allclose(loaded.beta_draws, np.asarray([[[0.1], [0.2]]]))
    assert loaded.latent_dataset is None


def test_legacy_forecast_requires_explicit_trusted_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "legacy_forecast.npz"
    _write_legacy_forecast(path)

    with pytest.raises(ValueError, match="legacy pickle-backed"):
        load_forecast_npz(path)

    loaded = load_forecast_npz(path, allow_legacy_pickle=True)
    assert loaded.variables == ["y"]
    assert loaded.quantiles[0.5][0, 0] == 1.5
    assert loaded.latent_draws is None


@pytest.mark.parametrize(
    "loader, artifact_kind", [(load_fit_npz, "fit"), (load_forecast_npz, "forecast")]
)
@pytest.mark.parametrize("marker_key", ["format_version", "artifact_kind"])
@pytest.mark.parametrize("allow_legacy_pickle", [False, True])
def test_incomplete_markers_never_use_legacy_retry(
    tmp_path: Path,
    loader,
    artifact_kind: str,
    marker_key: str,
    allow_legacy_pickle: bool,
) -> None:
    path = tmp_path / f"incomplete_{marker_key}_{allow_legacy_pickle}.npz"
    payload: dict[str, np.ndarray] = {"variables": np.asarray(["y"], dtype=object)}
    if marker_key == "format_version":
        payload[marker_key] = np.asarray(1, dtype=np.int64)
    else:
        payload[marker_key] = np.asarray(artifact_kind, dtype=str)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="incomplete format markers"):
        loader(path, allow_legacy_pickle=allow_legacy_pickle)


@pytest.mark.parametrize("loader", [load_fit_npz, load_forecast_npz])
def test_marked_invalid_artifacts_do_not_use_legacy_opt_in(tmp_path: Path, loader) -> None:
    wrong_kind = tmp_path / "wrong_kind.npz"
    np.savez_compressed(
        wrong_kind,
        format_version=np.asarray(1, dtype=np.int64),
        artifact_kind=np.asarray("forecast" if loader is load_fit_npz else "fit", dtype=str),
        variables=np.asarray(["y"], dtype=object),
    )
    with pytest.raises(ValueError, match="artifact_kind"):
        loader(wrong_kind, allow_legacy_pickle=True)

    malformed_marker = tmp_path / "malformed_marker.npz"
    np.savez_compressed(
        malformed_marker,
        format_version=np.asarray("1", dtype=str),
        artifact_kind=np.asarray("fit" if loader is load_fit_npz else "forecast", dtype=str),
        variables=np.asarray(["y"], dtype=object),
    )
    with pytest.raises(ValueError, match="format_version"):
        loader(malformed_marker, allow_legacy_pickle=True)

    unknown_version = tmp_path / "unknown_version.npz"
    np.savez_compressed(
        unknown_version,
        format_version=np.asarray(2, dtype=np.int64),
        artifact_kind=np.asarray("fit" if loader is load_fit_npz else "forecast", dtype=str),
        variables=np.asarray(["y"], dtype=object),
    )
    with pytest.raises(ValueError, match="unsupported format_version"):
        loader(unknown_version, allow_legacy_pickle=True)

    object_payload = tmp_path / "object_payload.npz"
    np.savez_compressed(
        object_payload,
        format_version=np.asarray(1, dtype=np.int64),
        artifact_kind=np.asarray("fit" if loader is load_fit_npz else "forecast", dtype=str),
        variables=np.asarray(["y"], dtype=object),
    )
    with pytest.raises(ValueError, match="missing required v1 field"):
        loader(object_payload, allow_legacy_pickle=True)


@pytest.mark.parametrize("loader", [load_fit_npz, load_forecast_npz])
def test_non_npz_and_missing_paths_do_not_retry_with_pickle(tmp_path: Path, loader) -> None:
    raw_pickle = tmp_path / "raw.pickle"
    raw_pickle.write_bytes(b"\x80\x04N.")
    with pytest.raises(ValueError, match="NPZ|parse"):
        loader(raw_pickle, allow_legacy_pickle=True)

    npy_path = tmp_path / "array.npy"
    np.save(npy_path, np.asarray([1.0]))
    with pytest.raises(ValueError, match="NPZ"):
        loader(npy_path, allow_legacy_pickle=True)

    with pytest.raises(FileNotFoundError):
        loader(tmp_path / "missing.npz", allow_legacy_pickle=True)


def test_load_run_dir_propagates_trusted_legacy_opt_in(tmp_path: Path) -> None:
    out = tmp_path / "legacy_run"
    out.mkdir()
    out.joinpath("config.yml").write_text(
        """\
model:
  p: 1
  include_intercept: true
prior:
  family: niw
  method: default
sampler:
  draws: 1
  burn_in: 0
  thin: 1
""",
        encoding="utf-8",
    )
    _write_legacy_fit(out / "fit_result.npz")

    with pytest.raises(ValueError, match="legacy pickle-backed"):
        load_run_dir(out)
    loaded = load_run_dir(out, allow_legacy_pickle=True)
    assert loaded.dataset.variables == ["y"]
