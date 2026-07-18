from dataclasses import FrozenInstanceError
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pytest

import srvar.artifacts as artifacts
from srvar.artifacts import ArtifactLoadLimits, load_fit_npz, load_forecast_npz, load_run_dir


def _write_zip(
    path: Path, entries: list[tuple[str, bytes]], *, compression: int = ZIP_STORED
) -> None:
    with ZipFile(path, "w", compression=compression) as archive:
        for name, data in entries:
            archive.writestr(name, data)


def _write_archive_size_case(path: Path) -> None:
    _write_zip(path, [("field.npy", b"x")])


def _write_member_count_case(path: Path) -> None:
    _write_zip(path, [("a.npy", b"x"), ("b.npy", b"x"), ("c.npy", b"x")])


def _write_member_size_case(path: Path) -> None:
    _write_zip(path, [("field.npy", b"xx")])


def _write_total_size_case(path: Path) -> None:
    _write_zip(path, [("a.npy", b"x"), ("b.npy", b"x")])


def _write_expansion_ratio_case(path: Path) -> None:
    _write_zip(path, [("field.npy", b"x" * 1024)], compression=ZIP_DEFLATED)


@pytest.mark.parametrize(
    ("write_case", "limit_kwargs", "message"),
    [
        (_write_archive_size_case, {"max_archive_bytes": 1}, "archive size .*max_archive_bytes"),
        (_write_member_count_case, {"max_member_count": 2}, "member count .*max_member_count"),
        (
            _write_member_size_case,
            {"max_member_uncompressed_bytes": 1},
            "member uncompressed size .*max_member_uncompressed_bytes",
        ),
        (
            _write_total_size_case,
            {
                "max_member_uncompressed_bytes": 2,
                "max_total_uncompressed_bytes": 1,
            },
            "total uncompressed size .*max_total_uncompressed_bytes",
        ),
        (
            _write_expansion_ratio_case,
            {"max_expansion_ratio": 2.0},
            "expansion ratio .*max_expansion_ratio",
        ),
    ],
)
@pytest.mark.parametrize("loader", [load_fit_npz, load_forecast_npz])
@pytest.mark.parametrize("allow_legacy_pickle", [False, True])
def test_limits_reject_before_any_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_case,
    limit_kwargs: dict[str, int | float],
    message: str,
    loader,
    allow_legacy_pickle: bool,
) -> None:
    path = tmp_path / "oversized.npz"
    write_case(path)
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("np.load must not run when metadata limits reject an artifact")

    monkeypatch.setattr(artifacts.np, "load", fail_if_called)

    with pytest.raises(ValueError, match=message):
        loader(
            path,
            allow_legacy_pickle=allow_legacy_pickle,
            limits=ArtifactLoadLimits(**limit_kwargs),
        )

    assert calls == 0


@pytest.mark.parametrize("loader", [load_fit_npz, load_forecast_npz])
@pytest.mark.parametrize("allow_legacy_pickle", [False, True])
def test_directory_is_preserved_before_any_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader,
    allow_legacy_pickle: bool,
) -> None:
    directory = tmp_path / "artifact-directory"
    directory.mkdir()
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("np.load must not run for a directory")

    monkeypatch.setattr(artifacts.np, "load", fail_if_called)

    with pytest.raises(IsADirectoryError):
        loader(directory, allow_legacy_pickle=allow_legacy_pickle)

    assert calls == 0


def test_artifact_load_limits_validate_and_are_immutable() -> None:
    limits = ArtifactLoadLimits(
        max_archive_bytes=2,
        max_member_count=3,
        max_member_uncompressed_bytes=4,
        max_total_uncompressed_bytes=5,
        max_expansion_ratio=6.0,
    )
    assert limits.max_total_uncompressed_bytes == 5

    with pytest.raises(ValueError, match="max_archive_bytes"):
        ArtifactLoadLimits(max_archive_bytes=0)
    with pytest.raises(ValueError, match="max_member_count"):
        ArtifactLoadLimits(max_member_count=-1)
    with pytest.raises(ValueError, match="max_member_uncompressed_bytes"):
        ArtifactLoadLimits(max_member_uncompressed_bytes=True)
    with pytest.raises(ValueError, match="max_expansion_ratio"):
        ArtifactLoadLimits(max_expansion_ratio=float("inf"))
    with pytest.raises(FrozenInstanceError):
        limits.max_archive_bytes = 10


def _write_legacy_fit(path: Path) -> None:
    np.savez_compressed(
        path,
        variables=np.asarray(["y"], dtype=object),
        time_index=np.asarray(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
        values=np.asarray([[1.0], [2.0]]),
        beta_draws=np.asarray([[[0.1], [0.2]]]),
        latent_values=None,
    )


def test_trusted_legacy_retry_reuses_the_same_binary_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "legacy_fit.npz"
    _write_legacy_fit(path)
    real_load = artifacts.np.load
    calls: list[tuple[object, bool]] = []

    def record_load(file, *args, **kwargs):
        calls.append((file, kwargs["allow_pickle"]))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(artifacts.np, "load", record_load)

    loaded = load_fit_npz(path, allow_legacy_pickle=True)

    assert loaded.dataset.variables == ["y"]
    assert [allow_pickle for _, allow_pickle in calls] == [False, True]
    assert calls[0][0] is calls[1][0]
    assert not isinstance(calls[0][0], (str, Path))


def test_load_run_dir_forwards_the_exact_limits_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run"
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
    out.joinpath("fit_result.npz").write_bytes(b"placeholder")
    limits = ArtifactLoadLimits(max_archive_bytes=2)
    captured: dict[str, object] = {}

    class SentinelError(Exception):
        pass

    def fake_load_fit(path, *, allow_legacy_pickle, limits):
        captured["path"] = path
        captured["allow_legacy_pickle"] = allow_legacy_pickle
        captured["limits"] = limits
        raise SentinelError

    monkeypatch.setattr(artifacts, "load_fit_npz", fake_load_fit)

    with pytest.raises(SentinelError):
        load_run_dir(out, limits=limits)

    assert captured["path"] == out / "fit_result.npz"
    assert captured["allow_legacy_pickle"] is False
    assert captured["limits"] is limits
