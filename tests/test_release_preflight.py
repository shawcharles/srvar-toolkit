from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = REPO_ROOT / "scripts" / "release_preflight.py"
SPEC = importlib.util.spec_from_file_location("release_preflight", PREFLIGHT_PATH)
assert SPEC is not None
release_preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_preflight
assert SPEC.loader is not None
SPEC.loader.exec_module(release_preflight)


def write_release_tree(
    root: Path,
    *,
    pyproject_version: str | int = "0.3.0",
    source: str = '__version__ = "0.3.0"\n',
    changelog: str = "## [0.3.0] - 2026-04-06\n\n### Added\n\n- Release.\n",
) -> None:
    quoted_version = (
        f'"{pyproject_version}"' if isinstance(pyproject_version, str) else str(pyproject_version)
    )
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            version = {quoted_version}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    package = root / "srvar"
    package.mkdir()
    (package / "__init__.py").write_text(source, encoding="utf-8")
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")


def errors_for(root: Path, tag: str = "v0.3.0") -> tuple[str, ...]:
    return release_preflight.validate_release(root, tag).errors


def test_valid_release_tree_passes(tmp_path: Path) -> None:
    write_release_tree(tmp_path)

    result = release_preflight.validate_release(tmp_path, "v0.3.0")

    assert result.ok
    assert result.version == "0.3.0"


def test_checkout_current_release_metadata_passes() -> None:
    result = release_preflight.validate_release(REPO_ROOT, "v0.3.1")

    assert result.ok


def test_rejects_malformed_tag_forms(tmp_path: Path) -> None:
    write_release_tree(tmp_path)

    bad_tags = [
        "0.3.0",
        "refs/tags/v0.3.0",
        "v0.3.0rc1",
        " v0.3.0",
        "v0.3.0 ",
        "v0.3.0; echo leaked",
    ]

    for tag in bad_tags:
        assert release_preflight.validate_release(tmp_path, tag).errors == (
            "tag must be exactly vX.Y.Z, for example v0.3.0",
        )


def test_rejects_pyproject_version_mismatch(tmp_path: Path) -> None:
    write_release_tree(tmp_path, pyproject_version="0.3.1")

    assert errors_for(tmp_path) == ("pyproject.toml version does not match tag version",)


def test_rejects_missing_pyproject(tmp_path: Path) -> None:
    write_release_tree(tmp_path)
    (tmp_path / "pyproject.toml").unlink()

    assert errors_for(tmp_path) == ("pyproject.toml is missing",)


def test_rejects_malformed_pyproject(tmp_path: Path) -> None:
    write_release_tree(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project\nversion = '0.3.0'\n", encoding="utf-8")

    assert errors_for(tmp_path) == ("pyproject.toml is malformed",)


def test_rejects_non_string_pyproject_version(tmp_path: Path) -> None:
    write_release_tree(tmp_path, pyproject_version=3)

    assert errors_for(tmp_path) == ("pyproject.toml [project].version must be a string",)


def test_rejects_non_table_pyproject_project(tmp_path: Path) -> None:
    write_release_tree(tmp_path)
    (tmp_path / "pyproject.toml").write_text('project = "not-a-table"\n', encoding="utf-8")

    assert errors_for(tmp_path) == ("pyproject.toml [project] must be a table",)


def test_rejects_missing_source_version(tmp_path: Path) -> None:
    write_release_tree(tmp_path, source='OTHER = "0.3.0"\n')

    assert errors_for(tmp_path) == (
        "srvar/__init__.py must contain exactly one __version__ assignment",
    )


def test_rejects_missing_source_file(tmp_path: Path) -> None:
    write_release_tree(tmp_path)
    (tmp_path / "srvar" / "__init__.py").unlink()

    assert errors_for(tmp_path) == ("srvar/__init__.py is missing",)


def test_rejects_malformed_source_file(tmp_path: Path) -> None:
    write_release_tree(tmp_path, source='__version__ = "0.3.0"\nif True print("bad")\n')

    assert errors_for(tmp_path) == ("srvar/__init__.py is not valid Python",)


def test_rejects_multiple_source_version_assignments(tmp_path: Path) -> None:
    write_release_tree(tmp_path, source='__version__ = "0.3.0"\n__version__ = "0.3.0"\n')

    assert errors_for(tmp_path) == (
        "srvar/__init__.py must contain exactly one __version__ assignment",
    )


def test_rejects_non_literal_source_version(tmp_path: Path) -> None:
    write_release_tree(tmp_path, source='__version__ = ".".join(["0", "3", "0"])\n')

    assert errors_for(tmp_path) == ("srvar.__version__ must be assigned as a literal string",)


def test_rejects_annotated_source_version(tmp_path: Path) -> None:
    write_release_tree(tmp_path, source='__version__: str = "0.3.0"\n')

    assert errors_for(tmp_path) == ("srvar.__version__ must be assigned as a literal string",)


def test_rejects_source_version_mismatch(tmp_path: Path) -> None:
    write_release_tree(tmp_path, source='__version__ = "0.3.1"\n')

    assert errors_for(tmp_path) == ("srvar.__version__ does not match tag version",)


def test_rejects_changelog_with_only_unreleased(tmp_path: Path) -> None:
    write_release_tree(tmp_path, changelog="## [Unreleased]\n\n- Pending.\n")

    assert errors_for(tmp_path) == (
        "CHANGELOG.md must contain a dated release heading: ## [0.3.0] - YYYY-MM-DD",
    )


def test_rejects_missing_changelog(tmp_path: Path) -> None:
    write_release_tree(tmp_path)
    (tmp_path / "CHANGELOG.md").unlink()

    assert errors_for(tmp_path) == ("CHANGELOG.md is missing",)


def test_rejects_undated_changelog_heading(tmp_path: Path) -> None:
    write_release_tree(tmp_path, changelog="## [0.3.0]\n\n- Release.\n")

    assert errors_for(tmp_path) == (
        "CHANGELOG.md must contain a dated release heading: ## [0.3.0] - YYYY-MM-DD",
    )


def test_rejects_invalid_calendar_date(tmp_path: Path) -> None:
    write_release_tree(tmp_path, changelog="## [0.3.0] - 2026-02-30\n\n- Release.\n")

    assert errors_for(tmp_path) == ("CHANGELOG.md release date is not a valid calendar date",)


def test_rejects_duplicate_changelog_release_headings(tmp_path: Path) -> None:
    write_release_tree(
        tmp_path,
        changelog=(
            "## [0.3.0] - 2026-04-06\n\n- Release.\n\n## [0.3.0] - 2026-04-07\n\n- Duplicate.\n"
        ),
    )

    assert errors_for(tmp_path) == ("CHANGELOG.md contains duplicate release headings for 0.3.0",)


def test_rejects_version_mentioned_only_in_changelog_prose(tmp_path: Path) -> None:
    write_release_tree(tmp_path, changelog="## [Unreleased]\n\n- Preparing 0.3.0 soon.\n")

    assert errors_for(tmp_path) == (
        "CHANGELOG.md must contain a dated release heading: ## [0.3.0] - YYYY-MM-DD",
    )


def test_cli_reports_success(tmp_path: Path) -> None:
    write_release_tree(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT_PATH), "--tag", "v0.3.0", "--repo-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "release preflight passed: v0.3.0 (0.3.0)"
    assert completed.stderr == ""


def test_cli_reports_category_without_file_content(tmp_path: Path) -> None:
    secret_like_text = "token=do-not-print"
    write_release_tree(
        tmp_path,
        changelog=f"## [Unreleased]\n\n- {secret_like_text}\n",
    )

    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT_PATH), "--tag", "v0.3.0", "--repo-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "CHANGELOG.md must contain a dated release heading" in completed.stderr
    assert secret_like_text not in completed.stderr


def test_cli_never_echoes_invalid_tag_or_observed_version_values(tmp_path: Path) -> None:
    secret_like_text = "token=do-not-print"
    write_release_tree(
        tmp_path,
        pyproject_version=secret_like_text,
        source=f'__version__ = "{secret_like_text}"\n',
    )

    mismatched = subprocess.run(
        [sys.executable, str(PREFLIGHT_PATH), "--tag", "v0.3.0", "--repo-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    invalid_tag = subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT_PATH),
            "--tag",
            secret_like_text,
            "--repo-root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert mismatched.returncode == 1
    assert invalid_tag.returncode == 1
    assert secret_like_text not in mismatched.stderr
    assert secret_like_text not in invalid_tag.stderr
