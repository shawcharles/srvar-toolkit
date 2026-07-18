"""Validate local release metadata before building a package release."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

TAG_RE = re.compile(r"\Av(?P<version>\d+\.\d+\.\d+)\Z")
CHANGELOG_HEADING_RE = re.compile(
    r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PreflightResult:
    """Result of a release metadata validation run."""

    tag: str
    version: str
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def version_from_tag(tag: str) -> str:
    """Return the release version encoded by a strict ``vX.Y.Z`` tag."""

    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError("tag must be exactly vX.Y.Z, for example v0.3.0")
    return match.group("version")


def read_pyproject_version(repo_root: Path) -> str:
    """Read ``[project].version`` from ``pyproject.toml``."""

    pyproject_path = repo_root / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except FileNotFoundError as exc:
        raise ValueError("pyproject.toml is missing") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError("pyproject.toml is malformed") from exc

    if not isinstance(project, dict):
        raise ValueError("pyproject.toml [project] must be a table")
    version = project.get("version")
    if not isinstance(version, str):
        raise ValueError("pyproject.toml [project].version must be a string")
    return version


def read_source_version(repo_root: Path) -> str:
    """Read the single literal ``__version__`` assignment from ``srvar/__init__.py``."""

    source_path = repo_root / "srvar" / "__init__.py"
    try:
        source = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError("srvar/__init__.py is missing") from exc

    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        raise ValueError("srvar/__init__.py is not valid Python") from exc

    candidates: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            _is_version_target(target) for target in node.targets
        ):
            candidates.append(node)
        elif isinstance(node, ast.AnnAssign) and _is_version_target(node.target):
            candidates.append(node)
        elif isinstance(node, ast.AugAssign) and _is_version_target(node.target):
            candidates.append(node)

    if len(candidates) != 1:
        raise ValueError("srvar/__init__.py must contain exactly one __version__ assignment")

    node = candidates[0]
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
        raise ValueError("srvar.__version__ must be assigned as a literal string")
    if not isinstance(node.value.value, str):
        raise ValueError("srvar.__version__ must be assigned as a literal string")
    return node.value.value


def changelog_has_release(repo_root: Path, version: str) -> bool:
    """Return whether ``CHANGELOG.md`` has exactly one dated heading for ``version``."""

    changelog_path = repo_root / "CHANGELOG.md"
    try:
        changelog = changelog_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError("CHANGELOG.md is missing") from exc

    matches = [
        match for match in CHANGELOG_HEADING_RE.finditer(changelog) if match["version"] == version
    ]
    if not matches:
        raise ValueError(
            f"CHANGELOG.md must contain a dated release heading: ## [{version}] - YYYY-MM-DD"
        )
    if len(matches) > 1:
        raise ValueError(f"CHANGELOG.md contains duplicate release headings for {version}")

    release_date = matches[0]["date"]
    try:
        date.fromisoformat(release_date)
    except ValueError as exc:
        raise ValueError("CHANGELOG.md release date is not a valid calendar date") from exc
    return True


def validate_release(repo_root: Path, tag: str) -> PreflightResult:
    """Validate all local release metadata for ``tag``."""

    errors: list[str] = []
    try:
        version = version_from_tag(tag)
    except ValueError as exc:
        return PreflightResult(tag=tag, version="", errors=(str(exc),))

    root = repo_root.resolve()

    try:
        pyproject_version = read_pyproject_version(root)
        if pyproject_version != version:
            errors.append("pyproject.toml version does not match tag version")
    except ValueError as exc:
        errors.append(str(exc))

    try:
        source_version = read_source_version(root)
        if source_version != version:
            errors.append("srvar.__version__ does not match tag version")
    except ValueError as exc:
        errors.append(str(exc))

    try:
        changelog_has_release(root, version)
    except ValueError as exc:
        errors.append(str(exc))

    return PreflightResult(tag=tag, version=version, errors=tuple(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, exactly vX.Y.Z")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to validate; defaults to this checkout.",
    )
    args = parser.parse_args(argv)

    result = validate_release(args.repo_root, args.tag)
    if result.ok:
        print(f"release preflight passed: {result.tag} ({result.version})")
        return 0

    print("release preflight failed", file=sys.stderr)
    for error in result.errors:
        print(f"- {error}", file=sys.stderr)
    return 1


def _is_version_target(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "__version__"


if __name__ == "__main__":
    raise SystemExit(main())
