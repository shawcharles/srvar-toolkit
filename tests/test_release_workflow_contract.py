from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
TEXT = WORKFLOW.read_text(encoding="utf-8")


def _run_bodies() -> list[str]:
    bodies: list[str] = []
    lines = TEXT.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^(?P<indent>\s*)run: \|$", line)
        if not match:
            index += 1
            continue
        run_indent = len(match.group("indent"))
        index += 1
        body_lines: list[str] = []
        while index < len(lines):
            next_line = lines[index]
            if next_line.strip() and len(next_line) - len(next_line.lstrip(" ")) <= run_indent:
                break
            body_lines.append(next_line)
            index += 1
        bodies.append("\n".join(body_lines))
    return bodies


def test_tag_trigger_is_routing_only() -> None:
    assert 'tags: ["v*"]' in TEXT
    assert "TAG_RE = re.compile" in TEXT
    assert r"\Av\d+\.\d+\.\d+\Z" in TEXT


def test_no_github_expressions_inside_run_scripts() -> None:
    for body in _run_bodies():
        assert "${{" not in body


def test_validates_dispatch_ref_matches_input_before_checkout() -> None:
    validation_index = TEXT.index("id: validated-ref")
    checkout_index = TEXT.index("name: Check out selected tag")

    assert validation_index < checkout_index
    assert "EVENT_NAME: ${{ github.event_name }}" in TEXT
    assert "WORKFLOW_REF: ${{ github.ref }}" in TEXT
    assert "DISPATCH_TAG: ${{ inputs.tag }}" in TEXT
    assert 'expected_ref = f"refs/tags/{tag}"' in TEXT
    assert "workflow_dispatch must run at the selected tag ref" in TEXT
    assert "GITHUB_OUTPUT" in TEXT


def test_downstream_steps_use_validated_tag_output() -> None:
    assert "ref: ${{ steps.validated-ref.outputs.tag }}" in TEXT
    assert "TAG: ${{ steps.validated-ref.outputs.tag }}" in TEXT
    assert "name: ${{ steps.validated-ref.outputs.artifact_name }}" in TEXT
    assert "name: ${{ needs.build.outputs.artifact-name }}" in TEXT


def test_checkout_identity_and_exact_payload_are_enforced() -> None:
    assert 'git rev-parse --verify "refs/tags/${TAG}^{commit}"' in TEXT
    assert "git rev-parse --verify HEAD^{commit}" in TEXT
    assert "rm -rf dist" in TEXT
    assert "exactly one wheel and one sdist" in TEXT
    assert "if-no-files-found: error" in TEXT
    assert "python -m twine check dist/*" in TEXT
    assert "python -m venv" in TEXT
    assert "import srvar" in TEXT


def test_publish_job_is_manual_artifact_only_oidc() -> None:
    publish = TEXT[TEXT.index("publish-pypi:") :]

    assert "needs: build" in publish
    assert "runs-on: ubuntu-latest" in publish
    assert "container:" not in publish
    assert "environment: pypi" in publish
    assert "id-token: write" in publish
    assert "contents: write" not in publish
    assert "actions/checkout" not in publish
    assert "password:" not in publish
    assert "secret" not in publish.lower()
    assert "github.event_name == 'workflow_dispatch'" in publish
    assert "inputs.confirm_publish == 'PUBLISH'" in publish
    assert "pypa/gh-action-pypi-publish@" in publish


def test_third_party_actions_are_full_sha_pinned() -> None:
    action_uses = re.findall(r"uses: ([^\s#]+)", TEXT)

    assert action_uses
    for use in action_uses:
        assert re.fullmatch(r"[-\w]+/[-\w]+@[0-9a-f]{40}", use), use
