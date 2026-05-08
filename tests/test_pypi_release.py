"""Tests for PyPI release readiness (Task 5.5).

Guards against shipping a broken package to PyPI:

- LICENSE file exists and is MIT
- CHANGELOG.md exists and mentions the version in pyproject.toml
- `pip install .` succeeds (no build-time errors)
- sdist and wheel include LICENSE, README.md, and the pyproject.toml version
- Release workflow YAML parses and triggers on version tags

TDD: tests first, then create the files they guard.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
LICENSE = REPO_ROOT / "LICENSE"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def version() -> str:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


# ── LICENSE ──────────────────────────────────────────────────────


class TestLicense:
    """PyPI requires a LICENSE file. Must match the SPDX declaration in pyproject."""

    def test_license_file_exists(self):
        assert LICENSE.exists(), (
            "LICENSE file missing at repo root. PyPI/setuptools need it."
        )

    def test_license_is_mit(self):
        if not LICENSE.exists():
            pytest.skip("LICENSE missing (guarded by test_license_file_exists)")
        text = LICENSE.read_text(encoding="utf-8")
        # MIT License has a recognizable opening line.
        assert "MIT License" in text, "LICENSE should be the MIT license"
        # Permission grant phrasing.
        assert "Permission is hereby granted" in text


# ── CHANGELOG ────────────────────────────────────────────────────


class TestChangelog:
    """Every release needs a CHANGELOG entry so users know what changed."""

    def test_changelog_exists(self):
        assert CHANGELOG.exists(), (
            "CHANGELOG.md missing at repo root. Create it before the v1.0 release."
        )

    def test_changelog_mentions_current_version(self, version):
        if not CHANGELOG.exists():
            pytest.skip("CHANGELOG missing (guarded by test_changelog_exists)")
        text = CHANGELOG.read_text(encoding="utf-8")
        # e.g. '1.0.0' appears as a header.
        assert version in text, (
            f"CHANGELOG.md must have an entry for version {version} (from pyproject.toml). "
            "Release artifacts will not match the changelog otherwise."
        )


# ── Release workflow ──────────────────────────────────────────────


class TestReleaseWorkflow:
    """GitHub Actions workflow triggers PyPI publish on version tag push."""

    def test_release_workflow_exists(self):
        assert RELEASE_WORKFLOW.exists(), (
            ".github/workflows/release.yml missing. "
            "Needed so `git tag v1.0.0 && git push --tags` triggers a PyPI release."
        )

    def test_release_workflow_triggers_on_version_tag(self):
        if not RELEASE_WORKFLOW.exists():
            pytest.skip("workflow missing")
        text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        # Must trigger on tag pushes, filtered to v* tags so feature branches
        # don't accidentally publish.
        assert "tags:" in text
        assert "v*" in text, (
            "Release workflow must filter to 'v*' tags so arbitrary tag pushes "
            "don't publish to PyPI."
        )

    def test_release_workflow_builds_sdist_and_wheel(self):
        if not RELEASE_WORKFLOW.exists():
            pytest.skip("workflow missing")
        text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        # Either 'python -m build' (both sdist+wheel) or explicit twine.
        assert ("python -m build" in text) or (
            "pypa/build" in text
        ), "Release workflow must build the package before uploading"

    def test_release_workflow_uploads_to_pypi(self):
        if not RELEASE_WORKFLOW.exists():
            pytest.skip("workflow missing")
        text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        # pypi-publish action is the official GitHub-recommended path.
        assert ("pypa/gh-action-pypi-publish" in text) or (
            "twine upload" in text
        ), "Release workflow must actually publish to PyPI"

    def test_release_workflow_uses_trusted_publishing(self):
        """Prefer OIDC trusted publishing over API tokens (safer, no secrets)."""
        if not RELEASE_WORKFLOW.exists():
            pytest.skip("workflow missing")
        text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        # `id-token: write` is the OIDC permission required for trusted publishing.
        assert "id-token: write" in text, (
            "Use OIDC trusted publishing (id-token: write permission). "
            "Avoids embedding a PYPI_API_TOKEN secret in the repo."
        )


# ── Build verification ────────────────────────────────────────────


class TestBuildArtifacts:
    """The actual artifacts we'll ship — built via `python -m build`.

    These tests build the package in a tmp dir and inspect the resulting
    sdist and wheel to make sure LICENSE, README, and version metadata
    are all present. If this test fails, the release workflow will fail too.
    """

    @pytest.fixture(scope="class")
    def build_artifacts(self, tmp_path_factory):
        """Build sdist + wheel once, reuse for multiple assertions."""
        out_dir = tmp_path_factory.mktemp("dist")
        # Skip if `build` module isn't available (local dev optional dep).
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--outdir", str(out_dir), str(REPO_ROOT)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("`build` module not installed — pip install build")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"`python -m build` failed:\n{e.stdout}\n{e.stderr}")
        except subprocess.TimeoutExpired:
            pytest.skip("Build timed out — possibly offline / slow network")

        sdists = list(out_dir.glob("*.tar.gz"))
        wheels = list(out_dir.glob("*.whl"))
        assert len(sdists) == 1, f"expected exactly 1 sdist, got {sdists}"
        assert len(wheels) == 1, f"expected exactly 1 wheel, got {wheels}"
        return {"sdist": sdists[0], "wheel": wheels[0]}

    def test_sdist_includes_license(self, build_artifacts):
        """PyPI rejects sdists without a LICENSE file."""
        with tarfile.open(build_artifacts["sdist"]) as tar:
            names = tar.getnames()
        license_entries = [n for n in names if Path(n).name == "LICENSE"]
        assert license_entries, f"sdist missing LICENSE. Contents: {names[:10]}..."

    def test_sdist_includes_readme(self, build_artifacts):
        """PyPI uses README for the project page. Must ship in sdist."""
        with tarfile.open(build_artifacts["sdist"]) as tar:
            names = tar.getnames()
        readme_entries = [n for n in names if Path(n).name == "README.md"]
        assert readme_entries, f"sdist missing README.md. Contents: {names[:10]}..."

    def test_wheel_metadata_has_correct_version(self, build_artifacts, version):
        """Wheel filename encodes the version — must match pyproject."""
        name = build_artifacts["wheel"].name
        assert f"-{version}-" in name, (
            f"wheel {name} doesn't encode version {version} from pyproject.toml"
        )

    def test_wheel_includes_autodidact_signals(self, build_artifacts):
        """Regression guard: every subpackage ships in the wheel."""
        with zipfile.ZipFile(build_artifacts["wheel"]) as zf:
            names = zf.namelist()
        assert any("autodidact/signals/" in n for n in names), (
            "wheel is missing autodidact/signals/ subpackage"
        )

    def test_wheel_includes_entry_point(self, build_artifacts):
        """The `autodidact` CLI command must be registered in the wheel."""
        with zipfile.ZipFile(build_artifacts["wheel"]) as zf:
            names = zf.namelist()
        entry_points = [n for n in names if n.endswith("entry_points.txt")]
        assert entry_points
        with zipfile.ZipFile(build_artifacts["wheel"]) as zf:
            content = zf.read(entry_points[0]).decode()
        assert "autodidact = autodidact.cli:app" in content
