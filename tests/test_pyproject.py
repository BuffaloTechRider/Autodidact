"""Tests for pyproject.toml — packaging configuration (Task 5.1).

These tests read pyproject.toml and verify that the packaging metadata
matches the requirements for a shipping v1.0 PyPI package:

- project name, version, description, license, python requires
- all runtime dependencies declared under [project].dependencies
- optional extras: [bedrock] for boto3, [openai] for openai, [pdf] for pymupdf
- CLI entry point: `autodidact = autodidact.cli:app`
- all autodidact submodules are discoverable (including signals/)

TDD: tests first, then the pyproject.toml is updated to pass them.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    """Parsed pyproject.toml."""
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


# ── Project metadata ─────────────────────────────────────────────


class TestProjectMetadata:
    """Required [project] fields."""

    def test_has_name(self, pyproject):
        assert pyproject["project"]["name"] == "autodidact"

    def test_has_version(self, pyproject):
        version = pyproject["project"].get("version")
        assert version, "version must be set"
        # Accept 1.x series for v1.0 release.
        assert version.startswith("1."), f"expected 1.x, got {version}"

    def test_has_description(self, pyproject):
        desc = pyproject["project"].get("description", "")
        assert len(desc) > 10, "description must be non-trivial"

    def test_has_readme(self, pyproject):
        assert pyproject["project"].get("readme") == "README.md"

    def test_has_license(self, pyproject):
        lic = pyproject["project"].get("license")
        # Accept either the SPDX string form or a {text=...} table.
        assert lic is not None
        if isinstance(lic, str):
            assert "MIT" in lic
        else:
            assert "MIT" in (lic.get("text") or "")

    def test_requires_python_310_or_newer(self, pyproject):
        req = pyproject["project"].get("requires-python", "")
        assert ">=3.10" in req or ">=3.11" in req, (
            f"must require Python 3.10+, got {req!r}"
        )


# ── Runtime dependencies ─────────────────────────────────────────


class TestDependencies:
    """All runtime dependencies are declared under [project].dependencies.

    This was a silent bug in the initial pyproject.toml where the `dependencies`
    list was written *after* [project.urls], making it a stray top-level key
    instead of a [project] field. Tests guard against that regression.
    """

    def test_dependencies_is_inside_project_table(self, pyproject):
        """Critical: dependencies must be a [project] key, not a stray top-level one."""
        assert "dependencies" in pyproject["project"], (
            "Runtime dependencies must be declared under [project].dependencies "
            "(not as a top-level key after [project.urls])."
        )

    def test_dependencies_is_list(self, pyproject):
        deps = pyproject["project"]["dependencies"]
        assert isinstance(deps, list)
        assert len(deps) > 0

    def test_required_deps_declared(self, pyproject):
        """All runtime deps imported by autodidact.* must be declared."""
        deps = pyproject["project"]["dependencies"]
        dep_names = {_package_name(d) for d in deps}

        required = {"numpy", "faiss-cpu", "pydantic", "requests", "rich", "typer", "pyyaml"}
        missing = required - dep_names
        assert not missing, f"Missing required deps: {missing}"


# ── Optional extras ──────────────────────────────────────────────


class TestOptionalExtras:
    """[project.optional-dependencies] must declare bedrock, openai, pdf, dev."""

    def test_has_bedrock_extra(self, pyproject):
        extras = pyproject["project"].get("optional-dependencies", {})
        assert "bedrock" in extras
        names = {_package_name(d) for d in extras["bedrock"]}
        assert "boto3" in names, "bedrock extra must include boto3"

    def test_openai_is_core_dependency(self, pyproject):
        deps = {_package_name(d) for d in pyproject["project"].get("dependencies", [])}
        assert "openai" in deps, "openai must be a core dependency (used by Google, OpenRouter, etc.)"

    def test_has_pdf_extra(self, pyproject):
        """[pdf] extra for document ingestion (R9 AC4)."""
        extras = pyproject["project"].get("optional-dependencies", {})
        assert "pdf" in extras, (
            "pdf extra required for PDF ingestion (R9 AC4). "
            "Install with: pip install autodidact[pdf]"
        )
        names = {_package_name(d) for d in extras["pdf"]}
        # Accept any of the common PDF-reading libs.
        assert names & {"pymupdf", "pdfplumber", "pypdf"}, (
            f"pdf extra must include a PDF reader; got {names}"
        )

    def test_has_dev_extra(self, pyproject):
        extras = pyproject["project"].get("optional-dependencies", {})
        assert "dev" in extras
        names = {_package_name(d) for d in extras["dev"]}
        assert "pytest" in names


# ── Entry points ─────────────────────────────────────────────────


class TestEntryPoints:
    """CLI entry point: `autodidact` → autodidact.cli:app."""

    def test_autodidact_script_registered(self, pyproject):
        scripts = pyproject["project"].get("scripts", {})
        assert "autodidact" in scripts
        assert scripts["autodidact"] == "autodidact.cli:app"


# ── Build config ─────────────────────────────────────────────────


class TestBuildConfig:
    """Build backend and package discovery must cover all submodules."""

    def test_build_system_uses_setuptools(self, pyproject):
        bs = pyproject["build-system"]
        assert "setuptools" in str(bs.get("requires", []))
        assert bs.get("build-backend") == "setuptools.build_meta"

    def test_packages_includes_all_autodidact_submodules(self, pyproject):
        """autodidact.signals must be picked up by package discovery."""
        tool = pyproject.get("tool", {})
        setuptools_cfg = tool.get("setuptools", {})
        packages = setuptools_cfg.get("packages", {})

        # Accept either explicit list or find-based discovery.
        if isinstance(packages, dict):
            find = packages.get("find", {})
            include = find.get("include", [])
            assert any("autodidact" in inc for inc in include), (
                f"package discovery must include autodidact; got {include}"
            )
        elif isinstance(packages, list):
            assert "autodidact" in packages
            assert "autodidact.signals" in packages


# ── URLs ─────────────────────────────────────────────────────────


class TestProjectUrls:
    """[project.urls] must point at the current repo (renamed to Autodidact)."""

    def test_repository_url_set(self, pyproject):
        urls = pyproject["project"].get("urls", {})
        # Case-insensitive key lookup.
        url_lower = {k.lower(): v for k, v in urls.items()}
        assert url_lower.get("repository") or url_lower.get("homepage"), (
            "must declare a Repository or Homepage URL"
        )

    def test_urls_use_current_repo_name(self, pyproject):
        """Repo was renamed from EvoAgent to Autodidact — URLs should reflect it."""
        urls = pyproject["project"].get("urls", {})
        for key, val in urls.items():
            # Any GitHub URL must use the current repo slug.
            if "github.com" in val.lower():
                # Should not still point at the old EvoAgent repo.
                assert "EvoAgent" not in val, (
                    f"{key} URL still points at old EvoAgent repo: {val}. "
                    "Update to BuffaloTechRider/Autodidact."
                )


# ── Helpers ──────────────────────────────────────────────────────


def _package_name(requirement: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string.

    e.g. 'numpy>=1.24' -> 'numpy'
         'openai>=1.0,<2' -> 'openai'
    """
    # Strip extras, version specifiers, environment markers.
    for sep in ("[", ">=", "<=", "==", "~=", ">", "<", "!=", ";"):
        if sep in requirement:
            requirement = requirement.split(sep)[0]
    return requirement.strip().lower()
