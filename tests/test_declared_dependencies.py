"""Every unconditional production import must be a declared dependency.

The defect this locks out has already happened once. ``quantagent.data.ashare.http``
imports ``requests`` at module scope, ``pyproject.toml`` did not declare it, and
every developer machine had it installed transitively -- so the whole suite
passed locally and CI died during collection with::

    ModuleNotFoundError: No module named 'requests'

The lesson is that "the tests pass here" says nothing about whether the
dependency set is honest. Two tests enforce it from different directions:

**Static audit (always runs, milliseconds).** Parse every production module,
collect the third-party modules imported *unconditionally at module scope*, and
require each to be declared in ``pyproject.toml``. Module scope is the right
boundary: an import inside a function or a ``try/except ImportError`` is an
optional capability the code is expected to degrade around, while a top-level
import is a hard requirement that must appear in the metadata.

**Clean-environment proof (opt-in, minutes).** Build a real virtualenv, install
only ``.[test]``, and run collection inside it. This is the only test that can
prove nothing is being inherited from the developer environment, but it costs a
full dependency install, so it is gated behind ``QUANTAGENT_CLEAN_ENV_TEST=1``
and run deliberately rather than on every PR.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"

#: Production trees whose module-scope imports must be declared. ``scripts`` is
#: excluded on purpose: scripts are operator entrypoints that legitimately
#: import optional extras (torch, akshare) and sibling scripts by filename, and
#: they are compiled but never imported during test collection.
PRODUCTION_TREES = ("src", "services")

#: Packages whose import name differs from their distribution name.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
}

#: First-party roots that are never external dependencies.
FIRST_PARTY = {"quantagent", "services", "scripts", "tests"}

#: Entitled vendor SDKs that are not on public PyPI and are therefore expected
#: to be absent in CI. Each must be imported defensively by the code that uses
#: it; the test below asserts that, so this set cannot be used as an escape
#: hatch for a genuinely missing declaration.
VENDOR_SDKS = {"tickflow", "xtquant", "MetaTrader5"}


def _declared_distributions() -> set[str]:
    """Distribution names declared anywhere in pyproject (core or extras)."""
    try:  # Python 3.11+
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs: list[str] = list(project.get("dependencies", []))
    for extra in (project.get("optional-dependencies") or {}).values():
        specs.extend(extra)

    names: set[str] = set()
    for spec in specs:
        # "uvicorn[standard]>=0.30" -> "uvicorn"
        name = spec.split(";")[0].strip()
        for boundary in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(boundary)[0]
        if name:
            names.add(name.strip().lower().replace("_", "-"))
    return names


def _module_scope_imports(path: Path) -> set[str]:
    """Third-party roots imported at module scope (not in a function or try)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in tree.body:  # top level only -- deliberately not ast.walk
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _production_modules() -> list[Path]:
    modules: list[Path] = []
    for tree in PRODUCTION_TREES:
        for path in (REPO / tree).rglob("*.py"):
            if "__pycache__" not in str(path):
                modules.append(path)
    return modules


def _undeclared_imports() -> dict[str, list[str]]:
    """Map undeclared third-party module -> the files importing it."""
    stdlib = set(sys.stdlib_module_names)
    declared = _declared_distributions()
    offenders: dict[str, list[str]] = {}

    for path in _production_modules():
        for module in _module_scope_imports(path):
            if module in stdlib or module in FIRST_PARTY or module.startswith("_"):
                continue
            if module in VENDOR_SDKS:
                continue
            distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
            if distribution.lower().replace("_", "-") not in declared:
                offenders.setdefault(module, []).append(
                    str(path.relative_to(REPO))
                )
    return offenders


class TestDeclaredDependencies:
    def test_no_undeclared_production_imports(self):
        """The exact regression that broke CI: an undeclared module-scope import."""
        offenders = _undeclared_imports()
        assert not offenders, (
            "these modules are imported unconditionally by production code but "
            "are not declared in pyproject.toml, so a clean environment will "
            f"fail during collection: {offenders}"
        )

    def test_requests_is_declared_because_it_is_imported_unconditionally(self):
        """Pin the specific dependency whose absence caused the CI failure."""
        source = (REPO / "src/quantagent/data/ashare/http.py").read_text(encoding="utf-8")
        assert "\nimport requests" in source, (
            "this test guards a module-scope requests import; if the transport "
            "moved to httpx, update the declaration and this test together"
        )
        assert "requests" in _declared_distributions()

    def test_vendor_sdks_are_imported_defensively(self):
        """Entitled SDKs are exempt from declaration only if they degrade safely.

        Otherwise VENDOR_SDKS becomes a way to hide a real missing dependency.
        """
        stdlib = set(sys.stdlib_module_names)
        for path in _production_modules():
            leaked = _module_scope_imports(path) & VENDOR_SDKS
            assert not leaked, (
                f"{path.relative_to(REPO)} imports vendor SDK(s) {leaked} at "
                "module scope; these are not installable in CI and must be "
                "imported lazily inside the function that needs them"
            )
        assert not (VENDOR_SDKS & stdlib)

    def test_audit_actually_detects_a_missing_declaration(self):
        """A gate that cannot fail proves nothing.

        Rather than trusting the audit, feed it a module that imports something
        certainly undeclared and assert it is reported.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.py"
            probe.write_text("import a_package_that_is_not_declared\n", encoding="utf-8")
            found = _module_scope_imports(probe)
        assert "a_package_that_is_not_declared" in found

        declared = _declared_distributions()
        assert "a_package_that_is_not_declared" not in declared
        # ...and the real declaration set is non-trivial, so an empty parse
        # cannot make the audit vacuously pass.
        assert {"numpy", "pandas", "pytest"} <= declared

    def test_declared_set_parses_extras_and_markers(self):
        declared = _declared_distributions()
        assert "uvicorn" in declared, "extras with [standard] must parse"
        assert "scikit-learn" in declared
        assert "pyqlib" in declared


@pytest.mark.skipif(
    os.environ.get("QUANTAGENT_CLEAN_ENV_TEST") != "1",
    reason="clean-environment install is slow; set QUANTAGENT_CLEAN_ENV_TEST=1 to run",
)
class TestCleanEnvironmentInstall:
    """Build a real venv, install only .[test], and prove collection works.

    This is the only check that can prove nothing leaks in from the developer
    environment, because it is the only one that does not run in it.
    """

    def test_clean_install_collects_the_suite(self, tmp_path):
        env_dir = tmp_path / "cleanenv"
        venv.create(env_dir, with_pip=True, clear=True)
        bindir = "Scripts" if os.name == "nt" else "bin"
        python = env_dir / bindir / ("python.exe" if os.name == "nt" else "python")

        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "--upgrade",
             "pip", "wheel", "setuptools"],
            check=True, cwd=REPO, timeout=900,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "-e", ".[test]"],
            check=True, cwd=REPO, timeout=3600,
        )

        # Import the production packages the suite touches. A clean env that
        # cannot import these will fail collection, which is the failure mode
        # this whole test exists to catch.
        probe = (
            "import quantagent.data.ashare.http as h; "
            "import quantagent.data.microstructure as m; "
            "import quantagent.governance as g; "
            "import services.quant_api.services.jobs as j; "
            "print('IMPORTS_OK', h.__name__, m.__name__, g.__name__, j.__name__)"
        )
        result = subprocess.run(
            [str(python), "-c", probe],
            capture_output=True, text=True, cwd=REPO, timeout=600,
        )
        assert result.returncode == 0, (
            f"clean environment cannot import production packages:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert "IMPORTS_OK" in result.stdout

        collected = subprocess.run(
            [str(python), "-m", "pytest", "tests/", "-q", "--collect-only"],
            capture_output=True, text=True, cwd=REPO, timeout=1800,
        )
        assert collected.returncode == 0, (
            f"collection failed in a clean environment:\n"
            f"{collected.stdout[-4000:]}\n{collected.stderr[-4000:]}"
        )
        assert "error" not in collected.stdout.lower().split("warnings")[0]
