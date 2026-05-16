"""V7 storage CLI: inspect and initialise the unified runtime layout."""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump
from quantagent.config.paths import quant_paths


@app.command("storage-info-v7")
def storage_info_v7(
    home: Path | None = typer.Option(None, "--home", help="Override QUANTAGENT_HOME for this invocation."),
    ensure: bool = typer.Option(False, "--ensure", help="Create any missing directories under the resolved home."),
) -> None:
    """Report the resolved storage layout used for all V7 large artefacts.

    The default home is ``<repo>/runtime`` on Windows and ``~/AI_quant`` on
    other platforms. ``QUANTAGENT_HOME`` overrides it. Use ``--ensure`` to
    create the directory tree.
    """
    layout = quant_paths(home=home)
    if ensure:
        layout = layout.ensure()
    payload = {
        "status": "passed",
        "ensured": ensure,
        "layout": layout.as_dict(),
    }
    typer.echo(json_dump(payload))


@app.command("setup-qlib-v7")
def setup_qlib_v7(
    target_dir: str = typer.Option(None, "--target-dir", help="Destination provider_uri; defaults to <home>/data/raw/qlib/cn_data."),
    region: str = typer.Option("cn", "--region"),
    run: bool = typer.Option(False, "--run", help="Attempt to run the official Qlib download command."),
    interval: str = typer.Option("1d", "--interval", help="Qlib data frequency: 1d or 1min."),
    allow_community_fallback: bool = typer.Option(
        False,
        "--allow-community-fallback",
        help="If --run fails, print the qlib-server / community mirror fallback instructions.",
    ),
) -> None:
    """Prepare Qlib CN data with an auditable setup path.

    Without ``--run`` this command only resolves the destination and
    prints the official Qlib download command, runnable verbatim inside
    a Qlib checkout. With ``--run`` it attempts to invoke pyqlib's
    ``GetData`` API directly so the dataset lands at the resolved path.

    Failures are reported with the actionable fallback instructions
    instead of being silently swallowed.
    """
    layout = quant_paths().ensure()
    if target_dir is None:
        target_dir = str(layout.raw / "qlib" / f"{region}_data")
    resolved = Path(target_dir).expanduser()
    payload: dict[str, object] = {
        "status": "manual_step_required",
        "target_dir": str(resolved),
        "region": region,
        "interval": interval,
        "official_command": (
            f"python scripts/get_data.py qlib_data --target_dir {resolved} "
            f"--region {region} --interval {interval}"
        ),
        "official_module_command": (
            f"python -m qlib.cli.data qlib_data --target_dir {resolved} "
            f"--region {region} --interval {interval}"
        ),
        "windows_powershell_command_chain": [
            "cd E:\\Project\\QuantAgent",
            "py -3.12 -m venv .venv",
            ".\\.venv\\Scripts\\Activate.ps1",
            "pip install -U pip",
            'pip install -e ".[training,research]"',
            "pip install pyqlib akshare polars lightgbm xgboost torch",
            '$env:QUANTAGENT_HOME = "E:\\Project\\QuantAgent\\runtime"',
            "quantagent storage-info-v7 --ensure",
            (
                f"quantagent setup-qlib-v7 --region {region} --interval {interval} "
                f"--target-dir {resolved} --run --allow-community-fallback"
            ),
            (
                f"quantagent check-qlib-v7 --provider-uri {resolved} --region {region} "
                "--symbols SH600519,SZ000001 --start-date 2018-01-01 --end-date 2020-09-25"
            ),
        ],
        "note": (
            "qlib CN instruments use uppercase exchange prefix (SH600519). "
            "The official free CN release covers 2000-01-04..2020-09-25; "
            "prepare a custom dump via scripts/dump_bin.py for newer data."
        ),
    }
    if run:
        try:
            from qlib.tests.data import GetData  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            payload["status"] = "qlib_unavailable"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["hint"] = "pip install -e .[research] then re-run --run"
            if allow_community_fallback:
                payload["community_fallback"] = _community_fallback_notes(resolved, region, interval)
            typer.echo(json_dump(payload))
            raise typer.Exit(code=1)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            GetData().qlib_data(
                target_dir=str(resolved), region=region, interval=interval, delete_old=False
            )
            payload["status"] = "downloaded"
        except Exception as exc:  # pragma: no cover - network path
            payload["status"] = "failed"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            if allow_community_fallback:
                payload["community_fallback"] = _community_fallback_notes(resolved, region, interval)
            typer.echo(json_dump(payload))
            raise typer.Exit(code=1)
    typer.echo(json_dump(payload))


def _community_fallback_notes(target_dir: Path, region: str, interval: str) -> dict[str, object]:
    """Provide manual recovery steps when the official mirror is unavailable.

    These point at well-known mirrors (qlib official release tarball,
    community ``qlib-server`` packages) without endorsing any specific
    third party; the operator decides which fallback is acceptable.
    """
    return {
        "notes": [
            "If the official mirror is unreachable, download the dataset tarball "
            "from a trusted community mirror such as the Qlib README referenced "
            "investment_data release, or from an internally approved mirror, "
            f"and extract it into {target_dir}.",
            "Alternatively prepare a custom dataset via scripts/dump_bin.py from CSV/Parquet "
            "OHLCV inputs you control.",
            "After preparing data, verify it via: quantagent check-qlib-v7 "
            f"--provider-uri {target_dir} --region {region}.",
        ],
        "community_mirror_example": (
            "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
        ),
        "custom_dump_bin_path": "Use Qlib scripts/dump_bin.py with your own PIT CSV/Parquet OHLCV dump.",
        "verify_command": (
            f"quantagent check-qlib-v7 --provider-uri {target_dir} --region {region}"
        ),
        "interval": interval,
    }
