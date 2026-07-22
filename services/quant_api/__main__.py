from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import uvicorn


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m services.quant_api",
        description="Launch the QuantAgent institutional workstation and API.",
    )
    parser.add_argument(
        "--runtime",
        default=os.environ.get("QUANTAGENT_HOME"),
        help="QuantAgent runtime root (defaults to QUANTAGENT_HOME/canonical storage settings).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("QUANT_UI_HOST", "127.0.0.1"),
        help="HTTP bind host (default: QUANT_UI_HOST or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=_port(os.environ.get("QUANT_UI_PORT", "8000")),
        help="HTTP port (default: QUANT_UI_PORT or 8000).",
    )
    reload_default = os.environ.get("QUANT_UI_RELOAD", "false").lower() == "true"
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument("--reload", dest="reload", action="store_true", help="Enable Uvicorn reload mode.")
    reload_group.add_argument("--no-reload", dest="reload", action="store_false", help="Disable Uvicorn reload mode.")
    parser.set_defaults(reload=reload_default)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default=os.environ.get("QUANT_UI_LOG_LEVEL", "info"),
        help="Uvicorn log level (default: info).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.runtime:
        os.environ["QUANTAGENT_HOME"] = str(Path(args.runtime).expanduser().resolve())

    uvicorn.run(
        "services.quant_api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
