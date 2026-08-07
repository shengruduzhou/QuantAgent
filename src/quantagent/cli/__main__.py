"""CLI process entry point.

Research rejections leave through here with their own exit status. Letting a
rejected hypothesis exit 1 alongside genuine faults is what made a completed,
correctly-gated run indistinguishable from a crash to everything downstream.

Catching here also suppresses Typer's rich traceback boxes, which are rendered
from ``sys.excepthook`` and only fire for uncaught exceptions. Those boxes wrap
the exception message across lines with box-drawing characters, which is
precisely what made failure messages unreadable in job logs.
"""

from __future__ import annotations

import sys
import traceback

from quantagent.cli import app
from quantagent.research.verdict import (
    CONFIGURATION_BLOCKED_EXIT_CODE,
    RESEARCH_REJECTED_EXIT_CODE,
    ResearchRejection,
    rejection_event,
)


def main() -> int:
    try:
        app()
    except ResearchRejection as rejection:
        verdict_path = rejection.persist()
        # One machine-readable line: the job layer reads the verdict from
        # stdout instead of scraping a traceback.
        print(rejection_event(rejection, verdict_path), flush=True)
        print(
            f"RESEARCH_VERDICT={rejection.verdict} code={rejection.code} :: {rejection.summary()}",
            file=sys.stderr,
            flush=True,
        )
        # A blocked run never tested anything; it must not be filed alongside
        # candidates a gate actually refused.
        if rejection.verdict == "blocked":
            return CONFIGURATION_BLOCKED_EXIT_CODE
        return RESEARCH_REJECTED_EXIT_CODE
    except SystemExit as exit_request:  # Click's standalone mode exits this way.
        code = exit_request.code
        return 0 if code is None else int(code)
    except Exception:
        # Engineering faults keep their traceback and their exit code.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
