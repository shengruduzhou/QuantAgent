"""Static audit of MQL5 sources for MQL4-style API misuse and unsafe patterns.

MetaEditor is the only thing that can *compile* MQL5, and it does not run on
this host. That is a real limitation and it is stated as one. What can be
checked without a compiler is the class of defect that survives compilation
anyway -- code that builds cleanly and then silently does the wrong thing:

* ``iMA(...)`` used as a value. In MQL4 it returns a price; in MQL5 it returns
  an integer handle. The port compiles and then compares a price against a
  handle id, which is typically a single-digit integer and looks plausible.
* An indicator handle created and never released, so repeated chart loads
  exhaust indicator resources.
* ``CopyBuffer`` return value ignored, so "not ready yet" reads as a value.
* ``trade.buy(...)`` lowercase, which is not the ``CTrade`` API.
* An EA with no real-account guard.
* Martingale / grid position-doubling as a default behaviour.
* FX lot assumptions applied to A-share share counts.

Findings are severities, not opinions: ``ERROR`` means the code is wrong,
``WARN`` means it is risky, ``INFO`` is advisory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ERROR = "ERROR"
WARN = "WARN"
INFO = "INFO"

#: Indicator factories that return a handle in MQL5 but a value in MQL4.
MQL5_HANDLE_FUNCTIONS: tuple[str, ...] = (
    "iMA", "iATR", "iRSI", "iMACD", "iBands", "iStochastic", "iCCI", "iADX",
    "iAO", "iMomentum", "iOBV", "iSAR", "iStdDev", "iCustom", "iAlligator",
    "iEnvelopes", "iForce", "iFractals", "iIchimoku", "iMFI", "iWPR",
)


@dataclass
class Finding:
    file: str
    line: int
    severity: str
    rule: str
    message: str
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_comments(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, code)`` with comments and strings blanked.

    Blanking rather than deleting keeps line numbers aligned, and prevents a
    rule from firing on the word ``trade.buy`` inside a docstring explaining
    why ``trade.buy`` is wrong.
    """
    out: list[tuple[int, str]] = []
    in_block = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw
        if in_block:
            end = line.find("*/")
            if end == -1:
                out.append((number, ""))
                continue
            line = " " * (end + 2) + line[end + 2:]
            in_block = False
        while True:
            start = line.find("/*")
            if start == -1:
                break
            end = line.find("*/", start + 2)
            if end == -1:
                line = line[:start]
                in_block = True
                break
            line = line[:start] + " " * (end + 2 - start) + line[end + 2:]
        slash = line.find("//")
        if slash != -1:
            line = line[:slash]
        line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
        out.append((number, line))
    return out


def audit_source(path: str | Path) -> list[Finding]:
    """Audit one ``.mq5`` / ``.mqh`` file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = _strip_comments(text)
    code = "\n".join(line for _, line in lines)
    findings: list[Finding] = []
    name = str(path)

    def add(line: int, severity: str, rule: str, message: str, excerpt: str = "") -> None:
        findings.append(Finding(name, line, severity, rule, message, excerpt.strip()[:160]))

    # -- MQL4-style indicator value usage
    handle_pattern = re.compile(
        r"\b(" + "|".join(MQL5_HANDLE_FUNCTIONS) + r")\s*\("
    )
    for number, line in lines:
        for match in handle_pattern.finditer(line):
            function = match.group(1)
            before = line[:match.start()].rstrip()
            # Assigning to an int handle, or adopting it, is correct usage.
            correct = re.search(
                r"(int\s+\w+\s*=\s*$)|(\bhandle\w*\s*=\s*$)|(Adopt\s*\(\s*$)"
                r"|(\breturn\s+$)|(=\s*$)",
                before,
            )
            assigns_to_double = re.search(r"\bdouble\s+\w+\s*=\s*$", before)
            in_comparison = re.search(r"[<>]=?\s*$|[-+*/]\s*$", before)
            if assigns_to_double or in_comparison:
                add(number, ERROR, "mql4_indicator_value",
                    f"{function}() returns an indicator HANDLE in MQL5, not a "
                    "value. Using it as a number compares against a handle id. "
                    "Read the value with CopyBuffer instead.", line)
            elif not correct and "=" not in before[-3:]:
                add(number, WARN, "mql4_indicator_value_suspect",
                    f"{function}() result is not obviously assigned to a handle; "
                    "confirm this is not MQL4-style value usage.", line)

    # -- handle lifecycle
    creates_handle = bool(handle_pattern.search(code))
    releases = "IndicatorRelease" in code or "Release()" in code
    if creates_handle and not releases:
        add(1, WARN, "missing_indicator_release",
            "indicator handles are created but IndicatorRelease is never called; "
            "repeated loads will exhaust indicator resources")

    # -- CopyBuffer result ignored
    for number, line in lines:
        if "CopyBuffer" in line:
            assigned = re.search(r"(=|return|if\s*\(|<|>|!)", line.split("CopyBuffer")[0][-30:])
            if not assigned:
                add(number, ERROR, "copybuffer_result_ignored",
                    "CopyBuffer's return value is discarded; a short or failed "
                    "copy then reads as a valid value", line)

    # -- lowercase CTrade methods (not the real API)
    for number, line in lines:
        if re.search(r"\b\w*[Tt]rade\s*\.\s*(buy|sell|positionopen|positionclose)\s*\(", line):
            add(number, ERROR, "ctrade_lowercase",
                "CTrade methods are Buy/Sell/PositionOpen/PositionClose; the "
                "lowercase spelling is not the MQL5 API", line)

    # -- EA-specific safety
    is_expert = "OnTick" in code and "OnCalculate" not in code
    if is_expert:
        if "ACCOUNT_TRADE_MODE_REAL" not in code and "AssertNotRealAccount" not in code:
            add(1, ERROR, "missing_real_account_guard",
                "expert advisor has no real-account guard; every EA must refuse "
                "a real account by default")
        if "OnTradeTransaction" not in code:
            add(1, WARN, "missing_trade_transaction_handler",
                "no OnTradeTransaction handler; asynchronous fills, partial "
                "fills and rejections will be missed")
        if "OrderSend" in code and "OrderCheck" not in code:
            add(1, WARN, "missing_order_check",
                "OrderSend without a prior OrderCheck")

    # -- martingale / grid
    for number, line in lines:
        if re.search(r"(volume|lot|lots|shares)\s*\*=\s*[2-9]", line, re.IGNORECASE):
            add(number, ERROR, "martingale_doubling",
                "position size is multiplied on loss (martingale); this has "
                "unbounded tail risk and is not permitted as a default", line)
        if re.search(r"\b(martingale|grid_step|averaging_down)\b", line, re.IGNORECASE):
            add(number, WARN, "martingale_grid_marker",
                "martingale/grid vocabulary present; such strategies belong in "
                "a quarantined educational category with tail-risk warnings", line)

    # -- FX assumptions applied to A-share sizing
    for number, line in lines:
        if re.search(r"\b(0\.01|0\.1)\s*;.*\b(lot|volume)", line, re.IGNORECASE):
            add(number, WARN, "fx_lot_assumption",
                "fractional lot sizing is an FX convention; A-share orders are "
                "whole share counts with a board-specific minimum", line)
        if "100000" in line and re.search(r"contract", line, re.IGNORECASE):
            add(number, ERROR, "fx_contract_size",
                "a 100,000-unit contract size is an FX default; an A-share "
                "custom symbol must use contract_size = 1 share", line)

    # -- DOM assumed available
    if "MarketBookAdd" in code and "MarketBookGet" not in code:
        add(1, WARN, "dom_subscribed_not_read",
            "MarketBookAdd without MarketBookGet: subscribing is not evidence "
            "that the broker publishes depth")

    return findings


def audit_tree(root: str | Path) -> dict[str, Any]:
    """Audit every MQL5 source under ``root``."""
    root = Path(root)
    sources = sorted(
        [*root.rglob("*.mq5"), *root.rglob("*.mqh")]
    )
    findings: list[Finding] = []
    for source in sources:
        findings.extend(audit_source(source))

    by_severity: dict[str, int] = {}
    for finding in findings:
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1

    return {
        "root": str(root),
        "files_audited": len(sources),
        "files": [str(s) for s in sources],
        "findings": [f.to_dict() for f in findings],
        "counts_by_severity": by_severity,
        "errors": [f.to_dict() for f in findings if f.severity == ERROR],
        "clean": not any(f.severity == ERROR for f in findings),
        "limitation": (
            "This is a static audit. MetaEditor is the only MQL5 compiler and it "
            "does not run on this host, so these sources are NOT verified to "
            "compile. The audit covers defects that survive compilation."
        ),
    }
