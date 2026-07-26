"""Canonical A-share security identity: symbol, exchange and board classification.

Single source of truth for how a six-digit A-share code maps onto an exchange
(SSE / SZSE / BSE) and a board (SH_Main / SZ_Main / ChiNext / STAR / BSE).
Every provider adapter converts its own vendor code format through this module,
so a symbol means exactly one thing everywhere in the pipeline.

Board rules (as published by the exchanges):

* 60xxxx        SSE main board
* 900xxx        SSE B share (main board)
* 688xxx / 689xxx  SSE STAR market (科创板)
* 000xxx / 001xxx / 002xxx / 003xxx  SZSE main board (002/003 = former SME board,
  merged into the main board on 2021-04-06)
* 200xxx        SZSE B share (main board)
* 300xxx / 301xxx / 302xxx  SZSE ChiNext (创业板)
* 43xxxx / 83xxxx / 87xxxx / 88xxxx  BSE legacy NEEQ-select codes
* 920xxx        BSE codes issued from 2023-11 onward (the current canonical range)

The BSE legacy/920xxx split matters: the vendor universe only serves 920xxx for
newly issued BSE names, while historical NEEQ-select codes stay in the 8xxxxx
range. Both are classified as ``BSE`` here and the caller decides fetchability.
"""

from __future__ import annotations

from dataclasses import dataclass

SSE = "SSE"
SZSE = "SZSE"
BSE_EXCHANGE = "BSE"

BOARD_SH_MAIN = "SH_Main"
BOARD_SZ_MAIN = "SZ_Main"
BOARD_CHINEXT = "ChiNext"
BOARD_STAR = "STAR"
BOARD_BSE = "BSE"
BOARD_OTHER = "OTHER"

ALL_BOARDS: tuple[str, ...] = (
    BOARD_SH_MAIN,
    BOARD_SZ_MAIN,
    BOARD_CHINEXT,
    BOARD_STAR,
    BOARD_BSE,
)

_SUFFIX_BY_EXCHANGE = {SSE: "SH", SZSE: "SZ", BSE_EXCHANGE: "BJ"}
_EXCHANGE_BY_SUFFIX = {"SH": SSE, "SS": SSE, "SZ": SZSE, "BJ": BSE_EXCHANGE, "BSE": BSE_EXCHANGE}


class SymbolError(ValueError):
    """Raised when a code cannot be resolved to a canonical A-share identity."""


@dataclass(frozen=True)
class SecurityIdentity:
    """Canonical identity of one A-share security."""

    code: str            # zero-padded six digit code, e.g. "600519"
    exchange: str        # SSE / SZSE / BSE
    board: str           # SH_Main / SZ_Main / ChiNext / STAR / BSE / OTHER
    security_type: str   # A_share / B_share
    symbol: str          # canonical "<code>.<SH|SZ|BJ>"

    @property
    def tencent_code(self) -> str:
        return f"{ {SSE: 'sh', SZSE: 'sz', BSE_EXCHANGE: 'bj'}[self.exchange] }{self.code}".replace(" ", "")

    @property
    def eastmoney_secid(self) -> str:
        """Eastmoney market prefix: 1 = SSE, 0 = SZSE and BSE."""
        return f"{'1' if self.exchange == SSE else '0'}.{self.code}"


def classify_code(code: str) -> tuple[str, str, str]:
    """Return ``(exchange, board, security_type)`` for a six-digit code."""
    c = str(code).strip()
    if not c.isdigit() or len(c) != 6:
        raise SymbolError(f"not a six-digit A-share code: {code!r}")
    p3, p2 = c[:3], c[:2]
    if p3 in {"688", "689"}:
        return SSE, BOARD_STAR, "A_share"
    if p3 == "900":
        return SSE, BOARD_SH_MAIN, "B_share"
    if c[0] == "6":
        return SSE, BOARD_SH_MAIN, "A_share"
    if p3 in {"300", "301", "302"}:
        return SZSE, BOARD_CHINEXT, "A_share"
    if p3 == "200":
        return SZSE, BOARD_SZ_MAIN, "B_share"
    if p3 in {"000", "001", "002", "003", "004"}:
        return SZSE, BOARD_SZ_MAIN, "A_share"
    if p3 == "920" or p2 in {"43", "83", "87", "88", "82", "89"}:
        return BSE_EXCHANGE, BOARD_BSE, "A_share"
    return SZSE if c[0] in "0123" else SSE, BOARD_OTHER, "A_share"


def identify(value: str) -> SecurityIdentity:
    """Resolve any common A-share code spelling into a canonical identity.

    Accepts ``600519``, ``600519.SH``, ``sh600519``, ``SH600519``, ``600519.XSHG``.
    The exchange is always re-derived from the code so a wrong vendor suffix can
    never silently move a security onto the wrong exchange; a suffix that
    contradicts the code raises instead of being accepted.
    """
    raw = str(value).strip()
    if not raw:
        raise SymbolError("empty symbol")
    suffix = None
    token = raw.replace("_", ".")
    if "." in token:
        head, _, tail = token.partition(".")
        tail_up = tail.upper()
        if tail_up in _EXCHANGE_BY_SUFFIX:
            suffix, token = _EXCHANGE_BY_SUFFIX[tail_up], head
        elif tail_up in {"XSHG", "XSHE", "XBSE"}:
            suffix = {"XSHG": SSE, "XSHE": SZSE, "XBSE": BSE_EXCHANGE}[tail_up]
            token = head
        else:
            token = head
    else:
        head = token[:2].upper()
        if head in _EXCHANGE_BY_SUFFIX and token[2:].isdigit():
            suffix, token = _EXCHANGE_BY_SUFFIX[head], token[2:]
    code = token.strip().zfill(6)
    exchange, board, sec_type = classify_code(code)
    if suffix is not None and suffix != exchange:
        raise SymbolError(
            f"symbol {value!r} carries exchange suffix {suffix} but code {code} belongs to {exchange}"
        )
    return SecurityIdentity(
        code=code,
        exchange=exchange,
        board=board,
        security_type=sec_type,
        symbol=f"{code}.{_SUFFIX_BY_EXCHANGE[exchange]}",
    )


def canonical_symbol(value: str) -> str:
    """Canonical ``<code>.<SH|SZ|BJ>`` form."""
    return identify(value).symbol


def board_of(value: str) -> str:
    return identify(value).board


def is_bse_legacy_code(value: str) -> bool:
    """True for a BSE security still carrying a pre-2023 NEEQ-select code."""
    ident = identify(value)
    return ident.board == BOARD_BSE and not ident.code.startswith("920")
