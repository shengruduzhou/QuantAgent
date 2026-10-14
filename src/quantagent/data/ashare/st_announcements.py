"""Dated risk-warning (ST / *ST) episodes from exchange-disclosure announcements.

The SZSE publishes a dated short-name register; the SSE and the BSE do not. All
three exchanges' listed companies must, however, announce on the CSRC-designated
platform (cninfo) both the implementation and the revocation of a risk warning,
and the SSE rules require that announcement to state the session from which the
warning applies (上交所股票上市规则 9.1.8; revocation: 9.3.10 — 停牌1天 on the
disclosure day, revoked 自复牌之日起).

This module is the pure part of that reconstruction: a title classifier, an
effective-date reader for the announcement body, and a per-security state
machine that turns events into episodes. Network access lives in
``scripts/u0_st_announcements.py``.

Semantics match the SZSE register: ``effective_start`` is the first session
traded under the warning (inclusive); ``effective_end`` is the first session no
longer under it (EXCLUSIVE). A span whose dates cannot be established is emitted
with ``st_state == "UNKNOWN"`` rather than guessed.

Method and acceptance thresholds: docs/audits/round30/roles/ENG_ST30_method.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

IMPLEMENT = "IMPLEMENT"
REVOKE = "REVOKE"
TRANSITION = "TRANSITION"
DELIST_PERIOD = "DELIST_PERIOD"   # 进入退市整理期: name becomes "XX退", loses its ST marker
IGNORE = "IGNORE"

#: Events that change (or may change) the risk-warning state.
STATE_EVENTS: frozenset[str] = frozenset({IMPLEMENT, REVOKE, TRANSITION, DELIST_PERIOD})

STATE_TRUE = "TRUE"
STATE_UNKNOWN = "UNKNOWN"

_WARN = r"(?:退市风险警示|其他风险警示|风险警示|特别处理|特別處理)"
_DO = r"(?:实施|实行|施行|被实施|被实行|叠加实施|叠加)"
_UNDO = r"(?:撤销|撤消|取消|解除)"

#: Title fragments that make an announcement commentary rather than the event.
_NOT_EVENT = re.compile(
    r"可能|预计|或将|将可能|风险提示|提示性公告|进展|申请|意见|法律|律师|核查|说明|"
    r"问询|回复|关注函|专项|自查|董事会关于|监事会|独立董事|会计师|审核|鉴证|"
    r"报告书|摘要|更正|补充|延期|未能|暂停上市|恢复上市|终止上市|摘牌|"
    r"情况的公告$|事项的公告$"
)
_DELIST_PERIOD = re.compile(r"进入退市整理期|退市整理期交易|退市整理期的公告|退市整理期首日")


def classify_title(title: object) -> str:
    """Classify an announcement title as a state event or IGNORE.

    Precedence: a revocation that keeps a warning in force is a TRANSITION
    (the episode continues); a plain revocation is a REVOKE; an implementation
    is an IMPLEMENT. Commentary (risk notices, applications, opinions, progress
    reports, corrections) is IGNORE — the effective event has its own notice.
    """
    text = re.sub(r"\s+", "", str(title or ""))
    if not text:
        return IGNORE
    if _DELIST_PERIOD.search(text) and not re.search(r"可能|风险提示|进展", text):
        return DELIST_PERIOD
    if _NOT_EVENT.search(text):
        return IGNORE
    undo = re.search(_UNDO + r".{0,12}?" + _WARN, text)
    do = re.search(_DO + r".{0,12}?" + _WARN, text)
    if undo:
        rest = text[undo.end():]
        keeps = re.search(r"(?:继续|仍|并|同时|后|暨).{0,8}?" + _DO + r".{0,10}?" + _WARN, rest) \
            or re.search(r"变更为.{0,6}?" + _WARN, rest) \
            or re.search(r"部分", text[:undo.end() + 4])
        return TRANSITION if keeps else REVOKE
    if do:
        return IMPLEMENT
    if re.search(r"变更为.{0,6}?" + _WARN, text):
        return TRANSITION
    return IGNORE


_NUM_DATE = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
_SLASH_DATE = r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})"


def _mk(y: str, m: str, d: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(int(y), int(m), int(d))
    except (ValueError, OverflowError):
        return None


_START_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("start_field", re.compile(r"(?:实施|实行|撤销|撤消)[^。；\n]{0,16}?起始日[为是：:\s]*" + _NUM_DATE)),
    ("start_field_slash", re.compile(r"(?:实施|实行|撤销|撤消)[^。；\n]{0,16}?起始日[为是：:\s]*" + _SLASH_DATE)),
    ("from_date", re.compile(
        r"(?:自|于|从)\s*" + _NUM_DATE + r"[^。；\n]{0,14}?(?:开市)?(?:时)?起[^。；\n]{0,20}?"
        r"(?:实施|实行|撤销|撤消|取消|被实施|被实行|复牌|恢复交易)")),
    ("resume_date", re.compile(r"(?:于|自)\s*" + _NUM_DATE + r"[^。；\n]{0,12}?复牌")),
)


def extract_effective_date(text: object, announcement_date: pd.Timestamp) -> tuple[pd.Timestamp | None, str]:
    """The session from which the announced change applies, read from the body.

    Only dates on or after the announcement date and within 60 days of it are
    accepted (a body also quotes report dates, rule dates and old events).
    Returns ``(None, "not_found")`` when nothing qualifies.
    """
    body = re.sub(r"[ \t　]+", "", str(text or ""))
    if not body:
        return None, "no_text"
    ann = pd.Timestamp(announcement_date).normalize()
    for basis, pattern in _START_PATTERNS:
        for match in pattern.finditer(body):
            when = _mk(*match.groups()[:3])
            if when is None:
                continue
            if ann <= when <= ann + pd.Timedelta(days=60):
                return when, basis
    return None, "not_found"


@dataclass(frozen=True)
class Calendar:
    """Exchange trading sessions (sorted, unique, normalised)."""

    sessions: pd.DatetimeIndex

    @classmethod
    def from_dates(cls, dates: Iterable) -> "Calendar":
        index = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
        return cls(index)

    def next_session_after(self, day: pd.Timestamp, n: int = 1) -> pd.Timestamp | None:
        pos = self.sessions.searchsorted(pd.Timestamp(day).normalize(), side="right") + n - 1
        return self.sessions[pos] if 0 <= pos < len(self.sessions) else None

    def on_or_after(self, day: pd.Timestamp) -> pd.Timestamp | None:
        pos = self.sessions.searchsorted(pd.Timestamp(day).normalize(), side="left")
        return self.sessions[pos] if pos < len(self.sessions) else None

    def distance(self, a: pd.Timestamp, b: pd.Timestamp) -> int:
        """Signed number of sessions from ``a`` to ``b`` (session positions)."""
        pa = self.sessions.searchsorted(pd.Timestamp(a).normalize(), side="left")
        pb = self.sessions.searchsorted(pd.Timestamp(b).normalize(), side="left")
        return int(pb - pa)


def resolve_effective_dates(events: pd.DataFrame, calendar: Calendar,
                            *, fallback_sessions: int | None = 1) -> pd.DataFrame:
    """Attach ``effective_date`` / ``effective_basis`` to classified events.

    ``events`` needs ``announcement_date`` and optionally ``text``. The body date
    wins; otherwise ``fallback_sessions`` sessions after the announcement date
    (the 停牌1天 / 自复牌之日起 mechanic) when given, else UNKNOWN (NaT). A body
    date that is not a trading session is moved to the next session.
    """
    out = events.copy()
    dates, bases = [], []
    texts = out["text"] if "text" in out.columns else pd.Series([None] * len(out), index=out.index)
    for ann, text in zip(out["announcement_date"], texts):
        when, basis = extract_effective_date(text, ann)
        if when is not None:
            session = calendar.on_or_after(when)
            dates.append(session)
            bases.append(f"body:{basis}")
            continue
        if fallback_sessions is not None:
            dates.append(calendar.next_session_after(ann, fallback_sessions))
            bases.append(f"rule:next_session+{fallback_sessions}"
                         + ("" if text else ":no_text"))
        else:
            dates.append(None)
            bases.append("unknown")
    out["effective_date"] = pd.to_datetime(pd.Series(dates, index=out.index))
    out["effective_basis"] = bases
    return out


EPISODE_COLUMNS = [
    "symbol", "effective_start", "effective_end", "st_state", "start_event_id",
    "start_announcement_date", "start_basis", "end_event_id", "end_announcement_date",
    "end_basis", "start_title", "end_title", "sec_name_at_start",
]


def build_episodes(events: pd.DataFrame, *, coverage_start: pd.Timestamp) -> pd.DataFrame:
    """Fold dated events into risk-warning episodes, per symbol.

    ``events`` columns: symbol, event, effective_date, announcement_date,
    announcement_id, title, sec_name. Rules:

    * IMPLEMENT opens an episode (a second IMPLEMENT inside an open episode is a
      kind change, e.g. ST → *ST, and keeps it open).
    * TRANSITION keeps an open episode open. Seen while *closed*, the security
      was already under a warning we never saw start: the span from
      ``coverage_start`` (or the previous close) to the transition is UNKNOWN and
      an episode is open from the transition.
    * REVOKE / DELIST_PERIOD close the open episode at their effective date
      (exclusive end). Seen while closed: the start is unobserved, so the span
      before it (from ``coverage_start`` or the previous close) is UNKNOWN.
    * An event without an effective date makes the affected boundary UNKNOWN:
      an undated IMPLEMENT opens an UNKNOWN span until the next dated event.
    """
    records: list[dict] = []
    frame = events[events["event"].isin(STATE_EVENTS)].copy()
    frame = frame.sort_values(["symbol", "announcement_date", "effective_date"], na_position="last")
    coverage_start = pd.Timestamp(coverage_start)

    for symbol, group in frame.groupby("symbol", sort=True):
        open_ep: dict | None = None
        last_close: pd.Timestamp | None = None
        for ev in group.itertuples():
            when = ev.effective_date if pd.notna(ev.effective_date) else None
            if ev.event == IMPLEMENT or (ev.event == TRANSITION and open_ep is None):
                if open_ep is not None:
                    continue                                   # kind change inside an episode
                if ev.event == TRANSITION:
                    lo = last_close if last_close is not None else coverage_start
                    hi = when if when is not None else pd.Timestamp(ev.announcement_date)
                    if hi > lo:
                        records.append(_ep(symbol, lo, hi, STATE_UNKNOWN, None, ev,
                                           "transition_without_observed_start"))
                if when is None:
                    open_ep = {"start": pd.Timestamp(ev.announcement_date), "ev": ev,
                               "state": STATE_UNKNOWN}
                else:
                    open_ep = {"start": when, "ev": ev, "state": STATE_TRUE}
            elif ev.event == TRANSITION:
                continue
            else:                                              # REVOKE / DELIST_PERIOD
                if open_ep is None:
                    lo = last_close if last_close is not None else coverage_start
                    hi = when if when is not None else pd.Timestamp(ev.announcement_date)
                    if hi > lo:
                        records.append(_ep(symbol, lo, hi, STATE_UNKNOWN, None, ev,
                                           "revoke_without_observed_start"))
                    last_close = hi
                    continue
                end = when
                if open_ep["state"] == STATE_UNKNOWN or end is None:
                    hi = end if end is not None else pd.Timestamp(ev.announcement_date)
                    records.append(_ep(symbol, open_ep["start"], hi, STATE_UNKNOWN,
                                       open_ep["ev"], ev, "undated_boundary"))
                    if end is None:
                        # we know it ended after the announcement, not when
                        nxt = pd.Timestamp(ev.announcement_date) + pd.Timedelta(days=30)
                        records.append(_ep(symbol, hi, nxt, STATE_UNKNOWN, open_ep["ev"], ev,
                                           "undated_end"))
                        hi = nxt
                    last_close = hi
                else:
                    if end > open_ep["start"]:
                        records.append(_ep(symbol, open_ep["start"], end, STATE_TRUE,
                                           open_ep["ev"], ev, None))
                    last_close = end
                open_ep = None
        if open_ep is not None:
            records.append(_ep(symbol, open_ep["start"], pd.NaT, open_ep["state"],
                               open_ep["ev"], None, None if open_ep["state"] == STATE_TRUE
                               else "undated_start"))
    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS + ["note"])
    return result


def _ep(symbol, start, end, state, start_ev, end_ev, note) -> dict:
    return {
        "symbol": symbol,
        "effective_start": pd.Timestamp(start),
        "effective_end": pd.Timestamp(end) if end is not None and pd.notna(end) else pd.NaT,
        "st_state": state,
        "start_event_id": getattr(start_ev, "announcement_id", None),
        "start_announcement_date": getattr(start_ev, "announcement_date", pd.NaT),
        "start_basis": getattr(start_ev, "effective_basis", None),
        "end_event_id": getattr(end_ev, "announcement_id", None),
        "end_announcement_date": getattr(end_ev, "announcement_date", pd.NaT),
        "end_basis": getattr(end_ev, "effective_basis", None),
        "start_title": getattr(start_ev, "title", None),
        "end_title": getattr(end_ev, "title", None),
        "sec_name_at_start": getattr(start_ev, "sec_name", None),
        "note": note,
    }


def row_states(panel: pd.DataFrame, episodes: pd.DataFrame,
               covered_symbols: Sequence[str] | None = None) -> np.ndarray:
    """TRUE / FALSE / UNKNOWN per panel row (start inclusive, end exclusive).

    UNKNOWN episodes override FALSE; TRUE overrides UNKNOWN only where a TRUE
    episode covers the row.
    """
    state = np.full(len(panel), "FALSE", dtype=object)
    if covered_symbols is not None:
        state[~panel["symbol"].isin(set(covered_symbols)).to_numpy()] = "UNKNOWN"
    if episodes.empty:
        return state
    dates = pd.to_datetime(panel["trade_date"]).to_numpy()
    sym = panel["symbol"].to_numpy()
    order = np.argsort(sym, kind="stable")
    sorted_sym = sym[order]
    for st_state in (STATE_UNKNOWN, STATE_TRUE):
        sub = episodes[episodes["st_state"] == st_state]
        for row in sub.itertuples():
            lo_i = np.searchsorted(sorted_sym, row.symbol, side="left")
            hi_i = np.searchsorted(sorted_sym, row.symbol, side="right")
            if lo_i == hi_i:
                continue
            idx = order[lo_i:hi_i]
            d = dates[idx]
            end = row.effective_end if pd.notna(row.effective_end) else pd.Timestamp.max
            hit = (d >= np.datetime64(row.effective_start)) & (d < np.datetime64(end))
            state[idx[hit]] = st_state
    return state
