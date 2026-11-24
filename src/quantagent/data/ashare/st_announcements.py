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

Method and acceptance thresholds: docs/research/preregistrations/ENG_ST30_preregistration.md.
"""
from __future__ import annotations

import re
import unicodedata
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
    r"可能|预计|或将|将可能|风险提示|进展|申请|意见|法律|律师|核查|说明|"
    r"问询|回复|关注函|专项|自查|监事会|独立董事|会计师|审核|鉴证|"
    r"报告书|摘要|更正|补充|延期|未能|暂停上市|终止上市|摘牌|"
    r"情况的公告$|事项的公告$|情形(?:已)?消除"
)
_DELIST_PERIOD = re.compile(r"进入退市整理期|退市整理期交易|退市整理期的公告|退市整理期首日")
#: "第十次提示性公告": a periodic reminder of a state already in force
_REMINDER = re.compile(r"第[一二三四五六七八九十百零〇两\d]+次")
#: "将被实施…的提示性公告": announced ahead of the implementation notice
_ADVANCE = re.compile(r"将(?:被)?(?:实施|实行|叠加)")
_NOT_STOCK = re.compile(r"债券|公司债|转债|优先股|票据")
_HEDGED_CLAUSE = re.compile(
    r"[、，,及和与暨]?(?:可能|或将|将可能|存在)[^、，,暨及的]{0,16}?" + _WARN)


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
    if _DELIST_PERIOD.search(text) and not re.search(r"可能|进展", text) and (
            not re.search(r"风险提示", text) or re.search(r"首日|第一次", text)):
        # the first-day / first risk notice is published on the first session of
        # the delisting period — the boundary itself; later reminders are not
        return DELIST_PERIOD
    # "被实施其他风险警示、可能被实施退市风险警示": the hedged clause is commentary,
    # the other clause is the event — drop only the hedged clause.
    text = _HEDGED_CLAUSE.sub("", text)
    if _NOT_STOCK.search(text) and "股票" not in text:
        return IGNORE                          # a bond's / preferred share's warning
    if _NOT_EVENT.search(text) or _REMINDER.search(text):
        return IGNORE
    undo = re.search(_UNDO + r".{0,12}?" + _WARN, text)
    do = re.search(_DO + r".{0,12}?" + _WARN, text)
    if undo:
        rest = text[undo.end():]
        keeps = re.search(r"(?:继续|仍|并|同时|后|暨|及|和|、).{0,8}?" + _DO + r".{0,10}?" + _WARN, rest) \
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
    ("delist_period_start", re.compile(
        r"退市整理期[^。；\n]{0,16}?(?:起始日|首个交易日|第一个交易日|首日)[为是：:\s]*" + _NUM_DATE)),
    ("delist_period_from", re.compile(
        r"(?:自|于|从)\s*" + _NUM_DATE + r"[^。；\n]{0,10}?(?:起)?[^。；\n]{0,6}?进入退市整理期")),
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
    kinds = out["event"] if "event" in out.columns else pd.Series([None] * len(out), index=out.index)
    for ann, text, kind in zip(out["announcement_date"], texts, kinds):
        when, basis = extract_effective_date(text, ann)
        if when is not None:
            session = calendar.on_or_after(when)
            dates.append(session)
            bases.append(f"body:{basis}")
            continue
        if fallback_sessions is not None:
            # SZSE DEV calibration (ENG_ST30 §2b): implement / revoke / transition
            # notices take effect the session after the announcement (the
            # announcement day is a one-day suspension); the first-day notice of a
            # delisting period is published on that first session itself.
            n = 0 if kind == DELIST_PERIOD else fallback_sessions
            dates.append(calendar.on_or_after(ann) if n == 0
                         else calendar.next_session_after(ann, n))
            bases.append(f"rule:next_session+{n}" + ("" if text else ":no_text"))
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


def consolidate_delist_period(events: pd.DataFrame, *, days: int = 30) -> pd.DataFrame:
    """One DELIST_PERIOD boundary per delisting period.

    A delisting period is announced several times (the decision notice some
    sessions ahead, then a notice on its first session). Within ``days`` of
    each other a symbol's DELIST_PERIOD events are one boundary; the kept event
    is the one dated from the body, else the first-day / first-reminder notice
    (published on that session), else the latest.
    """
    if events.empty or "event" not in events.columns:
        return events
    is_dp = events["event"].eq(DELIST_PERIOD)
    if not is_dp.any():
        return events
    dp = events[is_dp].sort_values(["symbol", "announcement_date"]).copy()
    gap = dp.groupby("symbol")["announcement_date"].diff().dt.days
    dp["_cluster"] = (gap.isna() | (gap > days)).cumsum()
    basis = dp.get("effective_basis", pd.Series("", index=dp.index)).astype(str)
    titles = dp.get("title", pd.Series("", index=dp.index)).astype(str)
    dp["_rank"] = (basis.str.startswith("body:").astype(int) * 4
                   + titles.str.contains("首日|第一次").astype(int) * 2)
    keep = (dp.sort_values(["_cluster", "_rank", "announcement_date"])
            .groupby("_cluster").tail(1).index)
    return pd.concat([events[~is_dp], events.loc[keep]]).sort_index()


def warning_kinds(title: object) -> frozenset[str]:
    """Which warning a title implements or revokes: ``delist`` (退市风险警示,
    *ST), ``other`` (其他风险警示 / 其他特别处理, ST) or ``generic`` (特别处理 /
    风险警示 without a kind, as used before 2003)."""
    text = re.sub(r"\s+", "", str(title or ""))
    kinds = set()
    if "退市风险警示" in text or re.search(r"退市风险[^警]{0,4}特别处理", text):
        kinds.add("delist")
    if re.search(r"其他风险警示|其他特别处理", text):
        kinds.add("other")
    return frozenset(kinds or {"generic"})


def kinds_from_name(sec_name: object) -> frozenset[str] | None:
    """Warning kinds a short name shows on the announcement date: ``*ST`` →
    退市风险警示 in force, ``ST`` → only 其他风险警示; ``None`` if the name
    carries no marker (it says nothing about which warning is in force)."""
    name = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(sec_name or ""))).upper()
    name = re.sub(r"^(?:G|S)(?=\*?S\s*T)", "", name)   # share-reform prefixes: G*ST, S*ST, SST
    if name.startswith("*ST"):
        return frozenset({"delist"})
    if name.startswith("ST"):
        return frozenset({"other"})
    return None


def drop_superseded_advance_notices(events: pd.DataFrame, *, days: int = 20) -> pd.DataFrame:
    """Drop a "将被实施…" advance notice when the implementation notice itself
    follows within ``days`` (the advance notice is not the effective event;
    alone, it is the only evidence and is kept)."""
    if events.empty or "title" not in events.columns:
        return events
    imp = events[events["event"].eq(IMPLEMENT)]
    advance = imp[imp["title"].astype(str).str.contains(_ADVANCE)]
    if advance.empty:
        return events
    drop = []
    later = imp[~imp.index.isin(advance.index)]
    by_symbol = {s: g["announcement_date"] for s, g in later.groupby("symbol")}
    for idx, row in advance.iterrows():
        dates = by_symbol.get(row["symbol"])
        if dates is None:
            continue
        gap = (dates - row["announcement_date"]).dt.days
        if bool(((gap >= 0) & (gap <= days)).any()):
            drop.append(idx)
    return events.drop(index=drop)


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
    frame = consolidate_delist_period(events[events["event"].isin(STATE_EVENTS)].copy())
    frame = drop_superseded_advance_notices(frame)
    frame = frame.sort_values(["symbol", "announcement_date", "effective_date"], na_position="last")
    coverage_start = pd.Timestamp(coverage_start)

    for symbol, group in frame.groupby("symbol", sort=True):
        open_ep: dict | None = None
        last_close: pd.Timestamp | None = None
        for ev in group.itertuples():
            when = ev.effective_date if pd.notna(ev.effective_date) else None
            kinds = warning_kinds(getattr(ev, "title", ""))
            if ev.event == IMPLEMENT or (ev.event == TRANSITION and open_ep is None):
                if open_ep is not None:
                    open_ep["kinds"] |= kinds                  # 叠加 / ST -> *ST inside an episode
                    open_ep.setdefault("added", set()).update(kinds)
                    continue
                if ev.event == TRANSITION:
                    lo = last_close if last_close is not None else coverage_start
                    hi = when if when is not None else pd.Timestamp(ev.announcement_date)
                    if hi > lo:
                        records.append(_ep(symbol, lo, hi, STATE_UNKNOWN, None, ev,
                                           "transition_without_observed_start"))
                start_kinds = {"other"} if ev.event == TRANSITION else set(kinds)
                if when is None:
                    open_ep = {"start": pd.Timestamp(ev.announcement_date), "ev": ev,
                               "state": STATE_UNKNOWN, "kinds": start_kinds}
                else:
                    open_ep = {"start": when, "ev": ev, "state": STATE_TRUE,
                               "kinds": start_kinds}
            elif ev.event == TRANSITION:
                open_ep["kinds"] = {"other"} if "other" in kinds else (open_ep["kinds"] or {"generic"})
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
                if ev.event == REVOKE and "generic" not in kinds:
                    # Revoking one warning while another stays in force is a
                    # transition. Which warnings are in force comes from the short
                    # name on the revocation date (*ST / ST) plus any 其他 warning
                    # explicitly added (叠加) inside this episode.
                    shown = kinds_from_name(getattr(ev, "sec_name", None))
                    active = set(shown) if shown is not None else {"delist", "other"} & open_ep["kinds"]
                    if "delist" in active and "other" in open_ep.get("added", set()):
                        active.add("other")
                    if active - kinds:
                        open_ep["kinds"] = active - kinds
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


def apply_name_observations(episodes: pd.DataFrame, names: pd.DataFrame, *,
                            settle_days: int = 3) -> pd.DataFrame:
    """Cut TRUE episodes that dated short names prove had already ended.

    ``names``: ``symbol, observed_on, sec_name`` — the short name printed on a
    cninfo disclosure of that date (a point-in-time fact; withdrawn filings,
    which show today's name, are excluded by the caller). Inside a TRUE episode
    ``[start, end)``, a name without an ST marker that is not followed by any
    ST-marked name in the episode (more than ``settle_days`` after the start)
    proves the warning had ended by then although no revocation was seen. The episode is kept TRUE up to and
    including the last ST-marked observation before it, the gap to the
    observation becomes UNKNOWN, and from the observation on the security is
    not under a warning (the rest of the span is dropped).
    """
    if episodes.empty or names.empty:
        return episodes
    obs = names.dropna(subset=["observed_on"]).copy()
    obs["observed_on"] = pd.to_datetime(obs["observed_on"]).dt.normalize()
    obs["kinds"] = obs["sec_name"].map(kinds_from_name)
    by_symbol = {s: g.sort_values("observed_on") for s, g in obs.groupby("symbol")}
    out: list[dict] = []
    one = pd.Timedelta(days=1)
    for row in episodes.to_dict("records"):
        seen = by_symbol.get(row["symbol"])
        if row["st_state"] != STATE_TRUE or seen is None:
            out.append(row)
            continue
        start = pd.Timestamp(row["effective_start"])
        end = row["effective_end"] if pd.notna(row["effective_end"]) else pd.Timestamp.max
        inside = seen[(seen["observed_on"] > start + pd.Timedelta(days=settle_days))
                      & (seen["observed_on"] < end)]
        # the cut is the first plain name after which no ST-marked name appears
        # in the episode: an isolated plain name between ST names is an anomaly
        marked_dates = inside.loc[inside["kinds"].notna(), "observed_on"]
        last_marked = marked_dates.max() if len(marked_dates) else pd.NaT
        plain = inside[inside["kinds"].isna()]
        if pd.notna(last_marked):
            plain = plain[plain["observed_on"] > last_marked]
        if plain.empty:
            out.append(row)
            continue
        cut = plain["observed_on"].iloc[0]
        marked = inside[inside["kinds"].notna() & (inside["observed_on"] < cut)]
        last_true = marked["observed_on"].iloc[-1] + one if len(marked) else start
        true_part = dict(row, effective_end=last_true, note="cut_by_dated_name")
        if last_true > start:
            out.append(true_part)
        if cut > last_true:
            out.append(dict(row, effective_start=last_true, effective_end=cut,
                            st_state=STATE_UNKNOWN, end_event_id=None,
                            end_announcement_date=cut, end_basis="dated_name_without_st",
                            note="ended_between_dated_names"))
    return pd.DataFrame(out, columns=episodes.columns)


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


# ---------------------------------------------------------------------------
# Evaluation against a dated register (ENG_ST30 §3–4) and the PIT interval
# contract consumed by ``quantagent.data.providers.st_pit``.
# ---------------------------------------------------------------------------

def merge_contiguous(episodes: pd.DataFrame) -> pd.DataFrame:
    """Merge a symbol's TRUE spans that touch or overlap (a rename inside a
    warning, e.g. ``ST国华`` → ``*ST国华``, is one episode, not two).

    Input columns: ``symbol, effective_start, effective_end`` (end exclusive,
    NaT = still open). Output has the same three columns.
    """
    if episodes.empty:
        return episodes[["symbol", "effective_start", "effective_end"]].copy()
    frame = episodes[["symbol", "effective_start", "effective_end"]].copy()
    frame = frame.sort_values(["symbol", "effective_start"])
    out: list[dict] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        cur_s = cur_e = None
        for row in group.itertuples(index=False):
            start, end = row.effective_start, row.effective_end
            if cur_s is None:
                cur_s, cur_e = start, end
                continue
            if pd.isna(cur_e) or start <= cur_e:
                if pd.isna(cur_e) or pd.isna(end):
                    cur_e = pd.NaT
                else:
                    cur_e = max(cur_e, end)
                continue
            out.append({"symbol": symbol, "effective_start": cur_s, "effective_end": cur_e})
            cur_s, cur_e = start, end
        out.append({"symbol": symbol, "effective_start": cur_s, "effective_end": cur_e})
    return pd.DataFrame(out, columns=["symbol", "effective_start", "effective_end"])


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    a_end = pd.Timestamp.max if pd.isna(a_end) else a_end
    b_end = pd.Timestamp.max if pd.isna(b_end) else b_end
    return bool(a_start < b_end and b_start < a_end)


def _overlap_days(a_start, a_end, b_start, b_end) -> float:
    a_end = pd.Timestamp("2262-01-01") if pd.isna(a_end) else a_end
    b_end = pd.Timestamp("2262-01-01") if pd.isna(b_end) else b_end
    return max(0.0, (min(a_end, b_end) - max(a_start, b_start)).days)


def evaluate_reconstruction(recon: pd.DataFrame, truth: pd.DataFrame, rows: pd.DataFrame,
                            calendar: Calendar, *, window: tuple[pd.Timestamp, pd.Timestamp],
                            symbols: Sequence[str]) -> dict:
    """ENG_ST30 §4 metrics for ``symbols`` (one validation half).

    ``recon``: reconstructed episodes (``st_state`` TRUE/UNKNOWN); ``truth``:
    register episodes (all TRUE); ``rows``: panel ``symbol, trade_date`` rows.

    Operationalisation fixed before any HOLDOUT evaluation (amendment A1):
    an episode "overlaps the window" when it covers any calendar day in it;
    matched pairs are formed per register episode with the reconstructed TRUE
    episode of largest calendar overlap; start agreement (M3) is measured on
    matched pairs whose register start lies in the window, end agreement (M4)
    on matched pairs whose register end lies in the window. Session distances
    use the exchange calendar.
    """
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    wanted = set(symbols)
    truth_m = merge_contiguous(truth[truth["symbol"].isin(wanted)])
    rec = recon[recon["symbol"].isin(wanted)]
    rec_true = merge_contiguous(rec[rec["st_state"] == STATE_TRUE])

    def in_window(frame: pd.DataFrame) -> pd.DataFrame:
        end = frame["effective_end"].fillna(pd.Timestamp.max)
        return frame[(frame["effective_start"] <= hi) & (end > lo)]

    truth_w, rec_w = in_window(truth_m), in_window(rec_true)
    rec_by_sym = {s: g for s, g in rec_true.groupby("symbol")}
    truth_by_sym = {s: g for s, g in truth_m.groupby("symbol")}

    recalled, start_ok, start_n, end_ok, end_n = 0, 0, 0, 0, 0
    misses: list[dict] = []
    start_errors: list[dict] = []
    end_errors: list[dict] = []
    for ep in truth_w.itertuples(index=False):
        cands = rec_by_sym.get(ep.symbol)
        best, best_days = None, 0.0
        if cands is not None:
            for c in cands.itertuples(index=False):
                days = _overlap_days(ep.effective_start, ep.effective_end,
                                     c.effective_start, c.effective_end)
                if days > best_days:
                    best, best_days = c, days
        if best is None:
            misses.append({"symbol": ep.symbol, "start": ep.effective_start,
                           "end": ep.effective_end})
            continue
        recalled += 1
        if lo <= ep.effective_start <= hi:
            start_n += 1
            d = calendar.distance(ep.effective_start, best.effective_start)
            if abs(d) <= 1:
                start_ok += 1
            else:
                start_errors.append({"symbol": ep.symbol, "truth": ep.effective_start,
                                     "recon": best.effective_start, "sessions": d})
        if pd.notna(ep.effective_end) and lo <= ep.effective_end <= hi:
            end_n += 1
            if pd.isna(best.effective_end):
                end_errors.append({"symbol": ep.symbol, "truth": ep.effective_end,
                                   "recon": None, "sessions": None})
            else:
                d = calendar.distance(ep.effective_end, best.effective_end)
                if abs(d) <= 1:
                    end_ok += 1
                else:
                    end_errors.append({"symbol": ep.symbol, "truth": ep.effective_end,
                                       "recon": best.effective_end, "sessions": d})

    precise = 0
    false_eps: list[dict] = []
    for ep in rec_w.itertuples(index=False):
        cands = truth_by_sym.get(ep.symbol)
        hit = cands is not None and any(
            _overlaps(ep.effective_start, ep.effective_end, c.effective_start, c.effective_end)
            for c in cands.itertuples(index=False))
        if hit:
            precise += 1
        else:
            false_eps.append({"symbol": ep.symbol, "start": ep.effective_start,
                              "end": ep.effective_end})

    sub = rows[rows["symbol"].isin(wanted)]
    sub = sub[(sub["trade_date"] >= lo) & (sub["trade_date"] <= hi)]
    truth_state = row_states(sub, truth_m.assign(st_state=STATE_TRUE))
    rec_state = row_states(sub, rec)
    t_true = truth_state == STATE_TRUE
    r_true = rec_state == STATE_TRUE
    n_t, n_r = int(t_true.sum()), int(r_true.sum())

    def ratio(a: int, b: int) -> float | None:
        return None if b == 0 else a / b

    return {
        "symbols": len(wanted), "rows": int(len(sub)),
        "register_episodes_in_window": int(len(truth_w)),
        "reconstructed_true_episodes_in_window": int(len(rec_w)),
        "M1_episode_recall": ratio(recalled, len(truth_w)),
        "M2_episode_precision": ratio(precise, len(rec_w)),
        "M3_start_within_1_session": ratio(start_ok, start_n), "M3_n": start_n,
        "M4_end_within_1_session": ratio(end_ok, end_n), "M4_n": end_n,
        "M5a_row_recall": ratio(int((t_true & r_true).sum()), n_t),
        "M5b_row_precision": ratio(int((t_true & r_true).sum()), n_r),
        "M5c_register_true_marked_false": ratio(
            int((t_true & (rec_state == "FALSE")).sum()), n_t),
        "register_true_marked_unknown": ratio(
            int((t_true & (rec_state == STATE_UNKNOWN)).sum()), n_t),
        "unknown_rows": int((rec_state == STATE_UNKNOWN).sum()),
        "M5_n_register_true": n_t, "M5_n_reconstructed_true": n_r,
        "examples": {"missed_episodes": misses[:40], "false_episodes": false_eps[:40],
                     "start_errors": start_errors[:40], "end_errors": end_errors[:40]},
    }


THRESHOLDS = {
    "M1_episode_recall": (">=", 0.95),
    "M2_episode_precision": (">=", 0.95),
    "M3_start_within_1_session": (">=", 0.95),
    "M4_end_within_1_session": (">=", 0.95),
    "M5a_row_recall": (">=", 0.98),
    "M5b_row_precision": (">=", 0.98),
    "M5c_register_true_marked_false": ("<=", 0.01),
}


def acceptance(metrics: dict) -> dict:
    """Apply the pre-registered HOLDOUT thresholds; a missing metric fails."""
    verdict = {}
    for key, (op, bound) in THRESHOLDS.items():
        value = metrics.get(key)
        ok = value is not None and (value >= bound if op == ">=" else value <= bound)
        verdict[key] = {"value": value, "threshold": f"{op} {bound}", "pass": bool(ok)}
    verdict["all_pass"] = all(v["pass"] for v in verdict.values() if isinstance(v, dict))
    return verdict


PIT_COLUMNS = ["symbol", "start_date", "end_date", "is_st", "available_at",
               "point_in_time_valid", "source", "evidence_id"]


def pit_intervals(episodes: pd.DataFrame, life: pd.DataFrame, *,
                  as_of: pd.Timestamp, coverage_start: pd.Timestamp) -> pd.DataFrame:
    """Complete ST/non-ST intervals in the ``st_pit`` interval contract.

    ``episodes``: ``symbol, effective_start, effective_end (exclusive, NaT=open),
    st_state (TRUE/UNKNOWN), available_at, source, evidence_id``.
    ``life``: ``symbol, first_day, last_day`` (inclusive) — listing to delisting.

    Every day of a symbol's life in ``[coverage_start, as_of]`` is TRUE, FALSE or
    UNKNOWN. TRUE and FALSE become rows with an INCLUSIVE ``end_date``; UNKNOWN
    spans get no row, so the strict loader refuses any market row inside one
    (absence is never read as FALSE). A FALSE span's ``available_at`` is the
    disclosure that ended the preceding warning, or the coverage/listing start.
    """
    as_of = pd.Timestamp(as_of).normalize()
    coverage_start = pd.Timestamp(coverage_start).normalize()
    one = pd.Timedelta(days=1)
    by_symbol = {s: g.sort_values("effective_start") for s, g in episodes.groupby("symbol")}
    out: list[dict] = []
    for life_row in life.itertuples(index=False):
        symbol = life_row.symbol
        first = max(pd.Timestamp(life_row.first_day).normalize(), coverage_start)
        last = min(pd.Timestamp(life_row.last_day).normalize()
                   if pd.notna(life_row.last_day) else as_of, as_of)
        if last < first:
            continue
        cursor, cursor_avail, cursor_src = first, first, "no_warning_disclosed"
        spans = by_symbol.get(symbol)
        rows = [] if spans is None else list(spans.itertuples(index=False))
        for ep in rows:
            start = pd.Timestamp(ep.effective_start).normalize()
            end_excl = (pd.Timestamp(ep.effective_end).normalize()
                        if pd.notna(ep.effective_end) else last + one)
            if end_excl <= first or start > last:
                continue
            start = max(start, first)
            end_incl = min(end_excl - one, last)
            if start > cursor:
                out.append({"symbol": symbol, "start_date": cursor, "end_date": start - one,
                            "is_st": False, "available_at": cursor_avail,
                            "source": cursor_src, "evidence_id": None})
            if ep.st_state == STATE_TRUE:
                avail = pd.Timestamp(ep.available_at) if pd.notna(ep.available_at) else start
                out.append({"symbol": symbol, "start_date": start, "end_date": end_incl,
                            "is_st": True, "available_at": min(avail, start),
                            "source": ep.source, "evidence_id": ep.evidence_id})
            cursor = max(cursor, end_incl + one)
            end_avail = getattr(ep, "end_available_at", pd.NaT)
            cursor_avail = (pd.Timestamp(end_avail) if pd.notna(end_avail) else cursor)
            cursor_avail = min(cursor_avail, cursor)
            cursor_src = ep.source
        if cursor <= last:
            out.append({"symbol": symbol, "start_date": cursor, "end_date": last,
                        "is_st": False, "available_at": cursor_avail, "source": cursor_src,
                        "evidence_id": None})
    frame = pd.DataFrame(out, columns=[c for c in PIT_COLUMNS if c != "point_in_time_valid"])
    frame["point_in_time_valid"] = True
    # available_at is a disclosure DATE; the loader compares it with 09:25 CST.
    frame["available_at"] = (pd.to_datetime(frame["available_at"]).dt.strftime("%Y-%m-%d")
                             + "T00:00:00+08:00")
    return frame[PIT_COLUMNS]
