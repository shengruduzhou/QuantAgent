"""Forward-anchored risk-warning (ST) state (ENG_ST31).

ENG_ST30 rebuilt ten years of SSE/BSE warnings from announcements alone and
failed its HOLDOUT: every large error was a stale state carried for months
because nothing re-anchored it. This module carries a *dated, full-universe*
state (an instrument-name snapshot) forward through classified announcements
and checks every later dated short name against the carried state. Where an
observation contradicts it, the sessions between the last confirmation and the
contradiction are UNKNOWN (no evidence row, so the strict loader refuses market
rows there) and the state restarts from the observation.

Pure functions only; network access lives in ``scripts/u0_st_forward.py``.
Pre-registration: ``docs/research/preregistrations/ENG_ST31_preregistration.md``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.data.ashare.st_announcements import (
    DELIST_PERIOD,
    IGNORE,
    IMPLEMENT,
    REVOKE,
    STATE_EVENTS,
    STATE_TRUE,
    STATE_UNKNOWN,
    TRANSITION,
    Calendar,
    consolidate_delist_period,
    drop_superseded_advance_notices,
    kinds_from_name,
    warning_kinds,
)

METHOD = "forward_anchored_ENG_ST31"
CST = "Asia/Shanghai"

# ENG_ST31 §2.2 pre-filters (from the ENG_ST30 HOLDOUT diagnosis, disclosed)
_NEGATED = re.compile(r"不(?:会|用|再|需)?被?(?:实施|实行)|无需(?:实施|实行)|不(?:涉及|触及)")
_CONTINUES = re.compile(r"继续(?:实施|实行)|并(?:继续)?(?:实施|实行)其他风险警示")
_UNDO = re.compile(r"(?:撤销|撤消|取消|解除)")


def prefilter_events(events: pd.DataFrame) -> pd.DataFrame:
    """Apply the two pre-registered title pre-filters to classified events.

    * an IMPLEMENT whose title says the warning is *not* applied → IGNORE;
    * a REVOKE whose title also continues a warning → TRANSITION.

    The changed rows carry ``prefilter`` = the rule name.
    """
    out = events.copy()
    out["prefilter"] = None
    titles = out["title"].astype(str).str.replace(r"\s+", "", regex=True)
    negated = out["event"].eq(IMPLEMENT) & titles.str.contains(_NEGATED)
    continues = out["event"].eq(REVOKE) & titles.str.contains(_CONTINUES)
    out.loc[negated, "event"] = IGNORE
    out.loc[negated, "prefilter"] = "negated_implement"
    out.loc[continues, "event"] = TRANSITION
    out.loc[continues, "prefilter"] = "revoke_continues_warning"
    return out


def is_marked(sec_name: object) -> bool:
    """True iff the short name carries an ST / *ST marker."""
    return kinds_from_name(sec_name) is not None


def _kinds_after_transition(title: object, current: frozenset[str]) -> frozenset[str]:
    text = re.sub(r"\s+", "", str(title or ""))
    match = _UNDO.search(text)
    rest = text[match.end():] if match else text
    tail = re.split(r"继续|并|后|暨|同时|变更为", rest, maxsplit=1)
    kinds = warning_kinds(tail[1]) if len(tail) == 2 else frozenset({"generic"})
    if kinds == frozenset({"generic"}):
        return current or frozenset({"other"})
    return kinds


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def snapshot_sessions(retrieved_at: pd.Timestamp, calendar: Calendar) -> tuple[pd.Timestamp, ...]:
    """Sessions a full-universe snapshot taken at ``retrieved_at`` can describe.

    Outside a session day's 09:15–15:00 window: the last session that ended
    before the instant and the next session (a vendor may already have rolled
    the name over for the next session). Inside it: that session and the next.
    """
    ts = pd.Timestamp(retrieved_at)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(CST).tz_localize(None)
    day = ts.normalize()
    minutes = ts.hour * 60 + ts.minute
    is_session = day in calendar.sessions
    if is_session and 9 * 60 + 15 <= minutes < 15 * 60:
        return tuple(s for s in (day, calendar.next_session_after(day)) if s is not None)
    if is_session and minutes >= 15 * 60:
        prev = day
    else:
        pos = calendar.sessions.searchsorted(day, side="left") - 1
        prev = calendar.sessions[pos] if pos >= 0 else None
    nxt = calendar.next_session_after(prev) if prev is not None else calendar.on_or_after(day)
    return tuple(s for s in (prev, nxt) if s is not None)


def name_sessions(disclosed_on: pd.Timestamp, calendar: Calendar) -> tuple[pd.Timestamp, ...]:
    """Sessions a cninfo short name printed on disclosure date D can describe:
    the last session strictly before D, the last session on or before D, and
    the first session after D."""
    day = pd.Timestamp(disclosed_on).normalize()
    pos_before = calendar.sessions.searchsorted(day, side="left") - 1
    pos_on = calendar.sessions.searchsorted(day, side="right") - 1
    found = []
    for pos in (pos_before, pos_on):
        if pos >= 0:
            found.append(calendar.sessions[pos])
    nxt = calendar.next_session_after(day)
    if nxt is not None:
        found.append(nxt)
    return tuple(sorted(set(found)))


@dataclass(frozen=True)
class Observation:
    """A dated short name of one security."""

    sessions: tuple[pd.Timestamp, ...]
    marked: bool
    kinds: frozenset[str]
    source: str
    ref: str
    observed_at: pd.Timestamp


def observations_for(frame: pd.DataFrame, calendar: Calendar) -> dict[str, list[Observation]]:
    """``frame``: ``symbol, kind ('snapshot'|'name'), observed_at, sec_name,
    source, ref`` → per-symbol observations sorted by their latest session."""
    out: dict[str, list[Observation]] = {}
    for row in frame.itertuples(index=False):
        sessions = (snapshot_sessions(row.observed_at, calendar) if row.kind == "snapshot"
                    else name_sessions(row.observed_at, calendar))
        if not sessions:
            continue
        kinds = kinds_from_name(row.sec_name)
        out.setdefault(row.symbol, []).append(Observation(
            sessions=sessions, marked=kinds is not None, kinds=kinds or frozenset(),
            source=str(row.source), ref=str(row.ref), observed_at=pd.Timestamp(row.observed_at)))
    for obs in out.values():
        obs.sort(key=lambda o: (o.sessions[-1], o.sessions[0]))
    return out


# ---------------------------------------------------------------------------
# Events → path
# ---------------------------------------------------------------------------

def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Pre-filters + the frozen ENG_ST30 de-duplication (one boundary per
    delisting period; an advance notice yields to the implementation notice)."""
    frame = prefilter_events(events)
    frame = frame[frame["event"].isin(STATE_EVENTS)]
    frame = consolidate_delist_period(frame)
    frame = drop_superseded_advance_notices(frame)
    return frame.sort_values(["symbol", "effective_date", "announcement_date"], na_position="last")


@dataclass
class _State:
    marked: bool
    kinds: frozenset[str]
    added: frozenset[str] = field(default_factory=frozenset)


def _apply(state: _State, ev) -> _State:
    kind = ev.event
    title_kinds = warning_kinds(getattr(ev, "title", ""))
    if kind == IMPLEMENT:
        if state.marked:
            return _State(True, state.kinds | title_kinds, state.added | title_kinds)
        return _State(True, title_kinds)
    if kind == TRANSITION:
        return _State(True, _kinds_after_transition(getattr(ev, "title", ""), state.kinds),
                      state.added)
    if kind == DELIST_PERIOD:
        return _State(False, frozenset())
    # REVOKE
    if not state.marked:
        return state
    if "generic" in title_kinds:
        return _State(False, frozenset())
    shown = kinds_from_name(getattr(ev, "sec_name", None))
    active = set(shown) if shown is not None else ({"delist", "other"} & set(state.kinds))
    if "delist" in active and "other" in state.added:
        active.add("other")
    remaining = frozenset(active - set(title_kinds))
    if remaining:
        return _State(True, remaining, state.added)
    return _State(False, frozenset())


def _path(sessions: pd.DatetimeIndex, i0: int, start: _State, events: list) -> tuple[np.ndarray, list]:
    """States (bool) for ``sessions[i0:]`` from ``start``, applying events
    effective strictly after ``sessions[i0]``; plus the change points."""
    n = len(sessions)
    marked = np.zeros(n - i0, dtype=bool)
    state = start
    changes = []
    cursor = i0
    for ev in events:
        when = ev.effective_date
        if pd.isna(when) or when <= sessions[i0]:
            continue
        j = int(sessions.searchsorted(when, side="left"))
        if j >= n:
            break
        marked[cursor - i0:j - i0] = state.marked
        new = _apply(state, ev)
        if new.marked != state.marked:
            changes.append((sessions[j], new.marked, ev))
        state, cursor = new, j
    marked[cursor - i0:] = state.marked
    return marked, changes


@dataclass(frozen=True)
class SymbolResult:
    symbol: str
    sessions: pd.DatetimeIndex
    state: np.ndarray            # 'T' / 'F' / 'U' per session
    restarts: tuple              # (observation, unknown_from, unknown_to)
    changes: tuple               # (session, marked, event-or-observation ref)


def forward_symbol(symbol: str, *, sessions: pd.DatetimeIndex, anchor_index: int,
                   start_marked: bool, start_kinds: frozenset[str], events: list,
                   observations: list[Observation]) -> SymbolResult:
    """Carry one security's state forward from ``sessions[anchor_index]``.

    ``anchor_index`` is the earliest session the start anchor can describe; the
    state there is the anchor's. Events effective after it are applied. Each
    observation (sorted by latest tolerated session) must match the carried
    state on at least one of its sessions; otherwise every session after the
    latest confirmed one up to the observation's latest session is UNKNOWN and
    the state restarts from the observation.
    """
    n = len(sessions)
    marked, changes = _path(sessions, anchor_index, _State(start_marked, start_kinds), events)
    current = np.zeros(n, dtype=bool)
    current[anchor_index:] = marked
    unknown = np.zeros(n, dtype=bool)
    confirmed = anchor_index
    restarts = []
    index = {s: i for i, s in enumerate(sessions)}
    all_changes = [(s, m, getattr(ev, "announcement_id", None)) for s, m, ev in changes]
    for obs in observations:
        tol = sorted(index[s] for s in obs.sessions if s in index and index[s] >= anchor_index)
        if not tol:
            continue
        hits = [i for i in tol if current[i] == obs.marked]
        if hits:
            confirmed = max(confirmed, hits[0])
            continue
        lo, hi = tol[0], tol[-1]
        unknown[confirmed + 1:hi + 1] = True
        restarts.append((obs, sessions[confirmed + 1] if confirmed + 1 < n else None, sessions[hi]))
        new, new_changes = _path(sessions, lo, _State(obs.marked, obs.kinds), events)
        current[lo:] = new
        all_changes = [c for c in all_changes if c[0] <= sessions[lo]]
        all_changes += [(s, m, getattr(ev, "announcement_id", None)) for s, m, ev in new_changes]
        confirmed = lo
    state = np.where(unknown, "U", np.where(current, "T", "F")).astype(object)
    state[:anchor_index] = None
    return SymbolResult(symbol, sessions, state, tuple(restarts), tuple(all_changes))


def run_forward(*, universe: pd.DataFrame, events: pd.DataFrame, observations: pd.DataFrame,
                calendar: Calendar, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward states for every security of ``universe``.

    ``universe``: ``symbol, anchor_session, start_session, start_name`` — the
    session the start anchor describes (state = ST-marking of ``start_name``),
    and the first session to emit. ``events``: classified + dated (ENG_ST30
    columns). ``observations``: see :func:`observations_for`.

    Returns ``(rows, log)``: rows = ``symbol, trade_date, state`` (TRUE / FALSE /
    UNKNOWN) for sessions in ``[start_session, as_of]``; log = one row per
    restart (contradicting observation and the UNKNOWN span it caused).
    """
    as_of = pd.Timestamp(as_of).normalize()
    prepared = prepare_events(events)
    by_symbol = {s: list(g.itertuples(index=False)) for s, g in prepared.groupby("symbol")}
    obs_by_symbol = observations_for(observations, calendar)
    first = pd.Timestamp(universe["anchor_session"].min())
    horizon_end = max([as_of] + [o.sessions[-1] for obs in obs_by_symbol.values() for o in obs])
    sessions = calendar.sessions[(calendar.sessions >= first) & (calendar.sessions <= horizon_end)]
    out_frames, log = [], []
    label = {"T": "TRUE", "F": "FALSE", "U": "UNKNOWN"}
    for row in universe.itertuples(index=False):
        anchor = pd.Timestamp(row.anchor_session)
        anchor_index = int(sessions.searchsorted(anchor, side="left"))
        kinds = kinds_from_name(row.start_name)
        result = forward_symbol(
            row.symbol, sessions=sessions, anchor_index=anchor_index,
            start_marked=kinds is not None, start_kinds=kinds or frozenset(),
            events=by_symbol.get(row.symbol, []), observations=obs_by_symbol.get(row.symbol, []))
        keep = (sessions >= pd.Timestamp(row.start_session)) & (sessions <= as_of)
        out_frames.append(pd.DataFrame({"symbol": row.symbol, "trade_date": sessions[keep],
                                        "state": [label[s] for s in result.state[keep]]}))
        for obs, lo, hi in result.restarts:
            log.append({"symbol": row.symbol, "observation_source": obs.source,
                        "observation_ref": obs.ref, "observed_at": obs.observed_at,
                        "observed_marked": obs.marked, "unknown_from": lo, "unknown_to": hi})
    rows = (pd.concat(out_frames, ignore_index=True) if out_frames
            else pd.DataFrame(columns=["symbol", "trade_date", "state"]))
    return rows, pd.DataFrame(log)


# ---------------------------------------------------------------------------
# Rows → strict PIT intervals; evaluation against a dated register
# ---------------------------------------------------------------------------

def rows_to_intervals(rows: pd.DataFrame, calendar: Calendar, *, as_of: pd.Timestamp,
                      source: str) -> pd.DataFrame:
    """TRUE/FALSE session runs → ``st_pit`` interval rows covering calendar days.

    A run starting on session s covers calendar days from s through the day
    before the next session (weekends/holidays inherit the last session's
    state); the last run ends at ``as_of``. UNKNOWN runs produce no row, and a
    run that follows an UNKNOWN run starts on its own first session.
    """
    as_of = pd.Timestamp(as_of).normalize()
    out = []
    for symbol, group in rows.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=True):
        dates = pd.to_datetime(group["trade_date"]).to_numpy()
        states = group["state"].to_numpy()
        start = 0
        for i in range(1, len(states) + 1):
            if i < len(states) and states[i] == states[start]:
                continue
            if states[start] != STATE_UNKNOWN:
                first = pd.Timestamp(dates[start])
                if i < len(states):
                    end = pd.Timestamp(dates[i]) - pd.Timedelta(days=1)
                else:
                    end = as_of
                out.append({"symbol": symbol, "start_date": first, "end_date": end,
                            "is_st": states[start] == STATE_TRUE,
                            "available_at": f"{first:%Y-%m-%d}T00:00:00+08:00",
                            "point_in_time_valid": True, "source": source,
                            "evidence_id": METHOD})
            start = i
    columns = ["symbol", "start_date", "end_date", "is_st", "available_at",
               "point_in_time_valid", "source", "evidence_id"]
    return pd.DataFrame(out, columns=columns)


def register_states(register: pd.DataFrame, symbols, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Register TRUE episodes (``effective_start`` incl., ``effective_end`` excl.,
    NaT = open) → ``symbol, trade_date, truth`` (bool) per session."""
    frame = pd.DataFrame([(s, d) for s in symbols for d in sessions], columns=["symbol", "trade_date"])
    truth = np.zeros(len(frame), dtype=bool)
    by_symbol = {s: g for s, g in register.groupby("symbol")}
    positions = {s: np.flatnonzero(frame["symbol"].to_numpy() == s) for s in by_symbol}
    dates = frame["trade_date"].to_numpy()
    for symbol, group in by_symbol.items():
        idx = positions.get(symbol)
        if idx is None or not len(idx):
            continue
        for ep in group.itertuples(index=False):
            end = ep.effective_end if pd.notna(ep.effective_end) else pd.Timestamp.max
            hit = (dates[idx] >= np.datetime64(pd.Timestamp(ep.effective_start))) & \
                  (dates[idx] < np.datetime64(pd.Timestamp(end)))
            truth[idx[hit]] = True
    frame["truth"] = truth
    return frame


THRESHOLDS = {"V1": (">=", 0.99), "V2": (">=", 0.99), "V3": (">=", 0.90),
              "V4": ("<=", 0.005), "V5": (">=", 0.95)}


def evaluate_forward(recon: pd.DataFrame, truth: pd.DataFrame, calendar: Calendar, *,
                     window: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    """ENG_ST31 §5 metrics.

    ``recon``: ``symbol, trade_date, state`` (TRUE/FALSE/UNKNOWN), including the
    session before the window when available (to see a flip on its first day).
    ``truth``: ``symbol, trade_date, truth`` over the same sessions.
    """
    lo, hi = (pd.Timestamp(x) for x in window)
    merged = truth.merge(recon, on=["symbol", "trade_date"], how="left")
    merged["state"] = merged["state"].fillna(STATE_UNKNOWN)
    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    in_window = merged["trade_date"].between(lo, hi)
    w = merged[in_window]
    emitted = w[w["state"] != STATE_UNKNOWN]
    tp = int(((emitted["state"] == STATE_TRUE) & emitted["truth"]).sum())
    fp = int(((emitted["state"] == STATE_TRUE) & ~emitted["truth"]).sum())
    fn = int(((emitted["state"] != STATE_TRUE) & emitted["truth"]).sum())
    truth_true = int(w["truth"].sum())
    unknown_rows = int((w["state"] == STATE_UNKNOWN).sum())
    metrics = {
        "rows": int(len(w)), "symbols": int(w["symbol"].nunique()),
        "emitted_true": tp + fp, "register_true": truth_true,
        "false_true_rows": fp, "missed_true_rows": fn, "unknown_rows": unknown_rows,
        "V1": tp / (tp + fp) if tp + fp else None,
        "V2": tp / (tp + fn) if tp + fn else None,
        "V4": unknown_rows / len(w) if len(w) else None,
        "V5": (truth_true - int((w["truth"] & (w["state"] == STATE_UNKNOWN)).sum())) / truth_true
        if truth_true else None,
    }
    # transitions
    flips, matched_n, examples = [], 0, []
    for symbol, g in merged.groupby("symbol", sort=False):
        t = g["truth"].to_numpy()
        s = g["state"].to_numpy()
        d = g["trade_date"].to_numpy()
        for i in range(1, len(t)):
            if t[i] == t[i - 1] or not (lo <= pd.Timestamp(d[i]) <= hi):
                continue
            new = STATE_TRUE if t[i] else "FALSE"
            old = "FALSE" if t[i] else STATE_TRUE
            ok = s[i] == STATE_UNKNOWN or s[i - 1] == STATE_UNKNOWN
            for j in (i - 1, i, i + 1):
                if 1 <= j < len(s) and s[j] == new and s[j - 1] == old:
                    ok = True
            flips.append(symbol)
            matched_n += int(ok)
            if not ok:
                examples.append({"symbol": symbol, "date": str(pd.Timestamp(d[i]).date()),
                                 "to": new})
    metrics["V3_n"] = len(flips)
    metrics["V3"] = matched_n / len(flips) if flips else None
    metrics["V3_misses"] = examples
    bad = w[(w["state"] == STATE_TRUE) != w["truth"]]
    bad = bad[bad["state"] != STATE_UNKNOWN]
    metrics["wrong_rows_by_symbol"] = bad.groupby("symbol").size().sort_values(
        ascending=False).head(20).to_dict()
    return metrics


def acceptance(metrics: dict) -> dict:
    result = {}
    for key, (op, bound) in THRESHOLDS.items():
        value = metrics.get(key)
        ok = value is not None and (value >= bound if op == ">=" else value <= bound)
        result[key] = {"value": value, "threshold": f"{op} {bound}", "pass": bool(ok)}
    result["all_pass"] = all(v["pass"] for v in result.values() if isinstance(v, dict))
    return result
