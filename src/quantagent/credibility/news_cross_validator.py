"""Deterministic news cross-validator.

The existing :func:`score_news_credibility` accepts pre-computed fields
like ``cross_validation_count`` and ``rumor_risk``. That makes it easy to
test, but in production we want the system to compute those fields by
*looking at the evidence stream itself*, not by trusting whatever a
single ingestor wrote.

This module produces, given a unified evidence frame (the one emitted by
:class:`DailyEvidenceJob`), a per-(symbol, theme, event_type) summary
with:

* ``confirming_sources`` — distinct source tiers that confirm the event.
* ``primary_confirmations`` — distinct PRIMARY sources (exchange,
  policy original text, company official).
* ``official_confirmations`` — distinct OFFICIAL sources (gov, regulator,
  exchange).
* ``contradiction_count`` — distinct sources that explicitly rebut.
* ``same_source_reposts`` — duplicate raw-hash count.
* ``after_close_only`` — whether the only confirmations are post-close
  publications.
* ``rumor_risk`` — derived from rumour-keyword flags.

The summary is then plugged into :func:`score_news_credibility` so the
final ``news_confidence`` reflects real corroboration instead of trusting
the inbound field.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd


_REFUTATION_PATTERNS = (
    "辟谣",
    "澄清",
    "否认",
    "并非",
    "不实",
    "回应称",
)


@dataclass(frozen=True)
class CrossValidationSummary:
    symbol: str
    theme: str
    event_type: str
    confirming_sources: int
    primary_confirmations: int
    official_confirmations: int
    contradiction_count: int
    same_source_reposts: int
    after_close_only: bool
    rumor_risk: float
    most_authoritative_source: str
    earliest_published_at: str
    latest_published_at: str


def cross_validate(
    evidence_frame: pd.DataFrame,
    rumour_keywords: Iterable[str] = (
        "传闻", "据传", "据悉", "未经证实", "市场传言", "传言",
    ),
) -> list[CrossValidationSummary]:
    """Group an evidence frame by (symbol, theme, event_type) and summarise."""

    if evidence_frame is None or evidence_frame.empty:
        return []
    required = {
        "symbol",
        "theme_candidates",
        "event_type",
        "source_name",
        "source_authority",
        "is_primary_source",
        "is_official",
        "published_at",
        "raw_hash",
        "body",
    }
    missing = required - set(evidence_frame.columns)
    if missing:
        for column in missing:
            evidence_frame = evidence_frame.copy()
            evidence_frame[column] = None
    rumour_keywords_tuple = tuple(keyword.lower() for keyword in rumour_keywords)
    out: list[CrossValidationSummary] = []
    flattened = _flatten_theme_candidates(evidence_frame)
    for (symbol, theme, event_type), group in flattened.groupby(["symbol", "theme", "event_type"], sort=False):
        if not symbol or pd.isna(symbol):
            continue
        confirming_sources = group["source_name"].astype(str).nunique()
        primary_confirmations = int(group["is_primary_source"].fillna(False).astype(bool).sum())
        official_confirmations = int(group["is_official"].fillna(False).astype(bool).sum())
        contradiction_count = int(_count_refutations(group["body"].fillna("").astype(str)))
        same_source_reposts = _count_same_source_reposts(group)
        after_close_only = _all_post_close(group["published_at"])
        rumor_risk = _rumor_risk(group["body"].fillna("").astype(str), rumour_keywords_tuple)
        most_authoritative = _most_authoritative(group)
        earliest = pd.to_datetime(group["published_at"], errors="coerce").min()
        latest = pd.to_datetime(group["published_at"], errors="coerce").max()
        out.append(
            CrossValidationSummary(
                symbol=str(symbol),
                theme=str(theme) if theme else "",
                event_type=str(event_type) if event_type else "",
                confirming_sources=int(confirming_sources),
                primary_confirmations=primary_confirmations,
                official_confirmations=official_confirmations,
                contradiction_count=contradiction_count,
                same_source_reposts=same_source_reposts,
                after_close_only=after_close_only,
                rumor_risk=rumor_risk,
                most_authoritative_source=most_authoritative,
                earliest_published_at=earliest.strftime("%Y-%m-%d") if pd.notna(earliest) else "",
                latest_published_at=latest.strftime("%Y-%m-%d") if pd.notna(latest) else "",
            )
        )
    return out


def attach_cross_validation_fields(
    news_scores: list,
    summaries: list[CrossValidationSummary],
) -> list:
    """Patch each NewsCredibilityScore with the values our validator derived.

    The function does not mutate inputs in place. It returns a new list of
    score dataclasses with ``cross_validation_count``, ``contradiction_flags``
    and ``rumor_risk`` overridden by the deterministic computation.

    The match key is ``(affected_symbol, affected_theme, event_type)`` —
    the score uses ``affected_symbols`` (a tuple) so we match on the first
    affected symbol. Scores whose key is not in the summary list are
    returned unchanged.
    """

    if not news_scores or not summaries:
        return news_scores
    summary_by_key: dict[tuple[str, str, str], CrossValidationSummary] = {}
    for summary in summaries:
        summary_by_key[(summary.symbol, summary.theme, summary.event_type)] = summary
    patched: list = []
    for score in news_scores:
        affected_symbols = tuple(getattr(score, "affected_symbols", ()) or ())
        symbol = affected_symbols[0] if affected_symbols else ""
        theme = str(getattr(score, "affected_theme", "") or "")
        event_value = getattr(score, "event_type", "")
        event_type = getattr(event_value, "value", str(event_value))
        summary = summary_by_key.get((symbol, theme, event_type))
        if summary is None:
            patched.append(score)
            continue
        new_flags = tuple(
            flag
            for flag in (
                f"contradictions={summary.contradiction_count}" if summary.contradiction_count else "",
                "after_close_only" if summary.after_close_only else "",
                f"same_source_reposts={summary.same_source_reposts}" if summary.same_source_reposts else "",
            )
            if flag
        )
        try:
            new_score = score.__class__(
                **{
                    field: getattr(score, field)
                    for field in score.__dataclass_fields__
                    if field not in {
                        "cross_validation_count",
                        "contradiction_flags",
                        "rumor_risk",
                    }
                },
                cross_validation_count=summary.confirming_sources,
                contradiction_flags=new_flags,
                rumor_risk=summary.rumor_risk,
            )
        except TypeError:
            patched.append(score)
            continue
        patched.append(new_score)
    return patched


def _flatten_theme_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "theme_candidates" not in frame.columns:
        data = frame.copy()
        data["theme"] = data.get("theme", "")
        return data
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        themes = str(row.get("theme_candidates", ""))
        theme_list = [item.strip() for item in themes.split(",") if item.strip()] or [""]
        for theme in theme_list:
            row_dict = row.to_dict()
            row_dict["theme"] = theme
            rows.append(row_dict)
    return pd.DataFrame(rows)


def _count_refutations(bodies: pd.Series) -> int:
    return int(sum(any(pattern in body for pattern in _REFUTATION_PATTERNS) for body in bodies))


def _count_same_source_reposts(group: pd.DataFrame) -> int:
    if "raw_hash" not in group.columns:
        return 0
    counts = group["raw_hash"].value_counts()
    return int((counts - 1).clip(lower=0).sum())


def _all_post_close(published_at: pd.Series) -> bool:
    parsed = pd.to_datetime(published_at, errors="coerce").dropna()
    if parsed.empty:
        return False
    times = parsed.dt.time
    return all((time.hour >= 15) for time in times)


def _rumor_risk(bodies: pd.Series, rumour_keywords: tuple[str, ...]) -> float:
    if bodies.empty:
        return 0.0
    flagged = sum(any(keyword in body.lower() for keyword in rumour_keywords) for body in bodies)
    return float(min(1.0, flagged / max(1, len(bodies))))


def _most_authoritative(group: pd.DataFrame) -> str:
    if group.empty:
        return ""
    authority = pd.to_numeric(group.get("source_authority", 0.0), errors="coerce").fillna(0.0)
    idx = authority.idxmax() if not authority.empty else None
    if idx is None:
        return ""
    return str(group.loc[idx, "source_name"])
