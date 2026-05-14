import pandas as pd

from quantagent.credibility.news_cross_validator import (
    attach_cross_validation_fields,
    cross_validate,
)
from quantagent.credibility.news_credibility_agent import (
    NewsCredibilityScore,
    score_news_credibility,
)


def _evidence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "theme_candidates": "consumer_recovery",
                "event_type": "earnings_growth",
                "source_name": "www.cs.com.cn",
                "source_authority": 0.78,
                "is_primary_source": False,
                "is_official": True,
                "published_at": "2026-04-29 09:30",
                "raw_hash": "h1",
                "body": "公司一季度营收同比 +20%",
            },
            {
                "symbol": "600519.SH",
                "theme_candidates": "consumer_recovery",
                "event_type": "earnings_growth",
                "source_name": "www.eastmoney.com",
                "source_authority": 0.55,
                "is_primary_source": False,
                "is_official": False,
                "published_at": "2026-04-29 14:00",
                "raw_hash": "h2",
                "body": "营收增长强劲",
            },
            {
                "symbol": "600519.SH",
                "theme_candidates": "consumer_recovery",
                "event_type": "earnings_growth",
                "source_name": "www.eastmoney.com",
                "source_authority": 0.55,
                "is_primary_source": False,
                "is_official": False,
                "published_at": "2026-04-29 14:00",
                "raw_hash": "h2",  # Same hash → repost
                "body": "营收增长强劲",
            },
            {
                "symbol": "002000.SZ",
                "theme_candidates": "ai_compute",
                "event_type": "sentiment_positive",
                "source_name": "xueqiu.com",
                "source_authority": 0.55,
                "is_primary_source": False,
                "is_official": False,
                "published_at": "2026-05-13 22:00",
                "raw_hash": "h3",
                "body": "据传公司将拿到大订单",
            },
            {
                "symbol": "002000.SZ",
                "theme_candidates": "ai_compute",
                "event_type": "sentiment_positive",
                "source_name": "www.cs.com.cn",
                "source_authority": 0.78,
                "is_primary_source": False,
                "is_official": True,
                "published_at": "2026-05-14 19:00",
                "raw_hash": "h4",
                "body": "公司发布澄清公告，否认市场传闻",
            },
        ]
    )


def test_cross_validate_counts_distinct_sources_and_reposts():
    summaries = cross_validate(_evidence_frame())
    by_key = {(s.symbol, s.event_type): s for s in summaries}
    earnings = by_key[("600519.SH", "earnings_growth")]
    rumor = by_key[("002000.SZ", "sentiment_positive")]

    # Two distinct sources for the earnings event, one repost
    assert earnings.confirming_sources == 2
    assert earnings.same_source_reposts >= 1
    # Rumour event has a refutation entry
    assert rumor.contradiction_count >= 1
    assert rumor.rumor_risk > 0.0
    assert rumor.after_close_only is True


def test_attach_cross_validation_fields_overrides_inbound_counts():
    evidence_frame = _evidence_frame()
    base_scores = score_news_credibility(evidence_frame)
    # Replace cross_validation_count / rumor_risk on the inbound scores so we
    # can verify the override actually replaces them with deterministic
    # computation. We line up the scores' affected_symbols / affected_theme
    # / event_type with the cross-validator summaries.
    patched_inputs: list[NewsCredibilityScore] = []
    for score in base_scores:
        patched_inputs.append(
            NewsCredibilityScore(
                **{
                    field: getattr(score, field)
                    for field in score.__dataclass_fields__
                    if field
                    not in {
                        "cross_validation_count",
                        "rumor_risk",
                        "affected_symbols",
                        "affected_theme",
                    }
                },
                cross_validation_count=99,
                rumor_risk=0.99,
                affected_symbols=("600519.SH",),
                affected_theme="consumer_recovery",
            )
        )
    summaries = cross_validate(evidence_frame)
    patched = attach_cross_validation_fields(patched_inputs, summaries)
    assert len(patched) == len(patched_inputs)
    # At least one score now reflects the deterministic count
    overridden = [score for score in patched if score.cross_validation_count != 99]
    assert overridden
