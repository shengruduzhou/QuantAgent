import pandas as pd

from quantagent.data.ingestion import (
    DailyEvidenceJob,
    DailyEvidenceJobConfig,
    DisclosureIngestor,
    EVIDENCE_COLUMNS,
    EvidenceIngestor,
    FinancialIngestor,
    NewsIngestor,
    OrderContractIngestor,
    PolicyIngestor,
    RegulatoryPenaltyIngestor,
    SourceCredibilityRegistry,
    SourceProfile,
    SourceTier,
    attach_source_profile,
    merge_user_profiles,
)


def test_source_registry_resolves_known_hosts_to_correct_tier():
    registry = SourceCredibilityRegistry()
    gov = registry.lookup("www.gov.cn")
    sse = registry.lookup("www.sse.com.cn")
    eastmoney = registry.lookup("www.eastmoney.com")
    csrc = registry.lookup("www.csrc.gov.cn")
    xueqiu = registry.lookup("xueqiu.com")

    assert gov is not None and gov.tier == SourceTier.OFFICIAL_PRIMARY
    assert sse is not None and sse.tier == SourceTier.EXCHANGE_DISCLOSURE
    assert csrc is not None and csrc.tier == SourceTier.REGULATORY_PENALTY
    assert eastmoney is not None and eastmoney.tier == SourceTier.TIER2_FINANCIAL_MEDIA
    assert xueqiu is not None and xueqiu.reliability < gov.reliability


def test_source_registry_user_override_extends_profiles():
    registry = SourceCredibilityRegistry()
    merge_user_profiles(
        registry,
        [
            {
                "name": "company_x_ir",
                "host_or_id": "ir.example.com",
                "tier": SourceTier.COMPANY_OFFICIAL.value,
                "is_primary": True,
                "is_official": False,
                "source_type": "disclosure",
                "reliability_override": 0.81,
            }
        ],
    )
    profile = registry.lookup("ir.example.com")
    assert profile is not None
    assert profile.reliability == 0.81


def test_daily_evidence_job_unifies_ingestor_outputs(tmp_path):
    class FakeIngestor(EvidenceIngestor):
        name = "fake_news"
        source_type = "news"

        def fetch(self, config, registry):
            return pd.DataFrame(
                [
                    {
                        "source_name": "www.gov.cn",
                        "url": "https://www.gov.cn/policy",
                        "title": "AI compute action plan",
                        "body": "Support AI compute infrastructure",
                        "published_at": "2026-05-13",
                        "theme_candidates": "ai_compute",
                        "event_type": "policy_support",
                        "confidence": 0.90,
                    },
                    {
                        "source_name": "www.eastmoney.com",
                        "url": "https://www.eastmoney.com/news/1",
                        "title": "Market commentary",
                        "body": "AI compute is hot",
                        "published_at": "2026-05-14",
                        "theme_candidates": "ai_compute",
                        "event_type": "sentiment_positive",
                        "confidence": 0.55,
                    },
                ]
            )

    job = DailyEvidenceJob(
        registry=SourceCredibilityRegistry(),
        ingestors={"news": FakeIngestor()},
    )
    config = DailyEvidenceJobConfig(
        as_of_date="2026-05-14",
        cache_root=str(tmp_path / "evidence"),
        enabled_sources=("news",),
    )
    result = job.run(config)
    assert len(result.frame) == 2
    # All EVIDENCE_COLUMNS must be present after normalisation
    for column in EVIDENCE_COLUMNS:
        assert column in result.frame.columns
    # PIT enforcement: nothing past 2026-05-14
    assert (pd.to_datetime(result.frame["available_at"]) <= pd.Timestamp("2026-05-14")).all()
    # The cache file should have been written
    assert result.metadata["cache_path"].endswith(".csv")


def test_policy_ingestor_reads_local_cache(tmp_path):
    cache_dir = tmp_path / "policy"
    cache_dir.mkdir(parents=True)
    (cache_dir / "policy1.csv").write_text(
        "source_name,url,title,body,published_at\n"
        "www.gov.cn,https://www.gov.cn/p1,AI compute pilot,Support 算力 infrastructure,2026-04-01\n",
        encoding="utf-8",
    )
    ingestor = PolicyIngestor(local_cache_root=str(cache_dir), allow_network=False)
    config = DailyEvidenceJobConfig(as_of_date="2026-05-14", cache_root=str(tmp_path / "evidence"))
    frame = ingestor.fetch(config, SourceCredibilityRegistry())
    assert not frame.empty
    assert "ai_compute" in frame["theme_candidates"].iloc[0]
    assert frame["source_reliability"].iloc[0] >= 0.90  # gov.cn is OFFICIAL_PRIMARY


def test_disclosure_ingestor_tags_event_types(tmp_path):
    cache_dir = tmp_path / "disclosure"
    cache_dir.mkdir(parents=True)
    (cache_dir / "ann.csv").write_text(
        "source_name,url,title,body,published_at,symbol\n"
        "www.sse.com.cn,https://example,关于签订重大合同的公告,与XX签订订单50亿元,2026-04-15,600519.SH\n"
        "www.sse.com.cn,https://example,关于收到警示函的公告,监管警示函,2026-04-20,000858.SZ\n",
        encoding="utf-8",
    )
    ingestor = DisclosureIngestor(local_cache_root=str(cache_dir))
    config = DailyEvidenceJobConfig(as_of_date="2026-05-14", cache_root=str(tmp_path / "evidence"))
    frame = ingestor.fetch(config, SourceCredibilityRegistry())
    events = set(frame["event_type"].tolist())
    assert "order_confirmed" in events
    assert "regulatory_penalty" in events


def test_news_ingestor_flags_rumours(tmp_path):
    cache_dir = tmp_path / "news"
    cache_dir.mkdir(parents=True)
    (cache_dir / "n.csv").write_text(
        "source_name,url,title,body,published_at\n"
        "www.eastmoney.com,https://www.eastmoney.com/n1,据传公司将获得大订单,据传 the order is coming,2026-05-13\n",
        encoding="utf-8",
    )
    ingestor = NewsIngestor(local_cache_root=str(cache_dir))
    config = DailyEvidenceJobConfig(as_of_date="2026-05-14", cache_root=str(tmp_path / "evidence"))
    frame = ingestor.fetch(config, SourceCredibilityRegistry())
    assert bool(frame.iloc[0]["rumor_risk_flag"]) is True
    # Confidence must be penalised after rumour flag
    assert float(frame.iloc[0]["confidence"]) < 0.55


def test_news_ingestor_caps_single_low_reliability_core_signal(tmp_path):
    cache_dir = tmp_path / "news"
    cache_dir.mkdir(parents=True)
    (cache_dir / "n.csv").write_text(
        "source_name,url,title,body,published_at\n"
        "xueqiu.com,https://xueqiu.com/n1,CATL 300750.SZ gets huge order,EV battery and energy storage demand rises,2026-05-13\n",
        encoding="utf-8",
    )
    ingestor = NewsIngestor(local_cache_root=str(cache_dir))
    config = DailyEvidenceJobConfig(as_of_date="2026-05-14", cache_root=str(tmp_path / "evidence"))
    frame = ingestor.fetch(config, SourceCredibilityRegistry())

    assert "300750.SZ" in frame.iloc[0]["affected_symbols"]
    assert "ev_supply_chain" in frame.iloc[0]["theme_candidates"]
    assert bool(frame.iloc[0]["core_pool_signal_allowed"]) is False
    assert float(frame.iloc[0]["confidence"]) <= 0.34
    assert "single_low_reliability_source" in frame.iloc[0]["risk_flags"]
