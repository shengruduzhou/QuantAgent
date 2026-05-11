from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.agents.commodity_agent import commodity_evidence_records
from quantagent.agents.news_agent import NewsAgent
from quantagent.agents.policy_agent import PolicyEvent, policy_evidence_records
from quantagent.agents.sentiment_agent import SentimentAgent
from quantagent.agents.views_schema import EvidenceRecord


@dataclass
class AgentCommittee:
    enable_sentiment: bool = True
    enable_news: bool = True
    enable_policy: bool = True
    enable_flow: bool = True
    enable_commodity: bool = True
    enable_financial_statement: bool = True
    news_agent: NewsAgent = field(default_factory=NewsAgent)
    sentiment_agent: SentimentAgent = field(default_factory=SentimentAgent)

    def run(
        self,
        trade_date: str,
        universe: pd.Index | list[str],
        news: pd.DataFrame | None = None,
        fund_flow: pd.DataFrame | None = None,
        fundamentals: pd.DataFrame | None = None,
        commodity: pd.DataFrame | None = None,
        sector_map: pd.Series | None = None,
    ) -> list[EvidenceRecord]:
        symbols = pd.Index(universe).astype(str)
        records: list[EvidenceRecord] = []
        sector_map = sector_map if sector_map is not None else pd.Series("market", index=symbols)
        if self.enable_news and news is not None:
            records.extend(self.news_agent.run(news))
        if self.enable_sentiment and news is not None and not news.empty:
            text = news.rename(columns={"summary": "text"}).copy()
            if "text" not in text.columns:
                text["text"] = text.get("title", "")
            records.extend(self.sentiment_agent.run(text[["symbol", "timestamp", "text", "sector"]].copy()))
        if self.enable_policy:
            event = PolicyEvent(
                published_at=str(trade_date),
                headline="V6 mock policy evidence routed as structured evidence",
                sectors=tuple(sorted(set(sector_map.dropna().astype(str).head(2)))),
                polarity=0.1,
            )
            records.extend(policy_evidence_records([event], sector_map, reference_date=pd.Timestamp(trade_date)))
        if self.enable_flow and fund_flow is not None and not fund_flow.empty:
            flow_latest = fund_flow.copy()
            flow_latest["trade_date"] = pd.to_datetime(flow_latest["trade_date"])
            flow_latest = flow_latest[flow_latest["trade_date"] <= pd.Timestamp(trade_date)]
            if not flow_latest.empty:
                records.extend(_flow_records(flow_latest, trade_date))
        records.extend(_sector_rotation_records(sector_map, trade_date))
        if self.enable_commodity and commodity is not None and not commodity.empty:
            latest = commodity.copy()
            latest["trade_date"] = pd.to_datetime(latest["trade_date"])
            latest = latest[latest["trade_date"] <= pd.Timestamp(trade_date)].groupby("commodity").tail(1)
            if "return" in latest.columns and not latest.empty:
                returns = latest.set_index("commodity")["return"].astype(float)
                records.extend(commodity_evidence_records(returns, sector_map, timestamp=str(trade_date), threshold=0.0))
        if self.enable_financial_statement and fundamentals is not None and not fundamentals.empty:
            records.extend(_fundamental_records(fundamentals, trade_date))
        return [record for record in records if record.symbol is None or str(record.symbol) in set(symbols)]


def _fundamental_records(fundamentals: pd.DataFrame, trade_date: str) -> list[EvidenceRecord]:
    data = fundamentals.copy()
    if "announcement_time" in data.columns:
        data["announcement_time"] = pd.to_datetime(data["announcement_time"])
        data = data[data["announcement_time"] <= pd.Timestamp(trade_date) + pd.Timedelta(hours=15)]
    records: list[EvidenceRecord] = []
    for _, row in data.groupby("symbol").tail(1).iterrows():
        quality = float(row.get("roe", 0.08)) - 0.5 * float(row.get("debt_to_asset", 0.4))
        direction = float(np.sign(quality))
        records.append(
            EvidenceRecord(
                source="financial_statement_agent",
                timestamp=str(row.get("announcement_time", trade_date)),
                symbol=str(row["symbol"]),
                event_type="financial_statement",
                horizon_days=20,
                direction=direction,
                magnitude=float(abs(np.tanh(quality))),
                confidence=0.55,
                rationale="PIT fundamental quality snapshot",
                raw_reference={"report_period": row.get("report_period", "")},
            )
        )
    return records


def _flow_records(flow_latest: pd.DataFrame, trade_date: str) -> list[EvidenceRecord]:
    latest = flow_latest.sort_values("trade_date").groupby("symbol", sort=False).tail(1)
    records: list[EvidenceRecord] = []
    for _, row in latest.iterrows():
        score = float(row.get("northbound_flow", 0.0)) / 1_000_000.0 + float(row.get("main_money_flow", 0.0)) / 2_000_000.0
        records.append(
            EvidenceRecord(
                source="flow_agent",
                timestamp=str(trade_date),
                symbol=str(row["symbol"]),
                event_type="fund_flow",
                horizon_days=5,
                direction=float(np.sign(score)),
                magnitude=float(abs(np.tanh(score))),
                confidence=0.50,
                rationale="latest PIT fund flow snapshot",
                raw_reference={"trade_date": str(row.get("trade_date", trade_date))},
            )
        )
    return records


def _sector_rotation_records(sector_map: pd.Series, trade_date: str) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for i, (symbol, sector) in enumerate(sector_map.items()):
        score = 0.05 if i % 2 == 0 else -0.02
        records.append(
            EvidenceRecord(
                source="sector_rotation_agent",
                timestamp=str(trade_date),
                symbol=str(symbol),
                sector=str(sector),
                event_type="sector_rotation",
                horizon_days=10,
                direction=float(np.sign(score)),
                magnitude=float(abs(score)),
                confidence=0.45,
                rationale="sector rotation neutral mock fallback",
                raw_reference={"sector": str(sector)},
            )
        )
    return records
