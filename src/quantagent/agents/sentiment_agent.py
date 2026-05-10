"""V5 lexicon-based Chinese financial sentiment agent.

Default implementation uses a small handcrafted Chinese-finance lexicon with
intensifier and negation handling. The output is cross-sectional z-score-normalized
so it plays well with downstream BL views (q is bounded, omega scales with
evidence_quality).

Upgrade path: drop in a Chinese financial BERT (e.g. FinBERT-Chinese) by setting
``backend="bert"`` and providing a callable scorer; the rest of the pipeline does
not need to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quantagent.agents.views_schema import EvidenceRecord

DEFAULT_POSITIVE_LEXICON = (
    "增长", "上涨", "突破", "利好", "超预期", "盈利", "扩张", "中标", "回购",
    "增持", "签约", "新高", "强劲", "改善", "复苏", "提价", "拓展", "受益",
)
DEFAULT_NEGATIVE_LEXICON = (
    "下跌", "下滑", "亏损", "减持", "退市", "立案", "处罚", "暴跌", "风险",
    "下调", "停产", "诉讼", "违规", "造假", "腰斩", "退货", "失败", "下行",
)
INTENSIFIERS = ("大幅", "显著", "强劲", "急剧", "巨额", "全面", "首次")
NEGATIONS = ("未", "不", "无", "没有", "并未")


@dataclass(frozen=True)
class SentimentAgentConfig:
    positive_lexicon: tuple[str, ...] = DEFAULT_POSITIVE_LEXICON
    negative_lexicon: tuple[str, ...] = DEFAULT_NEGATIVE_LEXICON
    intensifier_multiplier: float = 1.4
    negation_flip: bool = True
    base_evidence_quality: float = 0.55
    confidence_floor: float = 0.30
    confidence_ceiling: float = 0.95


@dataclass
class SentimentAgent:
    """Score a panel of (symbol, text) rows into EvidenceRecords.

    Inputs (DataFrame):
        - symbol
        - timestamp (str or datetime)
        - text (Chinese news / report excerpt)
        - sector (optional)
    """

    config: SentimentAgentConfig = field(default_factory=SentimentAgentConfig)
    backend: str = "lexicon"
    bert_scorer: Callable[[list[str]], list[float]] | None = None

    def run(self, frame: pd.DataFrame) -> list[EvidenceRecord]:
        if frame.empty:
            return []
        if self.backend == "bert" and self.bert_scorer is not None:
            scores = np.asarray(self.bert_scorer(frame["text"].astype(str).tolist()), dtype=float)
        else:
            scores = np.asarray([self._score_text(str(t)) for t in frame["text"]], dtype=float)
        scores = self._cross_section_normalize(scores)
        records: list[EvidenceRecord] = []
        for (_, row), score in zip(frame.iterrows(), scores):
            magnitude = float(np.clip(abs(score), 0.0, 1.5))
            direction = float(np.sign(score))
            confidence = float(np.clip(
                self.config.confidence_floor + magnitude * 0.4,
                self.config.confidence_floor,
                self.config.confidence_ceiling,
            ))
            records.append(
                EvidenceRecord(
                    source="sentiment_agent",
                    timestamp=str(row.get("timestamp", "")),
                    symbol=str(row["symbol"]),
                    sector=str(row.get("sector")) if row.get("sector") else None,
                    event_type="sentiment",
                    horizon_days=int(row.get("horizon_days", 5)),
                    direction=direction,
                    magnitude=magnitude,
                    confidence=confidence,
                    decay_half_life=float(row.get("decay_half_life", 5.0)),
                    rationale=f"lexicon_score={score:.3f}",
                    raw_reference={"raw_score": float(score)},
                )
            )
        return records

    def _score_text(self, text: str) -> float:
        if not text:
            return 0.0
        score = 0.0
        for token in self.config.positive_lexicon:
            count = text.count(token)
            if count == 0:
                continue
            local = count
            if self._has_intensifier(text, token):
                local *= self.config.intensifier_multiplier
            if self.config.negation_flip and self._has_negation(text, token):
                local *= -1.0
            score += local
        for token in self.config.negative_lexicon:
            count = text.count(token)
            if count == 0:
                continue
            local = count
            if self._has_intensifier(text, token):
                local *= self.config.intensifier_multiplier
            if self.config.negation_flip and self._has_negation(text, token):
                local *= -1.0
            score -= local
        denom = max(1.0, np.log1p(len(text) / 50.0))
        return score / denom

    def _has_intensifier(self, text: str, token: str) -> bool:
        idx = text.find(token)
        if idx < 0:
            return False
        window = text[max(0, idx - 4):idx]
        return any(w in window for w in INTENSIFIERS)

    def _has_negation(self, text: str, token: str) -> bool:
        idx = text.find(token)
        if idx < 0:
            return False
        window = text[max(0, idx - 4):idx]
        return any(w in window for w in NEGATIONS)

    @staticmethod
    def _cross_section_normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size <= 1:
            return scores
        std = scores.std(ddof=0)
        if std < 1e-9:
            return scores
        return (scores - scores.mean()) / std
