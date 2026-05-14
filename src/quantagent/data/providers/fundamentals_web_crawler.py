"""Public-web fundamentals crawler.

Pulls A-share fundamentals from public pages (东方财富 / 新浪财经 /
腾讯财经 / 巨潮资讯 / 交易所公告). Default behaviour is offline — the
crawler refuses to hit the network unless ``allow_network=True``. Each
fetched page is hashed and tagged with ``source_reliability`` so it
plugs straight into the V7 evidence schema.

This is a *complement* to :class:`TuShareFinancialProvider` /
:class:`AkShareFinancialProvider`, not a replacement. When a paid data
vendor is unavailable (no token, no quota, offline laptop), the
fundamentals crawler can still provide a coarse-grained set of fields by
parsing public HTML.

URL templates use ``{symbol}`` for the A-share ticker (no exchange suffix)
and ``{symbol_full}`` for the dotted form (e.g. ``600519.SH``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from html import unescape
import re
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


_DEFAULT_TEMPLATES: dict[str, str] = {
    "eastmoney_profile": "https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={symbol_full}",
    "eastmoney_business": "https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code={symbol_full}",
    "eastmoney_financial": "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/MainTargetAjax?code={symbol_full}",
    "sina_finance_profile": "https://money.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/{symbol}.phtml",
    "cninfo_announcement": "http://www.cninfo.com.cn/new/disclosure/detail?stockCode={symbol}",
}


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


@dataclass(frozen=True)
class FundamentalsCrawlConfig:
    allow_network: bool = False
    timeout_seconds: float = 10.0
    templates: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_TEMPLATES))
    user_agent: str = "QuantAgent-V7-Fundamentals-Bot/0.1"
    accept_language: str = "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"


@dataclass(frozen=True)
class FundamentalsDocument:
    template_id: str
    symbol: str
    url: str
    fetched_at: str
    title: str
    body: str
    raw_hash: str
    source_reliability: float
    extracted_fields: dict[str, float]


class FundamentalsWebCrawler:
    """Stdlib-only crawler for public A-share fundamentals."""

    def __init__(self, config: FundamentalsCrawlConfig | None = None) -> None:
        self.config = config or FundamentalsCrawlConfig()

    def fetch_for_symbol(
        self,
        symbol: str,
        as_of_date: str,
        template_ids: Iterable[str] | None = None,
    ) -> list[FundamentalsDocument]:
        if not self.config.allow_network:
            raise ProviderUnavailable(
                "Fundamentals web crawler is disabled; set fundamentals_pipeline.allow_network=true to enable"
            )
        template_ids = tuple(template_ids) if template_ids else tuple(self.config.templates.keys())
        documents: list[FundamentalsDocument] = []
        for template_id in template_ids:
            template = self.config.templates.get(template_id)
            if not template:
                continue
            url = template.format(symbol=_plain(symbol), symbol_full=_full(symbol))
            raw = self._fetch(url)
            if not raw:
                continue
            cleaned = _strip_html(raw)
            digest = sha256(f"{url}\n{cleaned}".encode("utf-8")).hexdigest()
            documents.append(
                FundamentalsDocument(
                    template_id=template_id,
                    symbol=symbol,
                    url=url,
                    fetched_at=as_of_date,
                    title=template_id,
                    body=cleaned[:10_000],
                    raw_hash=digest,
                    source_reliability=self._template_reliability(template_id),
                    extracted_fields=extract_numeric_fields(cleaned),
                )
            )
        return documents

    def fetch_bulk(
        self,
        request: ProviderRequest,
        template_ids: Iterable[str] | None = None,
    ) -> ProviderResult:
        if not request.symbols:
            raise ProviderUnavailable("fundamentals web crawl requires explicit symbols")
        rows: list[dict[str, object]] = []
        warnings: list[str] = []
        for symbol in request.symbols:
            try:
                documents = self.fetch_for_symbol(symbol, request.end_date, template_ids)
            except ProviderUnavailable as exc:
                warnings.append(f"{symbol}:{exc}")
                continue
            for doc in documents:
                rows.append(
                    {
                        "symbol": symbol,
                        "url": doc.url,
                        "template_id": doc.template_id,
                        "fetched_at": doc.fetched_at,
                        "title": doc.title,
                        "body": doc.body,
                        "source_reliability": doc.source_reliability,
                        "raw_hash": doc.raw_hash,
                        **doc.extracted_fields,
                    }
                )
        frame = pd.DataFrame(rows)
        return ProviderResult(
            frame,
            source="fundamentals_web_crawler",
            point_in_time=True,
            quality_score=0.65 if not frame.empty else 0.0,
            warnings=tuple(warnings) if warnings else (),
        )

    def _fetch(self, url: str) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept-Language": self.config.accept_language,
                "Accept": "application/json, text/html;q=0.9, */*;q=0.5",
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except URLError:  # pragma: no cover - depends on network availability
            return ""

    def _template_reliability(self, template_id: str) -> float:
        # Map known templates to the source-registry tier baselines
        if "cninfo" in template_id or "sse" in template_id:
            return 0.92
        if "eastmoney" in template_id:
            return 0.62
        if "sina_finance" in template_id:
            return 0.62
        return 0.55


_NUMERIC_FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pe_ttm", r"市盈率.*?(-?\d+\.?\d*)"),
    ("pb", r"市净率.*?(-?\d+\.?\d*)"),
    ("market_cap", r"总市值.*?(-?\d+\.?\d*)"),
    ("free_float_market_cap", r"流通市值.*?(-?\d+\.?\d*)"),
    ("roe", r"净资产收益率.*?(-?\d+\.?\d*)"),
    ("revenue_yoy", r"营业收入(?:同比|增速).*?(-?\d+\.?\d*)"),
    ("net_profit_yoy", r"净利润(?:同比|增速).*?(-?\d+\.?\d*)"),
    ("debt_to_asset", r"资产负债率.*?(-?\d+\.?\d*)"),
    ("gross_margin", r"毛利率.*?(-?\d+\.?\d*)"),
)


def extract_numeric_fields(text: str) -> dict[str, float]:
    """Best-effort regex extraction of common fundamentals from cleaned HTML text."""

    result: dict[str, float] = {}
    for field_name, pattern in _NUMERIC_FIELD_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            result[field_name] = float(match.group(1))
        except (TypeError, ValueError):
            continue
    return result


def _strip_html(raw: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(" ".join(text.split()))


def _plain(symbol: str) -> str:
    return str(symbol).split(".")[0]


def _full(symbol: str) -> str:
    text = str(symbol).upper()
    if "." in text:
        return text
    if text.startswith("6") or text.startswith("9"):
        return f"{text}.SH"
    if text.startswith("4") or text.startswith("8"):
        return f"{text}.BJ"
    return f"{text}.SZ"


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
