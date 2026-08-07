# Fuyao / HiThink Financial API integration

## Source of truth

Keep these URLs available to coding agents. Read `llms-full.txt` before modifying
provider schemas or adding endpoints.

- https://github.com/HiThink-Tech/Financial-API
- https://fuyao.aicubes.cn/
- https://fuyao.aicubes.cn/llms.txt
- https://fuyao.aicubes.cn/llms-full.txt
- https://fuyao.aicubes.cn/docs/
- https://fuyao.aicubes.cn/docs/api-reference/overview/

## Credential

```bash
HITHINK_FINANCE_API_KEY=<secret>
```

REST authentication uses `X-api-key`. Never commit, log, echo, persist, or put
the key in prompts. QuantAgent reads the repository `.env` or process environment
through the existing `quantagent.data.ashare.env` loader.

## QuantAgent flow

1. **Bulk history / full market**: use Market Dumps first.
   - `/api/dump/market-dumps/daily-k/download-url`
   - `/api/dump/market-dumps/daily-k-10d/download-url`
   - `/api/dump/market-dumps/adjustment-factors/download-url`
2. **Targeted / current / richer fields**: use REST through
   `quantagent.data.ashare.fuyao.FuyaoSource` / `FuyaoClient`.
3. **Canonical research data** still flows through `quantagent.data.ashare` U0.
   Fuyao does not bypass provenance, PIT `available_at`, raw-price, unit, coverage,
   or readiness gates.
4. Public Tencent/Sina/Eastmoney sources remain fallback/validation sources.
5. Full-market dump files land under `runtime/data/fuyao/` by default and can be
   consumed with `FuyaoDumpSource` without re-downloading per security.

Commands:

```bash
# Capability/entitlement smoke test
AI_quant_venv/bin/python3 scripts/fuyao_capability_probe.py --allow-network

# Preferred initial bulk sync: full daily K + 10-day delta + all adjustment events
AI_quant_venv/bin/python3 scripts/fuyao_market_dump_sync.py --allow-network --kind all

# Targeted canonical U0 acquisition with live REST fallback
AI_quant_venv/bin/python3 scripts/u0_acquire_bars.py --allow-network \
  --providers fuyao,tencent --symbols 600519.SH
```

## Coverage policy

Fuyao is used for every documented dataset family that materially improves the
research platform: daily prices, snapshot quotes, valuations, financial
statements/indicators, trading calendar, ticker metadata, adjustment/corporate
action events, index/sector data, limit-up/hot-stock/anomaly/dragon-tiger style
special datasets, and full-market Parquet exports.

Do **not** invent unsupported capabilities. If the current official capability
map does not expose minute bars/tick/order-by-order Level-2, macro series, or
full news/announcement text, keep the existing provider/source for that family.
A failed entitlement is explicit and fail-closed.

## Market workbench reference

The `/stock-replay` navigation entry is the **行情工作台 / Market Workbench**.
Follow the Financial-API examples as an information-architecture reference, not
as a pixel-for-pixel clone. Market-level views should prioritize:

- market breadth / strong-trend population;
- liquidity buckets and representative securities;
- index/sector breadth and constituent drill-down;
- daily K, price/volume context, and freshness/source identity;
- clear sample window, thresholds, and calculation rules;
- no fabricated recommendation when persisted evidence is absent.

Reference examples:

- https://github.com/HiThink-Tech/Financial-API/tree/main/examples/inspirations/06-marketdb-research
- https://github.com/HiThink-Tech/Financial-API/tree/main/examples/inspirations/13-price-volume-breakout
- https://github.com/HiThink-Tech/Financial-API/tree/main/examples/inspirations/15-short-term-reversal

## Endpoint-path note

For Market Dumps, the current Financial-API reference implementation and API
endpoint document use the `/api/dump/market-dumps/...` prefix. QuantAgent follows
that executable reference. If `llms-full.txt` and the reference repository ever
disagree, verify against the current official implementation before changing
production paths.
