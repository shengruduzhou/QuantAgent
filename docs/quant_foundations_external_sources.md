# QuantAgent External Quant Foundations / 外部基础设计与取数来源

> Status: pinned design/research references.  Agents (including Claude/Codex) should
> open the URLs below on the web before changing data contracts, factor fusion,
> model comparison, or the A-share market workbench.  Do not infer current API
> fields from this note when the official contract can be checked directly.

## 1. Fuyao / HiThink — primary official A-share data reference

Read in this order:

1. https://fuyao.aicubes.cn/llms.txt
2. https://fuyao.aicubes.cn/llms-full.txt
3. https://fuyao.aicubes.cn/docs/
4. https://fuyao.aicubes.cn/docs/api-reference/overview/
5. https://fuyao.aicubes.cn/
6. https://github.com/HiThink-Tech/Financial-API
7. https://github.com/HiThink-Tech/Financial-API/tree/main/docs/api
8. https://github.com/HiThink-Tech/Financial-API/tree/main/examples/inspirations
9. https://github.com/HiThink-Tech/Financial-API/tree/main/examples/inspirations/01-stock-overview

Credential contract:

- Environment variable: `HITHINK_FINANCE_API_KEY`.
- REST header: `X-api-key`.
- Never commit, print, persist in Runtime artifacts, or pass the key in prompts.
- Existing TickFlow / Google keys are independent credentials and do not replace
  the HiThink/Fuyao key.

Data responsibility in QuantAgent:

| Capability | Preferred source | Rule |
| --- | --- | --- |
| A-share daily snapshot / per-symbol daily history | TickFlow primary, Fuyao official fallback | raw/unadjusted is canonical market-panel input; adjusted view is separate |
| Full-market ~10y daily history | Fuyao Market Dump | do not loop over ~5000 symbols |
| Full-market recent daily increment | Fuyao `daily-k-10d` dump | UPSERT by `(thscode, date_ms)` |
| Full-market corporate-action events | Fuyao adjustment-factor dump | apply events as-of; do not fabricate daily factors |
| Financial statements | Fuyao / existing PIT providers | `available_at = report_date_ms`, never `period_end_ms` |
| Valuation / index / sector / fund / limit-up / hot-list / abnormal-trade data | Fuyao where public capability exists | capability probe first; retain provenance |
| Minute K / tick / depth | TickFlow | Fuyao public capability currently does not include these |
| News / announcement full text / research reports | existing evidence/news pipeline | Fuyao public capability currently does not include full text |
| Qlib research dataset / handler / workflow baseline | Qlib | fit learnable processors only on the training interval |

Market Dump endpoints currently pinned by the upstream API contract:

- `GET /api/dump/market-dumps/daily-k/download-url`
- `GET /api/dump/market-dumps/daily-k-10d/download-url`
- `GET /api/dump/market-dumps/adjustment-factors/download-url`

The presigned URL is short-lived capability data: obtain it immediately before
streaming the file and never persist or log it.

## 2. Qlib — AI quant workflow / processor reference

- https://github.com/microsoft/qlib
- https://qlib.org.cn/en/latest/
- https://qlib.org.cn/en/latest/reference/api.html

QuantAgent rules derived from the Qlib workflow:

- Distinguish raw data, inference processors, and learn-only processors.
- Any processor with learned parameters (normalisation, imputation statistics,
  feature selection, PCA, encoding, calibration, etc.) is fit on the training
  interval only, then applied unchanged to validation/test/live rows.
- Dataset segment boundaries do not make a globally pre-fit transformer safe.
- Store fitted processor state with the model artifact so inference uses the
  identical transformation.

## 3. Asset-pricing and factor-model foundations

### Classical pricing / cross-sectional factors

- Sharpe (1964), *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk*  
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1964.tb02865.x
- Black & Scholes (1973), *The Pricing of Options and Corporate Liabilities*  
  https://www.cs.princeton.edu/courses/archive/fall09/cos323/papers/black_scholes73.pdf
- Fama & French (1992), *The Cross-Section of Expected Stock Returns*  
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04398.x
- Carhart (1997), *On Persistence in Mutual Fund Performance*  
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb03808.x

### Formulaic alpha / ML asset pricing / platform workflow

- Kakushadze (2016), *101 Formulaic Alphas*  
  https://arxiv.org/abs/1601.00991  
  https://arxiv.org/pdf/1601.00991
- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*  
  https://academic.oup.com/rfs/article/33/5/2223/5758276
- Yang et al. (2020), *Qlib: An AI-oriented Quantitative Investment Platform*  
  https://arxiv.org/abs/2009.11189  
  https://arxiv.org/pdf/2009.11189

## 4. Nonlinear multi-factor model contract

Do **not** call every non-linear optimisation a nonlinear factor model.
QuantAgent keeps these model classes distinct:

1. `linear_additive`: `sum(w_j * x_j)`.
2. `rank_weighted_additive`: `sum(w_j * rank_t(x_j))`; cross-sectional rank is
   nonlinear in each raw value but the factor blend remains additive.
3. `factor_nonlinear_transform`: `sum(w_j * f_j(x_j))`; splines/buckets remain
   additive across factors.
4. `factor_interaction`: explicit `x_i * x_j` or equivalent conditional terms.
5. `regime_interaction`: `x_j * market_state_t`.
6. `nonlinear_learner`: tree / boosted tree / neural learner that can represent
   interactions directly.
7. `ensemble`: combines fitted models; ensembling additive models does not by
   itself create cross-factor interaction.

Research protocol:

- Baseline and nonlinear arms must share identical PIT rows, folds, labels,
  transaction costs, tradability constraints and evaluation windows.
- Pair/interactions are selected **inside each training fold** only.
- Preprocessing statistics are fit **inside each training fold** only.
- Hyperparameters/interaction sets are selected only on selection folds.
- Final holdout is read once and never used to choose an arm.
- Use purged walk-forward with effective gap at least the label horizon and the
  configured embargo.
- Factor lists are fail-closed against `forward_return*`, `future_return*`,
  `label*`, `target*` and equivalent label columns.
- Financial features become observable at disclosure time (`report_date_ms`).
- Report OOS RankIC/ICIR and economic metrics after costs; do not promote an arm
  because of in-sample fit.
- Keep interaction gains incremental versus the additive baseline. Complexity
  must pay for itself out of sample.

Relevant repository implementation:

- `src/quantagent/models/interactions.py`
- `src/quantagent/research/model_comparison.py`
- `src/quantagent/training/splitters.py`
- `src/quantagent/fusion/search.py`
- `src/quantagent/cli/fusion.py`

## 5. A-share market workbench reference

Primary information-architecture reference:

- https://github.com/HiThink-Tech/Financial-API/tree/main/examples/inspirations/01-stock-overview

Use the structure, not another company's branding.  QuantAgent keeps ATLAS/VNext
semantic tokens and A-share red-up/green-down semantics.

Required workbench hierarchy:

- identity + code + data timestamp + source + adjustment basis;
- latest price / change and a compact metric strip;
- dominant candlestick chart with MA20/MA60/MA120;
- linked volume;
- interval return / rolling drawdown / average turnover metrics;
- crosshair, zoom, pan and retained window state;
- evidence / decision / risk inspector beside the dominant chart where relevant;
- explicit unavailable states; no fabricated data.

The research/replay product may add QuantAgent-only layers (trade markers, factor
attribution, model score, decision chain and T+1 analysis), but these should not
obscure the basic market-reading hierarchy above.

## 6. PIT data-flow invariant

For a decision timestamp `t`, every model input row must satisfy:

```text
available_at <= t
```

and every learned transformation/model parameter used at `t` must have been fit
only from observations whose training information set is strictly before the
corresponding validation/test realization after purge + embargo.

Minimum provenance on persisted data rows/artifacts:

```text
source
source_endpoint
retrieved_at
available_at
quality_status
```

A provider switch inside one symbol history is a `SourceBoundary`; do not splice
vendors silently.  Missing evidence is `BLOCKED_BY_DATA`, not a synthetic fill.
