# QuantAgent SOTA Upgrade v3

This pass adds the Lopez de Prado AFML, Bailey 2014 PSR/DSR, Ledoit-Wolf 2004, Raffinot 2018 HERC, ICLR'24 iTransformer and conformal-prediction layers on top of v2. It also closes the A-share T+1 / price-limit gap and lays a QMT execution-gateway interface that is intentionally not yet wired to live trading.

## New modules (research and math)

```text
quant_math/triple_barrier.py      AFML 3 first-touch labels + sample uniqueness
quant_math/purged_cv.py           AFML 7 PurgedKFold + 12 CPCV + PBO
quant_math/realized_vol.py        Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang
quant_math/hrp.py                 HRP + HERC (vol or CVaR) clustering portfolios
quant_math/hmm_regime.py          Diagonal-Gaussian HMM with regime mixture multiplier
quant_math/conformal.py           Split conformal + CQR + width-based confidence
quant_math/factor_attribution.py  Gram-Schmidt orthogonalization + capacity curve
quant_math/ashare.py              Board limits, T+1 position state, suspension mask
```

## New modules (agents and execution)

```text
agents/bl_views.py            AgentSignal -> BL P, q, Omega -> posterior alpha
agents/debate.py              DebateRound + DebateOutcome + jsonl audit log
agents/policy_agent.py        Time-decayed policy events to per-symbol AgentSignal
agents/commodity_agent.py     Commodity move x sector-beta map signals
agents/flow_agent.py          Northbound + dragon-tiger flow signals
fundamental/scores.py         Piotroski F / Altman Z / Beneish M
models/itransformer.py        iTransformer + PatchTST alpha heads
backtest/engine.py            Event-driven A-share T+1 backtester
execution/broker_base.py      OrderSide / OrderType / OrderStatus / BrokerBase
execution/order_manager.py    Idempotent target_weights -> shares -> orders
execution/risk_kill_switch.py Daily loss / drawdown / position breach kill
execution/qmt_gateway.py      QMT stub respecting BrokerBase
```

## Existing modules upgraded

```text
training/losses.py            differentiable_spearman_loss + listmle_loss + soft_rank
quant_math/performance.py     PSR / DSR / Newey-West HAC t-stat
quant_math/ic_analysis.py     ic_summary now reports HAC-corrected t-stat
quant_math/covariance.py      ledoit_wolf_covariance optimal shrinkage
quant_math/risk_metrics.py    smooth gauss-decay drawdown_risk_multiplier
quant_math/transaction_cost.py A-share stamp duty + transfer fee + commission floor
quant_math/optimizer.py       fixed sector-grouping bug
agents/arbitration.py         numpy-backed softmax fixes pandas index mismatch
strategy/signal_fusion.py     renamed to score_fusion.py (avoid clash with quant_math)
configs/strategy.default.yaml  switched to A-share, includes board-limit table
```

## Wiring map

```text
prices + fundamentals
  -> features + multi-horizon labels + triple-barrier labels
  -> ic_analysis (HAC) + decay + capacity_curve filters
  -> short / long / event AlphaPrediction
  -> AgentSignal (technical, news, policy, commodity, flow, debate)
  -> precision_weighted_alpha + posterior_alpha_from_agents (BL)
  -> hmm_regime alpha multiplier + conformal confidence
  -> covariance (Ledoit-Wolf or HRP)
  -> optimizer (continuous MV) or hrp_weights / herc_weights
  -> ashare.enforce_tradability + constraints.weights_to_lot_shares
  -> EventDrivenBacktester (paper) or OrderManager + QMTGateway (live)
  -> risk_kill_switch
  -> persistence: agents/debate.persist_debate jsonl audit log
```

## Suggested experiment recipe

```text
1. Build features and triple-barrier labels.
2. PurgedKFold + Embargo to score 5+ short alpha models.
3. Use precision_weighted_alpha to fuse the survivors.
4. Run agent_reliability_weights on rolling IR / evidence_quality stats.
5. Convert AgentSignals to BL views, fold into posterior alpha.
6. Apply hmm_regime alpha multiplier and conformal confidence shrink.
7. Solve mean-variance with cost / turnover / sector / beta.
8. Compare to hrp_weights as a robust baseline.
9. Round to lot shares, enforce limit-up / suspension / T+1.
10. Backtest with EventDrivenBacktester. Report Sharpe / PSR / DSR / PBO / max DD / capacity curve.
11. Only after PSR > 0.95 vs SR=0 baseline and PBO < 0.5: paper trade.
12. After paper trading, wire QMTGateway and start with kill-switch on.
```

## Hard constraints kept

```text
LLM and agents emit AgentSignal, never orders.
Optimizer outputs target_weight, never orders.
QMT touched only inside execution/qmt_gateway.py.
No live trading without out-of-sample PSR / DSR / paper PnL.
```
