# QuantAgent v3 Factor-Centric Design

## Architecture

QuantAgent v3 makes factors the first-class research object. Raw data flows into factor registries, preprocessing, evaluation, composite scoring, agent evidence, sleeve allocation, target weights, and an A-share-aware backtest. Agents produce structured evidence, confidence, risk penalties, and target-weight adjustments only.

## Factor-First Principle

Models and agents never place orders. They emit `AgentSignal` or `TargetWeight`. The execution gateway remains a stub boundary. Every factor is point-in-time safe and must be evaluated with IC, Rank IC, ICIR, group returns, turnover, decay, capacity, neutralization, and transaction-cost-adjusted performance.

## A-Share Constraints

The pipeline respects T+1 sale availability, 100-share lots, limit-up and limit-down blocks, ST handling, suspensions, stamp duty, transfer fee, slippage, and liquidity participation limits. These constraints live in target-weight adaptation, risk gates, position state, and backtesting.

## Sleeve Allocation

Capital is split into `long_fundamental`, `short_event`, `sector_rotation`, `hedge`, and `cash_buffer`. Cash rises during drawdown and high volatility. Hedge rises with beta, drawdown, and volatility. Short-event exposure shrinks when factor ICIR weakens. Sector exposure requires breadth and flow confirmation.

## Sector Rotation

Sector factors include return strength, relative strength, volume expansion, breadth, limit-up count and ratio, turnover share, money-flow share, and combined rotation score. The sector rotation agent classifies market state as `main_trend`, `rotation`, `diffusion`, `exhaustion`, `defensive`, or `crash`.

## Fund Flow

Fund flow sources are normalized into a common schema for northbound holdings, northbound net buy, dragon-tiger list, main money flow, margin financing, ETF flow, block trades, institutional research visits, and public fund holdings. Features include z-score, acceleration, persistence, reversal, concentration, institution-retail imbalance, large-order pressure, margin balance change, and ETF sector flow.

## Stop-Loss State Machine

Position states cover new, normal, profit protection, pullback hold, breakeven exit, soft stop, hard stop, time stop, event stop, liquidity exit, and exited. Exits can be blocked by T+1 or limit-down conditions and are reported separately.

## Target Price Model

The fundamental stack standardizes statements, computes DuPont/ROIC, forensic accounting risk, DCF target ranges, reverse DCF implied growth, relative valuation targets, margin of safety, and bear/base/bull target-price bands.

## Backtest Validation Protocol

Use synthetic or point-in-time historical panels. Build factors, preprocess, evaluate IC and group returns, combine accepted factors, translate signals to target weights, apply sleeve allocation and lot/liquidity constraints, and run the A-share T+1 backtester. Review sleeve diagnostics, stop-loss diagnostics, blocked exits, and limit impact.

## How To Run

```bash
quantagent-build-factors prices.csv alpha101_factors.csv --library alpha101
quantagent-evaluate-factors prices.csv alpha101_factors.csv factor_eval.csv --horizon-days 5
quantagent-build-flow-features configs/flow.yaml flow_features.csv
quantagent-build-sector-rotation prices_with_sector.csv sector_rotation.csv
quantagent-run-v3-backtest weights.csv prices.csv backtest_diagnostics.csv
quantagent-generate-factor-report factor_panel.csv factor_report.csv --return-column forward_return_5d
quantagent-generate-valuation-report configs/fundamental/valuation.default.yaml valuation_report.csv
```

