# Strategy Operations Workbench

## Scope

This slice turns QuantAgent's existing real-data training, target-weight,
A-share backtest, risk and paper-report code into one governed Web workflow.
It does **not** claim access to any fund's private architecture and does not
enable live orders.

The public design references are:

- vn.py's explicit application boundaries for DataManager, DataRecorder,
  RiskManager and WebTrader (REST/WebSocket):
  <https://github.com/vnpy/vnpy/blob/master/README_ENG.md>
- Two Sigma's public guidance on versioning, testing and reproducing data as
  code: <https://www.twosigma.com/articles/treating-data-as-code-at-two-sigma/>
- Citadel's public description of real-time market data, interactive
  visualization, simulation and central risk platforms:
  <https://www.citadel.com/careers/engineering/>
- Optiver's public separation of pricing, risk and execution:
  <https://www.optiver.com/insights/technology-blog/engineering-the-three-pillars-of-trading-pricing-risk-and-execution/>
- MLflow's public model-registry, experiment-tracking and dataset-lineage
  contracts: <https://mlflow.org/docs/latest/ml/model-registry/>

## Canonical flow

```text
Strategy contract
  -> validate real paths and risk limits
  -> persist research-only manifest
  -> Human Gate
  -> allowlisted run-full-real-training-v7
  -> dataset / factors / rolling OOS training
  -> predictions / target weights
  -> A-share execution simulation
  -> risk and paper report
  -> RuntimeIndexer / model / backtest / evidence workstations
```

The CLI remains the execution source of truth. The UI does not duplicate
training or backtest logic and never derives performance from form inputs.

## Strategy contract

`quantagent.strategy.v1` persists:

- hypothesis and explicit invalidation criteria;
- point-in-time market, labels, sector and reviewed-factor paths;
- model, horizons, rolling/expanding split and fold count;
- portfolio objective, Top-K, concentration and turnover limits;
- declared research priorities for excess return, annual return and drawdown
  control;
- acceptance limits for drawdown, turnover and Sharpe;
- Human Gate state, content hash, validation result and research-only trust
  class.

The preference weights are configuration only. They are never rendered as
achieved returns.

## Multi-agent boundary

The Decision Council exposes structured roles for data quality, factor
research, model validation, portfolio, backtest, risk, challenger and human
approval. Data/model/risk/challenger/human roles can veto. Agent output is
advice/evidence only. It cannot create an order or bypass a Risk Gate.

## Realtime contract

The launcher returns a persisted job ID and opens
`GET /api/jobs/{job_id}/stream`. The SSE stream emits bounded log lines and
authoritative job snapshots. The browser closes a terminal stream and can
recover state from `/api/jobs`; no training state lives only in React.
`POST /api/strategies/launch` performs server-side validation, persists the
exact approved strategy version, and submits the allowlisted job atomically.
There is no direct strategy-pipeline submit route that can bypass Human Gate.

## Connector vault

The Web connector accepts only declared fields for TickFlow, TuShare, OpenAI
research and Alpaca market-data/paper sessions.

- Secrets are held in API-process memory or supplied by server environment.
- Secrets are never returned, stored in `jobs.json`, placed in argv, written to
  Runtime or kept in browser storage.
- The JobManager injects only variables associated with the command's declared
  provider.
- No arbitrary shell endpoint exists.
- Server restart clears session credentials.

This follows OWASP's guidance to minimize secret exposure and lifetime:
<https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html>.

## Safety invariants

- research/paper only; no live-order command;
- input paths confined to project/runtime and must exist;
- output paths confined to Runtime;
- no `mark_production_ready` parameter on the Web strategy command;
- strategy launch requires both real-input validation and Human Gate;
- declared maximum drawdown and minimum Sharpe are passed into the persisted
  paper/backtest acceptance gate rather than remaining display-only controls;
- missing evidence remains blocked/unavailable instead of being fabricated.
