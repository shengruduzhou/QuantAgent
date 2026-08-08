# QuantAgent isolated production audit — 2026-08-08

## Audit contract

This review intentionally separates **finding**, **cross-review**, **repair**, and
**acceptance**. The main repair role is not allowed to author evidence for its
own fix and does not count as an independent reviewer.

The machine contract is implemented in
`src/quantagent/governance/isolated_production_audit.py`:

- finding authors are isolated domain auditors, never `main_repair`;
- the author cannot review its own finding;
- P0/P1 findings require at least two independent approvals and one must be
  `testing_expert`;
- any independent rejection makes the finding `contested`;
- `needs_evidence` never counts as approval;
- only `accepted_for_main_fix` may be modified by the main role.

The audit roles are:

1. `backtest_expert`
2. `risk_expert`
3. `factor_strategy_expert`
4. `stock_selection_expert`
5. `testing_expert`
6. `quant_department_user`
7. `quant_expert_tester`
8. `design_testing_expert`
9. `main_repair` — repair only; no independent-review credit

This is an **isolated review protocol inside one QuantAgent engineering
workflow**, not a claim that nine external organisations or statistically
independent foundation models ran the review.

## Evidence baseline

Audit base: main commit `9279160f881c3ef3397e016812d0bb297771e80f`
(PR #50).

The current live-model certificate remains intentionally blocked. No finding in
this review is permitted to turn broker API reachability into model promotion or
real-money permission.

External architecture references were used as design evidence only. No
third-party source code was copied into QuantAgent. The review compared:

- AKQuant production broker, warm-start, parameter optimisation and architecture
  contracts;
- EasyXT QMT/miniQMT connectivity and data/broker integration patterns;
- TradingAgents-Astock A-share evidence-agent/risk patterns;
- QuantsPlaybook as a candidate strategy/factor hypothesis library;
- AKShare as a broad data-provider ecosystem that still requires QuantAgent's
  own integrity/PIT controls.

## Role 1 — backtest expert

### BT-001 — execution timing is not a simulator-level contract

**Severity:** P1  
**Finding:** the trusted `baseline_protocol.py` path applies the delay-1/T+1
semantics before calling the A-share simulator, but the generic simulator API
itself does not declare a named execution timing policy. A caller can therefore
reuse the simulator with a same-bar signal unless it knows the external
contract.

**Cross-review:**

- `testing_expert`: APPROVE finding; reproducible by constructing a same-date
  target-weight frame directly against the simulator.
- `quant_expert_tester`: APPROVE finding; production APIs should make timing
  semantics explicit rather than relying on caller memory.
- `factor_strategy_expert`: NEEDS_EVIDENCE on changing the trusted evaluator;
  changing execution timing can change historical comparability.

**Disposition:** accepted architectural issue, **no semantic migration in this
PR**. The main role must not rewrite the trusted baseline until a side-by-side
reproduction proves unchanged variant-C economics. Recommended next step:
introduce named fill policies (`next_open`, `next_close`, etc.) with a
compatibility mode and golden fixtures.

### BT-002 — impact/capacity remains a calibrated-model gap

**Severity:** P1  
**Finding:** the production A-share backtest has costs, tradeability and volume
constraints, but a strategy/account-calibrated nonlinear market-impact model is
not yet a universal production gate.

**Cross-review:** risk and testing both agree the gap is real; both reject
inventing an arbitrary square-root coefficient without broker/market evidence.

**Disposition:** NEEDS_EVIDENCE. Do not manufacture a capacity number. Calibrate
from actual turnover, participation, spread and broker execution data before a
live certificate can claim capacity.

## Role 2 — risk expert

### RISK-001 — kill state disappeared on process restart

**Severity:** P0  
**Finding:** `risk/kill_switch.py` kept only in-memory flags/reasons. Restarting
the process could therefore clear a severe loss, reconciliation, provider, or
audit-write block. Bare `release()` also cleared every reason.

**Cross-review:**

- `testing_expert`: APPROVE; restart reproduction is deterministic.
- `quant_department_user`: APPROVE; an operator cannot treat process restart as
  risk recovery.
- `quant_expert_tester`: APPROVE; corrupt persisted state must fail closed.

**Main repair:** accepted and implemented.

- optional durable `state_path` preserves research-mode compatibility;
- production state is atomically persisted with flush + fsync + replace;
- state restores on restart;
- corrupt state becomes `kill_switch_state_unreadable`, not green;
- persistent instances cannot use bare `release()`;
- full release requires explicit `release_all(confirm=True)`.

**Acceptance tests:** `tests/risk/test_kill_switch_persistence.py`.

### RISK-002 — no single research-to-broker arming boundary

**Severity:** P0  
**Finding:** before this audit there was no single object that AND-composed
model trust, operational kill state, broker preflight/health and global product
arming policy. Individual modules were correct in isolation but future callers
could compose them inconsistently.

**Cross-review:**

- `testing_expert`: APPROVE the gap and require a non-transmitting readiness
  object first.
- `quant_expert_tester`: APPROVE; query-only broker certification must remain
  possible while the model is blocked.
- `backtest_expert`: APPROVE, provided the new layer does not change backtest
  economics.

**Main repair:** accepted and implemented as
`quantagent.execution.live_session.LiveTradingSession`.

Current semantics are intentionally conservative:

- `query_only_ready = kill clear AND broker preflight AND broker health`;
- `economic_submit_allowed` additionally requires the governed model certificate
  AND an explicitly armed product live policy;
- order authorisation also invokes `RiskGate`;
- the class contains **no broker submit call**;
- current `LIVE_DISABLED` and the current blocked model therefore keep economic
  submission false.

This is a readiness/arming boundary, **not live certification**.

**Acceptance tests:** `tests/execution/test_live_trading_session.py`.

## Role 3 — factor & strategy expert

### FACT-001 — redundant factors could be promoted as independent ACTIVE signals

**Severity:** P1  
**Finding:** lifecycle reports computed `max_correlation_to_existing`, but
`recommend_factor_status` did not use it. A nearly duplicate factor could be
classified ACTIVE and double-count the same economic exposure.

**Cross-review:**

- `testing_expert`: APPROVE with a >0.99 correlation reproduction.
- `backtest_expert`: APPROVE redundancy gate; REJECT a blanket capacity number.
- `quant_expert_tester`: APPROVE fail-closed missing drift evidence.

**Main repair:** accepted and implemented.

- configurable `max_existing_correlation_for_active` (default 0.90);
- highly redundant factor remains `watch` rather than ACTIVE;
- insufficient drift history now returns NaN and cannot masquerade as zero
  drift;
- capacity remains recorded and can become an explicit production threshold via
  `min_capacity_rmb_for_active`, without breaking research tables that lack
  authoritative amount/capacity data.

**Acceptance tests:** `tests/factors/test_lifecycle_promotion_guards.py`.

### FACT-002 — pooled correlation is not a full factor-crowding model

**Severity:** P2  
**Finding:** current correlation is useful as a redundancy guard but is not a
full rolling cross-sectional exposure/crowding decomposition.

**Disposition:** NEEDS_EVIDENCE. Next-generation work should separate raw
correlation, neutralised residual correlation, rolling cross-sectional
correlation, style/industry exposure and portfolio marginal contribution. Do not
turn one correlation scalar into a false crowding certificate.

## Role 4 — stock-selection / fundamental expert

### SEL-001 — current sector snapshot could contaminate historical fundamental ranking

**Severity:** P0/P1  
**Finding:** the ranker documentation said sector rows without `available_at`
were not historical PIT, but the implementation still merged a
`coverage_status=current_snapshot` frame. Those rows then looked like
`sector_level_1` and could increase `real_sector_share`, potentially opening the
fundamental overlay gate.

**Cross-review:**

- `testing_expert`: APPROVE; direct synthetic reproduction.
- `backtest_expert`: APPROVE; historical industry membership is part of the PIT
  state.
- `quant_expert_tester`: APPROVE; missing PIT history must downgrade rather than
  silently use today's classification.

**Main repair:** accepted and implemented through a fail-closed public facade
`data/fundamental/safe_ranker.py`.

- sector maps without valid `available_at` are excluded from historical sector
  ranking;
- the ranker falls back to board-proxy buckets;
- a current snapshot therefore cannot inflate `real_sector_share` or open the
  sector-coverage overlay gate;
- a genuinely dated sector map remains eligible.

**Acceptance tests:** `tests/data/test_fundamental_ranker_pit_sector_guard.py`.

### SEL-002 — missing fundamental dimensions are renormalised

**Severity:** P1  
**Finding:** if only one dimension is present, the composite renormalises to that
dimension. This is mathematically explicit but can make a low-completeness name
look comparable to a complete one.

**Cross-review:** mixed. The stock-selection role recommends a minimum
completeness gate; the factor role notes that coverage varies by industry and
history.

**Disposition:** CONTESTED. No production change in this PR. A completeness
threshold must be calibrated by PIT coverage and sector, not chosen to make a
backtest look cleaner.

## Role 5 — testing expert

### TEST-001 — hosted CI is not a real MiniQMT certification

**Severity:** P0 evidence gap  
**Finding:** Linux CI can verify portable QMT contracts but cannot certify the
real Windows/MiniQMT process/account, broker restart, duplicate/out-of-order
callbacks, partial fills, or statement-level cash/fee reconciliation.

**Disposition:** NEEDS_EVIDENCE. Keep real-money arming disabled. Required
artifact remains a controlled-host, query-only soak followed by fault/recovery
and reconciliation drills.

### TEST-002 — safety static allowlist remains valuable

The existing safety test that allows broker order calls only in the audited QMT
adapter remains a valid invariant and must stay in CI. The new
`LiveTradingSession` intentionally does not add another broker call site.

## Role 6 — AI quant department employee / system user

### OPS-001 — restart should not change the operator's risk state

**Severity:** P0  
Validated `RISK-001`; fixed through persistent KillSwitch state.

### OPS-002 — operator needs separate readiness dimensions

**Severity:** P1  
A single green "ready" is unsafe. Operator surfaces should separately show:

- model trust: BLOCKED / PASS;
- broker: unavailable / query-only-ready / degraded;
- operational risk: CLEAR / KILLED + reasons;
- product policy: LIVE_DISABLED / explicitly armed;
- reconciliation: clean / divergent;
- certificate lineage/hash status.

**Disposition:** accepted UX issue; API/UI implementation remains a follow-up so
this PR does not conflate a new readiness display with actual arming.

## Role 7 — senior AI quant expert & tester

### EXP-001 — readiness certificate text contradicted the code

**Severity:** P1  
After PR #50, `readiness_tiers.py` still said the system had "no live order
path" even though an audited low-level QMT adapter existed.

**Cross-review:** testing and design both APPROVE; audit text must describe code
truthfully.

**Main repair:** accepted and implemented.

`live_trading_certificate()` now states:

- LIVE_TRADING_READY is not granted;
- controlled broker adapter exists;
- adapter is not armed;
- a fail-closed arming/readiness boundary exists;
- no product LIVE mode/web job/agent route is armed.

The distinction is intentional: **adapter present != live certified**.

### EXP-002 — model-trust certificate is still hand-editable evidence

**Severity:** P0 before live arming  
The verifier checks required fields, but the final production certificate still
needs a governed issuer binding git commit, model hash, dataset snapshot/hash,
config hash, benchmark/evaluation artifacts and holdout-read audit. A human-edited
JSON must never be sufficient to arm economic submission.

**Disposition:** NEEDS_EVIDENCE / remains open in production-readiness work.
Current blocked certificate makes this non-exploitable today.

## Role 8 — design & testing expert

### UI-001 — Governance surface does not show the new production safety dimensions

**Severity:** P1  
Current Governance UI is strong on shadow/S4/U0/lineage but does not visibly
separate model trust, broker query readiness, persistent kill state and product
arming. Some service payloads also still contain legacy
`liveTradingControls="absent"` wording.

**Disposition:** accepted UI truthfulness issue, not fixed by painting the panel
green. Required design is a four-state production-readiness strip with explicit
BLOCKED/QUERY-ONLY/KILLED/DISABLED semantics and no aggregate "ready" badge.

## Main-role repair ledger

| Finding | Status | Main-role action |
|---|---|---|
| RISK-001 | accepted | persistent fail-closed KillSwitch |
| RISK-002 | accepted | non-transmitting LiveTradingSession readiness/authorization boundary |
| FACT-001 | accepted | redundancy + missing-drift promotion guards; optional capacity threshold |
| SEL-001 | accepted | PIT-safe fundamental sector facade + overlay regression |
| EXP-001 | accepted | truthful live readiness certificate semantics |
| BT-001 | accepted issue / migration deferred | preserve trusted evaluator until golden comparison exists |
| BT-002 | needs evidence | calibrate impact/capacity; do not fabricate coefficient |
| FACT-002 | needs evidence | richer rolling/neutralised crowding research |
| SEL-002 | contested | no arbitrary completeness cutoff |
| TEST-001 | needs real host evidence | keep LIVE_DISABLED |
| EXP-002 | needs governed issuer | current model remains BLOCKED |
| OPS-002/UI-001 | accepted follow-up | separate operator readiness dimensions |

## Non-negotiable production conclusion

This audit does **not** certify the current strategy/model for live capital.
Software readiness, broker query readiness, research validity and model
promotion are independent evidence domains. The current live-model certificate
remains blocked and the product live policy remains disabled. Real-money arming
requires the remaining controlled-host, reconciliation, certificate-lineage,
fresh-OOS and capacity evidence to close without weakening any existing gate.
