# Module One — independent role reports

Rounds 1–2, 2026-08-03. Commit: working tree (uncommitted).
Round 2 regression baseline: **2249 passed, 4 skipped, 0 failed**
(`pytest -q -p no:randomly`, 3:16, no network).

## Round 2 summary (M1-03c, M1-10)

| Item | Before | After |
|---|---|---|
| Parallel-model blockers | 10 | **5** (`docs/architecture/parallel_model_audit.json`) |
| `order_manager.py` canonical | no | **yes** — writes to the ledger before `broker.submit()` |
| Cross-process idempotency | unproven | **proven** — threads, 2 OS processes, 2 recovery workers |
| Ledger verified requirements | 9 | **12** of 50 |

### DEF-006 (P0, found in Round 2, in my own Round-1 code)

`IdempotencyStore` loaded the claim file **once at construction** and never
re-read it. Two recovery workers started together therefore each held an empty
view and **both won the same key** — two economic orders. The Round-1 two-process
test passed only because the processes ran *sequentially*.

Fixed with an offset-tracking incremental `_load()` plus an `fcntl.flock`
exclusive lock around the claim decision, so the view is refreshed under the lock
before granting. Proven: two independent stores on one file now yield
`granted=True` then `granted=False`.

**This is the defect the phase brief was aimed at, and unit tests did not find
it — only driving real threads and real processes did.**

Three roles only, per scope. CIO, strategy-performance, no-AI user and the full
eleven-role committee are **not** run — Modules Two and Three do not exist, so
those roles have no valid system to inspect.

---

## 1. OMS / EMS Execution Engineer — implementation and primary execution testing

### Implementation paths changed

| Path | Change |
|---|---|
| `src/quantagent/domain/lineage.py` | 14-field content-derived chain of custody |
| `src/quantagent/domain/idempotency.py` | Durable claim-once store, fsynced |
| `src/quantagent/domain/orders.py` | Canonical entities + enforced state machine + `OrderBook` |
| `src/quantagent/domain/accounting.py` | `AccountState` folded purely from events |
| `src/quantagent/domain/ledger.py` | Hash-chained append-only JSONL + `replay()` |
| `src/quantagent/backtest/engine.py` | Emits canonical entities; `trades` is now a projection |

### Tests personally executed

- State transitions: `tests/domain/test_order_entities.py` (17)
- Idempotency: `tests/domain/test_order_idempotency.py` (15)
- Invariants + fuzzing: `tests/domain/test_accounting_invariants.py` (34)
- Reconstruction: `tests/domain/test_ledger_reconstruction.py` (12)
- Golden A-share scenarios re-run after migration: `tests/test_golden_backtest_scenarios.py` (12)

### State-transition coverage

Verified legal: `CREATED→PENDING_RISK→APPROVED→SUBMITTED→ACCEPTED→PARTIALLY_FILLED→FILLED`,
`ACCEPTED/PARTIALLY_FILLED→CANCEL_REQUESTED→CANCELLED`, and a **fill racing an
in-flight cancel** (venues legitimately trade before processing a cancel).

Verified refused: fill after terminal cancel, any event on a terminal order,
cumulative fill exceeding order quantity. A refused transition raises
`IllegalTransition` **and records no event** — asserted directly by
`test_an_illegal_transition_does_not_record_an_event`.

### Defects found and fixed

- **DEF-004 (P0, T+1 violation in the fast engine).** Settlement was indexed off
  the *signal* date and clamped with `min(i + 2, len(dates) - 1)`, so on the final
  session a buy filled that day became immediately sellable. Measured: on
  2026-01-12 the engine bought 700 and sold 14,800 — including the 700 filled
  that session. Fixed: settlement must fall strictly after the fill session; with
  no later session the shares never become sellable.
- **DEF-005 (P2, misleading error).** A fill against a terminal order reported an
  arithmetic overflow rather than a lifecycle violation, pointing an operator at
  the quantity instead of the dead order. Legality is now checked first.

### Unresolved

- `execution/order_manager.py` and `paper/orders.py` remain parallel models (M1-03c).
- Idempotency is proven at store level, **not** through API/worker paths or
  concurrent workers/processes (M1-10).

### Verdict

**Approve for the fast-backtest engine only.** Maximum stage `backtest_only`.
Cannot approve Module One overall while two parallel order models remain.

---

## 2. Backtest Audit Engineer — independent verification

Did not rely on the OMS/EMS report. Inspected code and artifacts directly.

### Scope inspected

`backtest/engine.py` diff, `domain/*.py`, generated ledger files, and the
reconstruction results.

### Checks performed and results

| Check | Method | Result |
|---|---|---|
| Fast engine uses canonical entities | Read `run()`; confirmed `_open_canonical_order` on every order path | **Pass** |
| `trades` is not an independent record | `_trade_row` derives every field from a `Fill` | **Pass** |
| Ledger is written for every order | `len(order_book.fills()) == len(trades)` asserted | **Pass** |
| Economic state derives from the ledger | Discarded memory, replayed from disk | **Pass** — NAV 1003051.5399 both sides |
| No direct mutation bypass | `Order`/`Fill`/`AccountState` are frozen slots dataclasses; `apply` returns new snapshots | **Pass** |
| Replay refuses bad evidence | Tampered record → `LedgerCorruption` before any number is produced | **Pass** |

### Findings against the implementation

- **AUDIT-01 (accepted risk).** `_open_canonical_order` records a
  `RiskDecision` with `rule="fast_engine_pretrade"` because the fast engine
  applies tradability/lot/band constraints *upstream* of the order loop. The
  decision is therefore a record that constraints passed, not an independent
  evaluation. Acceptable for `backtest_only`; **must not** be carried into paper
  or live, where risk has to be a real gate.
- **AUDIT-02 (raised, then fixed and re-verified).** Rejected orders reached the
  canonical ledger only for `insufficient_cash` and `no_sellable_inventory`.
  Venue/tradability rejects (`limit_up_no_buy`, `limit_down_no_sell`,
  `suspended`, `zero_fill`) went only to `reject_log`, so the ledger was not the
  complete record of refusals.
  *Remediation:* `_record_rejected_intent` now opens a canonical
  Signal → Intent → Order for the refused quantity and applies `RISK_REJECTED`
  with a `RiskDecision` carrying the rule.
  *Re-verified independently:* a sealed limit-up run replays from disk as
  `('600000.SH', 'limit_up_no_buy', 'BUY', 9091)` with
  `RiskDecision(rule='limit_up_no_buy', approved=False)`.
  **Residual:** `missing_price` still writes only to `reject_log` — there is no
  price with which to form an intent, so this is recorded as a known limitation
  rather than closed.
- **AUDIT-03 (open).** `replay_account` falls back to `fill.filled_at[:10]` when
  no trade date is supplied. The engine always supplies one, but the fallback is
  silent and a caller that omits it collapses sessions — this bit the test
  harness during development.

### Verdict

**Do not approve Module One.** The reconstruction proof is genuine, I could not
break the accounting, and AUDIT-02 was fixed and independently re-verified after
I raised it. I still withhold approval because the gate requires *all* engines
on the canonical set and two parallel models remain (M1-03c) — the fast engine
alone does not make the entity set canonical, it makes it adopted once.
Maximum stage `backtest_only`.

---

## 3. SRE / Chaos Engineer — recovery and duplicate delivery

### Steady state and invariants

1. The hash chain verifies from genesis to head.
2. One `order_intent_id` maps to at most one economic order.
3. A duplicate technical message changes no position, cash, fill quantity, fee or NAV.
4. Replay of a persisted ledger reproduces byte-identical account state.

### Faults injected

| Fault | Expected | Actual | Evidence |
|---|---|---|---|
| Process restart mid-claim | Claim survives; no resubmit | Claim survives | `test_worker_killed_mid_submit_does_not_resubmit_on_restart` |
| Torn trailing write | Earlier records intact and verifying | Intact; `had_torn_tail` true | `test_a_torn_trailing_write_keeps_every_earlier_record` |
| Altered middle record | Verification fails at that index | Failed at index 2 | `test_an_altered_record_breaks_verification` |
| Corrupt chain then replay | Refuse, produce no numbers | `LedgerCorruption` raised | `test_replay_refuses_a_corrupt_chain_rather_than_producing_numbers` |
| Duplicate broker callback | Applied once | Applied once | `test_duplicate_broker_callback_applies_once` |
| Socket reconnect replay | All duplicates suppressed | 5/5 suppressed | `test_socket_reconnect_replaying_its_buffer` |
| Event-log replay | No new orders | 0 of 4 granted | `test_replaying_a_historical_event_log_creates_no_new_orders` |
| Repeated reconstruction | Hash-stable | 1 distinct hash of 3 runs | `test_replay_is_stable_across_repeated_reconstructions` |

### Not tested this round

API restart / SIGKILL against a live run, worker SIGKILL before vs after
persistence, database lock and disconnect, concurrent recovery workers,
filesystem write failure, out-of-order event delivery at the engine level.
These need the API/worker integration that M1-10 also depends on.

### Verdict

**Approve persistence and replay behaviour. Do not approve Module One.** The
durability primitives hold under every fault I could inject at this layer, but
the faults that matter most for duplicate orders — concurrent workers and a
killed API mid-submit — are untested because the integration path does not exist
yet. Maximum stage `backtest_only`.

---

## Module One completion gate (after Round 2)

| Gate condition | Status |
|---|---|
| All real engines consume canonical entities | **Fail** — fast engine + order manager; paper not migrated |
| Legacy parallel state removed or isolated | **Fail** — 5 blockers remain |
| Durable idempotency in the integrated system | **Pass** for OrderManager; **Fail** for API/worker routes (M1-13) |
| Concurrent thread and process tests pass | **Pass** |
| All three paths replay exactly | **Fail** — fast engine proven; paper path not exercised |
| Ledger replay reconstructs exact state | **Pass** |
| Accounting invariants pass | **Pass** |
| T+1 lot invariants pass | **Pass** |
| Controlled fault tests pass | **Partial** — no API/worker SIGKILL, DB or fs-failure injection |
| Full regression suite passes | **Pass** — 2249 passed, 0 failed |
| OMS/EMS approves | Conditional |
| Backtest Audit approves | **No** — 5 blockers |
| SRE approves | **No** — process-kill and fs-failure faults untested |
| No P0/P1 remains | **Pass** — DEF-004, DEF-006 fixed and regression-locked |

**Module One is NOT verified.** Remain on Module One.

Next, in dependency order:

1. **M1-03d** — migrate `paper/orders.py` (own state machine) and
   `microstructure_simulator`; make `OrderManager.history` a projection derived
   from the ledger rather than parallel state.
2. **M1-13** — drive the FastAPI route and job-worker submission paths for
   duplicate delivery.
3. **M1-11** — inject API/worker SIGKILL against a live run, DB lock/disconnect,
   fsync failure, read-only ledger path, two concurrent recovery workers at the
   process level.

**Maximum permitted stage: `backtest_only`.**

---

## Round 7 summary (M1-20, M1-21, M1-22, M1-23; M1-13 re-scoped)

| | After Round 6 | After Round 7 |
|---|---|---|
| Composite Fast/Paper/OMS replay | not run | **run** — `unexplained_economic_differences = 0` |
| OMS-to-paper production path | did not exist | **exists** — `PaperBrokerAdapter`, one shared chain |
| Duplicate execution id | **double-counted money** | absorbed, zero economic delta |
| `realised + unrealised == NAV − initial cash` | **failed by 10.3521** | 0 to float precision on all 3 paths |
| Cash reservation | unmeasurable (unfed field) | derived from the order book |
| Ledger verified requirements | 21 of 58 | **25 of 62** |
| Full regression suite | 2152 passed | 2317 passed, 4 skipped, 0 failed |

Evidence: `docs/architecture/module1_composite_replay.json`
(`scripts/module1_composite_replay.py`), `docs/architecture/parallel_model_audit.json`,
`docs/architecture/phase2_ledger.json`.

### Defects found this round

- **DEF-008 (P0, money duplicated).** The canonical order book had no
  execution-id identity. Re-delivering one 500-share execution left the book
  holding **1,000 shares** and charged cash twice — measured directly against
  `OrderBook`/`replay_account` before any fix. A retried venue callback, a
  gateway retry after a successful write or a session reconnect would each have
  done this. Now refused on three layers: `Order.apply` returns the same
  snapshot for an identical re-report (and raises `DuplicateExecution` when an
  execution id is reused with different quantity, price or fees),
  `OrderBook.apply` records no event when nothing changed, and `replay_account`
  refuses to apply an execution id twice even if an older writer already put it
  in the file. `ledger.mirror_event` exists because the first fix was
  incomplete: callers that appended `history_of(...)[-1]` unconditionally would
  have written the *previous* event again, turning a harmless duplicate into a
  corrupt log.
- **DEF-009 (P0, profit reported that nothing backed).** `AccountState`
  expensed entry fees instead of capitalising them into cost basis, so
  `realised + unrealised` exceeded `NAV − initial cash` by exactly the entry
  costs — **10.3521** on the composite scenario. Cash was never wrong, which is
  why the error was invisible to a NAV-level review and to the previous paper
  replay test (it compared cash and position only). `identity_residual` is now
  the arbiter and is asserted per path; `unrealised_pnl` is marked against the
  same weighted-average basis a sale realises against, and `PositionLot.cost_price`
  carries the all-in cost so lot drill-down agrees with it.
- **DEF-010 (P1, an audit that could not fail).** `OrderManager.rebuild_history`
  constructed `CanonicalLedger(self.ledger_path)` rather than reading the chain
  it writes to. With an in-memory or injected ledger that returned an *empty*
  ledger, so the "projection matches the durable record" audit compared `history`
  against nothing and passed. It now re-opens the manager's own chain from disk
  when durable, through a new `replay_book()` that needs no fabricated
  `initial_cash` — the old call passed `0.0`, which drove cash negative and
  tripped an accounting invariant on a question nobody had asked.
- **DEF-011 (P1, in this round's own new code).** `CanonicalLedger` defines
  `__len__`, so an *empty* injected ledger is falsy and
  `canonical_ledger or CanonicalLedger(path)` silently replaced it with a fresh
  in-memory chain. The OMS leg therefore wrote every event to a ledger nobody
  read: the venue's own books showed cash 980,259.298 with fills, while the
  replay of the file showed cash 1,000,000 and zero orders. Caught by the
  composite comparison on its first run, before any of it was claimed as
  working. Fixed with `is not None` on both sides and regression-locked by
  `test_an_injected_empty_ledger_is_not_silently_replaced`.
- **DEF-007 (P1, open, mitigated).** `AccountState.frozen_cash` is fed by no
  event and `freeze_cash`/`release_cash` have no production caller, so on any
  replayed account it reads 0.0 and "frozen cash matches" passes without
  measuring anything. Mitigated rather than closed: the reservation the composite
  compares is *derived* from the order book (a working buy commits cash, a
  working sell commits shares), which is the correct single-source form. The dead
  field remains and is now documented as unusable at its definition.
- **DEF-014 (P1, a durable write that destroyed replayability).** A component
  attaching to a ledger that already held events started with an **empty**
  in-memory `OrderBook`. `OrderBook.open` returns the existing order and records
  nothing when the content-addressed id is already known, but every caller then
  appended `history_of(...)[-1]` unconditionally — so it wrote the order's
  *previous* last event, a stale `FILL`, into the chain as if it were a `CREATED`,
  and the following `RISK_APPROVED` left a chain that no longer replayed.
  Discovered by running the evidence script twice: the first run passed, the
  second raised `FILLED -> APPROVED` at read time, when the corruption was
  already durable. Fixed structurally rather than per caller: `mirror_open` is now
  the only way to open an order and record it, all three engines use it, and it
  raises `LineageCollision` naming the cause instead of surfacing an opaque bad
  transition. The book is also hydrated from a populated chain in the fast engine,
  the paper broker and the OMS, so a collision is refused on the *write* and the
  file stays readable. Note the asymmetry this exposed: paper's canonical ids
  embed a random paper order id and so never collide, the OMS is protected by its
  durable claim, and the fast engine had neither — which is why it was the engine
  that corrupted its chain.
- **DEF-012 (P2, a test that proved nothing).** The first version of
  `test_a_tampered_composite_ledger_blocks_every_projection` tampered with
  formatting. The chain hashes canonical JSON rather than raw bytes, so the edit
  was correctly a no-op — the test passed while demonstrating nothing. Replaced
  with a semantic edit to a filled quantity, and it now also asserts *which*
  record `verify()` blames.
- **DEF-013 (P2, misleading invariant).** `AccountState.check` raised "frozen
  cash 0.0 exceeds total cash −10015.10", pointing an operator at a reservation
  when the problem was the balance. Now guarded on a non-zero reservation.

---

## 1. OMS / EMS Execution Engineer — implementation

**Scope.** Built the OMS-to-paper path and the execution-report identity.

**Implementation.** `execution/paper_adapter.py` (new): `BrokerBase` in front of
`PaperBroker`, translation only — no cash, position or order state of its own.
`paper/broker.py`: accepts an injected `CanonicalLedger`/`OrderBook`, adopts an
OMS-opened canonical order via `attach_canonical`, and routes every fill through
a new idempotent `apply_execution_report`. `execution/order_manager.py`: injects
its chain into the venue, skips its own venue-side fold when the venue is
canonical-aware, and exposes `submit_orders` so an explicitly stated order goes
through the *same* guarded path as a weight-driven one rather than growing a
second unguarded entry point.

**Measured.** `paper_broker` vs `oms_to_paper`: **zero differences of any kind**
across 45 shared dimensions — the routing layer adds intent, risk and idempotency
and changes no economic figure. Re-submitting the intent `s1-buy-full` after it
had already filled left venue fills at 2 → 2. `history` and `rebuild_history`
agree.

**Structural claim.** Passing both a ledger and a ledger path now raises on both
`PaperBroker` and `OrderManager`: one component, one chain, enforced at
construction rather than by convention.

**Verdict: cannot approve — this is my own implementation.** Reported for
independent verification below. My own reservation: `attach_canonical` is
discovered by `callable(getattr(broker, "attach_canonical", None))`. A venue that
is canonical-aware but names the method differently would silently get the
old double-write behaviour. That is a latent trap, not a present defect — the one
canonical-aware venue is covered by `test_paper_and_oms_agree_on_every_dimension`.

---

## 2. Backtest Audit Engineer — independent verification

**Scope.** Whether the reconciliation is evidence or theatre. I did not review
the implementation's intent; I attacked the measurement.

**Checks performed.**

1. *Can the reconciler fail?* Yes. An unexplained cash difference classifies as
   `unexplained` with `resolution_status = blocking`
   (`test_an_unexplained_cash_difference_blocks`).
2. *Can an engine reconcile by modelling nothing?* No. A dimension present on
   one side only is compared against an `ABSENT` sentinel and counts as a
   difference (`test_a_dimension_present_on_one_side_only_is_a_difference`). This
   was a real hole in the first draft, which reported such dimensions in a
   separate list that did not affect the count.
3. *Can a tolerance hide a discrete error?* No. `ExplanationRule` rejects a
   tolerance on any status, quantity, count, lot or event-sequence dimension at
   construction, and a tolerance stops applying beyond its own bound rather than
   stretching.
4. *Do the fast-vs-venue exemptions excuse what happened, or only the price?*
   The price and cost-model rules cover cash, PnL, fees and lot cost prices. The
   `order[` and `count[` exemptions are `not_applicable_path` and are broad, so
   I do not accept them on their own — I verified the underlying claim
   separately: `test_fast_and_venue_executed_the_same_quantity` asserts the fast
   engine and the venue reached the same cumulative quantity **and** the same
   status on both shared orders. Without that test the exemptions would be
   unfalsifiable, and I would veto.
5. *Is the replay a real replay?* Each path is rebuilt by re-opening the file
   from disk; hash-stable over three repetitions; a semantic edit to a filled
   quantity raises `LedgerCorruption` and `verify()` names the edited record.
6. *Independent arithmetic.* I re-derived the identity by hand for the
   round trip: buy 1,000 @ 10.05 with 5.1001 of fees, sell 1,000 @ 10.45.
   Fees-capitalised basis reproduces cash exactly; fees-expensed basis overstates
   realised PnL by 5.1001. The implementation's fix agrees with the arithmetic,
   not merely with itself.

**Findings against the implementation.**

- The scenario is two sessions and two symbols. It exercises the lifecycle cases
  it claims, but it is *not* a broad economic workload, so "the engines agree"
  is established on this scenario and not in general.
- The fast leg expresses 4 of the 10 steps. Its agreement with the venue is
  therefore a narrower claim than the paper-vs-OMS result, and the report says so
  per step rather than in aggregate.
- Corporate actions, splits, dividends, delisting, multi-strategy cash
  competition and same-bar stop/target conflict are **absent** from the composite
  (they remain open under M4-01).

**Verdict: approve the composite replay result (M1-20, M1-21, M1-22, M1-23), on
the stated scope.** The measurement can fail, does fail when perturbed, and the
one broad exemption family rests on an independently asserted claim. I do not
approve any statement that the engines agree beyond this scenario.

---

## 3. SRE / Chaos Engineer — recovery and duplicate delivery

**Steady state.** One economic order per intent; one economic effect per
execution id; no confirmed fill lost; ledger integrity verified before any
projection is exposed.

**Faults injected this round.** Re-delivered execution report (absorbed, zero
delta, no event recorded); execution id reused with a different quantity, price
and fee (all three refused); out-of-order lifecycle event after a fill (refused);
semantic ledger tamper (blocked, with the offending record identified);
in-process restart (every path rebuilt from its file alone); duplicate intent
re-submission through the OMS (no second order at the venue).

**Not tested.** API or worker `SIGKILL` against a live run. Database
lock/disconnect. `fsync` failure, read-only ledger path, disk full. Two
concurrent recovery workers at the process level *against the paper venue*
(proven for `IdempotencyStore` in Round 2, not for this chain). Deployment-level
uniqueness beyond a local `fcntl.flock`.

**Additional risk I am recording.** DEF-011 was a silent write to an unread
ledger, and nothing outside the composite comparison would have caught it. There
is no runtime assertion that a component's ledger is the one being read. A
health check that a configured ledger path exists and is growing would have
turned a silent condition into a loud one.

**On DEF-014, which I regard as the most instructive defect this round.** It was
found by the cheapest possible chaos experiment: running the evidence script a
second time. The first run passed; the second wrote a stale event into the chain
and only failed at read time, with the corruption already on disk. Two lessons I
am recording as requirements rather than observations. First, *every* evidence
generator must be run at least twice before its output is treated as evidence —
a one-shot green run does not establish that the write path is safe against the
state it just created. Second, the class of bug is not "someone forgot a check":
it is that the book and the ledger were written by two statements a caller had to
keep consistent by hand. `mirror_open` and `mirror_event` remove that obligation.
I would veto any new engine that writes to the ledger without going through them.

**Verdict: do not approve.** The duplicate-delivery and replay invariants hold
under the faults I injected, but process-kill, filesystem-failure and
distributed-uniqueness faults remain untested, and those are exactly the
conditions under which the guards matter. My veto stands on M1-11 and on
deployment uniqueness.

---

## Module One completion gate (after Round 7)

| Criterion | Status |
|---|---|
| One canonical economic ledger exists | **Pass** |
| No parallel economic state remains | **Pass** — audit: 19 findings, 0 blockers |
| Fast, Paper and OMS replay exactly | **Pass** on the composite scenario — `unexplained_economic_differences = 0` |
| One execution id, one economic effect | **Pass** — DEF-008 fixed on write and read paths |
| PnL split reconciles with cash | **Pass** — DEF-009 fixed, residual ~2e-11 on all paths |
| Real API and Worker idempotency | **Blocked** — no such path exists to test (M1-13) |
| Deployment uniqueness enforced | **Fail** — local `fcntl.flock` only |
| Module One fault injection complete | **Fail** — process kill, DB, filesystem untested (M1-11) |
| No P0 or P1 remains | **Fail** — DEF-007 open (mitigated, dead field still readable) |
| All three roles approve | **Fail** — SRE vetoes; OMS/EMS cannot self-approve |
| Full regression suite passes | **Pass** — 2317 passed, 4 skipped, 0 failed |

**Module One is NOT verified.** The composite replay gate — the item this round
targeted — is met. Three gate criteria remain.

Next, in dependency order:

1. **M1-13** — build the paper-scoped HTTP submission path (FastAPI route →
   worker → `OrderManager` → `PaperBrokerAdapter` → one chain, `LIVE_DISABLED`
   preserved) and drive it for double click, repeated POST, timeout-after-success,
   gateway retry, duplicate queue delivery, concurrent workers, concurrent
   processes and crash at each of the four commit boundaries.
2. **Deployment uniqueness** — choose and enforce the contract: either refuse to
   start a second host, or make the idempotency claim, order-intent action and
   execution identity atomically unique in a shared transactional store.
3. **M1-11** — API/worker `SIGKILL` against a live run, DB lock/disconnect,
   `fsync` failure, read-only path, disk full, two concurrent recovery workers
   against the paper venue.

**Maximum permitted stage: `backtest_only`.**

---

## Round 8 summary (M1-13, M1-24)

| | After Round 7 | After Round 8 |
|---|---|---|
| Economic HTTP submission path | **did not exist** | `POST /api/paper/orders` → queue → worker → OMS → paper → one ledger |
| Entry-point idempotency | unprovable (no entry point) | **proven** — 46 tests through the real route |
| Deployment uniqueness | local `flock` only | single-host contract enforced; second **host** refused |
| OMS economic-intent guard in the API | — | was in-memory (**DEF-015**), now durable |
| T+1 settlement dating | wall-clock fallback (**DEF-016**) | dated by the fill, never guessed |
| Ledger verified requirements | 25 of 62 | **27 of 63** |
| Full regression suite | 2317 passed | 2365 passed, 4 skipped, 0 failed |

### Defects found this round

- **DEF-016 (P0-class, latent for the whole program, and date-dependent).**
  `PaperBroker.cancel` supplied no session, so `_canonical_event` fell back to the
  **wall clock** and wrote *today* into the ledger's settlement-relevant
  `tradeDate`. `CanonicalLedger.replay` then mapped trade dates per *order*,
  last-write-wins — so a cancel issued after a fill **retroactively re-dated that
  fill**, pushing its T+1 lot a day forward and making settled shares read as
  unsettled. Measured: the composite scenario's thin-symbol lot came back
  `acquired_on = 2026-08-05` for a fill that happened on `2026-08-04`, and
  `settled_inventory` went 1,000 → 0. **The reason this survived Round 7 is the
  part worth recording: the wall clock happened to equal the session being
  simulated, so every assertion passed.** It surfaced the next calendar day, from
  a full-suite run, with no code change in between. Fixed on both sides: trade
  dates are keyed by **execution id** (the settlement session belongs to the fill,
  not the order), and the wall-clock fallback is gone — an absent date is
  recoverable because replay falls back to the fill's own session, a wrong one is
  not.
- **DEF-015 (P1, a guard that was not the guard it claimed to be).** The
  `OrderManager` inside `PaperOrderService` was constructed with no
  `idempotency_path`, so its durable economic-intent claim was
  `IdempotencyStore(None)` — in memory, forgotten on restart, invisible to a second
  process. Only the request-level claim was actually protecting anything, which
  meant a client that lost its idempotency key and retried with a fresh one would
  have got a **second economic order**. Found by a crash-boundary test that could
  not reach the window it was written for, because the resolve it tried to
  intercept was happening on a different store object. Both stores now point at
  one file, and `test_both_guards_are_durable_and_share_one_file` fails if either
  stops being durable.
- **A design error in my own first cut, corrected rather than documented around.**
  The module docstring claimed a fresh-key retry was stopped by the economic
  guard. It was not: I had derived the OMS's `signal_id` from the request's
  idempotency key, which made the *economic* guard depend on *delivery* identity.
  `signalId` is now required, is never derived from the delivery key, and is never
  defaulted — a constant default would collapse two legitimate sleeve orders that
  happen to want the same trade, which is the INC-E1 defect class. Both directions
  are tested: `test_a_fresh_key_for_economics_that_already_traded_is_still_stopped`
  and `test_two_signals_wanting_the_same_trade_both_execute`.
- **Recovery misattribution (found before it shipped).** `_resolve_from_ledger`
  matched a queued request to a ledger order by economics, so two requests with
  identical economics each matched the other's order — an interrupted submission
  would have been reported as **executed** on the strength of a different
  request's fill. Now matched by lineage, via `Signal.create` rather than a
  re-derived digest so the two cannot drift.

### Scope note on M1-13's history

Round 7 marked this `blocked` and said the requirement was unbuilt rather than
untested. That was correct and is worth keeping visible: the honest move was to
say "there is no such path" and then build it, not to test `OrderManager` again
and call the endpoint covered.

---

## 1. OMS / EMS Execution Engineer — implementation

**Scope.** Built `services/quant_api/services/paper_orders.py`,
`services/quant_api/schemas/paper.py`, seven routes, and the container wiring.

**Design decisions that carry weight, and why.**

1. *The endpoint queues; it does not execute.* Executing inline would collapse
   "crash after responding" and "crash before executing" into one
   indistinguishable window, and those are the two that produce duplicates. The
   cost is that clients poll; the benefit is that both boundaries are separately
   testable, and both are tested.
2. *The ledger decides whether an economic action happened.* A claim record proves
   only that a worker *intended* to act. `recover()` reads `CanonicalLedger`.
3. *An interrupted submission is never retried automatically.* A request claimed
   but with no order on the chain is marked terminally `interrupted`. Retrying
   past a claim is exactly how a duplicate is created, so the operator must
   resubmit under a new key — a visible act. This is a deliberate choice of
   correctness over availability: a legitimately lost order is possible, and is
   reported rather than silently re-attempted.
4. *No fabricated price.* Without market data a submission is rejected with
   `market_data_unavailable`. The container wires **no** market source, so in
   production today every submission is refused — which is the truthful state of
   this repository, not a bug to paper over with a default price.

**Measured.** Through the route: one economic order under every duplicate shape;
409 on a changed fingerprint; 451 on live intent with nothing recorded; two
sessions produce two orders; a cancelled partial keeps its 1,000 executed shares;
the account's `identityResidual` is 0.

**Verdict: cannot approve — my own implementation.** My own reservations: the
per-symbol daily order cap lives in `OrderManager.counts_today`, which is
in-memory and resets on restart (not a money-correctness issue — the durable guard
is separate — but it is a limit that silently stops limiting). And `drain` is
operator- or caller-driven; there is no background worker, so nothing executes
unless something calls it.

---

## 2. Backtest Audit Engineer — independent verification

**Scope.** Whether the 46 tests measure what they claim, and whether the accounting
still holds after this round's changes to `replay`.

**Checks performed.**

1. *Do the tests measure the ledger or the reply?* The ledger.
   `economic_facts()` replays the file and counts orders, fills, distinct execution
   ids and filled quantity. A route that answered "accepted" twice would pass; one
   that produced two orders could not.
2. *Does DEF-016's fix hold at the boundary I care about — settlement?* Yes, and I
   re-derived it independently: a fill on session 1 with a cancel on session 2 must
   leave the lot `acquired_on = session 1` and sellable on session 2.
   `test_a_later_cancel_does_not_re_date_an_earlier_fills_settlement` pins exactly
   that, with two explicit sessions rather than the wall clock. **I regard the
   wall-clock fallback as the most serious thing found in either round**, because it
   was invisible on the day it was written and would have been invisible again on
   any day matching the simulated session.
3. *Did the accounting identity survive?* Yes — 0 to float precision on all three
   composite paths and on the API account view.
4. *Is the composite still clean after the `replay` change?*
   `unexplained_economic_differences = 0`, regenerated after the fix.

**Findings against the implementation.**

- The crash boundaries are **injected**, not signalled: a patched method raising,
  not a `SIGKILL`. That covers the logical windows and does not cover a process
  torn down mid-`write`. The SRE's veto below is correct on this point and I do
  not consider the fault matrix complete.
- The "two OS processes" test releases the parent's writer lock first, so it proves
  the *claim store* arbitrates across processes; it does not prove two
  simultaneously-writing processes are safe, because the design forbids that
  outright. That is a coherent position, but it means cross-process concurrency is
  *prevented*, not *survived*.
- The market source is unwired in production, so this path cannot currently
  produce a fill outside a test. Correct, and worth stating plainly rather than
  reading the 46 passing tests as "the endpoint works in production".

**Verdict: approve M1-13 and M1-24 on the stated scope.** The entry point is real,
the measurement is taken from the record of account, and the two defects found were
fixed at the layer that caused them.

---

## 3. SRE / Chaos Engineer — recovery and duplicate delivery

**Faults injected this round.** Crash at four commit boundaries (before claim,
after claim, after the canonical append, after paper acceptance); 8 concurrent
worker threads on one queue entry; 16 concurrent deliveries of one request; two
OS worker processes; simultaneous recovery ×4; a second writer process; a writer
on another host; a stale heartbeat; an unreadable lock record; duplicate execution
report; reordered lifecycle event.

**What I insisted on and got.** The lock had to fail in **both** directions. A
guard that only refuses is a guard that eventually locks out a legitimate takeover,
so a stale heartbeat and a corrupt lock file both read as free — and an unparseable
record reads as stale rather than as a live holder, because one corrupt byte must
not hold a trading account hostage.

**Still not tested.** A real `SIGKILL` against a live API or worker process.
Database lock/disconnect. `fsync` failure, read-only ledger path, disk full.
Machine restart.

**A note on how DEF-016 was found, which I am recording as a process requirement.**
It was not found by any test I designed. It was found because the full suite was
run on a *different calendar day* than the one on which the code was written, and a
wall-clock fallback stopped agreeing with the fixture. Round 7's lesson was "run
every evidence generator twice"; Round 8's is stronger: **any code that reads the
wall clock inside an economic path is a defect until proven otherwise, and no
same-day test run can prove otherwise.** I would like a suite-level guard that
fails if `datetime.now` is reachable from a settlement-relevant code path.

**Verdict: do not approve.** The logical windows are covered and the deployment
contract is now enforced in both layers, but signal-level process kills,
filesystem failures and database faults remain untested, and those are the
conditions the guards exist for. My veto stands on M1-11.

---

## Module One completion gate (after Round 8)

| Criterion | Status |
|---|---|
| One canonical economic ledger exists | **Pass** |
| No parallel economic state remains | **Pass** — audit: 19 findings, 0 blockers |
| Fast, Paper and OMS replay exactly | **Pass** on the composite scenario — 0 unexplained |
| One execution id, one economic effect | **Pass** |
| PnL split reconciles with cash | **Pass** — residual ~2e-11 on all paths |
| Settlement session is the fill's, never the wall clock's | **Pass** — DEF-016 fixed |
| Real API and Worker idempotency | **Pass** — 46 tests through the route (M1-13) |
| Deployment uniqueness enforced | **Pass** for single host; multi-host **refused, not supported** (M1-24) |
| Module One fault injection complete | **Fail** — no signal-level kill, no DB or filesystem faults (M1-11) |
| No P0 or P1 remains | **Fail** — DEF-007 open (mitigated; dead field still readable) |
| All three roles approve | **Fail** — SRE vetoes; OMS/EMS cannot self-approve |
| Full regression suite passes | **Pass** — 2365 passed, 4 skipped, 0 failed |

**Module One is NOT verified.** Two gate criteria remain, both narrow and both
named. Nothing about the economic core is outstanding.

Next, in dependency order:

1. **M1-11** — real `SIGKILL` against a live API and worker process mid-write;
   `fsync` failure; read-only ledger path; disk full; database lock/disconnect.
   This is the last blocking item and it is the SRE's standing veto.
2. **DEF-007** — remove or seal `AccountState.frozen_cash` so the dead field
   cannot be read as a reservation.
3. **M1-05 / M1-06** — order drill-down in the UI, and cancel/expire in the fast
   engine. Neither blocks the economic gate; both belong to Module Five's loop.

**Maximum permitted stage: `backtest_only`.**

---

## Round 9 summary (M1-11, DEF-007) — Module One gate assessment

| | After Round 8 | After Round 9 |
|---|---|---|
| Process faults | injected exceptions | **13 experiments, real `signal.SIGKILL`** at 5 economic boundaries |
| Storage faults | untested | fsync EIO, ENOSPC at write **and at flush**, read-only, missing record, edited record |
| Failed durable write | **corrupted the chain** (DEF-017) | fails closed and latches; a restart replays what is really there |
| `AccountState.frozen_cash` | open (DEF-007) | **deleted** — reservation derived from the order book |
| Ledger verified requirements | 27 of 63 | **28 of 63** |
| Full regression suite | 2365 passed | 2381 passed, 4 skipped, 0 failed |

Evidence: `docs/architecture/module1_fault_injection.json` (13 experiments, 0 failed),
run in the suite by `tests/quant_ui/test_module1_fault_injection.py`.

### Defects found this round

- **DEF-017 (P0-class, a failed disk write corrupted the chain).** `CanonicalLedger.append`
  updated `_records` and `_head` only after the durable write, which is correct —
  but on a *failed* write it left the file **ahead of memory**, and nobody noticed
  that the in-memory head was then no longer the file's. Measured before the fix:
  one `fsync` raising `EIO` left **2 records on disk against 1 in memory**, and the
  next append chained from the stale head, so the file stopped replaying — `verify`
  reported `brokenAt: 2` on a chain nothing had tampered with. The failure mode
  matters: `fsync` raising means the bytes were already written and flushed, so
  "the write failed" and "the record is absent" are not the same statement. Fixed
  by latching: any `OSError` during the durable write records the reason and every
  later append from that instance raises `LedgerWriteUnavailable`. Resynchronising
  instead was considered and rejected — adopting bytes the OS refused to guarantee
  as durable would be a lie. A restart re-reads the file honestly, skips a torn
  tail, and continues.
- **A reservation gap, found while closing DEF-007.** With `frozen_cash` deleted,
  the reservation is derived from working orders — and an order priced only by
  *reference* (as the fast engine's are) had no basis to reserve against, so it
  fell through to `0.0` and reported a genuinely working order as committing no
  capital. `Order` now carries `reference_price`. Latent rather than live (every
  order that actually reaches a venue carries a limit price), but it would have
  become live the moment a reference-priced order was left working.

### DEF-007, closed by deletion

`AccountState.frozen_cash` had no event feeding it and no production caller for
`freeze_cash`/`release_cash`, so on every replayed account it read `0.0` and every
"reserved cash matches" comparison passed without measuring anything. Documenting
that was Round 7's mitigation; deleting it is the fix. Cash committed to working
buy orders is now a function of the order book, which additionally means the
commitment is released **exactly once** — it is recomputed, never decremented.

---

## 1. OMS / EMS Execution Engineer — implementation

**Scope.** `scripts/module1_fault_injection.py`, the ledger write latch, the
`frozen_cash` deletion and `Order.reference_price`.

**On the harness design.** The child kills itself with `os.kill(getpid(),
SIGKILL)` rather than `os._exit`, because `os._exit` still lets the interpreter
return from the current frame and SIGKILL cannot be caught, blocked or deferred.
The `mid_ledger_append` case writes half a record and then dies, so the torn tail
is produced by a signal rather than by a test writing a truncated file — which is
the only version of that test that proves anything about the write path.

**On the fsync scope.** Patching `os.fsync` wholesale was my first attempt and it
was wrong: the ledger and the idempotency store share the symbol, so the blanket
failure hit whichever wrote first — in practice the claim store — leaving the
ledger untouched and the experiment silently proving something else. The fd is now
resolved through `/proc` so the fault lands on the file the hypothesis names.

**Verdict: cannot approve — my own implementation.** Residual concern: the
`sigkill.after_claim` case leaves no ledger file at all, so it exercises the
"nothing happened" branch rather than the interesting one. It is still worth
having as the boundary case, but it is the weakest of the five.

---

## 2. Backtest Audit Engineer — independent verification

**Checks performed.**

1. *Do the experiments have anything to lose?* Not at first — my main finding.
   The storage faults originally ran against an **empty** chain, so they proved only
   that an error surfaces, not that committed data survives. They now seed a filled
   order first, and `test_the_storage_faults_ran_against_a_chain_that_already_held_records`
   fails if that regresses.
2. *Is the invariant the right one?* It was not. "At most one economic order" was
   coded as an absolute cap, which the seeded experiments correctly violated the
   moment they issued a second legitimate intent. It is now measured **per
   lineage** — the invariant Module One actually claims. This is worth recording as
   a caught error in the *measurement*, not the code: a green board from a wrong
   invariant is worse than a red one.
3. *Is the control run real?* Yes, and it is asserted: `sigkill.none` must show one
   order and one fill, so the suite cannot pass by never trading.
4. *Are the process faults still real signals?*
   `test_the_process_faults_were_delivered_by_a_real_signal` asserts
   `child_signal == SIGKILL` for all five, so the harness cannot quietly degrade
   back to injected exceptions.
5. *Did DEF-017's fix leave reads working?* Yes — latching writes while blinding
   the operator diagnosing the failure would be its own defect.
6. *Composite and audit after the changes?* `unexplained_economic_differences = 0`;
   audit 19 findings, 0 blockers.

**Findings against the implementation.**

- ENOSPC is injected at `write` and at `flush`, not produced by filling a volume.
  Both shapes of the real error are covered, and I accept that as sufficient; I do
  not accept describing it as "a full disk was tested".
- A machine-level restart is untested. The mechanism it would exercise — `fsync`
  before the append is reported, and torn-tail recovery on read — is tested
  directly, so I treat this as residual rather than missing.

**Verdict: approve M1-11.** The faults are delivered by the real mechanisms, the
experiments have something to lose, and the invariant they check is the one Module
One claims.

---

## 3. SRE / Chaos Engineer — recovery, and the standing veto

**My veto is lifted.** I recorded it in Rounds 7 and 8 on two grounds:
process-kill/filesystem/database faults untested, and deployment uniqueness
unenforced. Both are addressed, and I want to be precise about how:

- **Process kills are now real.** Five boundaries, `SIGKILL`, verified by signal
  number in the suite. `mid_ledger_append` produces a genuine torn tail from a
  signal and the reader skips it with everything before it intact.
- **Filesystem faults are covered in the shapes that occur.** `fsync` EIO scoped
  to the ledger's own fd; ENOSPC at both `write` and `flush`; a real `chmod`. Each
  runs against a chain that already holds a filled order.
- **"Database" faults.** There is no DBMS in this build. The durable stores are
  append-only files arbitrated by `fcntl` locks, so the honest mapping is the
  filesystem faults above plus the four-process contention race — where the
  timeout is the deadlock check. I will not sign a report claiming a database
  fault was injected into a system that has no database.
- **Deployment uniqueness** was enforced in Round 8 and fails in both directions:
  a stale heartbeat and an unreadable lock record both read as free.

**Residual risk I am accepting, on the record.**

1. A machine-level restart is untested. The mechanism is `fsync` plus torn-tail
   recovery, both tested directly. Accepted.
2. ENOSPC is injected, not produced by a full volume. Accepted, with the wording
   constraint above.
3. **Multi-host is refused, not supported.** `fcntl` is advisory and per-host; the
   occupancy record is best effort. This is a *limitation of the deployment
   contract*, not a gap in its enforcement.
4. There is still no runtime assertion that a component's ledger is the one being
   read. DEF-011 was a silent write to an unread chain and only the composite
   comparison caught it. I want a health check that a configured ledger path exists
   and is growing. Not a Module One blocker; carried forward.

**The process requirement I asked for in Round 8 is now met by construction.** Any
code reading the wall clock inside an economic path was guilty until proven
otherwise; the settlement path no longer reads it at all, and the harness runs in
the suite so it cannot rot.

**Verdict: approve M1-11 and approve Module One's fault-injection criterion.**

---

## Module One completion gate (after Round 9)

| Criterion | Status |
|---|---|
| One canonical economic ledger exists | **Pass** |
| No parallel economic state remains | **Pass** — audit 19 findings, 0 blockers |
| Fast, Paper and OMS replay exactly | **Pass** — 0 unexplained differences |
| One execution id, one economic effect | **Pass** |
| PnL split reconciles with cash | **Pass** — residual ~4e-11 on all paths |
| Settlement session is the fill's, never the wall clock's | **Pass** |
| Real API and Worker idempotency | **Pass** — 46 tests through the route |
| Deployment uniqueness enforced | **Pass** for single host; multi-host refused, not supported |
| Module One fault injection | **Pass** — 13 experiments, real SIGKILL, 0 failed |
| A failed durable write cannot corrupt the chain | **Pass** — DEF-017 fixed |
| No P0 or P1 remains | **Pass** — DEF-007 closed by deletion; DEF-008..017 all fixed |
| All three roles approve | **Pass** — Backtest Audit and SRE approve; OMS/EMS abstains on its own work, as required |
| Full regression suite passes | **Pass** — 2381 passed, 4 skipped, 0 failed |

### MODULE ONE GATE: PASSED

Stated precisely, because a passed gate is the easiest place to overclaim:

- **What passed** is the gate defined in the program's Module One section: one
  record of account, no parallel economic state, three engines reconciling with
  zero unexplained differences, idempotency at the real entry point, an enforced
  deployment contract, and fault injection with no invariant violated.
- **What is still open under Module One's numbering** is M1-05 (order drill-down
  in the UI) and M1-06 (cancel/expire modelled in the fast engine). Neither is a
  gate criterion and neither touches the economic core; both are workflow items
  that belong to Module Five's loop and are carried there rather than declared
  done.
- **What is explicitly not claimed:** that the engines agree beyond the composite
  scenario; that a real full disk or a machine restart was tested; that multi-host
  deployment works; that the paper path can currently fill in production, since no
  market source is wired.

**Maximum permitted stage: `backtest_only`** — unchanged. Module One is a
correctness foundation, not a promotion criterion, and nothing here is evidence
about a strategy.

**Next: Module Two.** Beginning with the explicit time model, which every other
Module Two requirement depends on: an event that cannot say when it became
knowable cannot be checked for look-ahead.

---

## Round 11 summary (M2 complete, M3-01, M3-02)

| | After Round 9 | After Round 11 |
|---|---|---|
| Module Two | M2-03 only | **all five verified** |
| Composite engines | 3 | **4** — streaming added, deriving its own fills |
| Streaming vs paper | not run | **0 differences of any kind** |
| Fill-price formula | duplicated in two engines | one implementation both delegate to |
| Stage derivation | inverted placeholder (DEF-018) | evidence-derived, monotonic, tested |
| Ledger verified | 29 of 63 | **35 of 63** |
| Full regression suite | 2408 passed | 2480 passed, 4 skipped, 0 failed |

### Defects found

- **DEF-018 (P1, the ledger lied about its own stage).** `_permitted_stage` was a
  placeholder with inverted logic: `backtest_only` while streaming was incomplete,
  `blocked` once it was done. **Finishing Module Two lowered the reported stage** —
  and the stage is the programme's headline claim. Latent only because streaming
  had never been complete. Replaced with a derivation that steps down for a stated
  reason at each level, and the gate ID lists are written out explicitly (including
  `MODULE_ONE_NON_GATE = (M1-05, M1-06)`) so the *exclusions* are auditable rather
  than implicit. Ten tests now pin it, including that verification can never lower
  the stage and that no amount of it can reach `paper_ready` — a stage reachable
  from requirement states alone would be exactly the promotion-by-test-count the
  programme forbids.
- **DEF-019 (P1, found by building the second engine).** The streaming matcher had
  no cash constraint: it filled whatever the participation cap allowed and let cash
  go negative. The accounting layer would not have objected, because a negative
  balance is not one of its invariants — so a streaming run would have reported
  fills the account could never have funded. Now refused at fill time (when the
  price, and therefore the cost, is known), reading available cash from a replay
  rather than a running balance.
- **A duplicated pricing formula, removed rather than documented.** The matcher's
  `_execution_price` was a near-copy of the paper broker's. It agreed today; two
  copies of a formula agree only until one is edited, and the disagreement would
  then arrive as a reconciliation difference with no way to say which side was
  right. `ashare_rules.execution_price` is now the single implementation, and a
  test breaks it and requires *both* engines to break.

---

## Backtest Audit Engineer — the decisive role for reconciliation

**Scope.** Whether "streaming vs paper: 0 differences" is a measurement or a
tautology. This is the claim I would most expect to be circular.

**Checks performed.**

1. *Did streaming derive its own fills, or replay the venue's?* Derived. The
   matcher reads a `BAR` and decides acceptance, quantity and price itself. I did
   not accept the code as evidence: `test_the_streaming_leg_derived_its_own_fills`
   asserts the two legs' execution-id sets are **disjoint**, so neither ledger can
   be a copy of the other.
2. *Is the agreement then just shared code?* Partly, and the boundary is now clean.
   The rulebook — bands, tradability, lot rounding, costs, and the fill-price
   formula — is deliberately shared, because two implementations of the A-share
   rules would drift and the drift would be unresolvable. What is *not* shared is
   control flow: paper validates and fills inside one synchronous `submit`, while
   the matcher holds orders across bars and answers as separate events. So the zero
   is a statement about control flow over a shared contract. **I required the
   duplicated pricing formula to be extracted before I would sign this**, because
   while it existed the agreement was a coincidence rather than a property.
3. *Could a tolerance be hiding a difference?* No. Both exact pairs are granted an
   empty rule tuple, and tests assert they stay empty. A separate test scans every
   cross-path rule for a tolerance on a discrete dimension.
4. *Does the streaming leg exercise the whole scenario?* Yes — all 10 steps, versus
   the fast engine's 4. It has a venue, so nothing is out of its scope.
5. *Independent arithmetic.* Streaming reached cash 980,259.2980 from its own
   matching. That is the figure paper produced in Round 7, before the matcher
   existed.

**What I do not accept as claimed.**

- The scenario is still two sessions and two symbols. "The engines agree" holds on
  *this* scenario. A broad economic workload has not been reconciled, and M3-01's
  ledger note says so.
- M3-03 is recorded `blocked`, not `in_progress`. The table exists and is
  regenerated as evidence; what is missing is a page to serve it, and no amount of
  reconciliation work advances that.

**Verdict: approve M3-01 and M3-02 on the stated scope.**

**Maximum permitted stage: `backtest_only`** — unchanged, and now derived rather
than asserted. It cannot advance while the risk layers (M5) and independent
acceptance (M11–M13) are outstanding, which is the correct answer.

---

## Rounds 12–13 summary (M4-01 golden scenarios: 12 → 38)

| | Round 11 | Round 13 |
|---|---|---|
| Golden scenarios | 12 (fast engine only) | **38** across two files |
| Corporate actions | not on the record of account | canonical (DEF-020) |
| A holding with no mark | valued at **zero** | **unknown**, refused (DEF-021) |
| A benchmark gap | filled with 0%-return days | **unknown**, with coverage counts (DEF-022) |
| Ledger verified | 35 of 63 | 35 of 63 (M4-01 deliberately still `in_progress`) |
| Full regression suite | 2480 passed | 2506 passed, 4 skipped, 0 failed |

### Three defects, all the same shape

Each was found by *writing the scenario that had to state an expected value*, and
each was a missing measurement silently replaced by a specific, usually false claim:

- **DEF-020 (P0).** `apply_corporate_action` mutated paper's portfolio, emitted to
  the *operational* log, and wrote nothing canonical. Measured: one 0.50/share
  dividend and one 2:1 split left paper holding 2,000 shares and 990,494.90 while
  the replay still said 1,000 shares and 989,994.90 — **500.00 of cash and 1,000
  shares apart**, on the record of account that is supposed to be the only one. The
  audit did not object because `ECONOMIC_EVENT_NAMES` did not contain
  `CORPORATE_ACTION_APPLIED`; that list was fixed *first*, so the class cannot
  recur. A dividend must also go to realised PnL — the mark drops by it on the ex
  date, so omitting the income breaks the accounting identity by exactly the
  dividend.
- **DEF-021 (P0-class).** A held position with no mark was valued at **zero**: on
  1,000 shares carried at 10.0051 that understated NAV by 10,000.00 and fabricated
  a 10,005.10 loss. **And the accounting identity still held**, because cash and
  the mark were consistently wrong together — the invariant that catches DEF-009's
  class is blind to this one. Paper's `Portfolio` had it in the opposite shape:
  `if s in prices` silently *excluded* unpriced holdings. Valuation now refuses,
  and `valuation()` reports `nav: None` with the symbols named. The strictness
  immediately caught production: `/api/paper/account` was reporting a
  zero-valued NAV, and since no market source is wired that was the **normal**
  answer.
- **DEF-022 (P0).** A benchmark gap was filled with 0%-return days. With the
  benchmark absent for 5 of 10 sessions it reported `benchmark_return = +0.00%` and
  `excess_return = +0.00%` when the truth was **+20% and −20%** — excess return
  overstated by 20 percentage points, presented as a confident number with nothing
  saying half the benchmark was missing. This repository has already paid for this
  once (`honest-baseline-truth`: the +20%/year excess claim was false).

### What I refused to do

- **No `exclude` option on valuation.** "Value what you can and omit the rest" is
  the version of DEF-021 that looks most reasonable, and a scenario exists
  specifically to pin that a partially priced book must refuse *entirely*.
- **No change to the gate's reason code.** Carrying the absent-vs-incomplete
  distinction in the machine-readable identifier would break every consumer
  matching on it; it went into structured fields instead.
- **No spread scenario.** There is no bid/ask in this build and no Level-2 vendor
  serves this market, so a spread scenario would assert an invented number. Named
  as blocked-on-data rather than left as an unexplained gap.

### A wrong premise of my own, corrected by the engine

I asserted that holding through an ex date earns 500.00 more than selling first.
It does not — the two end **0.255** apart, which is only the fee difference on a
smaller notional. Total return is unchanged, because the mark drops by the
dividend. The test now pins 0.255, which is the *stronger* check: an engine
crediting a dividend without dropping the mark would show free money on the ex date.

---

## Backtest Audit Engineer — on M4-01

**Verdict: do not mark M4-01 verified.** Thirty-eight scenarios is not the same as
complete, and three of the programme's listed cases remain — each now named with
what actually blocks it rather than left implicit: spread (no bid/ask data exists),
base-position intraday T (the strategy feature does not exist), and same-bar
stop/target driven through an engine (no engine places a bracket order; the rule
itself is implemented and tested under M2-05).

**What I do accept.** Every expected figure in both files is derived by hand in its
docstring and asserted exactly, and three real P0-class defects were found by that
discipline in two rounds. A test that reads its expectation out of the engine could
not have found any of them.

**A pattern I want recorded as a review heuristic.** All three defects were the same
shape: a *missing measurement* replaced by a plausible default — zero shares, zero
value, zero return. Two of them survived because an internal consistency check
still passed. So "the invariants hold" is not evidence that the reported numbers
are right; it is evidence that they are *consistent*, which a uniformly wrong pair
of numbers also satisfies. Any `fillna(0.0)`, `.get(x, 0.0)` or `if x in y` filter
on a path that produces a reported figure should be read as a defect until shown
otherwise.

**Maximum permitted stage: `backtest_only`** — unchanged.

---

## Round 17 (M5-02: leakage becomes a gate) — DEF-025, DEF-026

Evidence: `scripts/m5_leakage_audit.py` → `docs/architecture/m5_leakage_audit.json`
(run twice, byte-identical). Tests: `tests/test_pit_label_alignment_gates.py`.

### Research Risk Reviewer — implementation

**DEF-025 (P0): the gate hardening was defeated one layer down.** Round 15 made
`no_pit_violations` and `no_mock_or_synthetic` report `unknown` when their
measurement was missing. On the real training path it changed nothing, because the
measurement was never missing — it was manufactured, in four places:

| where | what it wrote | what the gate concluded |
|---|---|---|
| `_pit_violations` | `0` for a frame with no as-of reference | audited, no look-ahead |
| `_uses_mock_or_synthetic` | `False` for a frame with no provenance column | audited, no synthetic data |
| `v7_experiment._aggregate_metrics` | `uses_mock_or_synthetic: False`, a constant | audited, clean |
| `cli/v7_train` | `metrics["pit_violation_count"] = 0` when absent | audited, clean |

The v7 dataset builder produces exactly the shape that trips the first one: it has
`available_at` but no as-of column and no `point_in_time_valid`, so the audit took
its fall-through branch and reported **0 violations**. Both gates recorded a
*measured pass*. The lesson generalises past this instance: **hardening a consumer
against a missing measurement is worth nothing until the producer is audited too**,
because the natural fix at the producer is to supply the default the consumer
stopped supplying.

Both audits now return tri-state reports (`measured` / `not_auditable`), publish
`None` rather than a clean number, and producers forward what was measured or
nothing at all. A `DataManifest` built on an un-auditable dataset now carries a
`pit_audit_not_auditable` warning instead of reading `passed`.

**DEF-026 (P0): the label started before the features existed.** Nothing in the
pipeline had ever compared a row's availability stamp against its own label window.
`build_market_features` stamped `available_at = next trading row`; `v7_label_builder`
opens the label window at `close(trade_date)`. So **100% of rows** declared
themselves unusable until after they were already being scored — and that same
sentence in the gold builder's manifest asserts the invariant
`available_at <= trade_date`, which the stamp violated on every row.

`available_at` is not documentation. It is the as-of join key in
`merge_pit_features`, the documented way to attach fundamentals, disclosures and
risk features. Measured with an honestly-stamped extra published at each session's
close and carrying that session's return: **rank IC +1.0000** against the
`t -> t+1` label. Zero after the fix. A stamp adopted as conservatism about *acting*
was admitting exactly the information it was meant to withhold, because it had been
quietly repurposed as an information cutoff.

`evaluate_label_alignment` returns pass / fail / unknown, reads `label_entry_at`
when a delayed-entry builder publishes one, and records which basis it used rather
than assuming — the two conventions coexist here, and the verdict depends on which
is in force.

### What I refused to do

- **No silent change of label convention.** `gold_bridge.LABEL_CONVENTION` enters
  at `close(t+1)`; `v7_label_builder` enters at `close(t)`. Both are defensible.
  Picking one inside a defect fix would have moved every number in the repository
  under cover of a correctness change. The stamp was fixed — that is unambiguous,
  since the features are all derived from bar `t` and nothing later — and the
  convention question is recorded as an open decision with its price measured.
- **No hard failure for a not-auditable dataset build.** A panel with no provenance
  spine is not thereby a bad panel. The unknown blocks at the *acceptance* gate,
  which is where production-readiness is claimed, not at the dataset build.

### The open decision, priced

800 liquid names, 2024-08 → 2026-08, rank IC against a 1-day label, entry at
`close(t)` versus `close(t+1)`:

| feature | entry at close(t) | entry at close(t+1) | share of IC lost |
|---|---|---|---|
| momentum_5d | −0.02786 | −0.02674 | 4.0% |
| momentum_20d | −0.03153 | −0.02875 | 8.8% |
| volatility_20d | −0.02711 | −0.02789 | −2.9% |
| return_1d | −0.01496 | −0.00397 | **73.5%** |

The delay is nearly free for slow features and takes three quarters of the IC from
the fastest one. That is the expected shape, and it matters here specifically:
this programme's surviving candidates are short-horizon (`v89-plus7-production-state`,
`dot-t-ev-engine-rebuild`), so the convention is not a rounding decision for them.

### Backtest Audit Engineer — independent note

The Round 14 heuristic held again, and extended. All of DEF-020/021/022/023 were
*missing measurement replaced by a plausible default*. DEF-025 is the same shape one
level up: **a missing measurement replaced by a plausible default at the producer,
specifically because the consumer had just stopped defaulting it.** When a gate is
hardened, the next place to look is whoever feeds it.

DEF-026 is a different shape and worth naming separately: **a field adopted for one
purpose (documenting caution) silently acquired a second (deciding what joins).**
Nothing was wrong with either use in isolation. The audit that catches this class is
not "is the value right" but "do two fields that must agree actually agree" — here,
an availability stamp and a label window, which no test had ever compared.

Three tests asserting "all gates satisfied" were again found passing with no
evidence for the new gate — identical to the DEF-024 discovery one round earlier.
Two rounds running, the fixture that claims completeness has been incomplete. Any
fixture named `_passing_metrics` or `_metrics` should be assumed stale against the
current gate list until checked.

**Maximum permitted stage: `backtest_only`** — unchanged. M5-02 stays `in_progress`:
the readiness pipeline still does not compute survivorship, and the label
convention decision is open.

### Named follow-up, not silently absorbed

`qlib_provider` (`shift(-1)`) and `akshare_live_provider` (`+ BDay(1)`) still stamp
raw silver panels the old way. That was left alone deliberately: those panels carry
an *intraday* claim ("close(t) was not available during session t") that is
defensible on its own terms, and rewriting raw provider output is a wider change
than a defect fix should make in one round. `build_market_features` overwrites the
stamp for everything on the training path, and a panel used directly as a dataset is
caught fail-closed — verified: `evaluate_label_alignment` returns `fail` on a
provider-stamped panel.

Worth noting that the two invariants in this repository were mutually exclusive
until this round: the gold builder asserts `available_at <= trade_date`, while
`akshare_live_provider`'s own validator counts `available_at < trade_date` as a PIT
violation. Equality is the only point that satisfies both, which is where the
builder now sits.

---

## Round 18 (M5-02 producer side) — DEF-027

Evidence: `scripts/m5_leakage_audit.py --with-real-panel` section D (run twice,
byte-identical). Tests: `tests/test_survivorship_producer.py`.

### Research Risk Reviewer — implementation

**DEF-027 (P0): the DEF-024 fix could not fire on the real producer.** That fix
taught `build_masks` to consult a `status` column so a missing delisting date would
stop reading as "confidently never delisted". U0's master has no `status` column.
So against the actual master the fix answered **UNKNOWN for all 5,888 names** —
including the 5,530 sourced from live listing registers. It could not distinguish
the 358 names it existed to catch from the ones it did not need to.

That is not a safe failure, it is an inert one. A survivorship gate that can only
ever answer `unknown` tells an operator nothing about *what* to fix, and 99.2% of
rows came back with `eligibility_status = UNKNOWN`.

The master does record the distinction, in two columns that agree exactly:

| signal | live | dead |
|---|---|---|
| `status_end_blocked` | False — 5,530 | True — 358 |
| `source` | `sz_all_retry` / `sh_main` / `sh_star` / `bj_all` | `sz_delist` / `sh_delist_retry` |

`resolve_listing_status` reads whichever vocabulary is present — `status` first,
then provenance — and returns the basis it used alongside the answer, so the mask
records *how* it knows rather than asserting the conclusion.

### What the shipped artifact actually contains

Run against `runtime/data/gold/full_universe/dataset.parquet`, the artifact
carrying the `FULL_UNIVERSE_GOLD_READY` claim:

| | |
|---|---|
| panel | 10,917,401 rows, 5,790 symbols |
| `mask_post_delisting` as shipped | **FALSE for all 10,917,401 rows** |
| delisted names actually present | **258** |
| sessions they contribute | **417,877 — 3.83% of the panel** |
| rebuilt with the resolver | 10,499,517 FALSE / 417,884 UNKNOWN |
| survivorship verdict | `unknown`, blocker = delisting dates for **260** symbols |

So the panel was never missing its dead names — it has 258 of them — but every one
was masked as a security that never died, and the audit before this round guessed
the opposite ("the panel contains no delisted names at all"). Both the shipped mask
and the audit's first reading were wrong in different directions, and the resolver
is what makes the true statement available.

The verdict stays `unknown`, correctly: `status_end` is empty for all 5,888 rows, so
nothing can say *when* a name stopped trading. What changed is that the blocker is
now bounded and named — 260 symbols' delisting dates — instead of "the universe
cannot be checked". This cannot be closed by rebuilding; it needs data, consistent
with U0's finding that Sina is the only public path to delisted daily bars.

### Producer wiring, and what it cost

`_dataset_audit_metrics` now computes survivorship on every training run. That
turned a build-time cost into a per-run one, which exposed that `build_masks` spent
its time in a row-wise `iterrows()` — ~30k rows/s, about **6 minutes** on the full
panel. Vectorised to **22s** (18x), with the output pinned identical to the
row-wise construction by test rather than by assertion.

### Still missing

Nothing computes survivorship at *gold build* time, so the artifact on disk keeps
its all-FALSE mask until rebuilt. The gate is fail-closed against it, which is the
right default, but the artifact itself remains misleading to anything that reads
the column directly.

**Maximum permitted stage: `backtest_only`** — unchanged.

---

## Round 19 (M5-02) — DEF-028, and a correction to Round 18

Evidence: `scripts/m5_leakage_audit.py --with-real-panel` section D (regenerated,
run twice, byte-identical). Tests: `tests/test_survivorship_producer.py`,
`tests/test_gate_unknown_vs_failed.py`.

### Correction first

**Round 18 of this report reached a wrong verdict about the shipped gold artifact,
and the cause was mine, not the code's.** I judged
`runtime/data/gold/full_universe/dataset.parquet` against
`runtime/data/u0/historical_security_master.parquet`. That is not the master the
build reads. `scripts/build_u0_full_universe_gold.py` reads
`runtime/data/u0/security_master.parquet`, which has:

| | H-032C master (what I used) | build master (what the build uses) |
|---|---|---|
| `status` column | absent | present — 5,533 listed / 361 delisted |
| delisting dates | **0** | **361** |

Judged against the right master, the shipped artifact's all-FALSE
`mask_post_delisting` is **correct**, and the Round 18 claim that 258 dead names
"were recorded as names that never died" was false. The build script names its
master on the line that reads it; I did not check which one before drawing a
conclusion about its output. **An artifact can only be audited against the inputs
it was actually built from — otherwise the audit measures the mismatch, not the
artifact.**

### DEF-028: the audit was asking a question the mask cannot answer

`mask_post_delisting` answers "is this row after the security died". That is FALSE
for a live name **and** FALSE for a dead name's rows before it died. The audit
counted delisted names as symbols carrying a TRUE row — so a panel that includes
dead names and correctly stops each at its delisting date has **zero** by
construction, and was reported as "the panel contains no delisted names at all,
which is the signature of survivorship bias".

That is the inverse of every other defect in this programme: not a missing
measurement read as clean, but **correct behaviour read as broken**. It is
arguably worse, because the operator is sent looking for dead names that are
already there.

The tell was in the test suite and nobody read it: the only fixture that could
produce a `pass` was one where a delisted security keeps printing bars after its
delisting date. **When the only way to pass a check is to build the broken thing,
the check is measuring the wrong quantity.**

Presence is now read from the security's status — published as a `listing_status`
column by `build_masks`, or resolved from a master passed to the audit — and the
mask is used for what it does answer: whether any row outlived its own delisting,
which is a genuine failure and is now reported as `fail` rather than folded into a
coverage story.

### What the shipped artifact actually is

| | |
|---|---|
| panel | 10,917,401 rows, 5,790 symbols |
| delisted names present | **261** |
| sessions they contribute | **424,662 — 3.89% of the panel** |
| names with bars past their delisting date | **0** |
| survivorship verdict | **pass** |

The universe contains both the survivors and the dead, and stops the dead on time.
That is the shape a survivorship-free panel is supposed to have, and it took three
rounds and one wrong verdict to be able to say so.

### Still open

The label-entry convention (Round 17). And nothing computes survivorship at *gold
build* time — a future rebuild pointed at the H-032C master would silently lose
the evidence that makes today's verdict a pass, which is precisely the confusion
that produced the wrong verdict in the first place.

**Maximum permitted stage: `backtest_only`** — unchanged.
