# ENG_ST31 — forward-anchored SSE/BSE risk-warning (ST) register: PRE-REGISTRATION

Status: PRE-REGISTERED 2026-11-25 (JST; 2026-11-24 CST). This file is committed
before any forward reconstruction is run on any exchange. Thresholds, windows,
the validation set and the decision rule below are not changed after a result is
seen. If validation fails, SSE and BSE stay `BLOCKED_BY_DATA` and the frozen fresh
window stays stalled.

## 0. Why, and what is different from ENG_ST30

ENG_ST30 rebuilt ten years of SSE/BSE risk-warning history from cninfo
announcements alone. Its HOLDOUT failed M5b (0.924 < 0.98). The post-hoc diagnosis
showed that every large error was a stale state carried for months. Causes were
a missed revocation, a negated title and a reversed transition, and nothing ever
re-anchored the state.

The frozen fresh window needs much less than ten years. The catch-up appends
sessions after the v7 panel's last date (2026-08-05). It refuses until every
symbol in its seed (2,063 `.SZ`, 1,210 `.SH`, 0 `.BJ`) has dated ST evidence for
every day of the window. ENG_ST31 therefore starts from a dated **full-universe**
state and carries it forward over a few months only. Every dated short name the
method can observe must agree with the carried state; where one does not, the
state is **UNKNOWN** rather than guessed.

## 1. What has been looked at before this commit (full disclosure)

* Everything disclosed in ENG_ST30 §0, A1, A2 and its Result section, including
  the HOLDOUT diagnosis.
* The anchor's provenance: `runtime/data/u0/security_master.parquet` (5,894
  rows) was built from TickFlow `exchanges.get_instruments`, and every row has
  `source_timestamp 2026-07-26T11:39:51Z`. The 5,533 TickFlow rows hold
  2,309 SSE, 2,894 SZSE and 330 BSE securities with status `listed`.
  `st_current_snapshot.parquet` is derived from those names: 135 SH, 195 SZ and
  3 BJ ST names.
* A new full-universe snapshot fetched for this study,
  `runtime/data/u0/pit/_st_forward/instrument_snapshots/instruments_20261124T231441.parquet`
  (TickFlow `exchanges.get_instruments`, retrieved 2026-11-24T15:14:41Z =
  23:14 CST, after the 2026-11-24 session). It holds 5,894 `stock` instruments
  (SH 2,469, SZ 3,068, BJ 357), including delisted ones with 退市/退 names.
  ST-marked names: SH 132, SZ 161, BJ 3. 359 stocks are absent from the
  master's listed set: 301 are delisted securities and 54 list after
  2026-07-26 (SH 12, SZ 14, BJ 28). One BSE security is in the master but not
  in the snapshot. No symbol-level comparison of SZSE anchor states has been made.
* While publishing ENG_ST30's SZSE rows, the 2026-11-24 SZSE register fetch was
  compared with the certified 2026-07-26 register. Every historical episode
  matched; the only differences were **14 episode closes and 5 new episodes
  after July 2026**. Those 19 boundaries are the validation transitions of §5;
  I have seen their count, not which symbols or dates they are.
* Counts of cninfo 特别处理和退市 / keyword announcements published since 2026-06-26:
  sh 170, sz 306, bj 38. No title of the post-July window has been classified
  or read.
* Fuyao/HiThink (`llms.txt`, re-read 2026-11-24): `stock-basics` is still
  "敬请期待"; nothing there carries short names, ST flags or name history.
  Qlib has no ST history.

## 2. Inputs (identical code for every exchange)

1. **Anchors (dated full-universe names).**
   * A0 = the security master's TickFlow names, retrieved 2026-07-26 19:39 CST
     (a Sunday).
   * A1 = the 2026-11-24 23:14 CST snapshot above.
   * A_t = each later snapshot written by `scripts/u0_st_forward.py snapshot`.
     The daily refresh takes one at 05:00 CST.
   * A snapshot never overwrites the master. A name is ST-marked iff
     `kinds_from_name` (ENG_ST30: NFKC, G/S share-reform prefix stripped,
     leading `*ST` or `ST`) returns a kind.
2. **Events.** cninfo announcements from the ENG_ST30 crawl (category
   特别处理和退市 plus the title-keyword search, all plates), extended to the run
   date. They are classified by the frozen ENG_ST30 classifier and dated by the
   frozen ENG_ST30 date reader and fallback (body date; else announcement + 1
   session, or + 0 for a delisting-period first-day notice). Two pre-filters
   are fixed now; they come from the ENG_ST30 HOLDOUT diagnosis, which is
   disclosed above:
   * a title stating the warning is **not** applied (`不(?:会|用|再|需)?被?(?:实施|实行)`,
     `无需(?:实施|实行)`, `不(?:涉及|触及)`) → IGNORE;
   * a REVOKE title that also continues a warning (`继续(?:实施|实行)` or
     `并(?:继续)?(?:实施|实行)其他风险警示`) → TRANSITION (the security stays
     under a warning).
3. **Dated short names (cninfo).** The `secName` cninfo prints on a disclosure
   is the name in force on the disclosure date (ENG_ST30 §A2). Names come from
   (a) every crawled event announcement and (b) per-company announcement lists,
   all categories, over `[A0 date − 10 days, run date]`. Lists are fetched for
   every symbol in the **active set**: ST-marked in any anchor, OR with any
   non-IGNORE event announced in `[A0 date − 45 days, run date]`, OR whose
   ST-marking differs between any two anchors. The active set is computed
   from anchors and events only, never from a register. Withdrawn filings
   (已取消) are excluded.

## 3. Method (frozen by the commit that records the DEV results, §4)

Per symbol. Sessions come from `runtime/data/u0/pit/trading_calendar.parquet`.

1. **Start.**
   * For a symbol in A0's TickFlow universe, the window starts at the first
     session after A0 (2026-07-27) with A0's state and kinds.
   * A symbol first seen in a later anchor whose listing date falls after A0
     starts at its listing session as not-ST, because a new listing carries
     no risk warning.
   * A symbol in neither case gets no rows.
2. **Path.** Apply events whose effective session is ≥ the start session, in
   order of effective date then announcement date.
   * IMPLEMENT: ST from the effective session on; kinds gain the title's kinds.
   * TRANSITION: ST.
   * REVOKE: ENG_ST30 kinds logic. The warnings still in force are those shown
     by the name on the revocation notice, plus 其他 if it was added inside
     the episode. The state stays ST while any warning remains; otherwise it
     is not-ST from the effective session.
   * DELIST_PERIOD: not-ST from the effective session (the "XX退" name carries
     no ST marker, as in the SZSE register).
   * An event effective before the start session is already reflected in the
     anchor and is skipped.
4. **Observation tolerance.** An observation constrains the state on a small
   set of sessions:
   * Snapshot at instant T (CST):
     * T outside a session day's 09:15–15:00: the last session ending before
       T, and the next session.
     * T inside 09:15–15:00: that session and the next one.
   * cninfo name on disclosure date D: the last session strictly before D,
     the last session on or before D, and the first session after D.
   * The observation is **consistent** if the path's state equals the
     observed state on at least one of those sessions.
5. **Inconsistency → UNKNOWN.** Observations are walked in time order.
   * When observation o is inconsistent, let p be the latest earlier
     consistent observation (the start counts as one).
   * Every session after p's earliest tolerated session, up to o's latest
     tolerated session, becomes UNKNOWN.
   * The path restarts at o's earliest tolerated session with o's state.
     Events effective after that session are re-applied, and the walk
     continues.
   * UNKNOWN sessions produce no evidence rows, so the strict loader refuses
     any market row inside them.
6. **Output.** Each TRUE and FALSE span becomes a row of the strict interval
   contract of `quantagent.data.providers.st_pit` (end date inclusive). Fields:
   * `available_at`: 00:00 CST of the span's first day.
   * `source` and `evidence_id`: the event or observation that opened the
     span.
   * The method: `forward_anchored_ENG_ST31`.
   SSE/BSE rows start at 2026-07-27; earlier days stay without evidence.

## 4. Development (SZSE DEV half only; no thresholds)

* Development uses SZSE securities with an **even** last digit, in
  historical windows only:
  * 2025-07-28→2025-11-24 and 2024-07-29→2024-11-22, the same season as
    validation;
  * 2025-03-03→2025-07-25, annual-report season, as a stress window.
* Each window uses a start anchor equal to the register state on the session
  before it (snapshot at 20:00 CST that evening) and an end anchor equal to
  the register state on its last session (snapshot at 23:00 CST).
* Truth is the certified `runtime/data/u0/pit/st_intervals.parquet`.
* Vendor-anchor check (DEV half only): A0's names against the register state
  on 2026-07-24.
* Code is tested and parameters may be tuned on DEV. Any change to §2–§3 is
  recorded as a dated amendment before validation. The DEV numbers at freeze
  are written into this file, and that commit freezes the method.
* Nothing on the validation window (§5) is computed during development. That
  includes any SZSE row state, any symbol-level SZSE A0/A1 comparison, and any
  register lookup.

## 5. Validation (evaluated once)

* **Set:** all SZSE A-share securities in A0's TickFlow universe, plus SZSE
  listings after A0 found in A1.
* **Window V:** sessions 2026-07-27 … 2026-11-24, which is 81 sessions. A
  symbol's rows run from its start session; listing dates come from A1.
* **Inputs:** exactly the production inputs of §2 (A0, A1, the cninfo crawl
  and per-company names), run by the frozen code.
* **Truth:** the SZSE dated short-name register (`akshare.stock_info_sz_change_name`,
  szse.cn), fetched fresh after the freeze. The state on a session is whether
  the register name in force on that session is ST-marked.
* **Guard:** `evaluate --split validation` refuses if a validation result
  exists or if the method files are uncommitted. It records the method
  commit.

Rows are symbol × session in V. Each row's reconstruction is TRUE, FALSE or
UNKNOWN. **All must hold:**

| id | metric | threshold |
|----|--------|-----------|
| V1 | precision: of rows emitted TRUE, share TRUE in the register | ≥ 0.99 |
| V2 | recall: of register-TRUE rows that are emitted (not UNKNOWN), share emitted TRUE | ≥ 0.99 |
| V3 | transitions: register ST↔not-ST flips on a session of V (the previous session may be 2026-07-24) that the reconstruction reproduces in the same direction within ±1 session, or whose true date lies in an UNKNOWN span | ≥ 0.90 of N (N reported; ~19 expected) |
| V4 | coverage: UNKNOWN rows / all rows | ≤ 0.005 |
| V5 | coverage of warnings: register-TRUE rows that are emitted (not UNKNOWN) / register-TRUE rows | ≥ 0.95 |

Power is limited: with N ≈ 19, V3 tolerates one miss. V1, V2 and V5 are row
measures over roughly 13–16 thousand register-TRUE rows. A single stale
two-month state breaches V1 or V2.

Reported, not gated:
* the vendor-anchor agreement on SZSE at the first session of V;
* the SSE/BSE count of symbols and sessions that are UNKNOWN;
* coverage of the FRESH seed.

## 6. Decision rule

* Validation fails any of V1–V5 → no SSE/BSE rows are written; both stay
  `BLOCKED_BY_DATA`, the FRESH catch-up keeps refusing, and every number is
  recorded here.
* Validation passes → `publish` writes:
  * SSE and BSE rows from 2026-07-27 onward;
  * SZSE rows from the exchange register, unchanged;
  * all of it into `runtime/data/u0/pit/st_pit_evidence.parquet` and
    `st_intervals_v2.parquet`, with `st_manifest_v2.json` naming this
    pre-registration and the validation result.

  UNKNOWN spans stay without rows. The catch-up's existing fail-fast check then
  decides: it appends only when every seed symbol is covered on every day.
  Otherwise it refuses and names the symbols. This study does not relax that
  check or exclude symbols.
* No threshold, window, set or rule changes after any validation number is seen.

## 7. Ongoing monitor (fail-closed, daily)

Every daily `publish` re-runs the frozen method on SZSE and compares it with
that day's register over all sessions from 2026-07-27 to the as-of date. That
span includes V, so one day cannot swing it.

* If V1, V2 or V5 falls below its threshold, or V4 exceeds its bound, SSE/BSE
  rows are withheld.
* The manifest records why.
* Reinstating them needs a new dated amendment or pre-registration.

## 8. Limits stated now

* The method assumes that SSE and BSE companies disclose risk-warning changes
  on cninfo and that cninfo short names are point-in-time in the same way as
  for SZSE companies. The rules for both are CSRC rules common to all
  exchanges. Validation measures SZSE only.
* A full implement→revoke round trip inside the window, with both notices
  missing from the crawl and no snapshot in between, would go unseen.
  Daily snapshots shrink this from 2026-11-25 on.
* `trading_calendar.parquet` ends 2026-12-31. The refresh must extend it before
  the year ends, or the build refuses.
* This study does not touch the certified `pit/st_intervals.parquet`, the
  ENG_ST30 results, or any quarantined evaluation window. FRESH bars are
  appended only by the existing catch-up under its existing rules.

## Amendment A1 — 2026-11-25, committed before validation (freezes the method)

Nothing in §2.1 (anchors), §5 (set, window, thresholds) or §6–§7 changes. Clarifications
and additions made while implementing and running DEV:

1. **Pre-filter scope.** The negation rule applies to events the frozen classifier marks
   IMPLEMENT, the continuation rule to events it marks REVOKE. A full revocation such as
   "撤销…退市风险警示且不实施其他风险警示" therefore stays a revocation.
2. **Fail closed on undated events.** An event announced on or after the first anchor
   session that has no effective session (possible only when the trading calendar ends
   first) makes the build refuse instead of skipping the event.
3. **Before validation, `build` runs on SSE/BSE only.** It refuses SZSE until a
   validation result exists (§4).
4. **Evaluation operationalisation**, as implemented in `st_forward.evaluate_forward`
   and `scripts/u0_st_forward.py evaluate`:
   * Rows include the anchor session (2026-07-24), but only so that a flip on the first
     session of V can be seen; row metrics count V only.
   * For a security listed after A0, truth rows start at its listing session.
   * Truth = `merge_contiguous` of the episodes that
     `u0_pit_intervals._st_intervals_from_name_history` folds from a fresh register fetch.
   * Information cut-off: events and names disclosed on or before 2026-11-24.
     Snapshots: A1 only.
   * Also reported, not gated: `V3_exact` (same session) and `V3_via_unknown`.
5. **Daily refresh.** `u0_st_register_refresh.sh` takes a snapshot, fetches company names
   and publishes through `u0_st_forward.py publish`. That command runs the unchanged
   ENG_ST30 SZSE publish first, then appends SSE/BSE rows per §6–§7. The ENG_ST30
   name-probe step is dropped, because nothing published depends on it.
6. §3 has no item 3: the committed numbering is kept so that references stay valid.
7. **DEV results at freeze** (`runtime/data/u0/pit/_st_forward/dev_eval_20261125T003641.json`;
   SZSE even codes; anchors simulated from the register; cninfo names fetched per company
   for the active sets — 103 / 132 / 122 companies):

   | window | securities × sessions | V1 | V2 | V3 (exact) | V4 | V5 | restarts |
   |---|---|---|---|---|---|---|---|
   | 2025-07-28→2025-11-24 | 1,384 × 80 | 1.000 | 1.000 | 7/7 (7) | 0 | 1.000 | 0 |
   | 2024-07-29→2024-11-22 | 1,375 × 78 | 1.000 | 1.000 | 6/6 (6) | 0 | 1.000 | 0 |
   | 2025-03-03→2025-07-25 | 1,380 × 100 | 1.000 | 1.000 | 56/56 (56) | 0 | 1.000 | 0 |

   Vendor anchor: A0's TickFlow names on the 1,393 DEV-half securities agree with the
   register on 2026-07-24 for every security (63 ST in both, 0 disagreements).
   Ablation on the stress window, for the record: with **no events**, the observations
   catch all 56 transitions as contradictions (56 restarts, 356 UNKNOWN rows = 0.26%,
   0 silently wrong rows); with events only (no observations) the result is unchanged.
   The events part is in-sample here — the ENG_ST30 classifier and date rules were
   calibrated on this half — which is why validation on unseen data is the test.
   The method is frozen by the commit that adds this line; validation runs once on it.
