# ENG_ST30 — SSE/BSE risk-warning (ST) register: method and PRE-REGISTRATION

Status: PRE-REGISTERED. This file is committed before any reconstruction is run
on any exchange. Thresholds and the decision rule below are not changed after
results are seen. If the hold-out fails, SSE/BSE stay `BLOCKED_BY_DATA`.

## 0. What has been looked at before this commit (full disclosure)

* `runtime/data/u0/pit/st_manifest.json` (sources already rejected; not re-litigated).
* Per-year *counts* of cninfo announcements in category 特别处理和退市
  (`category_tbclts_szsh`) by plate (sh / sz / bj). Counts only, no titles parsed:
  SH 4,345 (2001–2026), SZ 7,192 (2001–2026), BJ 55 (2025–2026 only).
* Five SZSE/SSE announcement records from 2023-04-25..2023-05-10 (probe), and the
  `secName` of 002102 announcements in four windows (to establish that cninfo's
  `secName` is the name *at announcement time*: 冠福家用 → *ST冠福 2012-05 →
  冠福家用 2013-05; ST冠福 2018-11..2023-05; 冠福股份 2023-06).
* One SSE announcement PDF (603268, 2025-04-28) whose body states
  "停牌日期为2025年4月28日 / 实施起始日为2025年4月29日".
* No SSE episode, SSE row state, or any SZSE agreement number has been computed.

## 1. Source

Primary: **cninfo 巨潮资讯网** (the CSRC-designated disclosure platform for
SSE, SZSE and BSE listed companies), announcement search endpoint
`POST http://www.cninfo.com.cn/new/hisAnnouncement/query`
(`column=szse, tabName=fulltext, category=category_tbclts_szsh, plate∈{sz,sh,bj}`),
monthly windows 2000-01 … 2026-07, every page. This is the endpoint wrapped by
`akshare.stock_zh_a_disclosure_report_cninfo`. Raw JSON is cached per
(plate, month, page). The announcement PDF (`http://static.cninfo.com.cn/<adjunctUrl>`)
is fetched for announcements the title classifier marks as candidate
state-change events, to read the effective date stated in the body.

`available_at` of an event = the cninfo announcement date (exchange disclosure
date; the stock is normally suspended that session). The effective date is the
first session traded under the new regime.

## 2. Reconstruction (identical code for every exchange)

1. Title classifier → `IMPLEMENT` (实施/实行 退市风险警示 / 其他风险警示 / 特别处理),
   `REVOKE` (撤销/撤消 … with no continuing warning), `TRANSITION`
   (撤销退市风险警示 并/后 实施其他风险警示, 变更为 …: stays inside the episode),
   or `IGNORE` (可能被…, 风险提示, 进展, 申请, 意见, 法律意见书, 核查, 说明, 补充/更正
   duplicates, …).
2. Effective date: (a) the date stated in the body (实施起始日 / 撤销起始日 / 自…起 /
   复牌日); (b) else a fallback rule calibrated on the SZSE DEV half only;
   (c) else the span is UNKNOWN.
3. Per-symbol state machine over events. A revoke with no observable
   implementation, an implementation whose date cannot be fixed, or contradictory
   sequences produce an **UNKNOWN** span, never a guessed TRUE/FALSE.
4. Semantics: `effective_start` inclusive, `effective_end` exclusive (the first
   session no longer under the warning) — the SZSE register's 变更日期 semantics.

## 3. Validation split (fixed now)

SZSE securities in `security_master.parquet`: **last digit of the 6-digit code
even → DEV, odd → HOLDOUT.** All method development, title rules and fallback
calibration use DEV only. The method code is committed (frozen) before HOLDOUT is
evaluated, and HOLDOUT is evaluated once.

Ground truth = the SZSE dated short-name register already in
`runtime/data/u0/pit/st_intervals.parquet` (start inclusive, end exclusive).

Window = the certified panel window **2016-01-04 … 2026-07-24**; rows = the
sessions present in `runtime/data/u0/panel/daily_bars_raw.parquet`.

## 4. HOLDOUT acceptance thresholds (all must hold)

| id | metric | threshold |
|----|--------|-----------|
| M1 | episode recall: register episodes overlapping the window that overlap a reconstructed TRUE episode of the same symbol | ≥ 0.95 |
| M2 | episode precision: reconstructed TRUE episodes overlapping the window that overlap a register episode | ≥ 0.95 |
| M3 | start-date agreement of matched episodes (trading sessions): exact or ±1 session | ≥ 0.95 |
| M4 | end-date agreement of matched closed episodes: exact or ±1 session | ≥ 0.95 |
| M5a | row recall: register-TRUE rows that the reconstruction marks TRUE (UNKNOWN counts as a miss) | ≥ 0.98 |
| M5b | row precision: reconstruction-TRUE rows that the register marks TRUE | ≥ 0.98 |
| M5c | register-TRUE rows the reconstruction marks FALSE (silent false negatives) | ≤ 0.01 |

## 5. SSE independent consistency checks (thresholds fixed now)

Price data: raw bars; main board only (SH_Main vs calibration SZ_Main);
2016-01-04 … 2026-07-03 (before the 2026-07-06 main-board ST band reform);
rows with volume > 0 whose previous row is the immediately preceding trading
session with volume > 0, and whose date is not an `ex_date` in
`corporate_actions.parquet`. ST band prices: `round_half_up(prev_close×(1±0.05), 2)`.

* **C1 hard contradiction** — share of reconstructed-TRUE SSE rows whose high or
  low lies outside the 5% band. PASS if ≤ max(2 × the same share on SZSE
  register-TRUE SZ_Main rows, 0.005).
* **C2 missed-ST screen** — an "ST-like month" (symbol × calendar month, ≥ 10
  valid rows) has ≥ 2 closes exactly at a 5% band price and no row outside the
  5% band. Unexplained = ST-like month in which < 50% of rows are TRUE.
  PASS if unexplained_SSE / ST-like_SSE ≤ max(2 × unexplained_SZSE /
  ST-like_SZSE (truth = register), 0.05).
* **C3 current snapshot** — of SSE symbols in `st_current_snapshot.parquet`
  (TickFlow names, 2026-07-26) with a panel row on 2026-07-24, share whose
  reconstruction is TRUE on that row ≥ 0.95; and of SSE symbols reconstructed
  TRUE on 2026-07-24, share not in the snapshot ≤ 0.05.
* D (informational only): `secName` of implement / revoke announcements
  (should be non-ST / ST respectively).

## 6. BSE

BSE uses the same code. BSE price bands are 30% for ST and non-ST, so C1/C2 do
not apply. BSE coverage is declared only if: HOLDOUT passes, C3 holds for BSE
(every snapshot ST name TRUE on 2026-07-24, no reconstructed-open name outside the
snapshot), and the crawl returns announcements under plate `bj`.

## 7. Decision rule

* HOLDOUT fails any of M1–M5c → no SSE/BSE rows are written; both stay
  `BLOCKED_BY_DATA`; every source and number is recorded.
* HOLDOUT passes and C1, C2, C3 pass → SSE episodes are written to a NEW file
  `runtime/data/u0/pit/st_intervals_v2.parquet` (+ `st_manifest_v2.json`), with
  per-row provenance (source, endpoint, announcement id/url, announcement date,
  available_at, effective_start/end, method) and UNKNOWN spans kept as UNKNOWN.
  SZSE rows are copied from the exchange register unchanged.
* HOLDOUT passes but any of C1–C3 fails for SSE → SSE stays `BLOCKED_BY_DATA`.
* BSE per §6, independently.
* No threshold, window or split is changed after any result is seen.

## Amendment A1 — 2026-11-24, committed before any DEV or HOLDOUT metric was computed

Nothing in §3 (split), §4 (thresholds), §5 (check thresholds) or §7 (decision rule)
changes. Additions and clarifications only:

1. **Second query on the same source.** A probe on 2026-11-24 showed that the BSE
   implementation notice of 920023 (2026-04-28, "关于公司股票交易将被实施退市风险警示的
   提示性公告") is *not* filed under category 特别处理和退市. The crawl therefore also pages
   the cninfo title-keyword search (`searchkey` ∈ {风险警示, 特别处理, 退市整理}, which
   matches titles only) per plate and month; both result sets are merged by
   `announcementId`. Disclosure of what was seen in that probe: the 14 titles cninfo
   returns for 920023 with keyword 风险警示 (2020→2026), the two 920023 titles of
   2026-04 that mention 警示/停牌, three titles per plate from 2026-10 and the
   per-plate keyword counts for 2024-04 and 2026-04. No SSE or SZSE episode was
   computed.
2. **Crawl end** extended to the run date (2026-11-24) so the register also covers
   the frozen fresh window. The evaluation window of §3 is unchanged.
3. **Metric operationalisation** (as implemented in
   `quantagent.data.ashare.st_announcements.evaluate_reconstruction`): register rows
   that touch (a rename inside one warning) are merged into one episode; an episode
   overlaps the window if it covers any calendar day of it; each register episode is
   matched to the same-symbol reconstructed TRUE episode with the largest calendar
   overlap; M3 is measured on matched pairs whose register start lies in the window,
   M4 on matched pairs whose register end lies in the window; distances are trading
   sessions of the SSE/SZSE calendar.
4. **HOLDOUT once.** `scripts/u0_st_announcements.py evaluate --split holdout` refuses
   to run if the method files are uncommitted or if a HOLDOUT result already exists,
   and records the method commit.
5. **Outputs.** In addition to `st_intervals_v2.parquet`, `publish` writes
   `runtime/data/u0/pit/st_pit_evidence.parquet` in the strict interval contract of
   `quantagent.data.providers.st_pit` (complete TRUE *and* FALSE intervals per listed
   A-share from 2016-01-04 to the as-of date; an UNKNOWN span gets no row, so any market
   row inside it is refused rather than read as FALSE). SZSE rows come from a freshly
   fetched SZSE register (`refresh-szse`); the certified `pit/st_intervals.parquet` of
   2026-07-26 is not modified and remains the HOLDOUT ground truth.
