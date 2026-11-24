---
paths:
  - "src/quantagent/data/**"
  - "scripts/u0_*.py"
  - "scripts/catchup_*.py"
  - "scripts/*st_register*"
---

# Data adapter rules

- Before changing an A-share adapter, re-read the upstream sources pinned in
  `CLAUDE.md` / `docs/quant_foundations_external_sources.md` (web access).
- Units are contracts verified on real payloads: volume = shares (vendor 手 ×100 at the
  adapter boundary), amount = CNY, panel prices raw. Detect payload shape per response.
- Every stored row carries `source`, `source_endpoint`, `retrieved_at`, `available_at`,
  `quality_status`. No cross-vendor splicing without a `SourceBoundary`.
- ST, names, industry and constituents must be dated PIT evidence; broadcasting a
  current snapshot into history is forbidden. Missing evidence fails closed.
- Readiness gates are derived from artifacts (`quantagent.data.ashare.readiness`);
  never constants, NOT_RUN ≠ PASS, partial coverage = BLOCKED.
- Cron jobs read these modules: check `crontab -l` and running processes before
  changing behaviour they depend on.
