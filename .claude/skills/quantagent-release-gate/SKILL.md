---
name: quantagent-release-gate
description: Release gate checklist for merging a QuantAgent branch into main - which independent reviews are required, which tests must run, and how to verify the pushed CI. Use before any merge or push to main.
---

# Release gate / 合入闸门

A branch merges only when every applicable line is satisfied. Record the result in the
merge commit message and in `docs/audits/HISTORY.md`.

| Gate | Required when | Evidence |
|---|---|---|
| Financial & logical correctness | trading rules, costs, portfolio, paper/execution | `math-stat-auditor` APPROVE |
| Mathematical / statistical validity | metrics, IC, DSR/PSR/PBO, bootstrap, labels | `math-stat-auditor` APPROVE with recomputed numbers |
| Data integrity / no leakage | data adapters, PIT fields, universes, datasets | `math-stat-auditor` or `data-engineer` (not the author) |
| Code quality & architecture | every substantial diff | `code-security-auditor` APPROVE |
| Security & permissions | workflows, deps, API, JobRunner, credentials | `code-security-auditor` APPROVE |
| Performance | full-universe paths | measured before/after (time, RSS) |
| Functional / UX | `apps/quant-ui` | tsc + vitest + build + browser screenshots |
| Reproducibility | research results | pre-registration commit precedes result commit; ledger row |
| Tests | always | full pytest counts; new tests fail on old code |
| CI | after push | pushed SHA green on GitHub Actions |

Hard rules: no weakened assertions, no deleted tests, no relaxed gate thresholds, no
quarantine/FRESH access outside a pre-registered final read, no synthetic data in
results, live trading stays `LIVE_DISABLED`.

Readiness levels are separate and must be stated separately: engineering, data,
research validity, paper-trading, live. A green release changes only the first.

Merge procedure (chair):

```bash
git -C /home/shanhefu/QuantAgent merge --no-ff cycle<N>/<topic>
AI_quant_venv/bin/python3 -m pytest tests/ -q -p no:cacheprovider   # on the merged tree
git push origin main      # SSH over 443 is the working remote path on this host
curl -s "https://api.github.com/repos/shengruduzhou/QuantAgent/actions/runs?per_page=8"
```
