---
name: quantagent-cycle
description: QuantAgent development-cycle protocol - how the lead coordinates departments, how each agent sets up an isolated worktree, commits, records evidence, and hands work to independent reviewers. Use at the start of any multi-step research or engineering cycle, or when spawning/briefing a department agent.
---

# QuantAgent development cycle / 开发周期协议

The organization (agents in `.claude/agents/`):

| Department | Agent | Writes code? | Approves? |
|---|---|---|---|
| Executive / architecture | the lead session (chair) | integration only | merge decision after reviews |
| Quantitative research | `quant-researcher` | yes (own worktree) | never its own work |
| Data engineering & governance | `data-engineer` | yes | never its own work |
| Software engineering / FDE | `software-engineer` | yes | never its own work |
| Product / UX | `product-ux` | yes | never its own work |
| Risk & audit — math/stat/PIT/finance | `math-stat-auditor` | no (read-only) | veto |
| Risk & audit — code/security/perf | `code-security-auditor` | no (read-only) | veto |
| QA & release | `qa-release` | no | release verification |

## Cycle

1. **Inspect** HEAD, `git worktree list`, cron/runtime state (`docs/OPERATIONS.md` §1),
   CI (`curl -s https://api.github.com/repos/shengruduzhou/QuantAgent/actions/runs?per_page=8`),
   `docs/ROADMAP.md`, recent `docs/audits/HISTORY.md`.
2. **Independent investigation**: auditors look before seeing proposals (no anchoring).
3. **Prioritize** in `docs/ROADMAP.md` (data integrity > research validity > engineering > UI).
4. **Assign** one owner per item and one worktree per owner; no two writers on the same files.
5. **Implement** with tests that fail on the old code.
6. **Review**: every substantial diff gets `code-security-auditor`; anything touching
   numbers, labels, data, evaluation or trading rules also gets `math-stat-auditor`.
   REJECT blocks merge; disagreements are recorded with evidence in HISTORY.
7. **QA** (`qa-release`) on the integrated branch; merge with `--no-ff`; push; verify the
   pushed SHA's GitHub Actions run is green.
8. **Reassess**: update ROADMAP/RESEARCH/DATA/HISTORY in place (no new report files),
   and memory notes for the next session.

Concurrency budget: ≤ 3 Opus agents at once (more hits usage limits); read-only audits
may use `sonnet`. Resume an interrupted agent with `SendMessage` instead of restarting.

## Worktree setup (every writing agent)

```bash
git -C /home/shanhefu/QuantAgent worktree add -b cycle<N>/<topic> /home/shanhefu/QuantAgent-wt/<topic> main
export PYTHONPATH=/home/shanhefu/QuantAgent-wt/<topic>/src:/home/shanhefu/QuantAgent-wt/<topic>
PY=/home/shanhefu/QuantAgent/AI_quant_venv/bin/python3   # editable install points at main; PYTHONPATH makes the worktree win
$PY -c "import quantagent; print(quantagent.__file__)"   # must print the worktree path
```

- Persistent worktrees are created by the chair under `/home/shanhefu/QuantAgent-wt/`
  (the built-in `isolation: worktree` deletes clean trees and loses context on interrupt).
- Commit after every meaningful step. **Never `git stash`** (refs/stash is shared by all
  worktrees). Never push from an agent; the chair integrates.
- Real data lives in `/home/shanhefu/QuantAgent/runtime/` (gitignored). Read it by
  absolute path; write new outputs to a new subdirectory; never overwrite certified
  artifacts. Tests must never write the real runtime.
- Long jobs: `nohup … > log 2>&1 &`, then poll; record wall time and peak RSS.
- If a file write is refused, return the content in your final message; the chair saves it.

## Evidence

- Raw audit material: `runtime/archives/audits/cycle<N>/` (gitignored).
- Conclusions: `docs/ROADMAP.md`, `docs/RESEARCH.md`, `docs/DATA.md`, `docs/audits/HISTORY.md`,
  `docs/research/ledgers/`, `docs/research/preregistrations/`, ADRs.
- Every quoted number: {window, trial count N, trust class, artifact path}.
