# Claude entrypoint

Read `AGENTS.md` first and follow every repository safety/PIT rule. Then read:

- `docs/ARCHITECTURE.md` (what actually runs), `docs/GOVERNANCE.md` (quarantine,
  trust classes, acceptance rules) and `docs/ROADMAP.md` (current priorities)
- `docs/quant_foundations_external_sources.md`

Write new findings into the existing documents (see AGENTS.md "文档只有一套");
do not create per-round audit directories or root-level reports.

Before modifying A-share data adapters, factor fusion, nonlinear/interactions,
model-comparison flow, or the market workbench, use web access to refresh the
pinned upstream sources in that document. In particular, re-read:

- https://fuyao.aicubes.cn/llms.txt
- https://fuyao.aicubes.cn/llms-full.txt
- https://fuyao.aicubes.cn/docs/
- https://github.com/HiThink-Tech/Financial-API
- https://github.com/microsoft/qlib
- https://qlib.org.cn/en/latest/

Do not read, print, commit, or echo values from `.env`. Credentials are consumed
only through allowlisted environment variable names. The Fuyao/HiThink variable
is `HITHINK_FINANCE_API_KEY`.

Do not use mock data to make a production/research result appear complete. A
missing source/capability remains explicit and fail-closed.
