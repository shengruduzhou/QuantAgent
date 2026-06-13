# Installing the QuantAgent daily health timer

## User-level install (recommended — no root required)

```bash
# Copy units to the user systemd directory
mkdir -p ~/.config/systemd/user
cp quantagent-health.service quantagent-health.timer quantagent-health-alert@.service \
   ~/.config/systemd/user/

# Edit WorkingDirectory / CONDA_PREFIX in the .service if paths differ
# Then enable and start
systemctl --user daemon-reload
systemctl --user enable --now quantagent-health.timer

# Check status
systemctl --user status quantagent-health.timer
systemctl --user status quantagent-health.service

# View logs
journalctl --user -u quantagent-health.service -n 50
```

## System-level install (root required)

```bash
sudo cp quantagent-health.service quantagent-health.timer quantagent-health-alert@.service \
     /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quantagent-health.timer
```

## One-shot manual run

```bash
# From the repo root:
./scripts/daily_health_check.sh

# Or via the CLI directly:
python -m quantagent.cli health-check-v7 --no-write

# Exit codes: 0=OK, 1=WARN, 2=FAIL
```

## Customising alerting

The `quantagent-health-alert@.service` unit writes a plaintext alert file to
`runtime/reports/daily_health/ALERT_<timestamp>.txt`.  Replace its `ExecStart`
line with a `curl` POST to a Slack webhook, an SMTP send, or any other
notification mechanism your team uses.

---

# QuantAgent daily + monthly research pipeline timers

Two additional timers drive the LLM+factor research loop:

- **quantagent-daily** — pre-open (Mon–Fri 09:00): 每日舆情短线推断 + 国家队/债市刷新.
  Outputs `runtime/reports/daily/sentiment_brief_<date>.md`.
- **quantagent-monthly** — 1st of month 00:00: 红头文件爬虫 + LLM 十五五政策研判 +
  投行研报 + 融合证据 + LLM+因子混合股池 + 月度研报.
  Outputs `runtime/reports/monthly/research_report_<YYYYMM>.md` (选股池参考).

```bash
mkdir -p ~/.config/systemd/user
cp quantagent-daily.service quantagent-daily.timer \
   quantagent-monthly.service quantagent-monthly.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now quantagent-daily.timer quantagent-monthly.timer
# run once now to verify:
systemctl --user start quantagent-daily.service
journalctl --user -u quantagent-daily.service -n 50 --no-pager
```

Notes:
- For a **0点 (midnight)** daily refresh instead of pre-open, set `OnCalendar=*-*-* 00:00:00` in `quantagent-daily.timer`.
- LLM env (`QUANTAGENT_LLM_*`, `google_API_KEY`) is read from the repo `.env`; the units also set the LLM vars explicitly.
- Pipelines are research-only and never emit live orders.

## Forward daily loop (post-close inference + A/B/C books + 反T watchlist)

```bash
cp quantagent-forward.service quantagent-forward.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now quantagent-forward.timer
# verify
systemctl --user list-timers | grep forward
# logs land in runtime/logs/forward/forward_<date>.log
```

Before first enable, run the one-time pipeline validation:

```bash
AI_quant_venv/bin/python3 scripts/forward_daily_inference.py --validate
# 2026-06-12 measured state: mean spearman ≈ 0.71 vs the training-run
# composite (NOT ≥0.95). Known cause: 11 alpha101 columns drifted in the
# v8.2 factor refactor after the dataset build — see the docstring of
# forward_daily_inference.py. Books keep rolling under this caveat; full
# fidelity needs the factor fix or a v8.9 retrain on rebuilt features.
```
