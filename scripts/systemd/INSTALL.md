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
