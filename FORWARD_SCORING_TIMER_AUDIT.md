# FORWARD_SCORING_TIMER_AUDIT — v8.8 forward 打分路径处置（P-F，2026-07-03）

## 1. 干跑调查结果（dry-run report）

| 检查项 | 实测 | 结论 |
|---|---|---|
| systemd（user + system）`list-timers` | 无任何 quantagent timer | **forward timer 从未安装/已卸载** |
| `scripts/systemd/quantagent-forward.{service,timer}` | 仅存在于仓库，指向 `run_forward_daily.sh` | 存在**误安装风险** |
| crontab | 仅 2 项：`run_dot_fundflow_daily.sh`(16:40) 与 `stage10_daily.sh`(17:30) | 两者均不调 forward 链（dot-fundflow 头注明确 "does NOT resurrect the full forward"） |
| `runtime/reports/v8/forward/` | 只有 `validate_merged.parquet`，mtime **2026-06-13** | forward 打分**已 3 周未运行**，无每日追加 |
| 受污染链定位 | `run_forward_daily.sh` → `forward_daily_inference.py`（硬钉 `v88_judgment_20260611_2015`）→ `forward_book_update / forward_rl_book / forward_paper_log` | 唯一 v8.8 消费路径 |

**结论：无活动进程可"停止"；处置目标从 stop 变为防复活（fail-fast 化）。**

## 2. 已实施（最小 diff）

1. `scripts/run_forward_daily.sh`：入口 fail-fast —— 默认打印禁用原因并 `exit 2`；唯一逃生阀 `QUANTAGENT_ALLOW_DEPRECATED_FORWARD=1`（显式、可审计）。验证：`bash -n` 通过；实跑 exit 2 + 双行说明。
2. `scripts/systemd/quantagent-forward.service`：Unit 头部加 DISABLED 注释块 + Description 改为 DISABLED 提示，防止未来 `systemctl enable` 时误装。
3. `forward_daily_inference.py` 的运行时警告（Stage C 已加）作为第三层防线。

## 3. 未动的健康任务（明确不受影响）

`quantagent-daily/weekly/monthly/health` 单元（未安装，保持原样）；cron 的 dot-fundflow 监控与 stage10 概念链日扫 —— 均不经过 v8.8 推理。

## 4. 替代路径（replacement note）

生产 blend 的唯一合法物化命令：

```bash
AI_quant_venv/bin/python3 scripts/materialize_production_composite.py --config configs/production_blend.json
```

真正的日频 forward 恢复前置条件 = **P6**（`PRODUCTION_RECONCILIATION_PATCH_PLAN.md`）：
① 修复 11 个不可复现 alpha 列的特征保真（overlap spearman >0.99）；② `forward_daily_inference.py` 改读 `configs/production_blend.json`（run dir + 2-sleeve rank blend）；③ forward 输出按 EVALUATION_PROTOCOL_V2 §5 进 append-only paper log。

## 5. 残余风险

- 有人手动 `QUANTAGENT_ALLOW_DEPRECATED_FORWARD=1` 跑旧链 → 输出仍带脚本 stderr 警告，但 parquet 本身无毒 stamp（P6 时把 `model_generation=v8.8-DEPRECATED` 写进输出行——已列 P6 范围）。
- `runtime/reports/v8/forward/ensemble_forward.parquet` 历史文件不存在于当前盘（仅 validate_merged）：历史 forward 分数已无消费方。
