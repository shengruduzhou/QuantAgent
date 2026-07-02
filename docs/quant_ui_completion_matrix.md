# Quant UI 完成矩阵 / Completion Matrix

> 本文件用于对照原始八步开发顺序与验收条件。状态必须以当前代码、真实 runtime 或测试结果为证据。

## 1. Development steps

| Step | Status | Evidence |
|---|---|---|
| 1. 只读分析 | Complete | `docs/quant_ui_code_map.md` |
| 2. 数据 schema | Complete | `docs/quant_ui_data_schema.md`、Pydantic schema tests |
| 3. Backend adapter/API | Complete | `services/quant_api/`、required-route audit、API smoke |
| 4. Frontend foundation | Complete | `apps/quant-ui/` React/TypeScript/Vite shell、API client、query cache、charts、empty/error states |
| 5. Core pages | Complete | 11 routes covering Dashboard through Settings |
| 6. Linked interactions | Complete | K-line/trade selection、Decision Inspector、filters、search routing、export、fullscreen |
| 7. Test and repair | Complete | Backend/core 28 tests、frontend typecheck/3 tests/build、desktop/mobile browser verification |
| 8. Final handoff | Complete | `docs/quant_ui_runbook.md`、`design-qa.md`、本矩阵 |

## 2. Backend capability evidence

| Requirement | Status | Current evidence |
|---|---|---|
| Runtime index | Complete | Cached metadata index; current live snapshot约 9,000 displayable artifacts |
| Parser registry | Complete | JSON、CSV、Parquet、log、metadata-only binary parsers |
| Backtest list/detail | Complete | 20+ current experiments discovered；live runtime continues to add runs |
| Equity/drawdown | Complete when artifact exists | `pnl.csv` / `nav.csv` mapping |
| Standard trades | Complete when order blotter exists | schema-gated `side/status/quantity/price` mapping |
| Research event safety | Complete | non-order `trades.csv` returns `unsupported_trade_schema` |
| Stock K-line | Complete | real silver market panel, symbol/date filtering, bounded window |
| Stock replay | Complete with availability flags | bars/trades/signals/positions/PnL only when source exists |
| Do-T analysis | Complete for available pair artifacts | 24 current indexed sources; daily-only artifacts remain explicitly partial |
| Factor catalog | Complete | 422 current factors from registry/source/runtime |
| Factor independent trades | Explicitly unavailable where absent | no multi-factor trades are reused as factor trades |
| Selection runs | Complete | 10 current hybrid selection runs |
| Model catalog | Complete | 20 current models across Deep Alpha、registered alpha、RL、T+1 bundles and generic artifacts；checkpoint binary content is not exposed |
| Risk center data | Complete for current artifacts | code-derived rules, events, stock PnL risk |
| Jobs | Complete | allowlisted backtest/train/infer commands、project path enforcement、SSE、Web Control Center |
| Runtime cleanup | Complete | backend-generated candidates、protected roots、DELETE confirmation、audit report；62.29 MB safe cleanup executed |
| Missing/empty data | Complete | `ready/partial/empty/error` envelope and empty-runtime tests |

## 3. Current verification

```bash
python3 -m pytest -q tests/quant_ui
# 20 passed

python3 -m pytest -q tests/quant_ui tests/test_v7_architecture_contracts.py tests/test_v8_pipeline_e2e.py
# 28 passed

python3 scripts/quant_ui_api_smoke.py
# all checks passed

cd apps/quant-ui
npm run typecheck
npm run test
npm run build
# all checks passed
```

Additional checks：

- `python3 -m compileall` passed。
- `git diff --check` passed。
- Required API paths are present；application currently exposes 60 routes。
- Quant UI source/docs/tests contain no project-external hardcoded local path。
- Browser verification covered Model Lab、Runtime Explorer、Safe Cleanup、Control Center、command palette and Stock Replay。
- Responsive verification covered Model Lab and Stock Replay at 390 × 844，document-level horizontal overflow=false。
- Design comparison evidence：`runtime/reports/quant_ui/qa/model-lab-side-by-side-1800x760.png`。

## 4. Acceptance notes

- 主视觉使用 Research Workbench，已融合 command-grid 交易明细/风险暴露与 signal-observatory 风险雷达/健康/漏斗/Top 贡献。
- 所有做 T 页面与提示统一为 A-share `T+1` 语义，不提供 T+0 execution。
- 当前 runtime 缺失的逐笔字段保持 empty/unavailable，不用模拟数据补齐。
- Live trading 仍默认关闭；Web job layer 不暴露 QMT live enable path。
