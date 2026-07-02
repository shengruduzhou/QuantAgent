# QuantAgent Quant UI Runbook

## Recommended startup

项目根目录执行一个命令：

```bash
./scripts/run_quant_ui.sh
```

脚本会在需要时安装 frontend dependencies、构建 production SPA，并由同一个
FastAPI process 同时提供 Web 与 API：

```text
Web:     http://127.0.0.1:8000
API:     http://127.0.0.1:8000/api
OpenAPI: http://127.0.0.1:8000/docs
```

不再要求同时维护 `8000 + 5173` 两个进程。控制中心页面还可以启动 allowlisted
backtest/train/infer research jobs。

## Backend API

### Install

在项目根目录执行：

```bash
python3 -m pip install -e ".[web]"
```

如当前环境已安装 FastAPI、Uvicorn 与 HTTPX，可直接启动。

### Start

```bash
./scripts/run_quant_ui_api.sh
```

等价命令：

```bash
python3 -m services.quant_api
```

默认监听：

```text
http://127.0.0.1:8000
```

环境变量：

```bash
QUANT_UI_HOST=127.0.0.1
QUANT_UI_PORT=8000
QUANT_UI_RELOAD=false
```

OpenAPI：

```text
http://127.0.0.1:8000/docs
```

### Smoke check

无需启动 HTTP port：

```bash
python3 scripts/quant_ui_api_smoke.py
```

### Runtime Explorer examples

```bash
curl "http://127.0.0.1:8000/api/system/runtime-index?kind=backtest&pageSize=20"
curl "http://127.0.0.1:8000/api/system/runtime-index?runId=v89_rankfix_20260613_1044&horizon=short_5d"
curl "http://127.0.0.1:8000/api/system/runtime-index?modifiedAfter=2026-06-01T00:00:00%2B00:00"
```

`strategy`、`model`、`symbol` filters 只匹配 artifact metadata/path，不会为搜索而扫描大型 Parquet 内容。
股票级检索使用 backtest/model/selection domain API。

### Research job example

Job API 不接受 shell command，只接受 allowlisted command ID。所有 input/output path 必须位于项目内部，
output 还必须位于 `runtime/`：

```bash
curl -X POST "http://127.0.0.1:8000/api/jobs/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "commandId": "run-strict-a-share-backtest-v8",
    "parameters": {
      "target_weights_path": "runtime/reports/example/target_weights.parquet",
      "market_panel_path": "runtime/data/v7/silver/market_panel/market_panel.parquet",
      "output_dir": "runtime/reports/quant_ui_jobs/example_backtest"
    }
  }'
```

任务状态、日志与 SSE：

```text
GET /api/jobs/{job_id}
GET /api/jobs/{job_id}/logs
GET /api/jobs/{job_id}/stream
```

### Safety

- API 默认 read-only。
- Job API 只接受 allowlisted QuantAgent research commands。
- Job output 必须位于项目 `runtime/`。
- 不提供 live-trading/QMT enable command。
- Model checkpoint 只返回 metadata，不返回 binary content。
- API path 输出统一转换为项目相对路径。

## Frontend

### Install

首次启动时在项目根目录执行：

```bash
cd apps/quant-ui
npm install
```

### Start

仅启动 frontend：

```bash
./scripts/run_quant_ui_frontend.sh
```

推荐的 production-style integrated startup：

```bash
./scripts/run_quant_ui.sh
```

默认访问：

```text
http://127.0.0.1:8000
```

仅在 frontend development mode 下使用 Vite：

```bash
./scripts/run_quant_ui_frontend.sh
```

Development mode 默认使用 `http://127.0.0.1:5173`，并代理 `/api` 到
`http://127.0.0.1:8000`。如 backend 不在默认地址，可设置：

```bash
VITE_API_BASE=http://127.0.0.1:8000/api npm run dev
```

生产构建与本地预览：

```bash
cd apps/quant-ui
npm run build
npm run preview
```

### Frontend verification

```bash
cd apps/quant-ui
npm run typecheck
npm run test
npm run build
```

## Runtime connection

- Backend 只索引项目 `runtime/` 和项目内已知 factor/source metadata。
- Runtime index 使用缓存，未变化的 artifact 不会重复解析。
- Parquet/CSV/JSON/log 通过 parser registry 读取；checkpoint 仅暴露 metadata。
- 标准交易必须通过 order blotter schema gate；研究事件不会伪装成真实成交。
- 缺失逐笔因子贡献、模型分数、T+1 fill 时间等字段时，API 返回 availability/issues，UI 显示「暂无数据」。

### Runtime cleanup

Runtime Explorer 的「安全清理」页通过 backend 重新计算候选，不接受任意路径：

```text
GET  /api/system/runtime-cleanup
POST /api/system/runtime-cleanup
```

执行前必须选择候选并输入 `DELETE`。Protected roots 包括 raw/silver/manifests、
canonical model registry 和当前关键 V8 baselines。

本次已删除 test-generated V7 registry、明确命名为 test/demo/smoke 的临时产物、
历史 smoke reports 和被新版 capture 取代的旧 Quant UI 截图。累计释放
`62,291,591 bytes`，审计记录：

```text
runtime/reports/quant_ui/cleanup/cleanup_20260619T184836Z.json
```

约 39.79 GB 的 superseded large training datasets 仍保留为「人工复核」候选，
没有自动删除。

## Existing QuantAgent integration

- 回测读取由 `services/quant_api/adapters/backtests.py` 适配现有 strict backtest、paper replay 与 stock-level artifact。
- 因子目录由 `services/quant_api/adapters/factors.py` 读取现有 factor registry、source definitions 和 runtime evaluation。
- 模型训练/推理由 `services/quant_api/adapters/models.py` 统一发现 Deep FT、recursive registry、RL policy、T+1 joblib bundle 和 generic binary artifact；映射 metrics、backtest evaluation、predictions、policy weights、feature importance 和 repository-relative artifact metadata。
- 选股与风控分别由 `selection.py`、`risk.py` 读取现有 hybrid pool、decision trace、risk events 和 code-derived rules。
- 新研究任务通过 `services/quant_api/services/jobs.py` 的 allowlist 调用已有 QuantAgent CLI；不复制训练、回测或风控实现。
- 如新增 artifact 格式，只需在 runtime parser registry 或对应 adapter 增加映射，前端 schema 无需跟随核心代码变化。

## Current implementation

- Dashboard：真实净值/回撤、风险雷达、系统健康、透明漏斗、风险暴露、Top 贡献标的、交易明细。
- Stock Replay：K 线、成交量、买卖点、T+1 做 T 点、交易表与 Decision Inspector 联动。
- Backtest Lab、Factor Center、Selection Logic、Model Lab、Risk Center、Runtime Explorer、Reports。
- Model Lab：所有已识别模型 family、搜索/过滤、能力覆盖、训练曲线、预测/权重、persisted evaluation、artifact inventory 和最多 6 模型横向对比。
- Global command palette：`Ctrl/Cmd + K` 页面导航与股票代码直达复盘。
- Responsive terminal：desktop high-density view；mobile 使用 icon rail、横向 catalog/tabs 和单列 analysis panels，不产生 document-level horizontal overflow。
- T+1 Analysis：pair 收益、成功/失败控制、逐笔数量与数据质量。
- Job API：allowlisted backtest/train/infer research jobs 与 SSE logs。

## Known limitations

- 当前 runtime 没有持久化的字段不会由 UI 猜测；例如逐笔 factor contribution、SHAP、risk score 可能为空。
- 单因子独立交易只在真实 artifact 存在时展示，不复用多因子交易冒充。
- 部分 T+1 research artifact 只有 pair-level 结果，没有可映射到 K 线的 intraday fill timestamp。
- 首次 runtime index 会受 artifact 数量和 Parquet metadata 读取速度影响，后续使用缓存。
- Web 系统不启用 live trading；所有研究任务仍受 QuantAgent safety gate 约束。
