# QuantAgent Web Quant Workstation：Terminal + Runtime/Data Manager 纵切片

日期：2026-07-22

分支：`agent/terminal-runtime-workstation`

基线：`main@b002cad91159eb87730e66699e4c2174237f8352`

## 1. 当前理解的系统结构

QuantAgent 继续是事实来源：研究、PIT 数据、因子、模型、严格 A 股回测、组合、风控、执行对象和 runtime artifacts 均保留原入口。本阶段只扩展现有 FastAPI `RuntimeIndexer` 和唯一 React/Vite 前端。

vn.py 的参考边界：

- 参考 [`MainWindow`](https://github.com/vnpy/vnpy/blob/master/vnpy/trader/ui/mainwindow.py) 的菜单、App 启动和监控器组织；不复制 Qt/PySide。
- 参考 [`BaseMonitor`](https://github.com/vnpy/vnpy/blob/master/vnpy/trader/ui/widget.py) 的高密度表格、字段映射和事件更新；QuantAgent 内部仍使用 typed API contract。
- 参考 [`DataManager`](https://github.com/vnpy/vnpy_datamanager) 的数据查询/下载/导入/概览工作流；不建立第二套数据库或数据引擎。
- 参考 [`CtaBacktester`](https://github.com/vnpy/vnpy_ctabacktester) 的参数、任务状态和结果联动；不替换 QuantAgent 严格 A 股回测。
- 参考 [`ChartWizard`](https://github.com/vnpy/vnpy_chartwizard) 的 K 线、成交标记和 drill-down；继续使用 ECharts。
- 参考 [`WebTrader`](https://github.com/vnpy/vnpy_webtrader) 的计算进程/Web 进程边界；不复制其直接 Web 下单和通用 `__dict__` 序列化。

## 2. 本阶段范围

完成第一条真实可运行纵切片：

1. 专业终端 AppShell：分组模块启动器、全局命令、持久化工作区标签、全局上下文、底部状态栏和可折叠任务 Activity drawer。
2. Runtime/Data Manager：Catalog、Runs、Lineage、Cleanup 四个联动工作区。
3. 后端扩展：manifest 元数据、facets、run 聚合、组合过滤/排序和显式 lineage。
4. 安全不变：live trading 仍禁用；研究结果明确不等于可执行订单；清理仍使用原 backend-approved candidate + `DELETE` confirmation。

未在本阶段实现 WebSocket。全局任务抽屉仅对已有持久化 `/jobs` 做 5 秒轮询，并明确标记；后续应在 typed event contract 稳定后统一替换为 WebSocket projection。

## 3. 读取和验证的关键文件

- `apps/quant-ui/src/App.tsx`
- `apps/quant-ui/src/components/AppShell.tsx`
- `apps/quant-ui/src/components/CommandPalette.tsx`
- `apps/quant-ui/src/pages/RuntimeExplorerPage.tsx`
- `apps/quant-ui/src/pages/SettingsPage.tsx`
- `apps/quant-ui/src/api/types.ts`
- `services/quant_api/routes/api.py`
- `services/quant_api/runtime_indexer/contracts.py`
- `services/quant_api/runtime_indexer/indexer.py`
- `services/quant_api/runtime_indexer/parsers.py`
- `tests/quant_ui/conftest.py`
- `tests/quant_ui/test_api.py`
- `tests/quant_ui/test_runtime_indexer.py`
- `apps/quant-ui/src/App.test.tsx`

## 4. 页面—功能—artifact 映射与处置

| 页面/模块 | 当前真实数据/API | 处置 | 理由 |
|---|---|---|---|
| 总览 | `/system/overview`、backtest/model/selection/risk adapters | 保留，纳入终端工作区 | 已有真实聚合，不新建 dashboard API |
| 市场复盘 | backtest bars/trades/signals/positions | 保留 | 已有 ECharts 联动，后续增强多窗格 |
| 选股研究 | selection runs/decision chain | 保留 | 现有透明选股链是 QuantAgent 优势 |
| 因子研究 | factor catalog/evaluation artifacts | 保留 | 不建立 vnpy.alpha 第二因子库 |
| 模型实验 | model registry/metrics/predictions/artifacts | 保留 | 不创建第二模型 registry |
| 回测实验 | strict A-share persisted artifacts | 保留 | vn.py 回测器仅参考 UI/任务状态 |
| 研究报告 | persisted reports | 保留 | 后续与 evidence/Serenity adapter 合并 |
| T+1 做 T | do-t persisted artifacts | 保留 | 与研究结论保持视觉隔离 |
| 风险监控 | RiskGate/risk event projections | 保留 | 不建立平行 RiskManager |
| Runtime / Data | `RuntimeIndexer`、manifest、preview、cleanup | **重构并增强** | 从文件浏览器升级为统一 catalog/store 控制面 |
| 系统控制 | allowlisted `/jobs` | 保留并纳入 Activity | 后续移出 API 进程，当前不复制 job manager |
| 独立 Data Manager 页面 | 无独立事实源 | 暂不新增 | dataset 先作为 Runtime Catalog capability view，避免第二数据入口 |
| Paper/OMS 页面 | 现有对象分散，缺统一 lifecycle | deferred | 先统一领域 contract/RiskGate invariant |
| Live trading 页面 | 未授权 | 拒绝 | live disabled；浏览器/Agent 不得直接下单 |

## 5. 新增、修改和废弃内容

新增：

- `workstation/modules.ts`：唯一 Web 模块 registry。
- `workstation/useWorkspaceLayout.ts`：localStorage 持久化的模块标签、启动器和 Activity 布局。
- `terminal.css`：终端框架和 Runtime Manager 独立样式边界。
- `RuntimeCleanupWorkspace.tsx`：从 Runtime 主页面提取安全清理职责。
- `GET /api/system/runtime-catalog`。
- `GET /api/system/runtime-index/{artifact_id}/lineage`。

修改：

- Runtime contract 新增 `declaredKind/runId/horizon/producer/qualityStatus/dataAsOf/rows/dateStart/dateEnd/upstreamPaths`。
- 类型与 run 识别改为 manifest 声明优先，路径启发式为兼容 fallback，并暴露 `kindSource/runIdSource`。
- Runtime 查询新增 trust、validation、freshness、capability、sort 字段。
- Catalog summary 新增 freshness/capability/status facets、runCount、manifestCoverage。

未废弃现有路由、页面、indexer、parser、cleanup service、backtester、model registry 或数据路径。

## 6. 为什么没有形成重复架构

- Catalog、Runs 和 Lineage 全部是同一个 `RuntimeIndexer.scan()` snapshot 的投影。
- 没有新数据库、文件 crawler、artifact registry 或 runtime root。
- `Run Catalog` 只聚合现有 `runId`；不持久化第二套 run state。
- lineage 只读取 manifest 中安全的 repository-relative upstream references；缺失时显示 `undeclared`，未解析时显示 `unresolved`。
- Cleanup 继续调用原 `RuntimeCleanupService`，没有前端文件删除逻辑。
- 前端继续复用 React、TypeScript、Vite、React Query、ECharts 和现有 routes。

## 7. 测试与验证结果

通过：

```text
tests/quant_ui:                         27 passed
Full Python suite:                      1280 passed, 17 skipped
Python compileall src services scripts: PASS
Frontend Vitest:                       4 passed
Frontend TypeScript typecheck:         PASS
Frontend production build:             PASS
git diff --check:                       PASS
HTTP /health:                           {"status":"ok"}
HTTP /system/runtime-catalog:           PASS
HTTP filtered runtime-index:            PASS (explicit empty state)
Vite /runtime HTML:                     PASS (QuantAgent Research Terminal)
```

CI 暴露的两个 shadow registry 测试原本读取未纳入版本控制的生产 runtime 文件。现已改为在 pytest `tmp_path` 内生成最小 append-only 哈希链、健康记录、市场覆盖率与订单/成交/加密文件，并调用真实 `shadow_day_registry.build()`；既保留 superseded-record 和 blinding 约束，又可在干净 checkout 中复现。另将 storage path 测试从硬编码仓库目录名改为校验当前仓库根目录下的 `runtime`，支持隔离 worktree。修复后全量测试无失败。

HTTP smoke 显式设置 `QUANTAGENT_HOME` 到隔离 worktree runtime；其中只有 1 个未分类、未验证的真实仓库文档 artifact，API 正确未授予 production capability。

Blocked：

- 浏览器自动化：`agent-browser` 在当前托管容器无法 bind 控制 socket，错误为 `Operation not permitted (os error 1)`。因此没有声称 browser visual/drill-down 验证通过；交互由 React Testing Library 覆盖，真实浏览器复验仍需在允许 Unix socket/Chrome 的环境执行。
- WebSocket reconnect：本阶段未引入 WebSocket，状态为 deferred，不伪装成已完成。

## 8. 对当前训练任务的影响

- 开发发生在独立 worktree/branch。
- 未修改主工作区、训练 branch、checkpoint、model、日志或大型 Parquet。
- 未停止 Claude、Python、CUDA 或训练任务。
- 启动验证仅使用隔离高位端口 `38491/38492`，完成后终止本次启动的进程。
- API smoke 的 `QUANTAGENT_HOME` 指向本 worktree runtime。

## 9. 已知问题

1. 历史 artifacts 多数没有 manifest；兼容期仍会显示 `path_heuristic` 和 `unclassified/unverified`。
2. 仅有显式 upstream path 的 artifacts 可建立 lineage；producer/job/output event 尚未形成统一 typed lineage edge。
3. `runId` 的历史 fallback 仍来自目录结构，后续 migration 应写回 manifest，不能长期依赖命名约定。
4. Activity 当前轮询 `/jobs`；后续应由 typed event + WebSocket 统一任务、日志、风险和执行状态。
5. Runtime 页面生产 chunk 约 37 KiB gzip；ECharts shared chunk 仍触发既有 500 KiB warning，未新增图表依赖。
6. 当前 API 无 auth/RBAC；在完成安全边界前只能绑定 loopback/internal network。

## 10. 下一阶段最合理工作

1. 定义 versioned typed event envelope 和 service lifecycle，但只服务现有 job/risk/audit，不创建 MainEngine clone。
2. 将现有 SSE job 状态接入统一前端 subscription，并实现 WebSocket heartbeat/reconnect/backpressure。
3. 建 `TrustedEvaluationService` facade，阻止 raw strict backtest 产物误获 production capability。
4. 完成真实 dataset artifact 的 schema/PIT/data-quality preview 和 Data Manager 下载/更新 job vertical slice。
5. 统一现有 RiskGate，随后实现 paper strategy start/status/stop/reconciliation/audit Web slice。

## 11. 启动与验证命令

```bash
cd /workspace/scratch/dea7d1cf11d0/QuantAgent-terminal
python -m pip install -e '.[web,test]'
QUANTAGENT_HOME="$PWD/runtime" QUANT_UI_PORT=8000 python -m services.quant_api
```

另一个终端：

```bash
cd /workspace/scratch/dea7d1cf11d0/QuantAgent-terminal/apps/quant-ui
npm ci
npm run dev
```

验证：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/api/system/runtime-catalog
curl -fsS 'http://127.0.0.1:8000/api/system/runtime-index?trustClass=production_ready&sortBy=sizeBytes&sortDirection=asc&pageSize=20'

python -m pytest -q -p no:cacheprovider tests/quant_ui
python -m compileall -q src services
cd apps/quant-ui
npm test -- --run
npm run typecheck
npm run build
```
