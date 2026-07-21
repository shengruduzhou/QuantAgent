# Quant UI typed events 与实时任务纵切片（2026-07-22）

## 1. 当前理解的系统结构

QuantAgent 的任务事实来源是 `JobManager` 及其 `runtime/jobs/quant_ui/jobs.json`，REST `/api/jobs` 提供列表，既有 SSE `/api/jobs/{id}/stream` 轮询单任务日志。Quant UI 的 Activity 原先每 5 秒轮询 REST。此阶段保留这些入口与持久化格式，只在服务边界增加 typed event bridge。

## 2. 本阶段处理范围

- 定义版本化 `quantagent.event.v1` envelope。
- 增加进程内、有界队列的 topic broker，以及显式 start/stop 生命周期。
- 将既有 `JobManager` 状态变化桥接为 `job.status`，不复制 job state。
- 增加 `/api/events/ws?topics=jobs`，包含初始 snapshot、heartbeat、gap 通知和断开清理。
- Activity 使用 WebSocket 更新 React Query cache；断线指数退避重连，保留 5 秒 REST fallback。

## 3. 读取和验证过的关键文件

- `services/quant_api/services/jobs.py`
- `services/quant_api/services/container.py`
- `services/quant_api/routes/api.py`
- `services/quant_api/app.py`
- `apps/quant-ui/src/components/AppShell.tsx`
- `apps/quant-ui/src/hooks/useApi.ts`
- `apps/quant-ui/src/api/client.ts`
- `tests/quant_ui/test_jobs.py`
- `tests/quant_ui/test_api.py`

## 4. QuantAgent 与 vn.py 的能力映射

| 能力 | QuantAgent 本阶段 | vn.py 参考 | 决策 |
|---|---|---|---|
| 事件契约 | `quantagent.event.v1` typed envelope | `Event(type, data)` | 借鉴事件解耦，但保留更强版本、来源与关联 ID |
| 分发 | 有界、topic-filtered、进程内 broker | EventEngine queue/handler | 兼容桥接；不引入第二个 MainEngine |
| 生命周期 | `ServiceContainer.start/stop` | engine/app lifecycle | 合并到现有 FastAPI service container |
| Web 推送 | jobs snapshot/status/heartbeat/gap | Web/RPC event distribution | WebSocket adapter，不改变任务事实来源 |
| 降级 | REST 5 秒 fallback | 模块独立查询 | 保留现有 REST，显式显示 reconnecting/stale/unavailable |

参考实现边界来自 [FastAPI WebSockets 官方文档](https://fastapi.tiangolo.com/advanced/websockets/) 与 [vn.py EventEngine 源码](https://github.com/vnpy/vnpy/blob/master/vnpy/event/engine.py)。

## 5. 新增、修改和废弃的内容

新增：

- `services/quant_api/events/`：contract、broker、WebSocket route。
- `apps/quant-ui/src/hooks/useJobEvents.ts`：cache bridge、heartbeat watchdog、reconnect。
- 后端 broker/backpressure/WebSocket reconnect tests 与前端 reconnect/cache test。

修改：

- `JobManager` 在原有持久化完成后发布状态事件。
- `ServiceContainer` 管理事件服务生命周期。
- Activity 根据实时连接状态切换 WebSocket 与 REST polling。

废弃：无。单任务 SSE 保持兼容。

## 6. 为什么没有造成重复架构

- job 状态仍只存于 `JobManager` 和现有 `jobs.json`；broker 不持久化业务状态。
- WebSocket snapshot 直接调用 `JobManager.list()`，事件 payload 直接来自 `_public(record)`。
- REST 和 SSE 未重写；WebSocket 是 adapter/service boundary。
- 未引入 Redis、消息队列、第二个 scheduler 或第二套前端状态仓库。

## 7. 测试与验证结果

```text
Targeted backend event/API tests:       11 passed
Python compileall src/services/scripts: PASS
Full Python suite:                      1282 passed, 17 skipped
Frontend Vitest:                        5 passed
Frontend TypeScript typecheck:          PASS
Frontend production build:             PASS
WebSocket reconnect/snapshot test:      PASS
Broker backpressure/gap accounting:     PASS
git diff --check:                       PASS
```

## 8. 对当前训练任务的影响

- 从已合并 `main` 建立独立 worktree `QuantAgent-events` 和分支 `agent/typed-events-realtime-jobs`。
- 未停止或修改训练进程、checkpoint、模型、Parquet、runtime 日志或端口。
- 测试只使用 pytest 临时目录和内存 broker。
- live trading 仍禁用，WebSocket 不能提交任意命令或订单。

## 9. 已知问题

1. Broker 当前是单 API 进程内实现；未来多 worker 需要 RPC/外部 event bridge，但本阶段不提前引入。
2. 当前只开放 `jobs` topic；risk、audit、execution 必须在各自 typed contract 和权限边界明确后接入。
3. 日志仍通过原 SSE/REST 获取；没有把高吞吐 stdout 逐行复制到 broker。
4. API 尚无 auth/RBAC，因此部署边界仍限 loopback/internal network。

## 10. 下一阶段最合理的工作

将现有 RiskGate/KillSwitch 审计记录适配为 typed `risk.event`，通过同一 event bridge 推送到 Risk Center，并加入事件确认、来源 artifact、run ID 与 replay link；不允许事件通道绕过风险门提交订单。

## 11. 启动与验证命令

```bash
python -m services.quant_api
cd apps/quant-ui && npm run dev

python -m pytest tests/quant_ui/test_events.py tests/quant_ui/test_jobs.py tests/quant_ui/test_api.py -q
python -m pytest tests/ -q
python -m compileall -q src services scripts
cd apps/quant-ui && npm test && npm run typecheck && npm run build
```
