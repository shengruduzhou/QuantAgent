# QuantAgent × vn.py × Serenity Phase 1 审计与能力矩阵

> 审计日期：2026-07-22（Asia/Tokyo）  
> QuantAgent：`main@21aecb53e7bf6092541d0d282f5a8f37efb92efc`  
> vn.py core：`master@1b78494979deb4c4996f6b864f234d9839f2f239`（README version 4.4.0）  
> serenity-skill：`main@c2fe93deedfd0d1bd9fe7ef0601ea1b9c20ea24a`  
> 工作分支：`codex/quantagent-vnpy-audit-20260722`（独立审计分支）

## 1. 结论先行

QuantAgent 已经具备完整度较高的 A-share PIT research、FT-Transformer、严格回测、target weights、VirtualBroker、QMT dry-run、RiskGate/KillSwitch、FastAPI 和 React/ECharts 工作台。正确方向不是复制 vn.py，而是保留这些核心，并吸收 vn.py 的 `MainEngine / Gateway / App / lifecycle / event bridge / process boundary` 思想。

大规模事件化之前必须先修四个可信性问题：

1. `train-v8-deep` 的长周期 label purge 没有消费 `label_end_*`，long-horizon 训练/比较可能跨 validation 边界。
2. `train-v8-deep` 自带 Step 7 将 `trade_date=t` 的 close-derived feature/prediction 交给同日 close 成交；只有 `baseline_protocol.py` variant C 做了 delay-1。
3. Web API 强制读取 repository-local `runtime/`，绕过 `QUANTAGENT_HOME`，可能看不到真实 runtime。
4. runtime index 依赖路径/文件名分类，不读 manifest/schema/trust；污染 holdout 和 research artifact 可能被 UI 作为 `ready` 最新结果展示。

因此首个 vertical slice 应是 **Trusted Artifact Control Plane**：同一真实 artifact 经过 manifest/schema/trust capability detection、API typed response、UI trust/stale/unavailable 展示；同时先封堵上述 label/serving integrity 问题。EventEngine、WebSocket 和 paper lifecycle 在可信 artifact contract 之后实施。

## 2. 环境与并发隔离

本次工具实际可访问的是 Ubuntu 24.04.3 x86_64 容器 `a8395b8fbb41`，`HOME=/root`，可用磁盘约 55 GB；它不是用户桌面 Mac，也无法代表 Buzz。

- 当前容器未发现 Python/Node/Claude 训练或服务进程；仅有 Codex 进程。
- 未发现 NVIDIA GPU；未启动训练、API、frontend，也未占用新端口。
- 监听端口只有 `127.0.0.1:39069` 和 `127.0.0.1:44519`，未识别为 QuantAgent 服务。
- `ssh buzz` 失败：`Could not resolve hostname buzz`。
- QuantAgent 使用全新独立 clone；原始 `main` clean，审计文档写入独立 branch。
- 未停止 Claude、Python、CUDA 或训练任务；未读取/修改 checkpoint、真实 runtime、大型 Parquet 或凭证。

仓库中的 `ARCHITECTURE_AUDIT.md` 记录了另一台 20-core/62-GiB/RTX-3090、83-GB runtime 主机的历史状态。该记录有参考价值，但本次无法直接验证其当前进程、GPU、磁盘或 artifacts，不能把它描述为本次实测。

## 3. 当前真实路径与边界

### 3.1 可信研究/生产候选路径

代码和配置当前指向：

```text
PIT silver + executable labels
  -> gold training dataset / feature schema
  -> train-v8-deep FT-Transformer sleeves
  -> materialize_production_composite.py
  -> configs/production_blend.json
  -> baseline_protocol.py variant C (eligible + delay-1)
  -> strict_v8.py / ashare_execution_simulator.py
  -> target weights / orders / fills / failed orders / risk events / audit
```

关键证据：

- `src/quantagent/cli/v8_deep.py`、`models/ft_transformer.py`、`training/ft_transformer_trainer.py`：训练与 bundle。
- `scripts/materialize_production_composite.py`：配置驱动物化并生成带 input/output SHA、argv、git、trust 的 manifest。
- `configs/production_blend.json`：short/mid 权重为 1/1、long 为 0；`trust=likely_overfit`，38.6% 明示 contaminated。
- `scripts/baseline_protocol.py`：quarantine guard、variant C 和 delay-1。
- `src/quantagent/backtest/strict_v8.py`、`ashare_execution_simulator.py`：A-share 成交约束与审计。
- `src/quantagent/cli/__init__.py`：稳定 CLI；部分旧 CLI 仅在 `QUANTAGENT_ENABLE_LEGACY_CLI` 开启。

`configs/production_blend.json` 已被 `materialize_production_composite.py` 消费；旧文档中“production config 无消费者”的结论已经过时。

### 3.2 不可当 production evidence 的路径

- `scripts/forward_daily_inference.py` 仍钉死旧 v8.8，并记录 feature reproducibility 不完整；它不是当前 production-serving 真值。
- `train-v8-deep` 内置 Step 7 没有 variant C delay-1，输出只能视为 diagnostic，不能引用为 trusted performance。
- `run_strict_backtest_v8` 可被大量 scripts 直接调用；只有 `baseline_protocol.py` fail-closed quarantine。直接调用面需要 capability token 或统一 trusted evaluator facade。
- `models/v7_deep_alpha.py`、`v7_multi_horizon.py` 是 heuristic/baseline；不能因命名被当成 production deep model。
- committed `artifacts/v7_alpha/registry/latest.json` 指向缺失文件和 pytest temp path，且 `production_ready=false`；它是 fixture/history，不是真实 runtime 证据。
- 当前 clone 的 `runtime/` 只有测试生成的 dry-run audit log；真实外置 runtime 未挂载，本次不声称已验证真实模型或收益。

## 4. Integrity 风险

| Priority | 问题 | 代码证据 | 影响 | 建议 |
|---|---|---|---|---|
| P0 | horizon-specific label purge 缺失 | `v7_label_builder.py` 生成 `label_end_120d`；`cli/v8_deep.py` 使用固定 BDay embargo 且训练前删除 label_end | long label 可跨 validation；模型比较不可信 | splitter 接受每行 `label_end`，按 sleeve fail-closed；增加 boundary tests |
| P0 | same-close execution | dataset 使用 close(t)，`v8_deep.py` 同日 target，simulator 同日 close fill | 内置回测 look-ahead | training command 只产 predictions；所有收益评测强制走 trusted evaluator + delay-1 |
| P0 | trust 未进入 UI | `adapters/backtests.py` 将可解析结果均标 `ready` | contaminated/research 结果可能成为 dashboard headline | schema 增加 `trustClass/validationStatus/capabilities`，默认 unclassified 非 production |
| P1 | PIT gate blind spot | `v7_quality_gates.py` 缺 as-of 时只看布尔 flag；v8 deep 读 available_at 但不校验 | 错误 flag 可绕过时间约束 | 数据入口必须提供 decision time，并验证 `available_at <= decision_at` |
| P1 | survivorship | feature registry 明示 current SW snapshot；static universe/current sector map | universe/sector bias | 历史 membership interval schema；缺失时降 trust |
| P1 | quarantine 可旁路 | direct strict backtest 只 stamp metadata；约 41 个 scripts 直接调用 | holdout 被重复消费 | `TrustedEvaluationService` 作为唯一公开入口；raw engine 标 internal |
| P1 | EvidenceStore CSV fallback 隐形 | write 可 fallback CSV，read 只 glob Parquet | 写成功读不到，可能 silent data loss | 统一 format manifest；read 对 manifest fail-loud |

## 5. 能力矩阵

| 能力领域 | QuantAgent 现状 | vn.py 对应设计 | 差距 | 决策 | 复用模块 | 主要风险 |
|---|---|---|---|---|---|---|
| PIT 数据 | 强：available_at、manifest、provider/router、gold schema | `vnpy.alpha.AlphaLab` bar/component store | vn.py PIT 财务/证据弱；QA sector/universe 仍有 snapshot | **保留 QA；不替换** | `data/*`, `manifest.py`, providers | available_at gate、survivorship |
| 行情 | 日/分钟 panel 和 provider 多；未统一 live subscription contract | `BaseGateway.subscribe/query_history` + Tick/Bar | research DataFrame 与 live object 两套语义 | **设计兼容 MarketDataSource/Gateway adapter** | providers、QMT/VirtualBroker | symbol/frequency/timezone 语义漂移 |
| Evidence/Serenity | EvidenceRecord/Store、industry chain、credibility 已有 | vn.py 无对应强项 | Skill 输出尚无正式 capability/trust schema | **新增 Skill adapter，不新增研究 pipeline** | evidence、themes、agents | 主观观点覆盖 quant/risk |
| 事件 | 有 research `EventStore`，无 service event bus | `EventEngine` string type + `Any` data | 缺 typed contract/lifecycle/replay；vn.py event 无背压与 handler isolation | **借鉴后重实现 typed dispatcher** | audit JSONL、API schema | 慢 handler 阻塞、事件丢失 |
| 领域对象 | research/v7/execution/API 多套 dataclass/Pydantic | `object.py` 统一 Tick/Order/Trade/Position/Account/Contract | QA 缺统一 Instrument/Account/Portfolio/ExecutionState | **合并规范，不引入第二套** | `broker_base.py`, `domain/schemas.py` | 迁移破坏 artifacts/API |
| 因子 | Alpha101/CICC/DSL/registry/lifecycle/诊断成熟 | `vnpy.alpha.dataset` Polars functions | QA registry 与 runtime factor artifact 未统一 capability | **保留 QA；借鉴 lab UX** | `factors/*`, feature registry | eval/旁路、multiple testing |
| 模型 | FT-Transformer sleeves、registry、walk-forward、schema | AlphaModel Lasso/LGB/MLP + lab persistence | 多 registry/旧模型命名；UI lineage 不完整 | **统一现有 registry** | trainers、production blend | pickle/未知 schema 不可引入 |
| 研究 Lab | CLI/scripts 丰富但分散 | `AlphaLab` dataset/model/signal workflow | 196 scripts，入口和可信等级难辨 | **以 registry + job template 收口** | stable CLI、experiment ledgers | 新 pipeline 蔓延 |
| 回测 | strict A-share engine 强，另有多套 legacy engine | CTA/alpha backtester、optimization UI | 可信入口可旁路；vn.py 默认规则不等同 A 股现实 | **只借鉴 UI/engine lifecycle** | strict_v8、simulator、baseline protocol | 回测口径混用 |
| 策略 | decision chain、rule signals、sleeves | CTA StrategyTemplate callbacks | QA strategy state/lifecycle 未统一 | **先定义 StrategyState，不复制 CTA** | decision_chain、portfolio | LLM/strategy 直接变订单 |
| 组合 | target weights、optimizer、sector/risk constraints | PortfolioStrategy/App、PortfolioManager | 多 allocator/overlay；状态持久化分散 | **以 target_weights 为唯一执行边界** | `v7_target_weights.py`, optimizer | 两套组合真值 |
| Order intent | BrokerBase/OrderManager 已有且幂等 | OrderRequest/MainEngine routing | Risk result 只是字符串，调用层未强制证书 | **扩展 typed RiskApproval envelope** | OrderManager | bypass risk gate |
| Order/Trade | dry-run order state、fills、failed-order audit | OMS caches order/trade and active orders | 状态/reject mapping 尚未服务化 | **借鉴 OMS state store** | broker objects、audit | duplicate/out-of-order events |
| Account/Position | VirtualBroker ledger；QMT query skeleton | AccountData/PositionData + OMS | 无统一 snapshot/version/reconciliation lifecycle | **规范 AccountSnapshot/PositionSnapshot** | ledger、reconciliation | stale broker state |
| RiskGate | 多套 risk gate/decision chain | RiskManager flow/order/trade/cancel limits | 重复实现；入口缺强制组合 | **统一现有 risk core，不复制 vn.py** | risk/*、execution constraints | gate 顺序/语义分叉 |
| KillSwitch | manual/stale/reject/loss/drawdown checks | RiskManager active switch | 尚未作为 gateway hard invariant | **在 execution service 强制** | risk kill switch | API/脚本绕过 |
| Paper trading | VirtualBroker、paper CLI/report | PaperAccount app | 无 Web start/status/stop/replay lifecycle | **包装现有 VirtualBroker** | VirtualBroker、paper report | 把模拟当实盘 |
| Gateway | QMT defaults safe；live submit intentionally unimplemented | BaseGateway + multiple adapters | 无 registry/lifecycle/health/reconnect contract | **设计 QA Gateway protocol；可写 vn.py adapter（optional）** | QMTGateway、BrokerBase | 实盘权限、安全凭证 |
| RPC | 无统一 RPC；Web jobs 本地 Popen | ZeroMQ RPC/RPCService | process boundary 缺失 | **采用 authenticated typed JSON/msgpack；拒绝 pickle RPC** | service schemas | RCE、trusted-network 假设 |
| Web API | FastAPI adapter、多 read API、job allowlist | WebTrader REST→RPC | runtime root 错位；无 auth/RBAC；jobs 与 API 同进程 | **演进现有 API** | routes/adapters/jobs | 改绑 0.0.0.0 后高风险 |
| 实时推送 | backend SSE log stream | WebTrader WebSocket event push | frontend 未使用 SSE/WS，也无 reconnect test | **新增 WS subscription + resume cursor** | typed API envelopes | 丢事件/重复事件 |
| Job 调度 | allowlist + path confinement + daemon Thread/Popen | app engine threads、optimization workers | 无 cancel/PID/process group/locks/limits；重启 orphan | **拆 JobRunner service** | command registry | 训练重复启动/资源争抢 |
| UI/图表 | 11 页 React/TS/Vite/ECharts，专业工作台基础已在 | Qt monitors/backtester chart/WebTrader | data quality/evidence/paper lifecycle/service health 缺页；无实时订阅 | **继续单一 frontend** | existing pages/components | mock/fixture 混为真实 |
| Loading 状态 | ready/partial/empty/error API envelope | monitors/events | `StateView` 只有 loading/empty/error，无 stale/unavailable/partial | **补完整状态机** | StateView、React Query | stale 结果看似最新 |
| Artifact | index/parser/adapters 已有 | AlphaLab dataset/model/signal listing | 按文件名猜 schema；不校验 hash/trust/version | **ArtifactIndex v2 + migration** | runtime indexer/adapters | checkpoint/污染结果暴露 |
| 日志/审计 | JSONL audit、failed orders、cleanup audit | EVENT_LOG/LogEngine | 结构/关联 ID 不统一；无敏感字段脱敏 | **统一 AuditEvent schema** | AuditLogger | 泄露路径/凭证 |
| 配置 | `quant_paths` 单一来源目标；多 JSON/YAML | SETTINGS + app JSON | Web bypass env；legacy configs 多 | **修复 API 使用 quant_paths()** | config paths | 读错 runtime |
| 插件 | factor/model 局部 registry | BaseApp + add_app + engine_class | 无通用 app discovery/lifecycle | **小型 explicit registry，拒绝自动 import 全目录** | existing registries | plugin side effects |
| Health | diagnostics + static `/health` | event timer/app status | `/health` 不检查 runtime/job/gateway/data freshness | **HealthSnapshot + service probes** | daily health | false green |
| 部署 | shell/systemd scripts；API/frontend run scripts | process-separated WebTrader/RPC | 缺统一 deployment manifest、auth、resource limits | **先 paper-only process topology** | systemd examples | 端口/孤儿进程/权限 |
| Live trading | 明确 disabled，QMT live submit未实现 | MainEngine 可直接 send_order | 不应因融合而打开 | **明确拒绝本阶段启用** | dry-run guard | 资金安全 |

## 6. vn.py 代码证据与取舍

### 6.1 值得借鉴

- `vnpy/event/engine.py`：register/unregister、timer、dispatcher 的最小概念。
- `vnpy/trader/engine.py`：`MainEngine.add_engine/add_gateway/add_app/close` 和 `OmsEngine` 的统一状态查询。
- `vnpy/trader/gateway.py`、`object.py`：Gateway contract 与 Tick/Bar/Order/Trade/Position/Account/Contract 分层。
- `vnpy/trader/app.py`：App metadata 和 engine/widget 关联。
- `vnpy_ctastrategy`、`vnpy_portfoliostrategy`、`vnpy_algotrading`：start/stop/callback/state UI 的生命周期组织。
- `vnpy_ctabacktester`：后台 backtest/optimization 与结果图表/明细联动。
- `vnpy_datamanager`：数据 import/export/download 的管理面。
- `vnpy_riskmanager`：订单流、成交量、撤单等 execution-time risk hooks。
- `vnpy_paperaccount`：paper account 独立 app 边界。
- `vnpy_webtrader`：交易进程与 FastAPI 进程隔离、REST 主动操作、WebSocket 推送的拓扑。
- `vnpy.alpha`：dataset/model/signal/lab 的用户工作流，以及 train/valid/test segment 和 component membership interval 思路。

### 6.2 不直接复制

- `Event.type: str`、`Event.data: Any`；单消费线程、无背压、无 handler exception isolation。
- `MainEngine.__init__` 修改全局 cwd；不适合 Web/service runtime。
- core RPC `recv_pyobj/send_pyobj` 使用 pickle；RPCService README 仅承诺可信网络，不能作为公网或跨信任边界协议。
- WebTrader `SECRET_KEY="test"`，WebSocket token 放 query string；不能复制其认证实现。
- `vnpy_websocket` 默认 `ssl.CERT_NONE` 关闭证书验证；这是明确拒绝项。
- REST client 默认请求无 timeout，错误输出可能包含 headers/body；不能直接进入 Gateway production path。
- `vnpy.alpha.lab` pickle dataset/model；不能作为可审计 artifact schema。
- `vnpy.alpha.dataset.utility` 对 expression 使用 `eval`；不能替换 QuantAgent 受限 DSL。
- Alpha processor 在未给 fit 时间时可使用整段统计量，`LassoModel.fit` 合并 TRAIN+VALID；不能把其默认结果当严格 OOS。
- database/datafeed 缺失时的 SQLite/空实现 fallback 不符合 QuantAgent real-data fail-loud 约束。
- vn.py alpha/backtest 没有 QuantAgent 完整的 A-share T+1、板块涨跌停、ST、停复牌、PIT、quarantine 和 trust discipline。
- CTA/Algo app 能直接下单；QuantAgent 必须继续保持 Agent/LLM→evidence/view→target weights→RiskGate→OrderManager 的边界。

### 6.3 本次抓取的官方仓库版本

| Repository | Commit |
|---|---|
| `vnpy/vnpy` | `1b78494979deb4c4996f6b864f234d9839f2f239` |
| `vnpy/vnpy_webtrader` | `1d4416cbb1181d89aca06bdd85f326a02fba0af4` |
| `vnpy/vnpy_ctabacktester` | `d2c31ffd4d52a34f678751f20afc403a8f48011d` |
| `vnpy/vnpy_ctastrategy` | `6ef76981624bf55b2ea978f8587f74d633aafc72` |
| `vnpy/vnpy_riskmanager` | `55ae48eab3c8c4b686eeff10a027b320ea47daca` |
| `vnpy/vnpy_datamanager` | `cbae6768f6ee8fd20ec8441baae13b70b21b1ff1` |
| `vnpy/vnpy_paperaccount` | `fcfe2b58965a0dc99b5cdbe075d5a372d8ef3ac2` |
| `vnpy/vnpy_portfoliostrategy` | `164d94f35e75c1b3c5c9a62f0b94de78e3e9662c` |
| `vnpy/vnpy_rpcservice` | `dc6d7fd53147f840cdfdb0641ec51daf4677d6c9` |
| `vnpy/vnpy_rest` | `0f1a2d15b091c654afc3916693304b70fc1bf9a5` |
| `vnpy/vnpy_websocket` | `e8eab0aaa1f3954cf8974c0f11d8c25dde7bd2fb` |
| `vnpy/vnpy_portfoliomanager` | `c71af6407af1337fd92cc97d51107aecc6997b93` |
| `vnpy/vnpy_algotrading` | `4133987530eb28f3538d1983545d81c4f83d7d59` |

## 7. Serenity Skill 安装审计

### 7.1 已完成

ChatGPT/Codex 受管个人 Skill 已安装并完成对账：

| 项目 | 结果 |
|---|---|
| 上游 commit | `c2fe93deedfd0d1bd9fe7ef0601ea1b9c20ea24a` |
| 安装时间 | `2026-07-22 04:17:56 JST` |
| 载荷 | 19 files；保留 SKILL.md、LICENSE、references、assets、scripts、examples、agents |
| 上游 validation | 安装前目录名为 `serenity-skill` 时通过 |
| Codex validation | 对账后 `quick_validate.py` 通过；scorecard smoke 通过 |
| 工作区 | clean；active 19 paths；disabled 0 |

兼容适配：上游 frontmatter 的 `compatibility` 不是 Codex validator 支持字段，因此只在安装副本删除该字段；研究方法、触发词、风险边界和脚本未改。对账后目录名由系统规范化，上游 validator 会因父目录名变化而失败；Codex validator 和运行脚本仍通过。

上游 `SHA256.txt` 中 `README.md`、`README.zh-CN.md` 哈希陈旧；README 不属于运行安装载荷，不能声称上游完整 SHA 清单全绿。

### 7.2 Blocked / unverified

| 目标 | 状态 | 原因 |
|---|---|---|
| 当前容器 Claude | blocked | 无 `claude`，`/root/.claude` 不可写 |
| Buzz Codex/Claude | blocked | hostname 无法解析、无 SSH config |
| 用户桌面 Mac Codex/Claude | unverified | 本次执行器无本机终端通道 |
| 当前已打开会话动态触发 | unverified | Skill 列表有缓存；active path/frontmatter/metadata 已验证 |

Serenity 只作为高优先级 research Skill：输出 evidence、opposing evidence、confidence、failure conditions 和 research priority。它不得直接输出 Order/OrderIntent，不得覆盖 quantitative validation、RiskGate 或 KillSwitch，也不得成为唯一研究框架。

## 8. API/UI 现状

现有 frontend 是唯一应继续演进的 UI：Dashboard、Stock Replay、Backtest Lab、T+1、Factor Center、Selection、Model Lab、Risk、Runtime Explorer、Reports、Settings 共 11 页。现有 API 覆盖 runtime/backtest/factor/model/selection/risk/do-T/jobs。

主要缺口：

- 无 data quality/evidence/paper strategy lifecycle/account/reconciliation/service-health 工作流。
- backend 有 SSE job log endpoint，但 frontend 没有 EventSource/WebSocket/polling。
- `StateView` 缺 stale/unavailable/partial；React Query 未设置 freshness/refetch contract。
- Settings 的 train template 使用 `horizon=short`，CLI 只接受 `short_5d/mid_5d_30d/long_30d_120d`；模板数据 lineage 也与当前 plus7clean 不一致。
- job allowlist/path confinement 可复用，但缺 cancel、PID/process group、GPU/CPU lock、concurrency、orphan adoption。
- API 当前 loopback 是安全前提；若绑定外网，job/cleanup write API 缺 auth/RBAC/CSRF，必须 fail-closed。

## 9. 渐进实施顺序

### Phase 0 — Integrity blockers

- horizon-specific label-end purge。
- 禁止 training command 自带同日成交绩效进入 trusted UI。
- `TrustedEvaluationService` 封装 quarantine + variant C + delay-1。
- 对应 PIT/look-ahead/quarantine regression tests。

### Slice 1 — Trusted Artifact Control Plane

- `ArtifactManifestV1`、`ArtifactCapability`、`TrustClass` typed schema。
- `ManifestResolver -> SchemaRegistry -> MigrationRegistry -> TrustPolicy`。
- API `default_settings()` 尊重 `quant_paths()` / `QUANTAGENT_HOME`。
- 一个真实 artifact 从外置 runtime 被索引，hash/schema/trust 校验，经 API 返回并在现有 UI 展示 source time/run ID/trust/stale。
- 历史无 manifest 结果标 `unclassified`，不得 production-ready。

### Slice 2 — Job + realtime

- JobRunner 独立服务；API 只提交 typed command。
- PID/process group、cancel、resource lock、idempotency key、restart reconciliation。
- typed JobEvent/AuditEvent；WebSocket subscription + cursor/reconnect；SSE 保留兼容。

### Slice 3 — Paper lifecycle

- 包装 VirtualBroker、OrderManager、RiskGate、KillSwitch、reconciliation。
- Web start/status/stop；risk event 全链路到 UI；audit replay。
- live feature flag 继续硬关闭；QMT live submit 不实现。

### Slice 4 — Gateway/app registry

- explicit GatewayRegistry/ServiceRegistry/AppManifest；不扫描目录自动 import。
- adapters：QuantAgent provider、VirtualBroker、QMT dry-run；可选 vn.py gateway adapter 仅在 schema/security contract 通过后引入。

## 10. 验证结果

本次实际执行：

| Check | Result |
|---|---|
| Python compileall (`src`, `services`) | pass，464 compiled files |
| backend/API/safety selected tests | 59 passed |
| strict A-share/PIT/risk selected tests | 55 passed, 1 skipped |
| warnings | 12 pandas FutureWarning，来自 `full_pipeline_backtester.py` fillna downcast |
| frontend typecheck | pass |
| frontend Vitest | 3 passed |
| frontend production build | pass |
| frontend build note | EChart chunk ~609 kB，超过 500 kB warning |
| vn.py core + audited official apps compileall | pass |
| `git diff --check`（写文档前） | pass |

未完成验证：真实外置 runtime integration、真实 API process smoke、WebSocket reconnect、真实 job cancel/restart、真实 GPU training、Buzz/Mac/Claude 安装。原因是对应 runtime/主机/进程在当前环境不可访问，而不是默认判定通过。

## 11. 下一阶段验收命令

当前审计副本的轻量验证：

```bash
cd /workspace/scratch/9da906f3b19f/QuantAgent_audit
PYTHONPYCACHEPREFIX=/tmp/quantagent-pycache python -m compileall -q src services
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/quantagent-test-deps:src:. \
  python -m pytest -q -p no:cacheprovider \
  tests/quant_ui \
  tests/test_v7_architecture_contracts.py \
  tests/test_v7_execution_and_cli.py \
  tests/test_v7_no_synthetic_fallback.py \
  tests/test_quarantine_guard.py \
  tests/backtest/test_strict_v8.py \
  tests/test_v7_full_pipeline_backtester.py \
  tests/test_v8_full_pipeline_e2e.py \
  tests/test_v7_datahub_pit.py \
  tests/test_v7_pit_financial.py \
  tests/test_v7_qlib_pit.py \
  tests/test_v7_walk_forward_splitters.py \
  tests/test_order_dedup_regression.py \
  tests/risk \
  tests/execution/test_risk_events_output.py
cd apps/quant-ui
npm run typecheck
npm test -- --run
npm run build
```

本机/Buzz 的 Serenity 安装必须固定到已审计 commit，并在目标主机分别记录 host、OS、home、target、commit、time、validation。取得 Buzz SSH host/user 或本机终端通道后再执行；当前不能标记为已完成。

## Appendix A — 本机/Buzz Codex CLI + Claude 幂等安装

以下命令适用于 macOS/Linux。Codex CLI 使用上游规定的 `~/.agents/skills`；Claude Code 使用 `~/.claude/skills`。它只更新名为 `serenity-skill` 的目录，不删除其他 Skills，不产生嵌套副本，并固定到本次已审计 commit。

```bash
bash <<'BASH'
set -euo pipefail

EXPECTED=c2fe93deedfd0d1bd9fe7ef0601ea1b9c20ea24a
SRC="${XDG_CACHE_HOME:-$HOME/.cache}/serenity-skill/source-$EXPECTED"
MANIFEST="$HOME/.local/state/serenity-skill/install.log"

printf 'host=%s os=%s home=%s\n' "$(hostname)" "$(uname -srm)" "$HOME"
df -h "$HOME"
test -w "$HOME"
command -v git
command -v python3

mkdir -p "$(dirname "$SRC")"
if test -d "$SRC/.git"; then
  test -z "$(git -C "$SRC" status --porcelain)"
  test "$(git -C "$SRC" remote get-url origin)" = \
    "https://github.com/muxuuu/serenity-skill.git"
  git -C "$SRC" fetch origin main
elif test -e "$SRC"; then
  printf 'Refusing non-Git source path: %s\n' "$SRC" >&2
  exit 20
else
  git clone --filter=blob:none \
    https://github.com/muxuuu/serenity-skill.git "$SRC"
fi

git -C "$SRC" checkout --detach "$EXPECTED"
test "$(git -C "$SRC" rev-parse HEAD)" = "$EXPECTED"

install_one() {
  agent="$1"
  root="$2"
  target="$root/serenity-skill"
  mode=new

  if test -L "$target"; then
    printf 'Refusing symlink target: %s\n' "$target" >&2
    exit 21
  fi
  if test -e "$target"; then
    mode=safe-update
    test -d "$target"
    test -f "$target/SKILL.md"
    grep -Eq '^name:[[:space:]]*serenity-skill[[:space:]]*$' \
      "$target/SKILL.md"
  fi

  mkdir -p "$target"
  cp -R \
    "$SRC/SKILL.md" \
    "$SRC/LICENSE" \
    "$SRC/references" \
    "$SRC/assets" \
    "$SRC/scripts" \
    "$SRC/examples" \
    "$SRC/agents" \
    "$target"/

  python3 "$target/scripts/validate_skill.py" "$target"
  PYTHONDONTWRITEBYTECODE=1 python3 \
    "$target/scripts/serenity_scorecard.py" \
    "$target/assets/bottleneck-scorecard.json" \
    --format json >/dev/null

  printf 'agent=%s target=%s mode=%s validation=pass smoke=pass\n' \
    "$agent" "$target" "$mode"
}

install_one codex "${CODEX_SKILLS_ROOT:-$HOME/.agents/skills}"
install_one claude "${CLAUDE_SKILLS_ROOT:-$HOME/.claude/skills}"

mkdir -p "$(dirname "$MANIFEST")"
printf 'host=%s commit=%s time=%s validation=pass\n' \
  "$(hostname)" "$EXPECTED" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | \
  tee -a "$MANIFEST"
BASH
```

Buzz 连接信息可用后：

```bash
ssh <buzz-user>@<buzz-host>
# 在 Buzz shell 中运行上面的完整安装块
```

Claude 发现性检查：询问 `What skills are available?`，再调用 `/serenity-skill`。如果技能目录是在当前 Claude Code 会话启动后首次创建，重新打开该 Claude Code 会话以注册顶层监听目录。
