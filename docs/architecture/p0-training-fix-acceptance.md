# 验收矩阵：`run-full-real-training-v7` P0 修复

日期 2026-08-02。环境：Linux，RTX 3090 24GB，62GB RAM，`AI_quant_venv/bin/python3`。
数据：`runtime/data/gold/full_universe/`（10,917,401 行 / 5,790 只 / 2016-01-04→2026-07-22 / 2,562 交易日）。

所有"实际结果"均来自真实运行产物，不是单元测试的模拟值。

---

## A. 缺陷复现与修复

| # | 测试目标 | 真实输入 | 启动方式 | 实际结果 | 产物 | 通过 |
|---|---|---|---|---|---|---|
| A1 | 复现出厂默认失败 | gold 全宇宙面板 + 60 股 pilot；UI 默认参数（nSplits=5, 80+20 协议） | CLI `run-full-real-training-v7` | **退出码 3**，`insufficient_oos_dates`：observed 80 / required 100。**仅产出 4 折**（fold_001..004），训练窗口 2016-06-30→2016-11-01，验证窗口 **2017-02→2017-06** | `repro_ridge/research_verdict.json`、`walk_forward_predictions.csv` | ✅ 复现 |
| A2 | 定位根因 | 同上产物 | 读 `walk_forward_predictions.csv` | fold_000 被 `v7_experiment.py` 的 `len(train_idx) < max(...)` 静默丢弃；因 `min_train_days=120` 被 `embargo 5 + purge 120` 吃掉，首折只剩 **2 个交易日**训练数据 | 见上 | ✅ |
| A3 | 修复后同配置重跑 | 与 A1 **完全相同**的输入与参数 | CLI | **5 折**（fold_000..004），**100 个 OOS 交易日**；训练窗口 = 滚动 756 天 2022-06→2025-12；验证窗口 **2026-02-12→2026-07-16** | `fix_ridge/training/walk_forward_predictions.csv` | ✅ |
| A4 | 端锚定回归检测 | A3 产物 | 按 horizon 分组统计 OOS 日期 | 5 个 horizon 的验证窗口**两两不相交（intersection = 0）**，跨期限融合无日期可融合，OOS 面板塌到 ~10 股/日 | 同上 | ⚠️ 发现回归 |
| A5 | 回归修复验证 | 60 股 pilot + FT-Transformer | CLI（GPU） | 5 个 horizon 验证窗口**逐日相同**：2025-08-22→2026-01-21，各 100 天 | `ft_gpu/training/walk_forward_predictions.csv` | ✅ |

## A-bis. 第二个同族缺陷：`selection_min_oos_days` 的单位歧义

| # | 测试目标 | 实际结果 | 通过 |
|---|---|---|---|
| A6 | 修复后 5 折仍失败 | 400 股 pilot，nSplits=5 产出**恰好 100** 个 OOS 日；治理阶段报 `insufficient early OOS days: observed=79, required=80` —— 在整个组合搜索付费之后 | ⚠️ 发现新缺陷 |
| A7 | 定位根因 | 切分器与预检按**交易日**计数，过拟合治理按**日收益观测数**计数；`nav.pct_change().dropna()`（`v7_train.py:1515`）恰好消耗 1 个观测。80 个选择日 ⇒ 79 个收益 | ✅ |
| A8 | 修复 | `required_oos_days()` 成为唯一定义并预留该日：80+20+**1** = **101** ⇒ 最少 **6 折**。出厂默认 nSplits 由 5 改为 **6**（UI draft + API schema） | ✅ |
| A9 | 6 折全程跑通 | `contract → dataset → factor_screening(312→78) → oos_budget(6/6=120 ≥ 101, pass) → training → prediction → portfolio → risk → evidence` **全部阶段完成** | ✅ |

### A9 产出的真实证据

| 闸门 | 实测值 | 结论 |
|---|---|---|
| rank_ic_mean | 0.0553 | pass |
| rank_ic_stability | 0.560 | pass |
| turnover_adjusted_net_return | 0.0877 | pass |
| max_drawdown | −3.38% | pass |
| selection_pressure | 8.167（阈值 ≥3.0） | pass |
| training / prediction symbols | 400 / 356 | pass |
| **excess_return_after_costs** | **0.0** | **fail** |

`QUANT_ACCEPTANCE_STATUS=failed`，唯一失败项 `excess_return_after_costs_failed`：本次未指定 benchmarkSymbol，超额收益定义上恒为 0。这是**研究结论「未通过验收」**，不是工程故障。
产物：`portfolio/target_weights.parquet`、`reports/walk_forward_backtest.json`、`reports/paper_report/`、`reports/acceptance_report.json`。

回测订单统计：**skipped_orders=2291，failed_orders=20** —— A股可交易性约束过滤掉的委托量级远大于失败量级，这条信息目前只存在于 JSON 里，网页端不可查询。

## B. 失败分类（blocked ≠ rejected ≠ failed）

| # | 测试目标 | 真实输入 | 启动方式 | 实际结果 | 产物 | 通过 |
|---|---|---|---|---|---|---|
| B1 | 不可行配置在训练前中止 | 真实 gold 数据集，nSplits=2（40 天 OOS < 协议 100 天） | CLI | **退出码 4**，`verdict=blocked`，stage=`preflight`，约 90 秒内中止，**未进入训练** | `blocked_test/research_verdict.json` | ✅ |
| B2 | remediation 区分两种成因 | 构造两组参数 | 单元测试 | 请求太小 → "把 nSplits 提高到至少 5 折…数据本身够用"；数据太短 → "数据跨度不足…单纯提高 nSplits 不会有帮助" | `tests/quant_ui/test_blocked_configuration.py` | ✅ |
| B3 | 可行配置不误报 | 60 股 pilot，nSplits=5 | CLI（FT GPU 运行） | `oos_budget` 阶段输出 `5/5 folds = 100 days (protocol needs 100)`，status=pass，继续训练 | `ft_gpu.log` | ✅ |

## C. 运行路径

| # | 路径 | 实际结果 | 通过 |
|---|---|---|---|
| C1 | Ridge / CLI | A1→A3 全程验证，训练→预测→组合阶段均执行 | ✅ |
| C2 | FT-Transformer GPU | `--require-gpu --ft-device cuda`，5 折全部在 CUDA 上训练完成（实测 762 MiB 显存 / 50% 利用率） | ✅ |
| C3 | 网页入口与 CLI 是否同一管线 | `services/quant_api/services/jobs.py` 构造的仍是同一条 CLI 命令；**未在真实浏览器中端到端跑通网页启动** | ⚠️ 未验证 |

## D. 已知未解决问题（本次未修）

| 现象 | 证据 | 影响 |
|---|---|---|
| `dropna(subset=[label, *features])` 要求**全部**特征非空，特征越多有效股票越少 | `factor-screening-mode off` + 320 特征 ⇒ 每日仅 ~4 只可选（`ft_gpu/predictions/predictions.parquet`） | 有效宇宙塌陷，组合无法构建 |
| acceptance gate `min_effective_universe_by_date=1`（非 production 模式） | `v7_train.py:1760` | 该闸门在研究模式下形同虚设 |
| 60 股 pilot 对本管线过小 | `no_viable_portfolio_candidate`，即使 top_k=5 也"覆盖了整个可选宇宙" | pilot 规模需与 top_k / max_weight 匹配，目前无预检 |
| 请求 fundamental 选股但未提供 fundamentals-root | `ft_gpu.log`：`fundamental stock selection requested but no PIT fundamental metrics are available` | 可预检但未预检 |

## E. 测试

| 范围 | 结果 |
|---|---|
| 前端 `vitest` | 70 passed / 1 skipped |
| 前端 `tsc --noEmit` | 通过 |
| 后端 `tests/quant_ui/` + 折数预算 | 207 passed |
| 后端全量 `pytest` | **2119 passed, 4 skipped, 0 failed**（3 分 12 秒，无 deselect） |

修复前后的全量套件耗时对比本身就是一个结论：

| 运行 | 结果 | 耗时 |
|---|---|---|
| 初次（原样） | 1943 passed, 2 failed，随后**挂死**，约 172 个用例未执行 | 48:30（含约 10 分钟挂死，需 SIGINT） |
| deselect 两个联网用例后 | 2116 passed, 1 failed | 35:05 |
| 加 `tests/conftest.py` 关闭环境联网后 | **2119 passed, 0 failed** | **3:12** |

35:05 → 3:12（**11 倍**）说明发起真实网络请求的**远不止那两个用例**，而是散布在整个套件里。

### 全量测试原本跑不完，根因是仓库根目录的 `.env`

首次全量运行：`1943 passed, 2 failed`，然后进程卡在 `ssl.py`，与 `172.217.116.4:443`（Google）
保持 ESTABLISHED 连接约 10 分钟无输出，**其后约 172 个用例根本没有执行**，必须 SIGINT 才能取回结果。

根因（已实证）：仓库根目录 `.env` 同时含有 `TICKFLOW_API_KEY` 和
`QUANTAGENT_LLM_ENABLED=1` / `QUANTAGENT_LLM_ALLOW_NETWORK=1` / `provider=gemini`；
`src/quantagent/agents/llm_skill_client.py:267` 的 `load_dotenv(..., override=False)`
会把整个 `.env` **注入进程级 `os.environ`**。实测：

```
before import: TICKFLOW_API_KEY in os.environ -> False
after  _load_dotenv_once(): -> True
```

一个根因同时造成两个失败：

1. `tests/test_v7_theme_research_pipeline.py` 的两个 `run_daily_v7_research` 用例
   因此发起**真实 Gemini 网络调用**（`.env` 里 timeout 180s，但叠加 fallback 链后挂了约 10 分钟）。
2. `test_connection_vault_never_returns_or_persists_secret` 失败在
   `disconnected...["connected"] is False` —— `ConnectionManager._public()` 认为
   变量出现在 `os.environ` 即为已连接，于是 `.env` 注入的 `TICKFLOW_API_KEY`
   让**断开连接之后仍然显示已连接**。这不是单纯的用例顺序问题，
   而是"在界面上断开某个数据源，其实并没有断开"的真实行为。

已加 `tests/conftest.py`：在 collection 之前把这几个变量置空
（置空而非删除 —— `override=False` 只跳过已存在的键，删除反而会被 `.env` 重新注入），
并保留 `QUANTAGENT_TEST_ALLOW_NETWORK=1` 作为显式联网开关。
修复后这两个用例 2.9 秒通过。

**未验证的一点：** 这两个用例在联网状态下究竟断言了什么，无法判断 —— 它们在本机从未跑完过。
关闭 LLM 后它们走的是离线/拒绝路径。

### 本次改动引入并已修复的回归

`tests/test_v7_realdata_pipeline.py::test_v7_training_experiment_writes_validation_artifacts`
—— 该 fixture 只有 25–29 个可用日期，原先能跑通**正是因为首折训练窗口被 embargo+purge 悄悄削掉**；
间隔改为额外预留后需要 `min_train_days+embargo+purge+valid_size`。
已把 fixture 从 50 天提高到 120 天，并把 "produced no walk-forward folds" 的报错改成
直接给出「现有 N 个可用交易日，一折需要 M 个（各项构成）」。

## F. 独立验证阶段（2026-08-03）

### F1. 闸门必须区分「测到了坏结果」与「根本没测」

`excess_return_after_costs` 只在有基准时才有定义。`paper_report.py` 正确返回 `None`，
但闸门用 `metrics.get(x, 0.0)` 把它**强制成 0.0**，再判 `0.0 > 0.0` 失败，
写下 `actual: 0.0` + `excess_return_after_costs_failed`。
读报告的人看到的是「该候选实测超额为零」，而真相是**没有任何测量发生**。

修复：每个闸门带显式 `status` = `pass` / `fail` / `unknown`。
`unknown` 永不算通过（`passed=False`），但与实测失败区分开，并带 `detail` 说明如何补齐证据。
5 个测试锁定。**注意这是系统性问题**：所有闸门都用 `metrics.get(x, 0.0)`，本次只修了超额收益这一处。

### F2. 断开连接必须真的断开

仓库 `.env` 被 `load_dotenv` 注入进程级 `os.environ`，而 `ConnectionManager._public()`
把「变量存在于 os.environ」当作已连接 ⇒ 操作员在界面点断开之后，
状态仍显示已连接，`environment_for()` 仍然把凭据发给作业。
修复：显式断开的 provider 进入抑制集合，压过环境变量；重新 connect 才解除。6 个测试锁定。

### F3. 2291 条 skipped orders 的真实构成

| 类别 | 条数 | 实情 |
|---|---|---|
| 纯 no-op | **1,971 (86%)** | 订单循环遍历**每一个有价格的标的**；无持仓且无目标权重的标的走进买入分支、取整为 0 股，被记成 `skipped_invalid_lot` |
| 真实但不足一手 | 320 | 其中 315 条隐含股数 < 100；中位数 **0.0025 股**，来自 ~1e-7 的目标权重尾巴 |

两者都不是交易所规则，却都用一个暗示交易所规则的名字上报。

修复：无意图不再产生审计行（`negligible_weight` 下限）；真实的不足一手改名
`skipped_below_min_lot` 并带 `implied_shares`。**真实数据验证**（同一 400 股 pilot）：

| | 修复前 | 修复后 |
|---|---|---|
| skipped | 2,291 | **153** |
| 原因 | 全部 `skipped_invalid_lot` | 全部 `skipped_below_min_lot` |
| 可下钻 | 无 | implied_shares 中位数 44.2，区间 −100 ~ +148.6 |
| 成交 | 157 | 157（未受影响） |

### F4. 全市场训练的资源闸门（基于实测而非估计）

系数来自实测：400 股 / 1,020,000 行 → 峰值 RSS **15.3 GiB**。
按标签面板真实行数（而非 股数×交易日，后者高估约 1/3）折算：

| 范围 | 行数 | 预计峰值 | 结论 |
|---|---|---|---|
| 400 股 pilot | 753,825 | 11.3 GiB | pass |
| 全宇宙 5,790 股 | 10,911,611 | **163.7 GiB** | **blocked**（可用 33.7 GiB） |

在数据集构建**开始前**中止，退出码 4 / `blocked`，而不是跑几十分钟后被 OOM 杀掉。5 个测试锁定。

### F5. 黄金回测场景（Section 7）—— 发现两个 P0 级回测正确性缺陷

原有 `tests/test_backtest_engine.py` 只有 1 个测试，断言 `nav_curve.notna().all()` 与
`trades.shape[0] > 0`。这种断言无法区分「正确的模拟器」和「印花税算错、T+0 结算、
涨停照买」的模拟器。改写为 12 个**手工可验算**的场景后立刻暴露：

**P0-1：滑点被收了两次。**
`AShareFillModel.fill()` 把成交价按 `slippage_bps(2.0) + impact` 移动，并算出
`slippage_cost` —— 这个值**被丢弃**。随后 `_execute_buy` / `_execute_sell`
在**已经滑过的价格**上再按 `cost.slippage_bps(5.0)` 收一笔。
实际滑点 ≈ 7 bps，而任何配置都声明 5 bps。
修复：滑点只体现在成交价里，记录不再重复计费；手续费一律按成交价计算。

**P0-2：卖出被跌停挡住时，系统会买入。**
`enforce_tradability` 在卖出受阻时把目标**权重**钉在信号日价格上；
以跌 10% 后的成交价重解该权重需要**更多股数** ⇒ 一笔被拒绝的卖出变成了
**向下跌停股加仓 1,100 股**（实测）。
修复：受阻的卖出直接拒绝并记录，不再产生反向委托。

**P0-3（可审计性）：涨跌停拦截原本不留任何记录。**
`enforce_tradability` 在订单循环之前就把目标中和掉，`delta == 0` 直接 `continue`，
显式的 `limit_up_no_buy` 分支**不可达**。委托既没有成交也没有原因，直接从审计链里消失。
修复后：`2026-01-06 600000.SH limit_up_no_buy` 会出现在 rejects 中。

## E. 未做（明确声明）

用户需求中的以下部分**本次完全未实现**，不得视为已交付：

- 网页端完整研究闭环（第三节）
- 流式 / 事件驱动回测引擎与订单级对账（第五节）
- 订单作为一等实体的查询与钻取（第六节）
- 四层风控体系（第七节）
- 自动策略研究实验室（第八节）
- 策略与实验生命周期管理（第九节）
- 模拟盘 / 影子 / canary / 实盘准入分级（第十节）
- 员工可用性、引导模式、权限与审批（第十一节）

**系统当前不具备实盘能力，也不应被描述为"私募企业级已完成"。**
