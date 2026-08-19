# Round 24 — R9 代码治理：收尾删除、装饰性旋钮、根目录归档 / Code Governance

- 日期 / Date: 2026-08-20
- 基线 / Baseline: `main` @ `3d4ecd8`
- 分支 / Branch: `agent/round24-cleanup`（独立 worktree）
- 角色 / Role: R9（可写：执行 Round 23 未完成的删除 + P1 旋钮 + 根目录归档）
- 上一轮报告 / Previous: `docs/audits/round23/09_deadcode.md`

## 0. 本轮纪律 / Discipline

- **禁用 `git stash`**（`refs/stash` 是仓库级共享的，Round 22 实测它会弹出别的
  worktree agent 的工作）。本轮零 stash。
- **不触碰** `src/quantagent/factors/**`、`src/quantagent/data/**`、
  `src/quantagent/rl/**`（主角色与另外两个角色正在改）。
- 每个删除附 `AGENTS.md:38` 要求的零引用 grep 命令与其输出。
- **分批**（删除 / 旋钮 / 归档），每批跑一次全量 pytest，全绿才做下一批。
- **不得修改任何现有测试的断言**。

### 0.1 基线说明 / Baseline note

本 worktree 起初检出在 `b313153`（Round 22 合并点），而 `main` 已推进到
`3d4ecd8`（含 Round 23 的两个删除 commit `766f9e6` / `3f79735`）。因此
`agent/round24-cleanup` **从 `main@3d4ecd8` 建立**，所有复核都在当前树上重跑，
不复用 Round 23 的快照结论。

本 worktree 没有 `AI_quant_venv/`（只在主检出里），故命令一律用绝对路径
`/home/shanhefu/QuantAgent/AI_quant_venv/bin/python3`。`pyproject.toml` 的
`pythonpath = ["src", "."]` 以 rootdir 为基准，已实测确认 pytest 导入的是本
worktree 的 `src/`：

```
$ PYTHONPATH=<worktree>/src .../AI_quant_venv/bin/python3 -c "import quantagent; print(quantagent.__file__)"
<worktree>/src/quantagent/__init__.py
```

---

## 1. 收尾删除：Round 23 剩下的三个 / Deletions carried over from Round 23

复核脚本（本轮实际使用的那一条；`$REL` = `src/quantagent/` 下的模块路径，
`$STEM` = 文件名，`$DOTTED` = 点分模块名，外加该模块**每一个顶层导出符号**）：

```bash
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=AI_quant_venv \
     --exclude-dir=node_modules --exclude-dir=runtime --exclude-dir=rd-agent \
     --exclude-dir=.claude \
     -e "$DOTTED" -e "\b$STEM\b" -e "\b<每个顶层符号>\b" . \
  | sed 's|^\./||' \
  | grep -v "^src/quantagent/$REL\.py:"                  # 自身定义行
  | grep -v "^src/quantagent\.egg-info/SOURCES\.txt:"    # setuptools 构建产物
```

命中里唯一被**约定排除**的是 `docs/audits/round21/09_debug.md` 与
`docs/audits/round23/09_deadcode.md` —— 那是提出删除提案的报告本身。把提案文档
算作引用会让任何提案永远无法执行。除此之外所有文档命中都计入。

### 1.1 `src/quantagent/strategy/weight_adapter.py`（142 行）— 复核通过 ⇒ **删除**

```
$ verify.sh strategy/weight_adapter horizon_alpha short_signal_to_weight long_score_to_weight \
      combine_short_long_weights combine_sleeve_target_weights apply_lot_liquidity_constraints
--- grep (imports / CLI / tests / README / AGENTS / docs), self + build artifact excluded ---
docs/audits/round21/09_debug.md:376:| 10 | `src/quantagent/strategy/weight_adapter.py` | 142 | 2 | ... | **删除** | 无 |
docs/audits/round23/09_deadcode.md:65:| 10 | `src/quantagent/strategy/weight_adapter.py` | 142 | 0 | **删除** |
docs/audits/round23/09_deadcode.md:103:$ verify.sh strategy/weight_adapter horizon_alpha short_signal_to_weight long_score_to_weight \
docs/audits/round23/09_deadcode.md:104:      combine_short_long_weights combine_sleeve_target_weights apply_lot_liquidity_constraints
--- END (empty above == zero external references) ---
```

四条命中全部是 Round 21 / Round 23 的提案表格与本复核命令的回显本身。
**零外部引用，复核通过。**

### 1.2 `src/quantagent/cli/fuyao_research.py`（152 行）— 复核通过 ⇒ **删除**

```
$ verify.sh cli/fuyao_research run_fuyao_research_backtest
--- grep (imports / CLI / tests / README / AGENTS / docs), self + build artifact excluded ---
docs/audits/round21/09_debug.md:377:| 11 | `src/quantagent/cli/fuyao_research.py` | 152 | 2 | ... | **删除**（见 B.2） | ...
docs/audits/round21/09_debug.md:388:### B.2 `cli/fuyao_research.py`：一个**永远无法被调用**的 CLI 命令
docs/audits/round21/09_debug.md:399:`run_fuyao_research_backtest` 不在其中，`quantagent/cli/__init__.py` 从未 import
docs/audits/round23/09_deadcode.md:66:| 11 | `src/quantagent/cli/fuyao_research.py` | 152 | 0 | **删除**（CLI 未注册）
docs/audits/round23/09_deadcode.md:108:$ verify.sh cli/fuyao_research run_fuyao_research_backtest
docs/audits/round23/09_deadcode.md:120:### 1.2 `cli/fuyao_research.py`：CLI 未注册的复核
--- END (empty above == zero external references) ---
```

CLI 未注册在**当前树上重跑**（不是引用 Round 21/23 的快照）：

```
$ PYTHONPATH=<worktree>/src .../python3 -c "
from quantagent.cli import app
names=[c.name for c in app.registered_commands]
print('registered commands:', len(names))
print('fuyao commands:', sorted(n for n in names if 'fuyao' in str(n)))
print('run-fuyao-research-backtest registered:', 'run-fuyao-research-backtest' in names)"
registered commands: 109
fuyao commands: ['audit-fuyao-coverage', 'fetch-fuyao-capability', 'fetch-fuyao-daily', 'fetch-fuyao-market-dump', 'sync-fuyao-all']
run-fuyao-research-backtest registered: False
```

`quantagent/cli/__init__.py` 从未 import 该模块 ⇒ 那 152 行 `typer.Option` 实现
在任何入口都跑不到。**复核通过。**

### 1.3 `src/quantagent/training/composite_loss.py`（110 行）— 复核通过 ⇒ **删除**

```
$ verify.sh training/composite_loss CompositeLossWeights v4_composite_loss
--- grep (imports / CLI / tests / README / AGENTS / docs), self + build artifact excluded ---
docs/audits/round21/09_debug.md:378:| 12 | `src/quantagent/training/composite_loss.py` | 110 | 1 | 仅 `SOURCES.txt` | **删除** | ...
docs/audits/round23/09_deadcode.md:67:| 12 | `src/quantagent/training/composite_loss.py` | 110 | 0 | **删除** |
docs/audits/round23/09_deadcode.md:112:$ verify.sh training/composite_loss CompositeLossWeights v4_composite_loss
--- END (empty above == zero external references) ---
```

**零外部引用，复核通过。**

### 1.4 ⚠️ 派生候选：`src/quantagent/training/losses.py`（71 行）—— 本轮**不删**，只登记证据

`composite_loss.py` 是 `quantagent.training.losses` 的**唯一**消费者：

```
$ grep -rIn ... -e "training.losses" -e "differentiable_spearman_loss" -e "pinball_loss" .
src/quantagent/training/composite_loss.py:25:    from quantagent.training.losses import differentiable_spearman_loss, pinball_loss
src/quantagent/training/composite_loss.py:35:        rank_loss = differentiable_spearman_loss(alpha, target_alpha)
src/quantagent/training/composite_loss.py:41:        quantile = pinball_loss(q_low, target_alpha, 0.1) + pinball_loss(q_high, target_alpha, 0.9)
```

对 `losses.py` 的全部六个顶层符号做同样扫描
（`alpha_multi_task_loss` / `soft_rank` / `differentiable_spearman_loss` /
`listmle_loss` / `daily_rank_correlation_loss` / `pinball_loss`）：除上面三行外，
其余命中**全部是英文单词 `losses` 作局部变量名**（`risk_metrics.py:14` 的
`losses = -returns.dropna()` 之类），不是引用。

也就是说：`losses.py` 今天已经有 **4/6 个符号零引用**，删掉 `composite_loss.py`
之后变成 **6/6 零引用** —— 删除会**当场制造**一个新的零引用模块，正是
Round 23 §1.6 记录的「屎山」生成机制。

**本轮仍不删**，理由是它不在主角色本轮授权的三个文件之内，而本轮指令明确要求
保持 Round 23 那种谨慎。证据已经完整，删除只需一条命令。
**请主角色在合并时裁定**：与 `composite_loss.py` 同批删除（二者构成一个 2 节点
死子图），或者明确保留并说明它服务于哪条未来路径。

### 1.5 上一轮已保留的四个 —— 本轮不重开

Round 23 §1.3–1.5 保留了 `fundamental/dupont.py`（被 `docs/audits/round21/03_factor.md`
引用）、`quant_math/factor_attribution.py` 与 `portfolio/sector_etf_allocator.py`
（被 `v7/agent_contracts.py` 当作扩展点**路径字符串**声明）、
`training/oom_safe_trainer.py`（主角色指令保留；但 Round 23 已证明它的保留理由
——「被 docs 引用」——依据的是 gitignore 掉的本地产物 `reports/**`，在版本控制
口径下它是真正的零引用）。本轮无新证据，维持原状。

---

## 2. P1：`OrderManagerConfig.max_participation_rate` —— 一个没人读的风控旋钮

Round 21 A-09 / Round 22 Q-01 提出，本轮执行。

### 2.1 现象复核（当前树重跑）

```
$ grep -c "max_participation_rate" src/quantagent/execution/order_manager.py
1
$ grep -n "max_participation_rate" src/quantagent/execution/order_manager.py
135:    max_participation_rate: float = 0.05
```

**全文件只出现一次，就是它自己的声明行。** `OrderManager` 从不读它。

写入方在当前树上是**两个**（不是主角色简报里说的一个）：

```
$ grep -rn "max_participation_rate" --include="*.py" . | grep -v node_modules
src/quantagent/execution/order_manager.py:135:    max_participation_rate: float = 0.05          # 声明，从不读
src/quantagent/backtest/ashare_execution_simulator_impl.py:151:            max_participation_rate=config.volume_participation_cap,
src/quantagent/paper/continuous_execution.py:1225:                    max_participation_rate=config.max_participation_rate,
src/quantagent/cli/paper.py:74:    max_participation_rate: float = typer.Option(...)      # 另一个同名字段，见 2.2
src/quantagent/cli/paper.py:112:        max_participation_rate=max_participation_rate,
src/quantagent/paper/continuous_execution.py:89:    max_participation_rate: float = 0.05                   # 另一个同名字段，见 2.2
src/quantagent/paper/continuous_execution.py:1197:            config=BrokerConfig(participation_cap=config.max_participation_rate),
tests/cli/test_paper_continuous_runtime.py:58:        max_participation_rate=0.05,
tests/paper/test_continuous_pending_execution.py:73:        max_participation_rate=0.05,
```

### 2.2 三个同名概念必须分开（这正是这个 bug 存活的原因）

| 字段 | 归属 | 状态 |
|---|---|---|
| `ContinuousExecutionConfig.max_participation_rate`（`paper/continuous_execution.py:89`） | 连续 paper 环的配置 | **活的**：喂给 `BrokerConfig(participation_cap=…)`，由 venue 真正计量成交 |
| `ExecutionConstraintSet.max_single_stock_participation_rate`（`execution/constraints.py:122`，默认 **0.10**） | pre-trade 约束 DSL | **活的**：`OrderManager` 经 `constraint_evaluator` 真正执行，缺 day-volume 时记 `unmeasured` 并 fail-closed |
| `OrderManagerConfig.max_participation_rate`（`order_manager.py:135`，默认 **0.05**） | OMS 配置 | **死的**：零读取 |

死字段的默认值 **0.05** 还与真正生效的 **0.10** 不同 ⇒ 读代码的人会以为
production 收的是 5%，实际是 10%。这不只是死字段，是**误导性死字段**。

### 2.3 为什么不能「接上」（遵主角色裁定）

这个字段的语义是「一笔单吃掉一根 bar 的多少」，是**成交定量**概念；而 OMS 不撮合。
把它接到 pre-trade 闸门上会重演本仓已论证过的问题 —— 生产代码自己就把这条写在
注释里（`paper/continuous_execution.py:1204-1209`）：

> ``max_participation=1.0``: the venue's ``participation_cap`` above already
> meters how much of a bar one order consumes. Setting the pre-trade limit to
> the same number would reject every order large enough to leave a remainder,
> making a partial fill unreachable.

⇒ **正确修法 = 从 `OrderManagerConfig` 移除，同时改两个写入方。**

### 2.4 「行为不变」的实测（删除前采样）

删除前，用同一组 target weights / prices / positions 走
`OrderManager.target_weights_to_order_intents`，把旋钮分别设成 **0.0 / 0.05 / 0.99**：

```
$ python3 knob_before.py            # 全量输出见 scratchpad
--- max_participation_rate=0.0  ---  { orders: [...], rejections: [...] }
--- max_participation_rate=0.05 ---  { orders: [...], rejections: [...] }
--- max_participation_rate=0.99 ---  { orders: [...], rejections: [...] }
identical across 0.0 / 0.05 / 0.99: True
```

三次的订单与拒单**完全一致**：

| 订单 | 拒单 |
|---|---|
| `("000001.SZ", buy, 1600, w=0.02, px=12.0)` | `("002594.SZ", buy, 0, skipped_below_min_lot, 500.0)` |
| `("000002.SZ", sell, 1000, w=0.0, px=9.0)` | `("300750.SZ", buy, 0, skipped_invalid_price, 0.0)` |
| `("600519.SH", sell, 200, w=0.4, px=1600.0)` | |

这就是「移除后行为不变」的实测依据：**旋钮从不参与任何判断**。
同样这组值被钉进新测试 `tests/execution/test_order_manager_participation_knob_removed.py`
（删除后重跑必须逐字相同）。

### 2.5 改动

1. `src/quantagent/execution/order_manager.py`：删掉 `max_participation_rate` 字段。
2. `src/quantagent/backtest/ashare_execution_simulator_impl.py`：删掉写入行；
   `volume_participation_cap` 继续经 `FillSimulator(participation_rate=…)` 走**唯一**
   真正计量成交的那条路。
3. `src/quantagent/paper/continuous_execution.py`：删掉写入行；
   `ContinuousExecutionConfig.max_participation_rate` 保留，它继续喂 venue 的
   `BrokerConfig(participation_cap=…)`。
4. 新测试 4 条（**不改任何现有测试的断言**）：
   - `test_order_manager_config_has_no_participation_knob` —— dataclass 字段面
   - `test_order_manager_never_reads_a_participation_rate` —— 源码面（防止改回来）
   - `test_routing_is_byte_identical_to_the_pre_removal_tree` —— 2.4 的钉值
   - `test_the_enforced_participation_limit_is_the_constraint_set_one` —— 说明真正
     生效的是哪个（移除的是 no-op，不是 enforcement）

---

## 5. 分批与全量 pytest / Batches & full-suite runs

命令：`/home/shanhefu/QuantAgent/AI_quant_venv/bin/python3 -m pytest tests/ -q -p no:cacheprovider`

| 批 | 内容 | passed | skipped | 耗时 |
|---|---|---|---|---|
| B-0 基线 | `main@3d4ecd8`，未改动 | **3522** | 47 | 615.82s |
| B-1 删除 | `strategy/weight_adapter.py`、`cli/fuyao_research.py`、`training/composite_loss.py`（−404 行） | **3522** | 47 | 640.38s |
