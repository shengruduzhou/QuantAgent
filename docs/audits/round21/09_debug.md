# Round 21 — R9 Debug 与代码治理 / Debug & Code Governance

- 日期 / Date: 2026-08-18
- 基线 / Baseline: `main` @ `057f8cf`
- 角色 / Role: R9（只读审计；不得改 `src/` `apps/` `services/`，不得删除文件，不得 commit）
- 状态 / Status: **完成**（增量落盘；A 段 9 条、B 段 12 个删除提案 + 14 个归档提案、C 段 7 项裁定）

## 排除范围 / Already-known, not re-reported

DEF-033 风控约束 fail-open、DEF-034 冲击成本未收、DEF-035 clean_room 缺 engine.py、
DEF-036 RL 零动作加杠杆、DEF-037 非线性命令不在白名单、interior-bar NAV 错位、
RiskGate/paper.RiskEngine 零生产调用点、认证面板只有 15 特征、RL reward 无风险项、
`[all]` extra 名不副实。

---

## A. 真 bug 清单 / Real defects

### A.0 汇总 / Summary

| # | 优先级 | 一句话 | 位置 | 实测复现 |
|---|---|---|---|---|
| A-01 | P1 | `sector_usable_for_diagnostics` 恒为 `True`，生产者写死常量 + 消费者默认 `True` | `data/sector/sector_mapping.py:398`、`diagnostics/sector_audit.py:95` | ✅ 2 个脚本 |
| A-04 | P1 | 治理审计哈希链并发下静默分叉，160 条写入只有 1 条可达 | `governance/audit.py:59-72` | ✅ 4 进程实测 |
| A-05 | P1 | `quantagent.governance` 决策协议零生产调用点 | `governance/{protocol,agents,envelopes,audit}.py` | ✅ grep |
| A-06 | P1 | 工作站回测台 `ffill` 价格面板 ⇒ 用从未打印的价成交、缺口记 0% ⇒ 总收益虚增 7.67pp | `services/.../market_playbooks_v3.py:177-178` | ✅ 600 日双资产 |
| A-09 | P1 | 生产 `OrderManagerConfig.max_participation_rate` 从未被读取（装饰性风控旋钮） | `execution/order_manager.py:135` | ✅ `count == 1` |
| A-02 | P2 | 主题因子覆盖闸门：未测量的主题通过，测量为空的被拒 | `themes/stock_pool_gate.py:97` | grep |
| A-03 | P2 | 缺 `is_exchange_disclosure` 时默认按交易所公告计分（可信度 0.55→0.85） | `fundamental/order_contract_agent.py:17` | 静态 |
| A-07 | P2 | `_perf.annualReturn` 无评估天数、无短窗口警告（违反 AGENTS.md） | `services/.../market_playbooks.py:169-172` | 静态 |
| A-08 | P2 | 优化器把缺协方差填 0（=零风险资产） | `quant_math/optimizer.py:75,202,283` | 观察，未复现 |

三条实测复现脚本位于
`/tmp/claude-1001/-home-shanhefu-QuantAgent/49d74420-495c-4c61-9c21-9f9c33dbb797/scratchpad/`
（`repro_sector_gate.py`、`repro_sector_gate2.py`、`repro_audit_chain_fork.py`、
`repro_playbook_ffill.py`），按纪律**未放入仓库**。

### A.0b 已检查且判定为**正确**的路径（记录检查范围，避免下轮重复）

| 检查项 | 范围 | 结论 |
|---|---|---|
| 自定义 `__len__`/`__bool__` 参与 `or` 短路 | 6 个类（`paper.EventLedger`、`domain.CanonicalLedger`、`governance.AuditLog`、`domain.IdempotencyStore`、`EntitlementMatrix`、`CapabilityMatrix`） | **已闭合**。`order_manager.py:178` 与 `paper/broker.py:152` 现在对"同时传两个账本"直接 raise，不再 `or`。`governance.py:389,474` 的 `matrix` 是 JSON dict，非上述类 |
| 裸 `except:` | `src` + `services`（**AST 遍历**，非 grep） | **0 处** |
| 可变默认参数 `=[]`/`={}`/`=set()` | `src` + `services`（**AST 遍历**所有 `FunctionDef.args.defaults` 与 `kw_defaults`） | **0 处** |
| `except ...: pass/continue/return None` 在风控/执行/组合/回测/评估路径 | 6 个目录 | 逐个读过，均为**良性**（临时文件清理、可选依赖探测、类型窄化），未发现 fail-open |
| `datetime.now()` 裸调用污染结算日 | `src` + `services` | 未发现 DEF-016 复发；`paper/daily_loop.py:1112` 的 `date.today()` 只在用户显式传 `"today"` 时生效 |
| 原子写 / fsync | `os.replace` 10 处、`os.fsync` 26 处 | `paper/daily_loop.py:1053-1064`、`paper/pending_signal.py:285-298` 是**正确样板**（write→flush→fsync→replace→fsync(dir)）。唯一缺口是 A-04 |
| 生产逻辑用裸 `assert` | `src` + `services` | 6 处，**全部**是 `assert X is not None` 类型窄化，非业务判据 |
| `DEF-022`（基准缺口填 0） | `backtest/paper_report.py:155-166` | **已修且带注释说明**，未复发 |


### A-01【P1】`sector_usable_for_diagnostics` 是一个**恒为 True 的装饰性闸门**（两处叠加）

- `src/quantagent/data/sector/sector_mapping.py:398` — 生产者侧写死字面量
  `"sector_usable_for_diagnostics": True`，与紧邻的
  `"sector_usable_for_optimization": bool(usable_for_optimization)` 形成对照：
  同一函数里一个是实测推导，一个是常量。
- `src/quantagent/diagnostics/sector_audit.py:95` — 消费者侧 `.get(..., True)`，
  即 manifest 完全不存在时也返回 `True`；而同一构造器里的
  `sector_usable_for_optimization` 与 `st_usable_for_risk_filter` 默认都是 `False`。
  **三个同族标志，两个 fail-closed，一个 fail-open。**

⇒ 该标志在**所有可达状态**下都是 `True`：全仓只有这两个赋值点
（`grep -rn '"sector_usable_for_diagnostics"' src services scripts` = 2 命中，
其中一个是字面 `True`，另一个是默认 `True`）。它永远无法为 `False`，
因此它保护不了任何东西。违反 `AGENTS.md`「**禁止**把关卡写成常量 `True`，
NOT_RUN 不得当作 PASS」。

**失败场景（实测）**：

```
$ AI_quant_venv/bin/python3 scratchpad/repro_sector_gate.py
sector_usable_for_diagnostics : True
sector_usable_for_optimization: False
st_usable_for_risk_filter     : False
sector_reason                 : manifest_missing        <-- 理由说"清单缺失"，布尔却说"可用"
AssertionError: FAIL-OPEN: no manifest at all, yet the sector map is declared usable for diagnostics

$ AI_quant_venv/bin/python3 scratchpad/repro_sector_gate2.py     # 空 sector map
reason                        : sector_validation_failed,sector_level_1_coverage_below_threshold,sector_level_2_coverage_below_threshold
observed                      : {'sector_level_1_coverage': 0.0, 'sector_level_2_coverage': 0.0, ...}
sector_usable_for_optimization: False
sector_usable_for_diagnostics : True                     <-- 校验失败 + 覆盖率 0.0，仍报可用
```

**最小复现**：见上两个脚本（覆盖率 0.0 且 validation failed 仍返回 True）。

**建议修法**：`sector_coverage_gate` 用一条**比 optimization 宽松但仍可测**的判据
推导 diagnostics（例如 `validation["status"] == "passed" and level_1_coverage > 0`），
消费者侧默认改为 `False`，并沿用 `reason="manifest_missing"` 使三态可审计。
风险：接受此修法后 `tests/diagnostics/test_sector_audit.py:31`、
`tests/test_sector_mapping_data_layer.py:123`、
`tests/portfolio/test_sector_gate_no_contamination.py:14` 三处断言需同步更新
（它们目前钉住的正是这个常量）。

### A-02【P2】主题因子覆盖硬闸门：**未测量的主题通过，测量为空的主题被拒**

`src/quantagent/themes/stock_pool_gate.py:70-98`

```python
factor_coverage_by_theme = {r.theme_name: bool(r.applicable_factor_names) for r in selection_reports}
...
if config.require_factor_coverage and not factor_coverage_by_theme.get(member.theme, True):
```

`factor_coverage_by_theme` 只包含**有 selection report 的主题**。一个
`selection_reports` 里根本没出现的主题（= 从未被评估过因子适用性）走
`.get(theme, True)` ⇒ 视为"有覆盖" ⇒ 通过硬闸门；而一个**被评估过、
但适用因子为空**的主题 ⇒ `False` ⇒ 被拒。**没测过比测出空更容易过关。**
与本仓 DEF-023（未审计=记录为干净）同形。

**建议修法**：`.get(member.theme, False)`，并把缺席记为
`drop_log[symbol] = "factor_coverage_unknown"`（与 `no_factor_coverage_for_theme`
区分，使"未测量"和"测量为空"在审计日志里可分辨）。

### A-03【P2】证据可信度：缺少交易所披露标记时**默认按官方披露计分**

`src/quantagent/fundamental/order_contract_agent.py:17-19`

```python
official = bool(row.get("is_exchange_disclosure", True))
reliability = 0.85 if official else 0.55
confidence = min(0.95, 0.45 + 0.30 * official + 0.20 * amount_score)
```

字段缺失 ⇒ `official=True` ⇒ `source_authority_level` / `evidence_quality` /
`source_reliability` 全部从 0.55 抬到 **0.85**，`confidence` +0.30，
`source_type` 从 `NEWS` 变成 `COMPANY_ANNOUNCEMENT`，
`cross_validation_count` 默认从 0 变成 1（第 38 行同一条件）。
**来源不明的传闻被自动升格为交易所公告。** 修法：默认 `False`。


### A-04【P1】治理审计日志的哈希链在并发写入下**静默分叉**，159/160 条记录成为孤儿

`src/quantagent/governance/audit.py:59-72`

```python
def append(self, *, kind, actor, subject, payload) -> AuditEntry:
    last = self.last_entry()          # <-- 从磁盘读整个文件取链尾
    entry = AuditEntry(sequence=(last.sequence + 1) if last else 0, ...,
                       prev_hash=last.entry_hash if last else GENESIS_HASH)
    with open(self.path, "a", encoding="utf-8") as handle:   # <-- 无锁、无 fsync
        handle.write(json.dumps(...) + "\n")
```

三个问题叠加：

1. **无文件锁**。`append` 是 read-then-write，两个进程会读到同一条链尾，
   写出**相同的 `sequence` 和相同的 `prev_hash`** ⇒ 哈希链分叉。
   对比：本仓其他所有 append-only 写者（`paper/pending_signal.py:38`、
   `paper/account_identity.py:37`、`paper/execution_journal.py:40`、
   `execution/parent_child.py:47`）**都** import 了 `fcntl`/`msvcrt` 做锁；
   `governance/audit.py` 是唯一没有的一个，偏偏它是唯一带哈希链的。
2. **无 `os.fsync`**。`domain/ledger.py` 在 DEF-017 之后有 latch 保护，
   这里没有任何等价物。
3. **`append` 是 O(n)**（`last_entry()` 遍历全文件），n 次 append = O(n²)；
   docstring 自己写着"the log grows without bound"。

**失败场景（实测，4 进程 × 40 条）**：

```
$ AI_quant_venv/bin/python3 scratchpad/repro_audit_chain_fork.py
lines written        : 160  (expected 160)
distinct sequences   : 76   (expected 160)
duplicate sequences  : 40   e.g. [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
forked prev_hash     : 40   e.g. [4, 4, 4, 4, 4]
entries reachable by following the chain from genesis: 1
  => 159 governance records ORPHANED, no exception raised
```

**160 条治理决策写进了文件，从创世哈希出发只能走到 1 条**；159 条否决/批准
记录在任何按链遍历的验证器眼里都不存在，而写入方**没有收到任何错误**。
这正是本仓反复出现的形状：损坏发生了，但每一层的内部一致性检查都通过。

**最小复现**：上面的脚本（4 个进程、`mp.Barrier` 同步起跑）。

**建议修法**：`append` 全程持有排他文件锁（`fcntl.flock(LOCK_EX)`），
锁内 re-read 链尾 → 写入 → `flush` + `os.fsync`；Windows 走 `msvcrt.locking`。
另建议把 `last_entry()` 改为记忆化链尾（锁内校验），把 O(n²) 降到 O(n)。

### A-05【P1】整个 `quantagent.governance` 决策协议**零生产调用点**（与 R2 的 RiskGate 同形，但是另一个子系统）

`grep -rn "quantagent\.governance" --include=*.py . | grep -v src/quantagent/governance/`
的**全部**非测试命中只有两条：

- `scripts/verify_pr_isolated_audit.py:23` → 只用 `github_audit_gate`
- `docs/research/多Agent量化架构设计.md:115` → 文档

即 `protocol.py`（10.3 kB，四条否决规则：结构有效性 / 硬否决 / 强制覆盖 /
顺序）、`agents.py`（13.6 kB）、`envelopes.py`（5.2 kB）、`audit.py`（4.3 kB）
**只被 `tests/test_governance_protocol.py` 与包自己的 `__init__.py` 引用**。

`protocol.py` 的 docstring 明确写着"A missing mandatory agent is
`NEEDS_EVIDENCE`, never an implicit approval, because '没人查过' 和
'查过且没问题' 不能产生相同结果"——**这条铁律实现得很好，但它没有接到任何
生产决策路径上**。A-04 的哈希链缺陷之所以在生产上尚未造成损失，原因也正是
这个：没人在生产里写这份日志。

**不建议删除**（质量高、语义正确）；建议记入"待接线清单"，与 R2 的
`RiskGate` / `paper.RiskEngine` 并列。

### A-06【P1】工作站回测台（playbook 13/14/15）对多资产价格面板做 `ffill`，把**从未打印过的价格**当成成交价，并把缺口日记成 0% 收益日

`services/quant_api/services/market_playbooks_v3.py:177-178`

```python
close      = pd.concat(close_parts, axis=1).sort_index().ffill()
open_price = pd.concat(open_parts,  axis=1).reindex(close.index).ffill()
```

`pd.concat(axis=1)` 取的是所有资产的**日期并集**。任一资产在某日没有 bar
（停牌 / 不同交易日历 / 历史较短），`ffill` 把上一根 bar 的价格搬过来。后果三层：

1. **成交价造假**：`_momentum_case` 用 `open_to_open = open.shift(-1)/open - 1`
   标记损益，`assumptions` 里写着 `"execution": "T close target -> T+1 open;
   open-to-open marking"`。被 ffill 出来的 open **是一个从未在市场上打印过的
   价格**，策略却被记为在这个价格上成交。这直接踩 `AGENTS.md`
   「不允许用 mock data 让 production/research 结果看起来完整」。
2. **缺口日被记成 0% 收益日**：ffill ⇒ `pct_change` = 0 ⇒ 与 DEF-021/DEF-022
   完全同形（持仓但无行情被静默按平盘计价）。`_perf`（`market_playbooks.py:169-172`）
   的 `maxDrawdown` 因此看不到缺口期间的真实回撤。
3. **逆波动率权重被系统性拉偏**：`vol = daily_close_return.rolling(60).std()`，
   ffill 制造出一串 0 收益 ⇒ 波动被低估 ⇒ `inv_vol = active/vol` 抬高
   **数据最差的那只资产**的权重。`vol.replace(0.0, np.nan)` 只挡住精确 0，
   挡不住"被稀释到很小"。

**失败场景（实测）**：同一条真实价格路径，一份完整观测、一份每 5 个交易日
只打印 2 根 bar（其余缺失），喂给**同一个 `_momentum_case`**：

```
$ AI_quant_venv/bin/python3 scratchpad/repro_playbook_ffill.py
--- TRUE panel (no gaps) ---
  60d annualised vol  LIQUID=0.1609  GAPPY=0.2028
  reported totalReturn=+33.5102%   days scored as exactly 0.0%: 211/600
--- ffill panel (as shipped) ---
  60d annualised vol  LIQUID=0.1609  GAPPY=0.2057
  reported totalReturn=+41.1854%   days scored as exactly 0.0%: 216/600
```

**同一份底层价格过程，工作站报出的总收益从 +33.51% 变成 +41.19%（虚增 7.67 pp），
差额完全来自缺口填充口径。**

**最小复现**：`scratchpad/repro_playbook_ffill.py`（600 个交易日，两只资产，
`np.random.default_rng(7)`）。

**建议修法**：不 ffill 价格。缺 bar 的资产当日 `active=False` 且权重归 0（现金），
并在返回体里公布 `pricedSessions` / `missingSessions` 三态，与
`/api/market/stocks/.../overview` 已经做对的 `status:"unavailable"` 保持一致。
若坚持保留 ffill，必须在 `assumptions` 里显式声明"成交价可能为结转价"，
并把受影响的日期列出来 —— 现状是**既不修正也不声明**。

### A-07【P2】`_perf` 的 `annualReturn` 无评估天数、无短窗口警告

`services/quant_api/services/market_playbooks.py:169-172`

```python
def _perf(r):
    x = r.dropna().astype(float)
    ...
    return {"totalReturn":..., "annualReturn": float(nav.iloc[-1]**(252/max(1,len(x)))-1), ...}
```

返回体不含 `len(x)`，调用方（`_backtest` / `time_series_momentum` /
`windowSensitivity` 三处）也不补。`AGENTS.md`「Strategy Workbench」明写
**"短窗口年化必须带评估天数警告"**。20 个交易日的样本会被年化成一个看起来像
长期业绩的数字，而 UI 拿不到任何可据以警告的字段。
建议：`_perf` 增加 `"days": int(len(x))`，短于阈值时 `annualReturn=None`。

### A-08【P2】`quant_math` 的 `fillna(0.0)` 把"缺协方差"当成"零相关"

`src/quantagent/quant_math/optimizer.py:75, 202, 283`、`signal_fusion.py:60`：

```python
cov = covariance.reindex(index=symbols, columns=symbols).fillna(0.0)
```

`reindex` 到一个不在原协方差矩阵里的 symbol ⇒ 该行/列全 NaN ⇒ 填 0 ⇒
**该标的被当成与所有资产零相关、且自身方差为 0**。均值-方差优化器看到一个
"零风险"资产会给它极端权重；`optimizer.py:202` 的 `portfolio_variance` 也会
低估组合风险。与 A-06 第 3 点同一形状：缺失测量被替换成最有利的数值。
建议：缺协方差的 symbol 直接从优化宇宙里剔除并记录，或抛错，不得填 0。
（本条未跑复现，按章程记为**观察**，优先级 P2，供主角色核实后决定。）

### A-09【P1】生产 `OrderManagerConfig.max_participation_rate` 是一个**从未被读取**的死字段

`src/quantagent/execution/order_manager.py:135`

```python
@dataclass
class OrderManagerConfig:
    ...
    max_participation_rate: float = 0.05      # <-- 声明于此，全仓再无第二次读取
```

**零引用证明**：

```bash
$ grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=AI_quant_venv \
      --exclude-dir=node_modules --exclude-dir=runtime --exclude-dir=rd-agent \
      --exclude-dir=.claude "max_participation_rate" .
src/quantagent/execution/order_manager.py:135:    max_participation_rate: float = 0.05      <-- 唯一一次出现在 order_manager.py
src/quantagent/cli/paper.py:74,112                                                          <-- 传给 ContinuousExecutionConfig
src/quantagent/paper/continuous_execution.py:89,1196,1203,1219                              <-- 另一个同名字段，真的在用
src/quantagent/backtest/ashare_execution_simulator_impl.py:151                              <-- 又一个同名参数
tests/... (2 处)
```

`order_manager.py` 里 `max_participation_rate` **只出现 1 次**（就是这行声明）。
`OrderManager` 的任何下单路径都不读它。

**失败场景**：运维/研究员通过配置把 `OrderManagerConfig.max_participation_rate`
调到 0.01 想收紧参与率——**没有任何效果，也没有任何报错或警告**。生产
`OrderManager` 依然按上游给什么就下什么。这是一个**看起来在做风控、实际是
装饰品的旋钮**，与 A-01 的常量闸门同类。

**最小复现**：
```bash
AI_quant_venv/bin/python3 - <<'EOF'
import inspect, quantagent.execution.order_manager as om
src = inspect.getsource(om)
print("occurrences of max_participation_rate in order_manager.py:",
      src.count("max_participation_rate"))   # -> 1 (仅 dataclass 字段声明)
EOF
```

**建议修法**：二选一 —— (a) 删除该字段（它已被 `continuous_execution` 的同名
字段取代）；(b) 真正接线，且必须与 `ExecutionConstraintSet.
max_single_stock_participation_rate` 对齐并发布 `binding_constraint`。
**不允许保留现状**：一个存在但无效的风控参数比没有这个参数更危险。

---

## B. 死代码与归档提案 / Dead code & archive proposals

### B.0 先说结论：**模块级死代码远比预期少**

用 AST 建了完整 import 图（`scratchpad/import_graph.py`）：
`src/quantagent/` 共 **540** 个模块，**从未出现在任何 `.py` 的 AST import 里**的
只有 **48** 个，其中 20 个是 `__init__.py`（包入口，不算死）。

同时做了纯字符串扫描（`scratchpad/zero_ref.py`，对每个模块 stem 全仓 grep）：

```
TOTAL zero-reference modules: 0 / 540
```

⇒ **不存在"连名字都没人提过"的模块。** 用户"删掉不重要的代码"的诉求在模块
粒度上几乎没有可下手的目标。本仓真正的"屎山"不是死模块，而是
**(a) 活着但没接线的子系统**（A-05 治理协议、R2 的 `RiskGate`/`paper.RiskEngine`）、
**(b) 版本号写进文件名的分层链**（见 §C）、**(c) 写完就没人调用的孤立函数**（下表）。

### B.1 可删除清单（附 AGENTS.md 要求的零引用证明）

证明命令统一为（`$m` = 模块相对路径，`$stem` = 文件名）：

```bash
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=AI_quant_venv \
     --exclude-dir=node_modules --exclude-dir=runtime --exclude-dir=rd-agent \
     -e "quantagent.<dotted>" -e "\b$stem\b" . | grep -v "^\./src/quantagent/$m\.py:"
```

（`src/quantagent.egg-info/SOURCES.txt` 是 setuptools 生成的构建产物，
`.claude/worktrees/` 是 agent 临时工作树，两者**都不算引用**，已在下表注明。）

| # | 文件 | 行数 | 命中数（排除自身） | 命中内容 | 建议动作 | 风险 |
|---|---|---|---|---|---|---|
| 1 | `src/quantagent/agents/agent_router.py` | 84 | **0** | — | **删除** | 无 |
| 2 | `src/quantagent/factors/pipeline_v6.py` | 44 | **0** | — | **删除** | 无。名字里的 v6 已被 v7/v8 取代 |
| 3 | `src/quantagent/themes/policy_universe_builder.py` | 210 | **0** | — | **删除** | 无。最大的一块 |
| 4 | `src/quantagent/training/ablation_runner.py` | 16 | **0** | — | **删除** | 无 |
| 5 | `src/quantagent/quant_math/hmm_regime.py` | 157 | 1 | 仅 `SOURCES.txt` | **删除** | 唯一导出 `hmm_regime_alpha_multiplier` 全仓 1 次命中=自身定义 |
| 6 | `src/quantagent/strategy/rule_signals.py` | 61 | 1 | 仅 `SOURCES.txt` | **删除** | `add_short_horizon_rule_signals` 全仓 1 次命中=自身 |
| 7 | `src/quantagent/quant_math/hrp.py` | 115 | 1 | 仅 `SOURCES.txt` | **删除** | `hrp_weights` 全仓 1 次命中=自身 |
| 8 | `src/quantagent/quant_math/realized_vol.py` | 79 | 1 | 仅 `SOURCES.txt` | **删除** | `add_realized_vol_features` 全仓 1 次命中=自身 |
| 9 | `src/quantagent/fundamental/dupont.py` | 70 | 1 | 仅 `SOURCES.txt` | **删除** | `dupont_decomposition` 全仓 1 次命中=自身。⚠️ `docs/audits/round21/03_factor.md:74` 把它列为基本面特征来源 ⇒ 删前需与 R3 对齐 |
| 10 | `src/quantagent/strategy/weight_adapter.py` | 142 | 2 | `SOURCES.txt` + `.claude/worktrees/` 副本 | **删除** | 无 |
| 11 | `src/quantagent/cli/fuyao_research.py` | 152 | 2 | `SOURCES.txt` + worktree 副本 | **删除**（见 B.2） | 它定义了一个 typer 命令，但 `quantagent.cli.app` 的 **109** 个已注册命令里没有它 |
| 12 | `src/quantagent/training/composite_loss.py` | 110 | 1 | 仅 `SOURCES.txt` | **删除** | `v4_composite_loss` 全仓 3 次命中全在自身（定义 + 无 torch 存根 + 报错文案）；`ft_transformer_trainer.py:434` 的 `val_composite_loss` 是**字典键**，不是这个函数 |
| 13 | `src/quantagent/quant_math/factor_attribution.py` | 110 | 1 | `v7/agent_contracts.py:222` 把它写成 `existing_extension_points` 字符串 | **保留但标注** | 不是 import，是契约文档里的路径字符串；删除会让契约指向不存在的文件 |
| 14 | `src/quantagent/portfolio/sector_etf_allocator.py` | 49 | 1 | `v7/agent_contracts.py:199` 同上 | **保留但标注** | 同上 |
| 15 | `src/quantagent/training/oom_safe_trainer.py` | 84 | 5 | `reports/code_audit/{README,module_map,project_structure,training_backtest_flow}.md` | **不得删除** | AGENTS.md：被 docs 引用即不可删。四份代码地图都把它写成"GPU 训练入口"的组成部分 —— **文档描述的能力实际没有接线**，应先修文档或接线 |

**合计可安全删除（#1–#12）：`1174` 行。**

> ⚠️ 删除前主角色必须自己重跑上面的 grep 并确认命中为 0/仅构建产物 ——
> 本报告的证明基于 `057f8cf` 快照。

### B.2 `cli/fuyao_research.py`：一个**永远无法被调用**的 CLI 命令

```
$ AI_quant_venv/bin/python3 -c "from quantagent.cli import app; \
    print(len(app.registered_commands)); \
    print([c.name for c in app.registered_commands if 'fuyao' in str(c.name)])"
109
['fetch-fuyao-daily', 'fetch-fuyao-capability', 'fetch-fuyao-market-dump',
 'audit-fuyao-coverage', 'sync-fuyao-all']
```

`run_fuyao_research_backtest` 不在其中，`quantagent/cli/__init__.py` 从未 import
该模块。152 行带 `typer.Option` 的命令实现**在任何入口都跑不到**。这既是死代码
也是一条误导：读代码的人会以为这个命令存在。

### B.3 仓库根目录 42 个 `.md`：归档建议（**不删除**）

按"是否被 `AGENTS.md` / `CLAUDE.md` / `README.md` / `src` / `services` /
`scripts` / `tests` / `configs` / `apps` 引用"分层。命令：

```bash
grep -rIn --exclude-dir=.git ... -F "<FILE>.md" AGENTS.md CLAUDE.md README.md \
     src services scripts tests configs apps | wc -l
```

**零引用（全仓 0 次外部命中）—— 建议移入 `docs/archive/`：**

| 文件 | 外部命中 | 权威源命中 |
|---|---|---|
| `EXPERIMENT_HISTORY_SUMMARY.md` | 0 | 0 |
| `H009_DRAWDOWN_REGIME_OVERLAY.md` | 0 | 0 |
| `PBO_DSR_INPUT_CENSUS.md` | 0 | 0 |
| `PRODUCTION_REPRODUCIBLE.md` | 0 | 0 |
| `RESEARCH_LOG.md` | 0 | 0 |

**仅 1 次外部命中且权威源 0 次 —— 次优先归档：**
`CODEBASE_CLEANUP_REPORT.md`、`EXP010_CORRECTED_INC_E1.md`、`IDEA_QUEUE.md`、
`LONG_SLEEVE_DIAGNOSTIC.md`、`MODEL_FLOW_MAP.md`、`OUTPUT_ARTIFACT_AUDIT.md`、
`PBO_DSR_INPUT_VALIDATION.md`、`PRODUCTION_CONFIG_SCHEMA.md`、`design-qa.md`
（各自那 1 次命中都是另一份根目录 `.md` 的交叉引用，不是代码/配置）。

**必须保留**（被 `AGENTS.md` 直接点名或高引用）：
`ACCEPTANCE_RULES.md`(7/2)、`BASELINE_TRUST_CLASSIFICATION.md`(7/3)、
`HOLDOUT_CONTAMINATION_AUDIT.md`(12/8)、`HYPOTHESIS_REGISTRY.md`(24/18)、
`DEAD_CODE_AUDIT.md`(32/31)、`PRUNE_PLAN.md`(32/31)、
`EXPERIMENT_LEDGER.md`、`FRESH_HOLDOUT_FREEZE_MANIFEST.md` 等。

> **归档而非删除**的理由：`EXPERIMENT_LEDGER.md`(92 kB) 与
> `HYPOTHESIS_REGISTRY.md`(67 kB) 是预注册闸门的历史依据，删除等于让过去的
> 拒绝记录不可复核；同理，`*_CORRECTED_INC_E1.md` 系列记录的是**被更正过的
> 结论**，正是最不该消失的一类文档。

### B.4 已排除的误报（供主角色避免误删）

| 猜测 | 实测裁决 |
|---|---|
| `live_model_trust.py` / `_v2.py` / `_v2_policy.py` / `_v2_execution_policy.py` 是"版本化残骸" | **不成立**，见 §C.1 —— 是四层依次包裹的活链条，每层都有 import 者与测试 |
| `intraday_dot_*` 四件套是残骸 | **不成立**，实际只有**三**个文件，全部有 script/test/API adapter 引用，其中 `intraday_dot_ev_backtest` 还是 `jobs.py:241` 的受治理入口 |
| `portfolio_env.py` vs `pit_portfolio_env.py` 需二选一 | **不成立**，`PortfolioEnv` 是**受治理的弃用件**：默认构造即 `raise ValueError`，必须显式传 `acknowledge_untradable_reward=True` 才能实例化，docstring 完整说明了它的奖励为何不可交易。这是本仓弃用处理的正面样板，**建议保留** |

---

## C. 重复实现与权威版本裁定 / Duplicate implementations & authoritative version

### C.1 `live_model_trust` 四件套 —— **不是残骸，是四层依次包裹的活链条**

| 层 | 文件 | 职责 | 被谁 import |
|---|---|---|---|
| L0 | `live_model_trust_v2.py`（673 行） | digest-bound 证书原语（`ArtifactBinding` / `REQUIRED_ARTIFACT_ROLES`） | `_v2_policy`、`_v2_execution_policy`、`live_model_trust` |
| L1 | `live_model_trust_v2_policy.py`（340） | governed 层（窗口隔离） | `_v2_execution_policy:48`、2 个测试 |
| L2 | `live_model_trust_v2_execution_policy.py`（454） | trace-proven 层（`strict_target_weights` + `strict_execution_trace`） | `economic_model_gate:13`、`live_model_trust:22`、`cli/governance:14`、3 个测试 |
| L3 | `live_model_trust.py`（262） | v1 schema 读取（仅取证/BLOCKED 用）+ 委派 L2 做生产判定 | `economic_model_gate:12`、`live_session:20`、测试 |

**裁定：四个都保留。** 无一冗余，文件名里的 `v2` 是**证书 schema 版本**不是
代码版本。`live_model_trust.py` 的 docstring 明确写着 v1 "is not a production
trust root"，生产走 L2。建议的唯一改动是**改名以消除误读**（如
`trust_certificate_v2.py` / `trust_policy_governed.py` / `trust_policy_trace_proven.py`），
但那属于可选整洁工作，不是缺陷。

### C.2 `fusion.search` vs `fusion.search_corrected` —— 权威 = `search_corrected`

`fusion/__init__.py:52` `from quantagent.fusion.search_corrected import run_fusion_search`；
`search_corrected.py:33` 把 `search.run_fusion_search` 重命名为
`_legacy_run_fusion_search` 并在其上重建 PBO/DSR 的 OOS 记录
（不再把 horizon-return 当独立低频序列、不再拼接各自 reset 回 1.0 的折内 NAV）。
`cli/fusion.py:158` 从**包级**导入 ⇒ 走的是修正版 ✓。

**但存在一个静默旁路**：`search.py:558` 的 `__all__` 仍然公开导出
`run_fusion_search`，函数名与修正版**完全同名**。任何人写
`from quantagent.fusion.search import run_fusion_search` 都会拿到**未修正的
统计层**，且不会有任何报错或警告——两者签名一致、返回类型一致。
本轮实测目前只有 `search_corrected.py` 自己这么 import（1 处，合法）。

**建议（P2 加固）**：把 `search.py` 里的函数改名为
`run_fusion_search_uncorrected` 并从 `__all__` 移除，或加一个
`_ALLOW_UNCORRECTED` 关键字守卫，使旁路必须显式。理由与本仓
DEF-025 一致：**加固消费者在生产者仍公开旁路时是不够的。**

### C.3 参与率上限：**同一概念 5 个数、两个不同的值**

| 位置 | 名称 | 值 |
|---|---|---|
| `execution/order_manager.py:135` | `OrderManagerConfig.max_participation_rate` | **0.05** |
| `execution/constraints.py:122` | `ExecutionConstraintSet.max_single_stock_participation_rate` | **0.10** |
| `paper/broker.py:104` | `BrokerConfig.participation_cap` | 0.10 |
| `execution/board_fill_model.py:66` | `participation_cap`（集合竞价） | 0.10 |
| `streaming/matching.py` / `reconciliation/composite.py:305,570` | `MatcherConfig.participation_cap` | 0.10 |

**裁定**：0.10 是全仓一致的撮合口径；`OrderManagerConfig` 的 0.05 是唯一的
异类，**而且它根本没被读过**（见 A-09）。真正生效的 5% 来自另一个同名字段
`paper/continuous_execution.py:89`，它经 `:1196` 与 `:1203` 注入
`BrokerConfig(participation_cap=...)` 和 `RiskLimits(max_participation=...)`。

**建议**：删除 `OrderManagerConfig.max_participation_rate`（死字段，见 A-09），
并在 `ExecutionConstraintReport` 里显式发布 `binding_constraint`——
当撮合端 5%、风控端 10% 并存时，发布的限额与实际生效的限额不是同一个数。

### C.4 滑点/成本模型：**至少 6 个独立默认值**

| 位置 | 字段 | 值 |
|---|---|---|
| `BacktestConfig.cost.slippage_bps` | 声明值（R1 F-04 已记） | 5.0 |
| `FillModelConfig.slippage_bps` | 快引擎实际施加（R1 F-04 已记） | 2.0 |
| `streaming/matching.py:55` | `MatcherConfig.slippage_bps` | 5.0 |
| `execution/intraday_fill.py:27` | `CostConfig.slippage_bps` | **8.0** |
| `risk/retail_hft_risk.py:139,280` | `extra_slippage_bps = clip(5 + 60·penalty, 5, 80)` | 5–80（**加性叠加**） |
| `execution/reconciliation.py:48` | `max_abs_slippage_bps`（对账阈值） | 30.0 |

**裁定**：`execution/cost_model.py::AShareCostModel` 是唯一带 A 股印花税/过户费/
佣金最低 5 元语义的完整模型，应为**权威**；其余五处是各引擎自带的简化常数。
R1 已经记了 5.0 vs 2.0；本轮补充的是 **8.0（日内）与 5–80 bps（零售 HFT 惩罚）
这两条从未与前两者对账过**，且 `retail_hft_risk` 的输出是**加到**其他滑点上的，
所以同一笔交易在不同引擎下的成本可以差一个数量级。

> 另注：`risk/retail_hft_risk.py:283-289` 的 `institutional_dump_risk=0.30`、
> `quote_stuffing_risk=0.30`、`overnight_gap_risk=0.40`、`short_reversal_risk=0.20`、
> `block_trade_risk=0.20` 全是**写死的常量**，不随任何输入变化，
> 只有 `limit_volatility_risk` 在 0.30/0.10 之间二选一。这是一份看起来
> 逐标的定制、实际上除涨跌停外完全不含信息的风险报告。建议要么真正计算，
> 要么把这些字段发布为 `None`/`unknown`。

### C.5 NAV 计算：**至少 9 个实现，其中 5 个在 `cumprod` 前 `fillna(0.0)`**

| 位置 | 形式 |
|---|---|
| `backtest/engine.py` | 逐日持仓估值（interior-bar 错位，R1 F-01） |
| `clean_room/engine.py` | 时钟正确的参照实现（本轮 DEF-035 新建） |
| `paper/account_target_state.py:151,182` | `cash` / `cash + market_values.sum()` |
| `domain/ledger.py` + composite replay | 事件溯源账本 |
| `optimization/multi_objective_loss.py:184` | `(1 + returns.fillna(0.0)).cumprod()` |
| `ensemble/blend_optimizer.py:199` | `(1 + excess.fillna(0.0)).cumprod()` |
| `portfolio/walk_forward_sleeve_allocator.py:168` | `(1 + returns.fillna(0.0)).cumprod()` |
| `fusion/search.py:451` / `search_corrected.py:69` | 折内 NAV / 拼接 NAV |
| `services/.../market_playbooks*.py` | `(1+net).cumprod()`，`net` 已被 `fillna(0)`（A-06） |

**裁定**：没有单一权威 NAV，也不建议强行统一（事件溯源账本与研究用净值序列
本就该分开）。但**五处 `fillna(0.0)` 后再 `cumprod` 是同一个已知缺陷形状**
（DEF-021/DEF-022：缺失观测 → 平盘日）。建议主角色统一加一条不变量：
**任何进入 `cumprod` 的收益序列必须先声明缺失日的处置**（drop 并缩短评估期 /
报 `None` / 显式标注补零），禁止裸 `fillna(0.0)`。这一条可以写成一个
lint 规则钉住，比逐个修更省事。

### C.6 Walk-forward 切分：4 个独立实现

| 位置 | 函数 | 用途 |
|---|---|---|
| `training/splitters.py:146` | `split_walk_forward` / `purged_walk_forward_splits` | **权威**（`AGENTS.md` 的 New Modules 明确登记） |
| `quant_math/purged_cv.py:21,56` | `purged_kfold_split` / `combinatorial_purged_split` | CPCV，与上者互补而非重复 |
| `research/intraday_dot_walkforward.py:45` | `make_walk_forward_splits` | 日内专用 |
| `portfolio/walk_forward_sleeve_allocator.py:100` | `_walk_forward_splits` | sleeve 配权专用，私有 |

`fusion/search.py:311` 正确复用了 `purged_walk_forward_splits` ✓。
**裁定**：`training/splitters.py` 是权威。
`walk_forward_sleeve_allocator._walk_forward_splits:100-106` **确实用了**
`config.embargo_days`（本条已自我更正）；但
`intraday_dot_walkforward.make_walk_forward_splits:45-52` 的签名是
`(dates, *, train_days, validation_days, test_days, step_days)` —— **没有
embargo，也没有 purge**。建议不删除，但在其 docstring 里显式声明"本切分不做
purge/embargo，其 OOS 数字不可与 `splitters.py` 的结果并列比较"。

### C.7 `market_playbooks{,_v2,_v3}.py` —— 版本号进文件名的继承链

`v3(MarketPlaybookService) → v2(MarketPlaybookService) → v1(MarketPlaybookService)`，
三个类**同名**，只有 v3 被 `routes/fuyao_playbooks.py:71,84` 实例化。
**裁定：不可删**（v3 依赖 v2 依赖 v1 的方法与 `_perf`/`_records`/`_finite` 工具）。
但这是本轮遇到的最典型的"屎山"形态：

- 三个同名类靠 import alias 区分（`_BaseMarketPlaybookService` / `_V2MarketPlaybookService`）；
- `market_playbooks.py` 大量单行压缩写法（`p=...; cons=...; allowed=...` 一行 5 个语句，
  见 `:128`、`:148`、`:172`），A-06 与 A-07 两个缺陷正好都藏在这种行里；
- 版本号进文件名意味着下一次改进大概率产生 `_v4.py`。

**建议**：重构为 `market_playbooks/` 包（`base.py` / `perf.py` / `backtests.py`），
类只保留一个，差异用组合而非继承；这是**唯一**建议主角色投入重构成本的模块。

---

