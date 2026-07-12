# EVALUATOR_ORDER_DEDUP_BUG (INC-E1) — 执行模拟器跨日订单去重缺陷

**发现 2026-07-06（EXP-013 执行取证）。状态：已证实、已量化、**
**修复已于 2026-07-06 经用户批准 PROMOTED 为 trusted-evaluator 默认**
**（`fix_cross_day_order_dedup=True` 默认开启；置 False 可复现 pre-INC-E1 旧模拟器用于取证对比）。**
**验证：buy→cut→rebuy 场景默认 3/3 成交（旧路径 2/3）；回归契约测试 + 旧路径复现测试均绿。**
**再验证级联进行中：单元 → 信任锚 → EXP-008 折 → EXP-009..013 → PBO/DSR。**

## 1. 一句话

`OrderManager` 的幂等去重（为实盘重试设计）在回测里跨越**整个模拟期**生效：
同一 `(symbol, side)` 的 client_order_id 是确定性 sha1（`signal_id="manual"` 恒定），
`_submit_all` 对 `history` 里已有 id 的订单**静默丢弃**——每只股票在整个回测中
**最多买一次、卖一次**。另外 `reset_daily_counters()` 从未被模拟器调用（5 单/日上限
实为 5 单/终身，同向叠加）。

## 2. 复现（4 行目标权重）

```
tw = [0.50, 0.25, 0.50, 0.50]  # 单一股票，买→减半→加回→持有
实际成交: d1 buy 50000 股, d2 sell 25000 股, d3 rebuy **静默消失**
```
脚本：本文档 §复现代码（EXP-013 取证时已跑通）；仓位从 d2 起永远停在 25%。

## 3. 机制（代码位置）

- `src/quantagent/execution/order_manager.py:209 _make_id`：`signal_id` 真值
  （reconcile 默认 `"manual"`）→ `sha1(f"{symbol}-{side}-manual-unknown")[:10]`
  确定性后缀 → 同一 (symbol, side) **永远同 id**。
- `order_manager.py:191 _submit_all`：`if order.client_order_id in self.history: continue`
  —— 无任何 skip 记录，**审计不可见**。
- `ashare_execution_simulator.py`：全模拟期共用一个 manager；`reset_daily_counters` 零调用。
- 附带既有缺陷（独立但同向）：sell 整手下取整 + `min_order_value_yuan` 使小额衰减
  卖单永久跳过（`skipped_invalid_lot`，有审计）。

## 4. 量化影响（2026-07-06 实测）

| 探针 | 结果 |
|---|---|
| F1 · C3_ema0.7 · k10 素书（生产族 variant-C） | **意图订单值的 81.6% 被静默丢弃**（1,796 单 / 141.6M CNY vs 成交 31.9M；注：含逐日重生成的重复意图，占比按当日意图口径） |
| W2 k30 书 × 恒定 gross 0.5（2024-01 整月） | 首日正确建到 48.5%，随后**每日净买 4-5% NAV 直至 100% 满仓**（新入场名字首买成交，同名后续调整全丢 + 衰减卖单地板化） |
| W2 × R2a de-risk（F1，flip 2023-08-18 → 0.5） | **invested fraction 0.998 纹丝不动** —— overlay 完全没有表达 |

## 5. 影响面（诚实分级）

**全部经 `run_strict_backtest_v8 → simulate_ashare_target_weights` 的结果都经过此层**，
包括：baseline_protocol 全部 variant-C、v8.9 +17.3% 信任锚、EXP-000..013、PBO/DSR 重放、
UI 落盘回测。基准线（panel 收盘价等权）**不受影响**（不经模拟器）。

失真程度按书型分层：
- **快轮换素书（C1/C2/C3 每日重选）**：中度失真——多数交易是首次 (symbol,side)，
  但重入场、逐日 drift 再平衡全部被丢；k10 F1 探针显示意图值大头被丢。
- **平滑/部分调仓书（EMA、B3/W2、B1..B4）**：重度失真——增量调整几乎全被丢，
  实际交易 ≈"首次入场买一笔、首次衰减卖一笔"；EXP-011/012 的"低换手"
  部分是**订单被丢的伪影**，非规则本身。
- **gross overlay（EXP-009/010/013）**：de-risk 首次卖出可执行（首卖），
  **re-risk 回补 = 同名第二次买 = 全部被丢** —— EXP-009/010 的"缩放后"结果
  实为"减仓后不完全回补"的路径；EXP-013 在 W2 书上连首卖表达都被吞。
- **EXP-011 发现③（±3pp/bps 路径噪声、>20pp 盆地跳变）**：很大程度就是本 bug——
  毫厘成交差改变哪个订单先占据 history 槽位，随后级联。

**结论污染标记**：EXP-008..013 的门判定、换手数字、DD、fold 表在修复后必须全部重跑
才可再引用；在此之前全部戳 `pre-INC-E1`。生产配置比较（C2 vs C3_ema0.7）同理。

## 6. 拟议修复（未应用——等待批准）

最小外科补丁（保留实盘幂等语义）：

```python
# ashare_execution_simulator.py 日循环内（advance_trading_day 之后）：
manager.reset_daily_counters()

# reconcile 调用改为按日 signal_id（同日重试仍幂等，跨日不再相互吞单）：
states = manager.reconcile(adjusted, prices, nav)
#  → target_weights_to_order_intents(..., signal_id=f"bt-{date:%Y%m%d}")
```

（实现细节：`reconcile` 需透传 `signal_id`；或模拟器直接调
`target_weights_to_order_intents(signal_id=f"bt-{date}")` + 手工下单循环。）

## 7. 修复后的再验证顺序（提案）

1. 单元测试：buy→cut→rebuy 三段式必须 3 单全成交（本文档 §2 场景固化为回归测试）。
2. 重跑 v8.9 rankfix k50 +17.25% 信任锚与 plus7 生产 holdout 族 → 更新
   BASELINE_TRUST_CLASSIFICATION（预期：数字全变，方向未知）。
3. 重跑 EXP-008 折表（24 评测，CPU）→ 门判定重裁。
4. 重跑 EXP-009..013（overlay/书构建结论可能反转——特别是 R2a 的 F2 优势
   与 B3/W2 的"低换手"）。
5. PBO/DSR 全量重算。

预算：全部 CPU，≈2-3 小时机时。**在批准前不动任何一行评估器代码。**

## 8. 为什么测试没抓到

现有单测用逐日新建 manager 或唯一 signal_id 场景；重放保真检验
（EXP-000 Spearman 0.9922）比较的是**同一个坏模拟器对同一坏模拟器**。
本缺陷只在"同名多次同向交易"的策略形态下显形——恰是本周期书构建实验
第一次系统性触碰的形态。
