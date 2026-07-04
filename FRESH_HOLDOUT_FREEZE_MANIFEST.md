# FRESH_HOLDOUT_FREEZE_MANIFEST — 新鲜 holdout 冻结证书（H-003，2026-07-04）

## 冻结声明

| 项 | 值 |
|---|---|
| **窗口起点** | **2026-05-19**（上一个被搜索污染的日期 = 2026-05-18） |
| **当前可用终点** | **2026-07-02** |
| **可用交易日** | **32 天** |
| **正式首读门槛** | 120 交易日 → **当前远低于门槛，禁止任何读数** |
| **最早正式读日** | ≈ **2026-11 中旬**（120 交易日约在 2026-11-13±，读取前按实际日历精确复核） |
| **trust class** | `frozen_future_holdout` |
| **硬规则** | **不可用于模型/blend/top-K/因子/RL/风控的任何选择或比较；本次修复与 QC 全程未读任何策略表现** |
| 守卫 | `configs/quarantined_windows.json` 第 2 条（2026-05-19→2027-12-31），`bp.evaluate` fail-closed + `strict_v8` trust-stamp；守卫在数据落地**之前**已激活 |
| 读取流程 | 到期后按 `EVALUATION_PROTOCOL_V2.md` §2：预注册 ≤3 配置（`configs/preregistered_evals.json`）、每配置一次性评测、守卫经配置变更解除（不走 override） |

## 数据来源与时间戳

- Provider：TickFlow daily klines（vendor_api，reliability 0.9），`adjust="none"`（as-of-day 原始价基准，与 panel 逐分对齐验证：600519 2026-05-18 close 1323.00 / amount 6,594,983,723 / OHLC 全等）；volume 手→股 ×100（实测 49,661×100≈4,966,097）。
- 摄取批次：①首轮 `tickflow_daily_append` 73,284 行（2026-07-03 夜）②修复 `tickflow_daily_append_repair_20260704` 42,842 行（两遍：全量重扫 39,260 + 定向补漏 3,582）。
- `available_at = trade_date + 1d`；`point_in_time_valid = true`；ST 旗标 provenance = `current_snapshot_broadcast`（全 panel 既有约定，局限已知）。

## 修复记录（首轮为何不合格 → 如何修复）

1. **2026-05-19 整日缺失**：TickFlow 对 start_time 起点请求**掉首日**（两次独立复现）→ 修复脚本用 count=40 拉取绕过；updater 永久修复 = start 提前 3 天缓冲（本地 `> last` 过滤去重）。
2. **1,281/3,653 symbols 限流失败**：修复脚本 3 次退避重试 → 首轮后残余 386（116 零覆盖+270 部分）→ 定向第二遍 → **最终仅 35 个失败**（清单在 `runtime/logs/repair_pass2_20260704.log`，特征与退市/长停牌一致；占 seed 集 0.96%）。
3. **旗标错误**：05-20 的涨跌停旗标曾以 05-18 为 prev-close 推导 → 全窗旗标（limit_up/down、suspended、st）用补齐后的价格链**整体重算**。
4. 外科式插入：仅插缺失 (symbol, trade_date) 行，既有行零覆盖；panel tail 双备份（`market_panel.pre_20260702.tail.parquet` / `pre_repair_20260704.tail.parquet`）。

## QC 证书（scripts/analysis/fresh_window_qc.py，12/12 通过，报告 runtime/reports/fresh_window_qc/qc_report.json）

| 门 | 结果 | 关键数字 |
|---|---|---|
| 交易日缺失 | ✅ | 期望 32（TickFlow 日历）/实有 32，缺失 0（含 05-19） |
| 每日覆盖 | ✅ | min 3,606 / max 3,634（seed 3,639 的 99.1–99.9%；门槛 ≥93%） |
| (symbol,date) 重复 | ✅ | 0 |
| OHLCV 空值 | ✅ | open/close null=0 |
| 非正价格/负量额 | ✅ | 0 |
| **复权接缝** | ✅ | 05-18→05-19 收益 vs 板别涨跌幅带，越界 ≤容忍（原始价基准无断层） |
| volume 基准 | ✅ | 窗内/窗前中位量比中位≈1（×100 错误将呈 0.01——未出现） |
| amount 基准 | ✅ | amount≈volume×VWAP 带内占比 >98% |
| prev-close 连续性 | ✅（v2 门） | cap+2% 越界 143/116,116=0.123%（≤0.25%）；物理不可能移动（<−60%/>+45%）=0。**143 例已抽验定性**：Top 下跳 −35/−34/−33% 在前复权序列中仅 −5.6/−2.7/−5.6% ⇒ 除权除息（6 月分红季）；上越界 +10.9/+10.7% 两序列一致 ⇒ 摘帽股 10% 板 + 低价四舍五入。首版门槛按复权基准误设，v2 修正含论证注释——非静默放水 |
| 涨跌停旗标 | ✅ | 日均涨停率 2.56%，与收益一致性 >90% |
| ST/停牌 | ✅ | is_suspended ≡ volume≤0 一致率 >99.9% |
| provenance/schema | ✅ | schema sha256 `95411d85…`；窗内行全部来自两个已登记 source |

## 完整性与资源

- panel：**15,105,783 → 15,221,909 行**（+116,126 = 窗内行数精确一致）；文件 479 MB；post-repair sha256 `b6508f4df5418d38a558e4aba4f9e1e0aaffb1f17cd91772e078771a91a8660e`。
- 命令链：`update_market_panel_daily.py --end 2026-07-02` → `repair_fresh_window_20260704.py`（全量）→ 同脚本 `--symbols-file`（定向 386）→ `fresh_window_qc.py`。
- 资源峰值：摄取/修复 RSS ≤3.8 GiB；QC RSS 0.47 GiB。磁盘：+~0.5 GB（panel 增量+备份），仍 199G 可用（78%）。

## 残余已知局限（如实）

35 个 symbols 窗内零/部分覆盖（0.96%，疑退市/长停牌——正式读日前用披露数据核定处置：退市按真实退市处理，误漏则补取）；is_st 为快照广播（全 panel 既有约定）；TickFlow 限流使日更慢但当日增量（1 天×3.6k）可承受。
