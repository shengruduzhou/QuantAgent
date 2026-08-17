# Round 21 — R4 选股专家审计 / Selection & Portfolio-Construction Audit

- 角色 / Role: R4（选股专家：基本面 + 量价打分、组合构建）
- 基线 / Baseline: `b56ae57`（`main`，clean）
- 日期 / Date: 2026-08-18
- 性质 / Nature: **只读审计**（未修改 `src/`、`apps/`，未 commit/push）
- Python: `AI_quant_venv/bin/python3`
- 审计域 / Scope: `src/quantagent/portfolio/`（构建侧）、`universe/`、`fundamental/`、
  `concept/`、`themes/`、`strategy/`、`tests/portfolio`、
  `src/quantagent/execution/parent_child.py`（执行算法）

> 裁决规则沿用 `00_charter.md`：**证据缺失记 `unknown`，永不记 `pass`**。
> 本文件按确认顺序增量写入。

---

## 0. 网页调研 / External research（先做，再看代码）

| # | 来源 | 与本审计相关的结论 |
|---|---|---|
| 1 | https://akquant.akfamily.xyz/guide/cross_section_checklist/ | 横截面清单要求：**评分前**校验窗口长度并跳过样本不足标的（"评分前校验窗口长度"）；停牌/涨跌停/成交异常必须有降级协议；universe 来源必须**版本化**；调仓需容差带（tolerance band）；验证阶段必须跟踪 turnover / concentration / slippage 敏感度。 |
| 2 | https://akquant.akfamily.xyz/guide/strategy/ | 强调 reduce-first 调仓语义（先减后加，避免现金约束）；T+1 下 `get_position()` ≠ `get_available_position()`；显式警告"新上市或停牌恢复标的数据窗口不完整；**评分前**统一检查并跳过"。 |
| 3 | https://akquant.akfamily.xyz/textbook/09_funds/ | 组合构建方法谱系：固定权重 / 风险平价（按波动率倒数，纠正"60% 资金贡献 90% 风险"）/ 均值方差（有效前沿、切点组合）/ Black-Litterman / HRP。明确"大资金必须把流动性当作**硬约束**"，含冲击滑点。 |
| 4 | WebSearch `cross-sectional stock selection top-k portfolio construction turnover control hold band` | 业界两种换手控制范式：**TopK-DropN**（每期只换 N 只，而非整体重排）与 **band turnover regularization**（对平均换手偏离可接受区间加罚）。 |
| 5 | WebSearch（见 §5） | 风险平价 vs 均值方差、A 股行业中性。 |
| 6 | WebSearch（见 §2） | A 股基本面 PIT（PE/PB/ROE + ann_date）。 |

关键外部共识（用作本审计的判据）：
1. **不可交易标的必须在打分之前剔除**，否则 top-k 会被不可成交名字占满。
2. **行业分类必须版本化 / PIT**，当期快照回填历史 = 前视。
3. **流动性是硬约束**，大资金下必须进入选股阶段而非只在执行阶段。

---
