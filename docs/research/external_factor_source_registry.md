# external_factor_source_registry — 外部因子来源治理台账（fu_20260713）

> 规则（任务书 §5 + AGENTS.md）：只用一手/已授权来源的**精确公式**；专有公式不可得时记 `source_required`，**不得虚构**；不得给通用公式贴投行名。CSV 版本：`runtime/reports/full_universe/fu_20260713/source_registry.csv`。

## 一手学术来源（公式精确可得，confidence=high）

| 来源 | 因子族 | 公式可得性 | A股相关性 | 实施状态 |
|---|---|---|---|---|
| Bali, Cakici & Whitelaw (2011, JFE) "Maxing Out" | MAX 彩票需求 | 精确（max daily ret / month） | 高（A股彩票偏好文献一致） | **H-025 M1** |
| Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE) | 已实现偏度/峰度 | 精确 | 高 | **H-025 M2** |
| Ang, Chen & Xing (2006, RFS) downside risk | 下行波动/半方差 | 精确（日频代理） | 高 | **H-025 M8** |
| Baltussen, Van Bekkum & Van der Grient (2018, JFQA) vol-of-vol | 波动率的波动 | 精确（日频代理） | 中 | **H-025 M7** |
| Da, Gurun & Warachka (2014, RFS) "Frog in the Pan" | 信息离散度 FIP | 精确 | 中（A股动量弱，方向存疑，照实验证） | **H-025 M11** |
| Lou, Polk & Skouras (2019, JFE) overnight/intraday | 隔夜/日内收益分解 | 精确 | 高（中文文献记录 A 股隔夜溢价反转） | **H-025 M6** |
| Grinblatt & Han (2005, JFE) CGO | 未实现盈亏/筹码 | 精确公式需换手率加权参考价——**本库缺真实换手率** | 高 | **H-025 M12（日线 VWAP 近似，显式标注 approximation）** |
| Gervais, Kaniel & Mingelgrin (2001, JF) high-volume premium | 异常成交量 | 精确 | 高（A股方向反转假设：缩量溢价） | **H-025 M4/M9** |
| Amihud (2002) illiquidity | 非流动性 | 精确 | 高 | 已测（batch 1 D5，reject：多头=容量陷阱） |
| Chaikin A/D CLV（公开技术指标标准） | 收盘位置 | 精确 | 中 | **H-025 M5** |

## 券商研报族（族描述可得、精确清单/公式受限）

| 来源 | 状态 | 处置 |
|---|---|---|
| CICC 高频因子手册（79 个高频因子） | **source_required** — 精确清单非公开；本地无授权副本 | 不虚构。族级映射见 `reports/tickflow/factor_capability_matrix.csv`（highfreq 三行全部 BLOCKED_BY_DATA/REQUIRES_COLLECTION，数据先于公式已阻塞） |
| CICC 低频价量手册 | 族级公开（momentum/reversal/liquidity/vol/turnover/量价背离/价格路径） | 族内使用学术等价公式（上表）；换手率类 BLOCKED_BY_DATA（缺 float_shares） |
| CICC 基本面手册 | 族级公开 | 已覆盖：EXP-020 估值/质量/成长/杠杆/应计 PIT 面板；EXP-021/22 判定 raw-input 冗余 |
| 华泰金工筹码分布系列 | 族级公开（CGO 方法学在学术一手 Grinblatt-Han 已有） | M12 用学术公式 + 日线近似标注 |
| JPMorgan / GS / MS 因子研究 | **source_required** — 无授权一手材料在库 | 通用族（value/quality/momentum/lowvol/crowding）不得贴行名；其学术等价已在库或已测 |
| QuantsPlaybook（hugo2046） | 公开 repo，**无显式 LICENSE** | 作 hypothesis library 用（见 quantsplaybook_mapping.md）；不逐字移植代码；其引用的原始研报按上行规则处理 |

## 方法学参考（非公式来源）

Qlib / RD-Agent（已集成）；López de Prado（purged CV/PBO/DSR/triple-barrier —— PBO/DSR 已实装）；ASHA/Hyperband/BOHB（多保真思想 → 本任务 Stage A/B 分层即其实例）；AlphaForge/AutoAlpha（符号因子挖掘 —— 对应本库 factor_synthesis DSL 闭环）。
