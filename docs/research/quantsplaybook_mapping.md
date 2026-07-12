# quantsplaybook_mapping — QuantsPlaybook 研究库映射（fu_20260713）

> 实取核对 2026-07-13（github.com/hugo2046/QuantsPlaybook 结构 live fetch）：顶层 = A-量化基本面(2) / B-因子构建类(22+) / C-择时类(25+) / D-组合优化(2) / SignalMaker / hugos_toolkit。README 引用光大/华泰/招商/国信/东方等券商研报。**无显式 LICENSE ⇒ 不逐字移植代码；作 hypothesis/replication library 用，其历史收益不作为证据。**

## 映射裁定（族级）

| QuantsPlaybook 族 | QuantAgent 现状 | 裁定 |
|---|---|---|
| B-因子构建：价量类（量价背离、路径、动量/反转变体） | alpha101+cicc80+gtja191 已大面积张成；缺口=高阶矩/隔夜分解/量稳/FIP | **adapt** → H-025 M2/M6/M10/M11（学术一手公式，非移植） |
| B-因子构建：凸显性 STR / 球队硬币（行为类） | 无对应；公式在原研报（券商），repo 实现 provenance 可查但许可不明 | **defer**（source_required + license 不明；不移植） |
| B-因子构建：筹码分布（基于换手率的 CGO） | 缺真实换手率（float_shares BLOCKED_BY_DATA） | **adapt-degraded** → M12 日线 VWAP 近似（显式标注）；真 CGO 待 shares 数据采购 |
| A-量化基本面（FFScore、现金流） | EXP-020/021/022 已测：raw 输入冗余；crash-conditional 价值已记录 | **already_answered**（不重测） |
| C-择时类（RSRS/QRS/HHT/小波） | regime 线已走过（EXP-019/023）；择时指标属 regime 特征而非截面因子 | **defer**（若未来立项 regime v2，先做 RSRS 一项、预注册） |
| D-组合优化（差分进化/多任务） | 本库已有 GA/Optuna/policy search；SEARCH 窗过拟合风险已被 PBO 0.886 教育 | **reject**（无正当性再开搜索线） |
| SignalMaker / hugos_toolkit | 工具库 | 不需要（本库自有 DSL/评测栈） |

## 纪律声明

- 不以 README/notebook 宣称收益为目标或先验。
- 任何被 adapt 的族：公式取自学术一手来源（见 source registry），QuantsPlaybook 仅作"该族在 A 股有研究先例"的旁证。
- PIT 审计、tradability 验收、去相关门 —— 一律走本库 H-025 预注册协议，不继承外部结论。
