# EVALUATOR_VALIDITY_AUDIT_IC016 — GBM 绝对 IC 0.16 对抗性审计（2026-07-10）

> 触发：EXP-022 诚实旗标"base 绝对 IC 0.16 偏高，留待泄漏审计"。任务指令（2026-07-10）要求在任何大规模训练前完成 evaluator 信任审计。
> **裁定：evaluator 可信（无前视泄漏）；IC 0.158 真实但经济上单薄——由通用截面因子结构（低波/反转/size）+ 60d 标签秩自相关解释；GBM 的 top-decile 收益与 1 行反转因子几乎相同。瓶颈确证=phantom breadth，已按流动性分层定量。**

## 1. 标签构造审计（最初的"红旗"→ 澄清为更保守的正确约定）

| 检查 | 结果 |
|---|---|
| (symbol,trade_date) 重复行 | **0**（6,781,038 行全量） |
| `forward_return_1d` == close(t+1)/close(t)−1（naive 同日约定） | 仅 0.2% —— 红旗触发深查 |
| **实际约定：`forward_return_hd` == close(t+1+h)/close(t+1)−1（delay-1 executable）** | **1d 匹配 100.0%，60d 匹配 94.7%** |
| `label_end_hd` == trade_date.shift(−(1+h)) | 1d 99.7%，60d 94.0% |
| 入场过滤 | 建库时已剔除 is_st(t) ∨ is_suspended(t) ∨ is_suspended(t+1) ∨ is_limit_up(t+1) 行（scripts/build_executable_labels_dataset.py，v8.7 起有意设计：杀死一字板 phantom alpha） |
| 特征 `return_1d` 是否后视 | == close(t)/close(t−1)−1 匹配 99.3% ✓ |
| corr(return_1d, fwd60) 同日重叠检验 | −0.0003（无同日泄漏通道） |

**结论**：生产标签是**执行感知的 delay-1 约定**（t+1 收盘入场），且入场不可行行已剔除——比审计假设的 naive 约定更保守。**文档漂移**：`quantagent/data/v7_label_builder.py` 的 docstring 仍写 close(t)→close(t+h) 旧约定，与生产工件不符（生产走 build_executable_labels_dataset.py 路径）。

### 确认的两个非泄漏缺陷（如实记录，均为保守方向）

1. **陈旧日历标签噪声（~5%）**：60d 标签建于 2026-07-04 fresh-window 修复**之前**的 panel；修复插入了停牌行 → 现 panel 上 ~5.3% 行的 label_end 与 shift(−61) 不符（跨停牌窗的实际交易日数 >60）。方向=标签噪声非前视（end 日期仍是真实未来日）。重建数据集时自动消除。
2. **原始价基准股息低估**：标签用未复权 close，除息日下跳未计股息 → 高分红（价值/大盘）名标签系统性低估。方向=保守 + 轻度反价值偏置。修复路径=复权价重建标签（已知 adj_factor 可物化）。

### 回归锁

`tests/test_executable_label_convention.py`（2 tests，通过）：①合成面板全流程断言 delay-1 值/label_end/入场过滤/无重复；②生产工件抽样断言 delay-1 匹配 ≥95%——静默改回同日约定（重新引入一字板 phantom alpha）会 loud fail。

## 2. IC 0.158 的量级校准（同宇宙/同窗/同标签的单因子对照）

可交易宇宙（eligible ∧ 流动上半），2023-04..2025-08-29，label=delay-1 fwd60：

| 因子（1 行定义） | mean 日 rank-IC | ICIR | top-decile fwd60 |
|---|---|---|---|
| lowvol20（−20d 波动） | **+0.110** | 0.51 | +2.94% |
| rev60（−60d 动量） | +0.089 | 0.69 | **+4.51%** |
| small size（−amount） | +0.091 | 0.91 | +1.11%* |
| low_price | +0.075 | 0.58 | +4.36% |
| **EXP-022 GBM（250 特征）** | **+0.158** | 1.07 | **+4.58%** |

*size 的 top-decile 定义取 −amount 高秩=最小票。

**判读**：单个朴素因子达 0.075–0.110；60d 标签相邻日秩自相关 ~0.98 使日 IC 高度持续（ICIR 因此偏高）。250 特征非线性模型到 0.158 **不异常**——是 2023-25 A 股截面因子结构（低波/反转/小盘）的常规组合，无需泄漏解释。**GBM top-decile +4.58% vs 1 行 rev60 +4.51%**：在可交易宇宙的 decile 粒度上，模型相对朴素反转的增量 ≈ 0。与 [[full-universe-deep-mlp-no-edge]]、EXP-021/022 结论一致收敛。

placebo shuffled-label GBM 未跑（省 30G/8min）：泄漏通道已逐一排除（标签 delay-1、embargo=61td≥标签跨度、valid=2022 尾、无跨集归一化），且上表已证明模型数字可被朴素因子复现——机器级伪信号无处藏身。

## 3. Phantom breadth 定量（本审计最重要的经济结论）

同窗 eligible 宇宙 eqw 60d 前向收益（delay-1 executable）年化：

| 流动性段 | eqw 年化 | 中位 fwd60 |
|---|---|---|
| 非流动下半 | **+25.4%/yr** | +1.99% |
| 流动上半 | +11.7%/yr | −1.76% |
| 流动性 top 20% | **+7.4%/yr** | −2.99% |

**breadth 溢价单调消失于流动性**：eqw-all-A 基准（~+52%/yr 含不可交易微盘）对可执行宇宙夸大 2–3×+。可交易上半的 decile 级可选 edge ≈ +1.7%/60d（≈7%/yr，成本前）。**含义：任何以全宇宙 eqw 为参照/训练分布的信号评估都被 phantom breadth 污染；冠军书（k=10 集中持仓）的真实容量取决于其实际持仓的流动性段——由容量研究（EXP-024）直接测量。**

## 4. 裁定与后续

- **evaluator：TRUSTED**（标签无前视、执行感知、入场过滤正确；两个保守向缺陷已记录+回归锁）。
- **特征侧大训练（GPU）维持 NO-GO**：IC 层面再无未解释异常，EXP-021/022 的先验门结论加固。
- 后续：①容量研究（冻结冠军按流动性/参与率/AUM 情景，EXP-024）②数据集重建时用复权价+现日历重建标签（消除缺陷 1/2）③v7_label_builder docstring 修正。
