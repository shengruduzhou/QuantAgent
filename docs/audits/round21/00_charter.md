# Round 21 — 多角色隔离审计章程 / Isolated Multi-Role Audit Charter

日期 / Date: 2026-08-18
基线 / Baseline commit: `b56ae57`

## 组织形式 / Firm structure

本轮按 AI 私募量化基金公司的职能分工执行，11 个副角色各自**数据隔离**：
每个角色只读自己职责域的代码与文档，独立做网页调研，独立产出裁决，
不得引用其他角色未公开的中间结论。主角色（CIO/主程）负责汇总、裁决冲突、
实施修复、跑测试并合并到 `main`。

| Role | 职责 / Mandate | 报告文件 |
|---|---|---|
| Main | 量化/架构/全栈总负责，最终修复与合并 | `90_main_integration.md` |
| R1 | 回测专家（WFA/CPCV/PBO/DSR、执行语义） | `01_backtest.md` |
| R2 | 风控专家（VaR/CVaR/回撤/敞口/断路器） | `02_risk.md` |
| R3 | 因子与策略专家（有效性、生命周期、非线性融合） | `03_factor.md` |
| R4 | 选股专家（基本面+量价打分、组合构建） | `04_selection.md` |
| R5 | 测试专家（全栈测试、前后端、内部代码） | `05_test.md` |
| R6 | 量化部门员工（以用户身份使用系统） | `06_operator.md` |
| R7 | 量化部门专家（策略/风控/前后端联调） | `07_desk_expert.md` |
| R8 | UI 全栈与设计专家 | `08_ui.md` |
| R9 | Debug 与代码治理（屎山清理） | `09_debug.md` |
| R10 | 审计角色（复审其他角色，交叉否决） | `10_meta_audit.md` |
| R11 | 强化学习专家（奖励机制、可交易性） | `11_rl.md` |

## 通用裁决规则 / Shared adjudication rules

1. **证据缺失记 `unknown`，永远不记 `pass`。** 没跑过的检查不算通过。
2. 每条 finding 必须给出 `file:line`、**具体失败场景**（输入 → 错误输出）、
   **最小复现命令**，否则降级为 `观察` 不进修复队列。
3. 报告文件必须**增量写入**：每确认一条就落盘，不允许攒到最后一次性写。
4. 不得使用 mock/synthetic 数据让结论显得完整；缺能力就 fail-closed 记录。
5. 禁止读取 / 打印 `.env`。凭证只经允许的环境变量名消费。
6. 只读审计角色不得修改 `src/`、`apps/`；修复由主角色实施。

## 本轮必须闭环的历史遗留 / Carried-over open items

- PR #94（`agent/paper-shadow-next-session-state-v3`）closed-unmerged：结论化。
- RL 的 T+1→T+2 reward / tradability fail-closed 尚未进 `main`。
- account identity 的 API/UI 可观测性缺失。
- parent→child TWAP/VWAP/POV/iceberg + restart/TCA 尚未进 `main`。
- 计时修复后的真实历史数据重新回测尚未运行 ⇒ 旧 headline 年化/Sharpe 不可引用。
- AkShare 数据口径（用户反复反馈"数据不太对"）。
