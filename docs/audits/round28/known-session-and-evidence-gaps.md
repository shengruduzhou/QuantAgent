# Round 28 — 已知交易日与缺证据拒绝 / Known sessions and missing evidence

基线 main `22b3f6cbe7d8c069d9931824b7f4ee1e8b78aece`，日期 2026-09-20。
保持 **RESEARCH / NOT LIVE READY；RL NOT ENABLED**。

## 已复现 / Confirmed

- [#154](https://github.com/shengruduzhou/QuantAgent/issues/154)：R11 独立发现并由
  主角色复现，RL market clock 仅来自 price pivot。全体 bar 缺失但 gap 表明确记录
  `MISSING_UNEXPLAINED` 的 2024-01-11 被跳过；2024-01-09 signal 的奖励错误地
  从 01-10 算到 01-12，并把 +5% 当作一个 session 的 return。
- [#155](https://github.com/shengruduzhou/QuantAgent/issues/155)：R4 独立发现并由
  主角色复现，公开 stock pool gate 缺 selection report 时默认视为有 factor
  coverage；公开 PIT slice 缺 `available_at` 列时原样放行整张 evidence frame。
  正常 pipeline 通常提供 reports；未把 API 边界缺陷扩大为已发生的真实交易问题。

## 修改 / Changes

RL 将 panel dates 与经过校验的 session-gap dates 合并成时钟，并用该时钟对齐
内部 price matrix 和 lag features。缺口日期不会凭空生成行情 bar；执行和奖励
端仍通过 `_proven_close` 验证：只有逐标的 `SUSPENDED` 可 carry last traded
close，其它缺口拒绝。cutoff 仍按 reward end 审核，sparse book 仍拒绝。

默认启用 coverage 的 stock pool gate 遇到缺报告，记录
`missing_factor_coverage_report` 并丢弃对应 member；报告存在但为空仍记
`no_factor_coverage_for_theme`。既有显式 `require_factor_coverage=False` 行为保留。

非空 evidence frame 缺 availability 列时，PIT slice 抛出可操作的 `ValueError`。
已有 future/invalid/NaT 行过滤、可配置列名、空 frame 和输入不变性保持。

## 验证 / Validation

主角色定向执行 **76 passed，1 existing pandas dtype warning**，并通过
`git diff --check`：

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/rl/test_session_gap_clock.py \
  tests/rl/test_pit_portfolio_env.py \
  tests/rl/test_pit_self_financing.py \
  tests/rl/test_reward_clock_intervals.py \
  tests/test_v7_evidence_store_and_gate.py \
  tests/test_v7_exposure_and_pool_gate.py \
  tests/test_v7_theme_research_pipeline.py
```

新增 regression 覆盖全日缺口 execution/reward 端、全体 suspension 的准确时钟和
冻结持仓、缺少另一标的 suspension 证据、reward cutoff、sparse book，以及缺
coverage report/错误 theme/显式 opt-out/缺 availability 列和可见时间边界。
CI 与修改后独立审核以 PR 的实际提交记录为准。

## 未证明的事项 / Limits

全部反例是 synthetic engineering fixtures；没有读取真实行情、冻结 holdout，
没有训练或启用 RL。若某交易日既无 bar 也无 gap 记录，本次不能凭空发现它，
因此仍不宣称官方 calendar 完整。Session-gap 数据自身的来源认证是独立前提。

R4 对 DataHub 内部 helper 的 NaT 行为另有观察，但其上游 provider 已做过滤，
尚未复现正常 load 路径可触达，未当作第三项已确认修复。浏览器、真实 bundle
单位、MiniQMT 与生产 readiness 的既有缺证据状态保持。

## References

- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)：observation/reward
  接口定义；具体 T/T+1/T+2 与 fail-closed 规则来自本项目契约。
- [Pandas Index.union](https://pandas.pydata.org/docs/reference/api/pandas.Index.union.html)：
  日期集合并集；不把 API 语义当作交易日来源认证。
