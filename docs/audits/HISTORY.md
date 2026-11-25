# 审计历史 / Audit History

每轮审计一段：发现了什么、修了什么、留下什么。逐岗原始报告不再保存在工作树中；
**任何已删除的文件都可用 `git show 2c4e296:<原路径>` 取回**（2c4e296 = Round 31 之前的
`main`，本地另有标签 `round31-pre-consolidation`）。Round 31 的完整审计证据（六岗报告、
复现测试、基准日志）保存在 `runtime/archives/audits/round31/`。

## Cycle 32（2026-11-25）— 多 agent 组织落地；FRESH 补数修复；"数据当前"≠"管道恢复"

组织：`.claude/agents/`（7 岗，两名只读审计员有否决权；只读靠行为约束，Bash 仍在）、
`.claude/skills/quantagent-{cycle,release-gate}`、`.claude/rules/`、`.claude/settings.json`
（拒读 `.env`、拒 force-push 各变体与 `git stash`）。每个分支都经两名独立审计员
（math-stat / code-security）审查，作者按条件修复后复核，再整体回归后合并。

决定与依据：
- **补数缺陷（ef0f152 / 0715ed1 / f6464cf）**：`fetch_with_retry` 写死 `count=40`，而窗口有 73 个交易日
  ⇒ 会留下 33 天的内部空洞却报告 CURRENT。改为日历交易日数 + 10；逐日覆盖率 ≥ 93% 种子才合并；
  种子棘轮、原子写、rc=75 表示"已部分暂存、下轮重试"。06:00 运行第 5 轮收敛：73 天、每日 3,264–3,273/3,273，
  0 重复键，全部行带日期 PIT ST 出处，amount/(vol×close) 中位 1.001（主管独立复核，不信任作业自报）。
- **同日 orders=FAILED**：`fresh_blind_daily.py` 仍调用 #119（2026-08-11）已删除的
  `baseline_protocol._apply_delay`。因数据停摆，权重步骤一直是 BLOCKED_UPSTREAM，故 3.5 个月未暴露；
  08-11 起不存在盲测下单记录。修复（c4517df / b902d07）传未平移的 book：候选冻结时（07-13）
  模拟器在索引日按 close 成交、旧 runner 先平移 1 天 ⇒ T 信号 T+1 close 成交；现模拟器自己做 T→T+1，
  语义相同。`holdings_in_effect` 与旧记录 3,000 例模糊测试 0 差异。注意：07-13 之后加入的公司行动、
  退市核销、红利税与敞口选项使 FRESH NAV 与冻结时运行不逐位相同（权重与 weights_hash 不受影响）。
- **剩余阻塞**：v7 silver 面板用"缺行"表示停牌（`is_suspended` 全 0；2026-05-09 起 433 个内部空洞 / 140 只），
  08-09 起的严格模拟器对缺 bar 失败关闭（`missing_execution_bar`）。交数据工程 wave 2。
- **P1-3 合并重复实现**：13 处 rank-IC 迁到 `factors.evaluation.cross_sectional_rank_ic`，6 处 Sharpe/回撤迁到
  `quant_math.performance`；与冻结旧代码对拍 1e-12。改变数字的差异只登记不修（见 ROADMAP P1-3）。
- **P1-6**：非重叠 cohort t 统计量；筛选与 lifecycle `validated` 需 t ≥ 3（单侧、先定向）。重叠零假设下
  尺寸正确（审计复现 5.05% @1.96）；持续性零假设下任何非重叠 t 都偏大，不得当 N(0,1) 用。
- **P1-4 试验台账**：哈希链 JSONL，默认落主检出（worktree 内运行不再丢失）；运行前预留试验数、失败也计入；
  nonlinear_fusion 基线 146（逐项出处见 RESEARCH §3；原 134 把 R29 E1 只算了一次）。
- **P1-2 标签口径戳**：`build-labels-v7` 必须显式 `--label-convention`；delay1 走全局交易日历；消费者拒绝无戳标签；
  关闭 CSV 旁路与 parquet→csv 回退两条绕过路径；UI 默认 r29，且不能覆盖哈希钉住的认证标签。

被否决的方案：在 FRESH runner 里重新实现一次本地平移（会复活 #119 删除的重复时钟）；
放松模拟器的缺 bar 检查（把缺失测量当默认值）；delay1 默认值静默从 delay-0 改为 delay-1（会改变已有命令的数字）。

验证：集成树 3,985 passed / 0 failed，compile 与前端（tsc、vitest 149/1 skip、build）通过；main bd92b52 已推送。
遗留：`test_folds_never_reach_a_quarantined_label_window` 单测耗时 60 分钟（待剖析）；台账截断仍可验证通过
（需外部高水位）；RD-Agent 非重叠门的对齐问题需预注册后重基线。

## Round 31 续 3（2026-11-25）— CI 自 2026-10-09 起持续失败：Polars 2.0 与依赖漂移

GitHub CI 从 run #1793（main 2c4e296，2026-10-09）起每次都失败，没人注意到。Polars 2.0（2026-10-06）
删掉了 `rolling_*(min_periods=)` 参数；`pyproject` 写的是 `polars>=1.0`，干净环境解析到 2.1.0，
`factors/expr.py` 的 Polars 后端就报错。本机 venv 停在 1.40.1，只出 DeprecationWarning，所以本地全绿。
在干净 venv（UTC、空环境）复现：未修改的 main 恰好 1 个失败，与 GitHub 一致。

审计同时发现：本机 numpy 2.4.5 早已超出声明的 `<2.3`；CI 安装不锁版本，任何上游发布都能让无关的 push 变红。

修复：
- 改用 `min_samples`。语义不变：pandas 参考实现仍用 `rolling(window, min_periods=window)`。
  新增窗口内含缺失值的对拍测试，以及把 Polars DeprecationWarning 升级为错误的测试，
  使 1.x 运行时也能抓到将被删除的参数。两者在修复前的代码上都会失败。
- 依赖范围：`polars>=1.21,<3`（1.21 实测通过），`numpy<2.5`，`pydantic<3`。
- CI 改为按 `requirements/ci-constraints.txt` 精确安装；新增每周 `Dependency Drift` workflow。
- 依赖测试在任何环境检查"已装版本在声明范围内"。

验证：
- 干净环境，范围内最新版本：3,597 passed / 0 failed。
- Polars 1.21.0 下限：相关 389 个测试通过。
- Polars 2.x 之外，仓库代码没有其他 Polars 不兼容用法（逐一对照 2.0 升级指南实测）。

遗留：
- pandas 3 的 FutureWarning（`groupby.apply` 含分组列、`fillna` 降型、不兼容 dtype 赋值），由 `pandas<3` 挡住。
- 本机 venv 与 constraints 的偏差见 OPERATIONS.md §0。

## Round 31 续 2（2026-11-25）— ENG_ST31 前向锚定 ST 登记，FRESH 窗证据解除阻塞

用户选择预注册 ENG_ST31。核实 A0 真实带日期：安全主表名称来自 TickFlow `get_instruments`，2026-07-26 11:39:51Z 全市场
5,533 只上市证券（`st_current_snapshot` 只是它的 ST 子集）；新取 A1（2026-11-24 23:14 CST）。方法：锚点状态经 ENG_ST30
冻结分类器 + 两条预注册预过滤器的公告向前推进，所有快照与巨潮当日简称（246 家活跃公司逐家抓全部公告）必须一致，
否则记 UNKNOWN 并从观测重启。DEV（深市偶数尾号三个历史窗含年报季）全对；A0 与登记在 DEV 半区 0 分歧；
去掉事件的消融显示观测闸门把 56 个漏掉的变更全部转为 UNKNOWN、0 静默错误。**深市一次验证全过**（235,036 行 0 错、
19/19 同日、0 UNKNOWN）⇒ 沪/北 2026-07-27 起发布，3,273 个种子全覆盖。新增每日深市监控（失守即撤下沪/北）。
诚实边界：19 个变更的单侧 95% 下界约 0.85；沪市正确性靠同一披露规则 + 0 矛盾，而非沪市真值。

## Round 31 续（2026-11-24）— ENG_ST30 沪/北 ST 登记

用户选择选项 A。巨潮公告（类别 + 标题关键词，16,766 条）、正文生效日（3,973 份 PDF/HTML）、公告当日简称与定期报告
名称探针重建 ST 区间；方法只在深市 DEV 半区开发（A1/A2 修订均在评测前提交）。**HOLDOUT 一次评测：M5b 行精度
0.924 < 0.98 ⇒ 按预注册决策，沪/北不写入**。交付：深市每日刷新的严格 PIT ST 证据表、catchup 默认读取它、
06:00 supervisor 先刷新证据；FRESH 窗仍因沪市缺证据而停摆。事后诊断（未用于改方法）：否定式标题、
"撤销其他并继续退市"的转换方向、两条缺失的撤销公告使陈旧区间吞并了后续真实区间。

## Round 31（2026-11-24）— 全系统审计、收敛与修复

**流程**：六个独立审计岗（架构收敛、脚本/文档卫生、量化研究、回测风控、数据基础设施、全栈）
各自只读审计并以复现测试作证；主岗交叉核验后在 `agent/round31-consolidation` 上逐项提交；
独立验证岗复审合并候选。

**收敛（删除均经可达性证明：CLI 注册 ∪ FastAPI ∪ JobRunner 白名单 ∪ crontab ∪ CI ∪ 规范脚本，
含惰性导入、字符串路径与包重导出）**

- 脚本：删除 152 个无引用脚本（−20.8K 行：stage1–13 实验驱动、rankfix/做T 实验、v8 扫参启动器、
  已被取代的面板构建器、裁决已入账的一次性消融）与未安装的 systemd 调度。
- 源码：删除 110 个生产不可达模块（−22.6K 行）及只测试它们的 47 个测试文件：未接线的治理哈希链与
  agent 协议、clean-room / full-pipeline / T+1 玩具回测器、decision-chain/状态机/pareto 组合栈、
  只被测试引用的 specialist agents、做T 引擎、v7 agent 路径注册表、5 个从未接线的证据摄取器、
  过期（2026-10-01）的 legacy CLI 模块。
- 文档：185 个 Markdown → 约 20 个；根目录 40 个报告合并进 `docs/GOVERNANCE.md`、
  `docs/RESEARCH.md`、`docs/DATA.md`、`docs/OPERATIONS.md`、`docs/ARCHITECTURE.md` 与本文件。
- 合并 Round 30 未合入的工作：`round30-chair-fixes`（8 个提交）与 `round30-risk-exposure`
  （ExposurePolicy，默认关闭）。`round30-st-register`（草稿）与 `round30-fusion-e2`（预注册需修订）保留为分支。

**修复（每项均有修复前失败、修复后通过的测试）**

| 范围 | 缺陷 | 修复 |
|---|---|---|
| 纸面 P0 | 连续纸面循环从不入账分红送转：除权日价格下跌被记为亏损并触发日亏停买 | 每个执行日按认证面板 `ca_*` 规范入账（整股 + 零股折现、按 (symbol, ex-date) 幂等） |
| 纸面 | 除权日涨跌停基准用原始前收；回撤锁人工解除后下一交易日立即重新锁定；科创板 <200 股残余无法卖出 | 除权参考价；解除时把峰值重新锚定到当前权益（日志化，回放生效）；整仓卖出标记零股 |
| 执行 | 送转除权日整仓卖出按可用+冻结股数下单，整单被拒 | 卖出以当日可用股数为上限 |
| 组合 | 可交易持仓涨过单票上限且换手上限生效时，目标构建抛错 ⇒ 纸面当日无目标 | 裁剪到上限，强制裁剪不计入换手预算；冻结/锁定持仓仍 fail-closed |
| 数据泄漏 | 数据集构建器按黑名单选特征，`entry_close_t1`/`forward_return_20d`/广播 `is_st` 会成为特征 | 与因子 DSL 共用 post-decision 判定；钉住 schema 要求此类列即失败 |
| 隔离窗 | 融合搜索与 RD-Agent 验证段落在烧毁 holdout / 冻结新鲜窗内 | 先按标签窗剔除触及隔离窗的行 |
| 统计 | RD-Agent 接受门接受 48% 纯噪声因子 | 非重叠验证 IC t ≥ 3（噪声 0/144，植入信号仍被接受）；PBO 上限 0.50 → 0.25 |
| 测试隔离 | 测试把 14 行合成标签写进真实 runtime、导入 API 即获取真实纸面写锁 | conftest 隔离 `QUANTAGENT_HOME`；app 延迟构建 |
| 数据 | 工作台默认取最新文件（被证伪的 v7 面板 + 测试合成标签）；Web gold 重建跑旧构建器 | 认证优先的默认输入；指向 r29 构建器；gold 标签写出 `label_end_{h}d`；构建器剔除 B 股 |
| UI/API | 导出报告混用两次运行且无可引用性标注；/api/data/coverage 42 s 且把 191 个节假日报为缺口；索引重扫冻结 30–47 s；否定词误判 PASSED | 同一运行 + NOT CITABLE；精确去重 10 s + U0 交易日历；stale-while-revalidate；共用否定感知分类器 |
| 其他 | `walkforward_risk_backtest.py` 印花税写死 50 bps（10×）并由 API 提供结果 | 脚本、路由、未挂载页面一并删除 |
| 依赖 | lightgbm/cryptography/tickflow 等被导入却未声明；tabulate/transformers/vectorbt 声明却未用 | 按 extra 声明；删除未用项 |

**关键结论**：仓库内没有通过成本与统计闸门的可交易 edge（见 `docs/RESEARCH.md`）；FRESH 窗数据管道
自 2026-08-05 停摆；"生产 FT-Transformer" 训练在前视宇宙上，只能视为历史产物。

## Round 30（2026-10-14，未合并，Round 31 收编）

主岗修复（回合交易统计计入送转、手机宽度外壳、FRESH 窗权威日历、kill-switch 解除取账户锁、T+1 汇总、
因子流水线 PASSED 语义、Backtester `?backtestId=`、DSL 列白名单）与 ExposurePolicy（回撤阶梯 + 波动率目标，
默认关闭，预注册见 `docs/research/preregistrations/ENG_RISK30_preregistration.md`）。

## Round 29（2026-09-30 → 10-09）— 11 岗公司式审计

规范评估器默认改用认证 r29 执行面板（旧 v7 面板的 ST 为今日名单广播 = 前视）；印花税按法定日程；
先卖后买；历史峰值回撤 + 几何 CAGR；PSR/MinTRL/自助法 CI；纸面风控真正接线（回撤 20% 只减仓锁、
日亏 5% 当日停买、无行业图拒买、状态持久化）；AkShare 单位真相（东财"手"、腾讯科创 ×100、
qfq 拼接）；LLM 因子 AST 白名单。真实数据：透明量价 top-50 −11.73%/年 vs 等权 +6.89%；
RL 预注册 DO NOT ENABLE；E1 非线性 +0.0126 IC（t 3.51，毛）。

## Rounds 17–28（2026-08）

- DEF-025/026：闸门加固在生产者一侧被架空（伪造 `pit=0`/`mock=False`）；标签窗早于特征可用时点
  ⇒ 外部特征 rank IC = 1.0。DEF-027/028：幸存者审计问错了掩码。
- DEF-029：HAC lag 按样本量取 ⇒ h=20 噪声因子 23.6% 过 |t|>1.96；DEF-030/031：装饰性守卫与单 seed 正对照。
- DEF-032：akshare 腾讯日线把"手"标成 CNY 成交额（缩小 1.2e5 倍）。
- DEF-033~041：组合级限额结构性不可达、冲击成本写进证书却从未收取、RL 零动作自行加杠杆、
  interior-bar NAV、风控接到生产 venue、选股器谎报过滤器。
- Round 22：撮合判据 close vs open（189 次假阻断）、声明 5 bps 实收 2 bps、`.ffill()` 虚增 7.67pp。
- Round 23：认证面板补 Alpha101 + GTJA191（144 个不同因子，|IC|>0.02 仅 74 个；6 对跨族重复）。

## Module One / Two（2026-08-05 前后）

四引擎复合回放（canonical / paper / streaming / legacy）0 个无法解释的经济差异；修复重复
execution_id 记两次、手续费费用化、wall-clock 回退改写 T+1 结算日、fsync 失败断链、
corporate action 只改 portfolio 不写 canonical（DEF-020）、无行情持仓按零计价（DEF-021）、
基准缺口填 0（DEF-022）、空 metrics 下闸门 PASS（DEF-023）。证据 JSON 保存在 `docs/architecture/`。

## 更早（2026-05 → 07）

v7/v8 研究链建立；全宇宙深度 MLP 无 edge；+20%/年 超额被证明来自不可交易标的；做T 无可实现 edge；
v8.8 batch-rank 污染；生产评估协议 v2 与隔离窗（2026-07-03）。
