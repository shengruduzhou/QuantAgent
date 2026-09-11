# Round 27 — 审计修复与验收状态 / Audit repair status

本批基于 main `674b72308dd522488bb35f6210df183669f86695`。结论仍为 **RESEARCH / NOT LIVE READY；RL NOT ENABLED**。所有数值反例来自 synthetic unit fixtures，不是策略收益证明。未读取或调参冻结 holdout，未创建任何许可、证书或实盘授权。

## 修复范围 / Changes

| 路径 | 已复现问题 | 本批行为 |
|---|---|---|
| OMS / QMT / constraint DSL | NaN 限价比较可落空；无限 NAV/成交量可使比例限制失真 | 有限数检查；非法限价不抵达 broker |
| OMS restart | 重启清空日内订单数、turnover、participation 累计 | 将 risk intent 绑定进 canonical RiskDecision；恢复当日已提交记录，包含取消等终态；legacy 缺字段时 recovery-required |
| Council API / UI | unknown→warn 可进入 Human Gate；跨角色共享 draft | write/replay/aggregate 同时拦截；UI 限定 unknown 只能改 blocked，清理跨上下文 draft，submit 再检查 |
| Strict simulator | 只在调仓执行日估值，漏掉持仓间隔回撤 | 首次至末次映射执行日期间逐 observed session 估值，不增加调仓，不隐式扩大评测窗口；缺持仓价格拒绝 trace |
| Factor selection / RankIC | 整个缓存的未来缺失率影响过去列选择；非成对排名算错稀疏 Spearman | 限定至请求 keys 后筛列；对 jointly finite pairs 的两端重新排名 |
| Factor lifecycle | 同一 evidence digest 重放能累计退休 | 同一版本去重，不把重试当独立观察；severe semantic violation 仍优先隔离 |
| PIT RL env | 冻结资产仍为新买入提供虚假资金；无收益后 drift | 可行卖出后的现金先付费用再按比例分配买入；收益后 drift；observation 与未来 reward accounting 分离；稀疏执行序列暂拒绝 |
| Qlib / AkShare | Qlib normalized OHLCV 当 raw；AkShare canonical metadata 缺失 | Qlib 默认拒绝未知 raw contract，显式 factor restoration 与单位映射；AkShare 补 frequency/timezone/units/adjustment/PIT metadata |

## Evidence 与验收边界

本地执行环境失效之前，可直接确认的测试输出：
- 风控重启与未测约束：20 passed。
- RL accounting 与 lifecycle 专项（含独立 dollar-ledger reference tests）：37 passed。
- 前端 TypeScript 与 production build 完成；前端完整测试：123 passed，1 skipped。
- 扩展后端专项曾得到 1052 passed / 16 skipped / 3 failed。三项失败分别是新增测试对 public fail-closed 异常的预期，以及两项旧 RL 测试的免费费用/不漂移假设；已更正回归预期后启动全量测试。

**全量 Python 测试的最终结果未取回。** 浏览器连接随后中断，选定工作环境不可用。此 PR 从固定 base 和会话记录的修改重建；新 RL regression 文件保留四项产品现金/PIT oracle。此提交必须重新跑 CI，不能把上述本地结果冒充为此 SHA 的最终验收。浏览器实测、最终独立角色签核尚未完成，不合并。

真实数据探针：AkShare 1.18.84 的 EastMoney adapter 返回 000001.SZ 在 2024-01-02 至 2024-01-05 的 4 行原始日线；未绑定 calendar 的首次探针仍明确 point_in_time=false。后续 calendar-bound 探针结果未能取回，因此不宣称 PIT 数据认证通过。原日志目前无法从失效环境读取，不把本段当作可复核产物替代。

## 未完成与限制 / Outstanding

- 当前 simulator 的 calendar_source 仍是 observed_market_panel：若全市场共同缺一天，不能据此证明官方交易日完整。尾部持仓超出最后 mapped execution 的估值须单独声明窗口，不能直接沿整个输入文件延伸。
- OMS 本批恢复 consumed pre-trade limits；完整 open-order 操作投影、跨进程风控序列化、交易所时钟来源与 broker reconciliation 仍需专项验收。不能据此称可直接实盘。
- Qlib `raw_amount_field` 与 `volume_scale_to_shares` 必须来自实际 bundle 的证据。新 `build-market-panel-v7` 提供相应参数；未经配置的其它 Qlib 路径会明确报错或走已有 fallback，不默认猜单位。日期粒度 PIT 不提供盘中可用性证明。
- RL 仍是受限 weight environment，不是完整 lot/volume/fee/券商逐笔模拟；修复不能证明增量 alpha，旧奖励数字不得用于启用依据。
- 未来缓存修复限定于请求 keys；正式 TRAIN/OOS 全流程仍须冻结训练期 schema，不能把整个合并数据集上的统计筛选当作训练期 fit。
- 本批没有删除任何分支或文件，没有处理不存在的 open PR。任务起点核实 #129 已 merged，#94 已 closed。分支安全删除清单及实际清理尚未完成。
- 独立角色以隔离 checkout 开展了部分审计；不能把库存遍历称为对全部文件逐行语义理解，也不能把中断的角色当作 approve。

## Primary references / 一手资料

- [Qlib Data Layer](https://qlib.readthedocs.io/en/latest/component/data.html)：明确 normalized/adjusted OHLCV 与 `close / factor` 的 raw price restoration。
- [AKShare 股票数据接口](https://akshare.akfamily.xyz/data/stock/stock.html)：核对 endpoint 与量额单位；数据抓取仍由本项目 adapter 执行。
- [AKQuant testing guide](https://akquant.akfamily.xyz/guide/testing/) 与 [API](https://akquant.akfamily.xyz/reference/api/)：事件时钟与测试参考。
- [Boyd et al., Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf)：现金支付交易费用及收益后的权重传播。
- [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent)：因子/模型研究流程参考，不作为本项目收益证据。

用户给定链接中有不可读取页面及未能取回的视频内容；没有声称完整阅读所有网站与视频。
