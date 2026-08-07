# 参考平台能力矩阵

调研日期：2026-08-02。目的是提取**公开的**设计原则与操作模式，用于 QuantAgent 的 A 股研究工作站，不复制任何受保护的源码、页面文案或私有实现。

## 调研可达性（诚实记录）

| 站点 | 方式 | 结果 |
|---|---|---|
| NinjaTrader NT8 帮助中心 | WebFetch | ✅ 完整读取（walk-forward、Strategy Analyzer） |
| MetaTrader 5 帮助中心 | WebFetch | ✅ 完整读取（testing report 指标全集） |
| NautilusTrader 文档 | WebFetch | ✅ 完整读取（事件驱动回测架构） |
| Tradesea help center | WebFetch | ✅ 读取（pSpark 回测流程） |
| Deltapex / Prop Hub `dpj.deltapex.cn/replay` | 真实浏览器 | ✅ 读取（SPA，文本抓取为空，浏览器可见） |
| 王者 Quant `ptqmt.com` | 真实浏览器 | ⚠️ 仅首页/目录可读；子章节需逐页展开，本次未穷尽 |
| EasyXT（公开源码） | WebFetch | ✅ 读取（QMT/miniQMT 封装的 API 面） |
| 聚宽 `joinquant.com` | WebFetch | ❌ **被地域封锁**（"当前地区暂不支持访问"，非中国大陆 IP） |
| 投资科学 `touzikexue.com` | WebFetch | ❌ SPA 且抓取为空，本次未取得实质内容 |
| ProjectX gateway docs | WebFetch | ⚠️ 仅落地页；API 明细未取得 |

未取得内容的站点，本矩阵不做任何推测性陈述。

## 能力矩阵

| 能力 | 参考来源与要点 | QuantAgent 现状 |
|---|---|---|
| **Walk-forward 用"周期长度 + 总区间"配置** | NinjaTrader：用户配置 *Optimization period (days)* 与 *Test period (days)*，窗口在整个日期范围上滚动，产生多个测试段 | ❌→✅ **本次修复**。原先 `n_splits` 锚定在样本起点，5 折在 10 年面板上只验证 2017 年；现改为锚定末端并暴露 `plan_walk_forward` |
| **报告以订单/成交为一等实体** | MT5 testing report：净利、盈亏比、回撤（余额/净值）、Sharpe、期望收益、连胜连亏、MFE/MAE 相关性、持仓时长分布 | ⚠️ 部分：有 trade_blotter.csv 与 metrics.json，但缺 MFE/MAE、持仓时长分布、订单状态机 |
| **逐笔复盘 + 交易日志** | Deltapex：每笔含 日期/时间/品种/**时段**/方向/Entry/Exit/时长/数量/盈亏/**评分**；星期×小时热力图；日历、复盘日记、报告、账户分离 | ❌ 缺失。A 股可对应 集合竞价/早盘/尾盘 时段切分 |
| **事件驱动回测的严格时序** | NautilusTrader：按 `ts_init` 排序；每个数据点走 *交易所撮合 → 策略回调 → 结算* 三段，杜绝前视 | ⚠️ 现有回测未经此标准独立校验（**本次未做**） |
| **撮合与成交模型** | NautilusTrader：L1/L2/L3 book 类型；`prob_fill_on_limit`（排队位置）、`prob_slippage`；延迟模型把订单放入 inflight 队列 | ⚠️ 有 VirtualBroker 干跑路径，未按此维度审计 |
| **回测/实盘同构** | NautilusTrader：同一 actor/strategy 不改代码跑历史与实时 | ❌ 未验证 |
| **QMT/miniQMT 实盘接口形态** | EasyXT：行情 API / 交易 API（买卖、持仓、账户、撤单）/ 高级（组合再平衡、网格）；大 QMT 走信号桥、miniQMT 直连 | ⚠️ 已知 QMT 为 Windows+券商限定，本机 42 项能力全部 PLATFORM_BLOCKED（见既有 memory） |
| **策略代码迁移** | 王者 Quant：聚宽→PTrade/QMT 在线转换器 | ❌ 无此路径 |
| **回测→迭代的闭环叙事** | Tradesea pSpark：保存策略→回测→看胜率/盈亏/回撤/逐笔→加过滤器前后对比→回到对话迭代；实盘部署明确标注 "coming soon" | ⚠️ 部分：有候选比较，但"改一个约束会怎样"的反事实提示缺失 |

## 本次采纳的三条原则

1. **折数是数据的函数，不是愿望**（NinjaTrader）。请求的窗口数必须由日期跨度决定，且预估与实际必须由同一段算术产生 —— 已落地为 `plan_walk_forward`，前端预检与运行期共用。
2. **"跑不成"要和"被否决"分开**（本项目自身教训 + Tradesea 对未上线能力的诚实标注）。配置不可行 → `blocked`，训练前中止；假设被闸门拒绝 → `rejected`，证据保留。
3. **锚定最近数据**（NinjaTrader/MT5 的滚动语义）。滚动验证的意义在于逼近部署时点，锚在样本起点等于研究一个已消失的市场。

## 尚未采纳（明确记录为未做）

订单状态机、逐事件市场回放、MFE/MAE、时段热力图、流式回测与现有回测的订单级对账、QMT 回报对账 —— 均属参考站点的核心能力，本次**未实现**。
