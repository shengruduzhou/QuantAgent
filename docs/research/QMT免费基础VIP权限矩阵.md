# QMT 免费 / 基础 / VIP 权限矩阵

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **实现**：`src/quantagent/data/providers/qmt_entitlement.py`
- **产物**：`runtime/data/capabilities/qmt/entitlement_matrix.{json,csv}`

## 1. 本次结论

**42 项能力全部 = `PLATFORM_BLOCKED` / `PLATFORM_UNAVAILABLE`。**

> **本机未测量。** Linux 主机无法运行 MiniQMT，本节全部条目为 `PLATFORM_BLOCKED`／`PLATFORM_UNAVAILABLE`。任何权限等级、历史范围或覆盖率数字都必须在真实 Windows + 券商账户上实测后填入，**不得**从公开对比表抄录。

## 2. 为什么不用"免费"这个布尔值

"免费"至少混合了六种互不相同的情况。本矩阵用八个权限类替代它：

| 权限类 | 含义 |
| --- | --- |
| `BROKER_INCLUDED` | 券商随普通资金账户附带 |
| `BASIC_INCLUDED` | 基础（非 VIP）数据档位包含 |
| `VIP_INCLUDED` | 需要 VIP 数据档位 |
| `SEPARATE_PURCHASE_REQUIRED` | 需单独购买的数据产品 |
| `BROKER_DEPENDENT` | 因券商而异，无法一般性回答 |
| `UNKNOWN_UNTIL_PROBED` | **诚实的默认值**：尚未确立 |
| `UNAVAILABLE` | 已确认任何档位都拿不到 |
| `PLATFORM_BLOCKED` | 本操作系统根本无法触达 |

## 3. 探测状态：十级判定

任务要求区分"接口存在"到"适合生产"之间的十个层次。矩阵用 `probe_status` 承载：

| 状态 | 含义 |
| --- | --- |
| `SERVING` | 真实调用返回了真实、非空、语义明确的数据（**唯一**可称可用） |
| `EMPTY_VERIFIED` | 权限已确认，且该键确实无数据 |
| `EMPTY_UNVERIFIED` | 返回空但权限未确认——**绝不可当作"数据不存在"** |
| `PERMISSION_DENIED` | 供应商以权限理由拒绝 |
| `TRUNCATED` | 无报错但返回窗口窄于请求 |
| `PLATFORM_UNAVAILABLE` | 客户端无法在本机运行 |
| `CLIENT_DISCONNECTED` | xtdata 可导入但 MiniQMT 未应答 |
| `NOT_PROBED` | SDK 有此接口，本机未调用 |
| `ERROR` | 非权限原因的调用错误 |
| `UNKNOWN_SEMANTICS` | 返回内容含义未文档化 |

代码层面强制的两条不变量（有测试）：

1. `SERVING` 必须 `rows_returned > 0`，否则构造即抛异常；
2. `EMPTY_VERIFIED` 在权限为 `UNKNOWN_UNTIL_PROBED`/`PLATFORM_BLOCKED` 时**禁止**
   构造——权限未确立时的空只能是 `EMPTY_UNVERIFIED`。

## 4. 能力目录（42 项）

### 4.1 证券与日历（12）
instrument_master、instrument_type、active_stock_list、delisted_identities、
board_classification、listing_date、delisting_date、trading_calendar、
trading_sessions、security_status、suspension_flag、price_limits

### 4.2 日线与日内（10）
daily_raw、daily_front_adjusted、daily_back_adjusted、adjustment_factors、
minute_1m、minute_5m、tick、amount_field、volume_field、preclose_field

### 4.3 PIT 与公司数据（10）
**st_history**（U0 唯一剩余 PIT 阻塞项）、star_st_history、pt_history、
corporate_actions、dividends、rights_issues、financial_statements、
sector_membership、index_membership、announcements（xtdata 无对应接口，CNINFO 仍为来源）

### 4.4 实时与微观结构（10）
full_market_snapshot、single_symbol_subscription、level1_quote、five_level_quote、
l2quote、l2quoteaux、l2order、l2transaction、l2transactioncount、l2orderqueue

## 5. 官方文档中**确实**有的权限陈述

| 陈述 | 来源 | 类别 |
| --- | --- | --- |
| 「获取lv2数据时需要数据终端有lv2数据权限」 | xtdata 文档 | **事实（官方原文）** |
| 非 VIP 存在订阅数量限制；VIP 支持全市场推送 | innerApi 文档 | **事实（定性，无数字）** |
| 具体订阅位数、分钟/tick 历史长度 | — | **官方文档未给出** |

## 6. 待实测清单（Windows 主机就绪后）

```bash
python scripts/probe_qmt_entitlements.py --output runtime/data/capabilities/qmt
```

该脚本会为每项能力填入真实的 `permission_class` 与 `probe_status`，并测量
`earliest_date`/`latest_date`。在那之前，本矩阵的价值是**明确列出未知的范围**，
而不是给出未经验证的答案。
