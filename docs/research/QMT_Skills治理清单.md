# QMT Skills 治理清单

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **实现**：`src/quantagent/data/providers/qmt_skills.py`
- **产物**：`runtime/data/capabilities/qmt/skill_inventory.json`

## 1. 核心规则

**LLM 不得仅因为函数名存在就调用某个 skill。** 只有当能力证书表明该能力实测为
`SERVING`、且平台可达时，skill 才被授权执行。其余一律拒绝并给出机器可读原因。

## 2. 13 个只读 skill

| Skill | 所需能力 | 平台 | 需网络 |
| --- | --- | --- | --- |
| `qmt_probe_environment` | — | ANY | 否 |
| `qmt_probe_permissions` | — | WINDOWS | 是 |
| `qmt_list_periods` | — | WINDOWS | 否 |
| `qmt_list_ashare_symbols` | instrument_master | WINDOWS | 是 |
| `qmt_download_daily` | daily_raw | WINDOWS | 是 |
| `qmt_download_minute` | minute_1m | WINDOWS | 是 |
| `qmt_download_tick` | tick | WINDOWS | 是 |
| `qmt_download_st_history` | st_history | WINDOWS | 是 |
| `qmt_download_financials` | financial_statements | WINDOWS | 是 |
| `qmt_probe_level2` | — | WINDOWS | 是 |
| `qmt_export_canonical_partitions` | — | ANY | 否 |
| `qmt_reconcile_u0` | — | ANY | 否 |
| `qmt_report_coverage` | — | ANY | 否 |

每个 skill 声明：输入 schema、输出 schema、所需权限、平台要求、网络要求、
允许写入目录、超时、重试策略、只读标记。

**`trading_permitted` 恒为 `false`。**

## 3. 五层拒绝（含机器可读原因）

| 原因码 | 触发条件 |
| --- | --- |
| `UNKNOWN_SKILL` | 未注册的 skill 名 |
| `TRADING_NOT_PERMITTED` | skill 名或参数含下单语义（order/trade/买卖/委托/撤单…） |
| `PLATFORM_NOT_SUPPORTED` | 需要 Windows 但当前非 Windows |
| `ENTITLEMENT_NOT_GRANTED` | 所需能力的 `probe_status ≠ SERVING` |
| `INVALID_PARAMETERS` | 非法标的/日期/周期/未知参数 |
| `OUTPUT_PATH_NOT_ALLOWED` | 绝对路径、`..` 穿越、白名单外目录 |

**顺序是刻意的**：交易拒绝排在平台与权限之前，任何平台或权限状态都无法授权下单路径。

### 3.1 一个必要的例外

`l2order` 里含有 `order` 字样，但它是**只读的逐笔委托行情**，不是下单。因此
`qmt_probe_level2` 与 `qmt_download_tick` 在交易关键词匹配中豁免——否则探测
Level-2 委托流会被误判为提交订单。有测试锁定该行为。

## 4. 参数校验

- 标的必须匹配 `^[0-9]{6}\.(SH|SZ|BJ)$`；
- 日期必须 `YYYYMMDD`；
- 周期只允许 `1m|5m`；复权只允许 `none|front|back`；
- **未知参数直接拒绝**（防止夹带）；
- 路径参数经 `_safe_path()`：拒绝绝对路径、`..`、白名单外目录。

**无 shell、无任意文件访问、无交易权限。**

## 5. 审计

每次授权决定（无论允许还是拒绝）都写入 `SkillAudit`：skill 名、时间、是否允许、
拒绝原因、平台、所需能力、**该能力当时的实测状态**、被接受的参数。

## 6. 当前状态

本机非 Windows，故所有 `WINDOWS` 类 skill 一律 `PLATFORM_NOT_SUPPORTED`，
拒绝信息中明确要求上报 **`NOT_RUN_PLATFORM`** 而非"探测失败"。
