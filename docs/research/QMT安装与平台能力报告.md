# QMT 安装与平台能力报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **探测脚本**：`scripts/probe_qmt_entitlements.py`
- **产物**：`runtime/data/capabilities/qmt/environment.json`

## 1. 结论

**环境判定 = `PLATFORM_UNAVAILABLE`。**

本机为 Linux（Ubuntu 20.04.6，内核 5.4.0-216），无 Wine，未安装 xtquant，无 MiniQMT。QMT/MiniQMT 是 Windows 桌面客户端，xtquant 的原生扩展仅有 Windows 版本。因此本机**无法测量任何 QMT 权限**，结论一律为 `PLATFORM_BLOCKED` / `PLATFORM_UNAVAILABLE`。

关键区分（本报告全篇遵守）：

- **未测量 ≠ 不可用**。本报告不对"某券商是否提供某数据"下任何结论；
- **文档中有接口 ≠ 本账户有权限**。

## 2. 实测环境

| 项 | 实测值 |
| --- | --- |
| 操作系统 | Linux 5.4.0-216-generic（Ubuntu 20.04.6 LTS） |
| 是否 Windows | **否** |
| Wine | 未安装（`which wine`/`wine64` 无输出） |
| xtquant 已安装 | **否**（`ModuleNotFoundError: No module named 'xtquant'`） |
| xtdata 可导入 | 否 |
| MiniQMT 连接 | 否 |
| MiniQMT 路径 | 未发现 |
| 已授权市场 | 无法查询 |

### 2.1 补充证据：xtquant wheel 平台普查

此前会话下载并解包 `xtquant-250807.1.2-py3-none-any.whl`（34.4 MB）：

| 类型 | 数量 |
| --- | --- |
| Windows `.pyd` | **22** |
| Windows `.dll` | **6** |
| Linux `.so` | **0** |

wheel 标签为 `py3-none-any`（声称平台无关），实际只带 Windows 二进制。在隔离
venv 中真实安装后，`from xtquant import xtdata` 于 `datacenter` 处抛
`ImportError`。

产物：`runtime/data/capabilities/qmt/xtquant_wheel_platform_census.json`

## 3. 官方文档要点（事实，2026-07-28 访问）

| 主题 | 来源 | 关键事实 |
| --- | --- | --- |
| 行情/合约/财务/板块 API | <https://dict.thinktrader.net/nativeApi/xtdata.html> | `get_instrument_detail`、`get_trading_dates`、`download_history_data`、`get_financial_data`、`get_sector_list`、`get_index_weight` 等 |
| 周期字面量 | 同上 | `tick`,`1m`,`5m`,`15m`,`30m`,`1h`,`1d`,`1w`,`1mon`,`1q`,`1hy`,`1y` |
| **Level-2 权限** | 同上 | 原文：**「获取lv2数据时需要数据终端有lv2数据权限」** |
| 订阅限制 | <https://dict.thinktrader.net/innerApi/data_function.html> | 提及非 VIP 存在"订阅数量限制"、VIP 支持全市场推送，**但未给出具体数字** |
| 历史长度限制 | 同上 | **官方文档未给出**分钟/tick/日线的明确历史长度上限 |

> **重要**：任务背景中提到的"100 个订阅位 / 1 年分钟 / 1 个月 tick / VIP 300 位 /
> 3 年分钟"等数字，**在我能获取到的官方文档中并未出现**。因此本仓库**不会**把这些
> 数字写进任何报告作为事实——它们必须以真实账户实测为准。

## 4. 能力目录（已建立，待实测）

已建立 **42 项**能力目录，覆盖任务要求的四大族：

| 族 | 能力数 |
| --- | --- |
| security（证券与日历） | 12 |
| bars（日线与日内） | 10 |
| pit（PIT 与公司行动） | 10 |
| microstructure（实时与微观结构） | 10 |

每项记录：`capability / api / documented / platform / permission_class /
probe_status / earliest_date / latest_date / symbols_requested / symbols_returned /
rows_returned / fields / sample_hash / error / source_url`。

**当前 42/42 全部为 `PLATFORM_UNAVAILABLE`。**

## 5. 网关设计（已实现，等待 Windows 主机）

`src/quantagent/data/providers/qmt_gateway.py`：健康检查、版本检查、MiniQMT 路径
探测、连接状态、权限探测、断点续传、限速、数据哈希、schema 校验、规范 Parquet 导出、
manifest 生成、写入目录白名单、**不记录任何凭证**、**只读**。

只读性由测试在**导入图**层面强制：`xtquant.xttrader` 与 `XtQuantTrader` 均不得
出现在导入图中。

## 6. 未决阻塞

1. **无 Windows 主机 + 无券商 QMT 账户** → 全部 42 项能力维持未测量；
2. 因此本任务**未能**取得任何真实 QMT 数据、任何真实权限等级、任何真实历史范围。

## 7. 复现命令

```bash
python scripts/probe_qmt_entitlements.py --output runtime/data/capabilities/qmt
```
