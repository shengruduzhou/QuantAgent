# MT5 A股行情能力实测报告

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **分支**：`agent/ashare-mt5-tick-multiagent`
- **探测脚本**：`scripts/probe_mt5_capability.py`
- **实现模块**：`src/quantagent/data/providers/mt5_capability.py`
- **产物目录**：`runtime/data/capabilities/mt5/`

## 1. 结论（先说结论）

**本机 MT5 分类结果 = `TERMINAL_UNAVAILABLE`。**

这不是"MT5 没有 A 股数据"，而是**本机根本无法测量 MT5 的任何能力**。两者必须
严格区分：前者是关于券商的结论，后者是关于运行环境的事实。本报告只给出后者。

因此以下问题在本机全部为 **UNMEASURED（未测量）**，不得在任何后续文档中被写成
已知结论：

- MT5 是否存在真实的 A 股交易所行情；
- tick 历史深度与字段可用性；
- DOM（市场深度）档位数；
- Strategy Tester 真实 tick 与生成 tick 的差异。

## 2. 实测证据

### 2.1 运行环境

| 项 | 实测值 |
| --- | --- |
| 操作系统 | Linux 5.4.0-216-generic（Ubuntu 20.04.6 LTS） |
| Python | 3.12 / 3.10（`AI_quant_venv`） |
| `MetaTrader5` 包可导入 | **否** |
| 导入错误 | `ModuleNotFoundError: No module named 'MetaTrader5'` |
| Wine 是否存在 | 否（`which wine` / `wine64` 均无输出） |
| `terminal64.exe` | 全盘搜索无结果 |

### 2.2 关键证据：该包在本平台**不存在任何可安装版本**

```bash
pip download MetaTrader5 --no-deps
```

实测输出：

```
ERROR: Could not find a version that satisfies the requirement MetaTrader5
       (from versions: none)
ERROR: No matching distribution found for MetaTrader5
```

`from versions: none` 是决定性证据：不是"版本不兼容"，而是 PyPI 对本平台
**不提供任何发行版**。MetaTrader5 Python 包仅发布 Windows wheel。

### 2.3 官方文档佐证

来源：<https://www.mql5.com/en/docs/python_metatrader5>（2026-07-27 访问）

官方原文：

> "MetaTrader package for Python is designed for convenient and fast obtaining
> of exchange data via **interprocessor communication directly from the
> MetaTrader 5 terminal**."

**事实**：该包通过 IPC 与**本机运行中的终端**通信。
**推论（本报告作出，非官方原文）**：因此"装上包"与"有行情"是两个独立问题；即使
在 Windows 上装好包，若无已登录终端，`initialize()` 仍会失败。

官方安装说明引用的是 `python.org/downloads/**windows**`，与 2.2 的实测一致。

## 3. 探测器设计（为何是 fail-closed）

`run_probe()` 在终端不可达时仍然写出全部产物，且每一个 capability cell 记为
`CLIENT_UNAVAILABLE`，而**不是** `NOT_OFFERED`。这两个状态的区别是本仓库的硬性
要求：

- `CLIENT_UNAVAILABLE` = 我们无法运行客户端 → 能力未知；
- `NOT_OFFERED` = 已确认该供应商不提供该数据。

把前者写成后者，等于把"没测"伪造成"没有"。

探测器输出中显式包含以下免责声明（见 `probe_result.json`）：

> "No MT5 terminal was reachable, so every downstream MT5 question (A-share
> symbols, tick history, DOM depth, real vs generated ticks) is UNMEASURED on
> this host. This is not evidence that brokers do or do not offer A-share feeds."

## 4. 若在 Windows 主机上续测，探测器会做什么

以下逻辑已实现并有单元测试覆盖（`tests/data/test_market_data_providers.py`），
只是在本机没有执行机会：

1. **品种名穷举**：对每个规范代码尝试 6 种券商拼写
   （`600000.SH` / `SH600000` / `600000.SSE` / `600000` / `600000.sh` / `SSE:600000`），
   **不假设**任何一种成立。
2. **五板块覆盖**：沪主板、深主板、创业板、科创板、北交所各取代表标的，
   而不是只测一只蓝筹。
3. **真伪判定**：仅当 `exchange` 字段非空**且**计价货币为 CNY 时，才可能被判为
   交易所行情；否则一律 `BROKER_CFD_OR_SYNTHETIC`。券商把合约命名成 `600000`
   不构成证据。
4. **DOM 必须"读到"而非"订阅成功"**：`market_book_add()` 返回 true 只说明订阅
   被接受；探测器会连续 `market_book_get()` 5 次并取最大档位数，档位为 0 即判
   `A_SHARE_TICK_NO_DEPTH`。
5. **分级结论**：`>5` 档 → `A_SHARE_LEVEL2_CANDIDATE`（候选，仍需与交易所口径
   对账才能升级）；`1..5` 档 → `A_SHARE_DOM_SNAPSHOT`。

## 5. 架构影响

由于本机 MT5 不可用，且**即使可用也无法先验假定其有 A 股源**，本仓库的架构决策为：

- MT5 **不进入**权威数据链路，只作为自定义品种回放与 MQL5 实验工作站；
- 规范数据底座保持供应商中立（`quantagent.data.microstructure`）；
- MT5 相关模块一律不在模块级 import `MetaTrader5`，保证无终端主机上仍可导入与测试。

详见 `docs/research/MT5自定义品种导入验证.md`。

## 6. 产物清单

| 文件 | 内容 |
| --- | --- |
| `runtime/data/capabilities/mt5/terminal.json` | 终端与运行环境状态 |
| `runtime/data/capabilities/mt5/accounts.json` | 账户状态（本机为不可用） |
| `runtime/data/capabilities/mt5/symbols.parquet` | 品种探测明细 |
| `runtime/data/capabilities/mt5/ashare_symbol_mapping.parquet` | 规范代码 ↔ 券商拼写映射尝试 |
| `runtime/data/capabilities/mt5/tick_probe.parquet` | tick 探测结果 |
| `runtime/data/capabilities/mt5/dom_probe.parquet` | DOM 探测结果 |
| `runtime/data/capabilities/mt5/capability_matrix.{json,csv}` | 能力矩阵 |
| `runtime/data/capabilities/mt5/probe_result.json` | 完整结构化结果 |

## 7. 事实与推论分离

| 陈述 | 类别 |
| --- | --- |
| 本机无 `MetaTrader5` 包，PyPI 对本平台无任何发行版 | **事实（实测）** |
| 官方文档称该包通过 IPC 直连本地终端 | **事实（官方原文）** |
| 本机无法测量 MT5 的 A 股能力 | **事实（由上二者直接得出）** |
| MT5 券商普遍不提供 A 股交易所行情 | **未验证，不作断言** |
| 即使有终端，仍需按 §4 流程逐项取证 | **推论（本报告）** |

## 8. 未决阻塞

1. 没有 Windows MT5 主机 → 全部 MT5 能力问题维持 UNMEASURED。
2. Linux/Wine 路径**未评估**，且按任务要求不作为首选生产假设。
3. 因此"MT5 是否可作为 A 股 tick 源"这一问题**本任务未能结案**，只能结论为
   "本机不可测，且架构上已不依赖它"。
