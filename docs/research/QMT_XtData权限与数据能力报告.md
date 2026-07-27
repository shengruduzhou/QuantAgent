# QMT / XtData 权限与数据能力报告

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **探测脚本**：`scripts/probe_xtdata_capability.py`
- **实现模块**：`src/quantagent/data/providers/xtdata_market_provider.py`
- **产物目录**：`runtime/data/capabilities/qmt/`

## 1. 结论

**本机 QMT/XtData 分类 = `CLIENT_UNAVAILABLE`（11 个数据族全部）。**

关键区分：

- `xtquant` 这个 **Python 包本身可以装在 Linux 上**（wheel 标记为 `py3-none-any`）；
- 但它的**原生扩展只有 Windows 版本**，`from xtquant import xtdata` 在 Linux 上
  必然抛 `ImportError`；
- 因此本机**无法测量**任何 QMT 权限，包括 Level-2 是否被授权。

## 2. 决定性证据：wheel 平台普查

下载 `xtquant-250807.1.2-py3-none-any.whl`（34.4 MB）并逐项统计其原生模块：

| 类型 | 数量 |
| --- | --- |
| Windows `.pyd` 扩展 | **22** |
| Windows `.dll` | **6** |
| Linux `.so` | **0** |

`.pyd` 清单（节选）：

```
xtquant/datacenter.cp310-win_amd64.pyd
xtquant/datacenter.cp311-win_amd64.pyd
xtquant/datacenter.cp312-win_amd64.pyd
xtquant/datacenter.cp313-win_amd64.pyd
xtquant/xtpythonclient.cp39-win_amd64.pyd
...
```

`.dll` 清单：`datacenter_shared.dll`、`libeay32.dll`、`log4cxx.dll`、
`msvcp140.dll`、`ssleay32.dll`、`vcruntime140.dll`

**结论**：wheel 标签 `py3-none-any` 是误导性的——它声称平台无关，实际只带
Windows 二进制。

产物：`runtime/data/capabilities/qmt/xtquant_wheel_platform_census.json`

## 3. 实证导入失败

在隔离虚拟环境中**真实安装**该 wheel 后运行探测器：

```
package_importable : true
xtdata_importable  : false
platform_supported : false
import_error       : ImportError: cannot import name 'datacenter' from 'xtquant'
```

失败位置：`xtquant/xtdata.py:278` → `xtquant/xtdatacenter.py:5` →
`from . import datacenter as __dc`。

同理 `xtquant.xttrader` 失败于 `from . import xtpythonclient as _XTQC_`。

产物：`runtime/data/capabilities/qmt/installed_wheel_runtime_evidence.json`

## 4. XtData 声称的能力面（读源码所得，非官方文档转述）

`xtdata.py` 是纯 Python 且可读。以下接口**确实存在于 SDK**，但**是否对本账户
授权，本机无法验证**：

| 接口 | period 字面量 | 规范事件族 | 声称的数据类 |
| --- | --- | --- | --- |
| `get_l2_quote` | `l2quote` | book_snapshot | `LEVEL2_SNAPSHOT`（按价位聚合） |
| `get_l2_order` | `l2order` | order_event | `EXCHANGE_ORDER_EVENT`（逐笔委托） |
| `get_l2_transaction` | `l2transaction` | trade_event | `EXCHANGE_TRADE_EVENT`（逐笔成交） |
| `get_l2thousand_queue` | `l2thousand` | book_snapshot | `LEVEL2_ORDER_BOOK`（千档队列） |
| `get_full_tick` | `tick` | quote_event | `LEVEL1_QUOTE`（盘口快照） |

补充接口：`get_fullspeed_orderbook`（全速盘口）、`get_broker_queue_data`（港股经纪队列）、
`get_divid_factors`（除权因子）、`get_instrument_detail`、`get_index_weight`、
`get_financial_data`。

> **重要**：接口存在只说明 QMT **售卖** Level-2，不说明本账户**被授权读取**。这
> 两件事在本仓库的能力矩阵里是两个独立字段（`status` 与 `entitlement`）。

## 5. 与 U0 PIT 阻塞项的直接关联（本次最有价值的发现）

`xtdata` 暴露了：

```python
get_his_st_data(stock_code)
download_his_st_data(...)
```

读源码可见其实现为读取本地 QMT 数据目录下的 `SH_XXXXXX_2011_86400000.csv`，
按 `(股票代码, 生效日, 状态标记)` 还原 **ST 状态区间**。

这正是 U0 当前**唯一剩余的 PIT 阻塞项**：

```
st_intervals: BLOCKED_BY_DATA — PARTIAL: 906 dated episodes over 651 securities
              from SZSE; no dated register for BSE, SSE
```

**推论（本报告作出）**：若取得券商 QMT 权限并在 Windows 主机运行，
`get_his_st_data` 有可能补齐 SSE 与 BSE 的 ST 历史区间，从而解除 U0 的 PIT 阻塞、
放行全宇宙训练。这是一条**明确的、可执行的**后续路径，但**尚未验证**。

其余可能有用的接口：`get_divid_factors`（公司行动）、
`get_market_data_ex` 的 `suspendflag` 字段（停牌历史）。

## 6. 读写隔离（架构硬约束）

本模块**只读**，且由测试强制：

- `test_module_never_imports_the_trader` 解析模块的 **AST 导入图**（不是文本
  grep），断言 `xtquant.xttrader`、`xtquant.xtconstant`、`xtquant.xttype` 与
  `qmt_gateway` 均未被导入，且 `xtquant.xtdata` 被导入；
- 用 AST 而非文本，是为了让模块能在文档里**说明**这条隔离，而不会被自己的说明
  文字触发告警；
- 执行侧 `quantagent.execution.qmt_gateway` 的 `dry_run=True` /
  `live_trading_enabled=False` 默认值**本次未改动**。

行情读取路径的缺陷与下单路径的缺陷是两类事故，不允许共享隐式状态。

## 7. 归一化实现（无客户端也可测试）

`normalise_transactions` / `normalise_orders` 是纯函数，可在无 QMT 的主机上用
构造载荷测试：

- 逐笔成交的买卖方向由 `bidOrder` 与 `askOrder` 的先后推出，故
  `side_method = ORDER_ID_MATCHED`（**观测**，非 tick-rule 猜测）——这与腾讯分笔
  的 `QUOTE_RULE_INFERRED` 形成对比；
- 若载荷缺少订单号（如 SSE 的部分 l2transaction），`side` 置空且
  `side_method = UNKNOWN`，**不猜方向**；
- `tradeIndex` 进入 `sequence`（交易所序列号），与 `ingest_sequence` 分离。

本次修复了一个真实缺陷：原实现用 `DataFrame.get()` 取列，缺列时返回标量而非
Series，在缺 `askOrder` 的载荷上直接崩溃。已改为显式的 `_column()` helper，缺列
返回全空 Series。

## 8. 未决阻塞

1. **无 Windows 主机 + 无券商 QMT 权限** → Level-2 全部维持 `CLIENT_UNAVAILABLE`。
2. 因此本任务**未能**取得任何真实 Level-2 记录，这一点在所有下游文档中如实记录。
3. `get_his_st_data` 能否补齐 SSE/BSE 的 ST 历史 —— **待验证**，是解除 U0 PIT
   阻塞的首选路径。

## 9. 复现命令

```bash
python scripts/probe_xtdata_capability.py --output runtime/data/capabilities/qmt
```

在装有 `xtquant` 的环境中运行会额外产出
`installed_wheel_runtime_evidence.json`（本次即以隔离 venv 取得）。
