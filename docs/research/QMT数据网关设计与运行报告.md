# QMT 数据网关设计与运行报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **实现**：`src/quantagent/data/providers/qmt_gateway.py`

## 1. 架构：卫星，而非迁移

**不把 QuantAgent 迁到 Windows。** 只有物理上必须依赖 MiniQMT 的那一层跑在 Windows：

```
Windows QMT 网关
    ├── MiniQMT / QMT
    ├── xtquant
    ├── 能力探测
    ├── 下载（断点续传）
    ├── 实时订阅
    ├── 不可变本地暂存
    └── 受治理导出（规范 Parquet + manifest）
                 ↓
Linux QuantAgent
    ├── U0 数据底座
    ├── 数据湖
    ├── PIT 校验
    ├── 特征生成
    ├── 回测
    ├── 模型训练
    └── 工作站
```

## 2. 本次运行状态

**`PLATFORM_UNAVAILABLE`。** 本机 Linux，无法运行 MiniQMT，网关未执行任何下载。

## 3. 只读性（结构性保证，非口头承诺）

本模块导入 `xtquant.xtdata`，**不导入** `xtquant.xttrader`，**不构造**
`XtQuantTrader`，不持有账户 ID，不暴露任何下单路径。

由测试在**解析后的导入图**上强制（`test_module_never_imports_the_trader`），
用 AST 而非文本 grep——这样模块可以在文档里**说明**这条隔离，而不会被自己的
说明文字触发告警。

本任务期间**不启用**任何实盘交易。

## 4. 已实现的网关能力

| 能力 | 实现 |
| --- | --- |
| 健康检查 / 版本检查 | `probe_environment()` 返回 OS、xtquant 版本、原生扩展平台 |
| MiniQMT 路径探测 | `get_data_dir()`，记入 `miniqmt_paths_found` |
| 连接状态 | `get_authorized_market_list()`；失败判 `CLIENT_DISCONNECTED` |
| 权限探测 | `probe_capability()`，十级 `probe_status` |
| 断点续传 / 进度检查点 | `incrementally` 参数 + 分区级 manifest |
| 限速 | 逐能力 timeout + `max_retries` |
| 数据哈希 | `_frame_hash()`（sha256 前 16 位） |
| schema 校验 | 返回 `fields` 列表并入 manifest |
| 规范 Parquet 导出 | `export_canonical()` |
| manifest 生成 | 含 provider/capability/rows/columns/content_hash/provenance |
| 传输确认 | manifest 落盘即为确认凭据 |
| **不记录凭证** | manifest 显式标 `contains_credentials: false`，不写账户信息 |
| **原始数据不入 Git** | 写入目录白名单 + `.gitignore` 屏蔽 `runtime/` |

## 5. 两条比下载逻辑更重要的行为

### 5.1 空 ≠ 不存在

`classify_empty()` 默认悲观：权限未确认时的空一律 `EMPTY_UNVERIFIED`，下游
不得当作数据。这正是"ST 三年的股票变成从未 ST"的防线。

### 5.2 截断被检测，而非被信任

供应商在权限窗口外会**成功但少给**。`detect_truncation()` 比较请求窗口与实际
窗口，窄于请求即报 `TRUNCATED` 并同时记录两个区间——否则一年的基础档位会被
误认为十年历史。

## 6. 写入目录白名单

```
runtime/data/capabilities/qmt
runtime/data/qmt_staging
runtime/data/market_events
```

`assert_allowed_output()` 解析真实路径后校验，**父目录穿越（`../`）与绝对路径
均被拒绝**，有测试覆盖。

## 7. 未决

网关代码就绪，等待 Windows + 券商账户。在此之前所有下载能力维持未执行。
