# QMT 历史 ST 权限与覆盖报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **产物**：`runtime/data/capabilities/qmt/st_probe.json`

## 1. 结论

**ST 历史未探测，权限判定 = `PLATFORM_UNAVAILABLE`。**

本机为 Linux，无 Wine／xtquant／MiniQMT；QMT 为 Windows 客户端，其原生扩展仅有 Windows 版本。 因此本次**没有**取得任何 ST 区间数据。

**`st_intervals` 仍然是 U0 严格 PIT 的强制未满足字段，未被移除、未被弱化。**

## 2. 为什么这件事最关键

U0 的严格全宇宙就绪当前**只被这一项**阻塞：

```
st_intervals: BLOCKED_BY_DATA — PARTIAL: 906 dated episodes over 651 securities
              from SZSE; no dated register for BSE, SSE
```

只有深交所有带日期的简称变更登记，沪市与北交所没有。**部分覆盖算 BLOCKED，不算通过。**

## 3. 拟执行的探测（代码已实现）

```python
from xtquant import xtdata
xtdata.download_his_st_data()
result = xtdata.get_his_st_data("000004.SZ")
```

**绝不假设成功。** `QmtGateway.probe_st_with_controls()` 的设计如下。

### 3.1 阳性对照是必需的（本报告最重要的设计）

`get_his_st_data` 在两种**相反**的情况下都返回 `{}`：

1. 该证券**从未** ST；
2. 账户**无权读取** ST 数据集（或数据集未下载）。

若不加区分，第二种会把整个宇宙写成"从未 ST"——这正是任务禁止的
"Never treat empty permission-denied data as a valid empty dataset"。

因此探测强制使用对照组：

| 对照 | 标的 | 作用 |
| --- | --- | --- |
| 阳性 | `000004.SZ`、`000005.SZ`、`600149.SH` | **已知曾经 ST** |
| 阴性 | `600519.SH`、`000651.SZ` | 预期从未 ST |

判定规则：

- 阳性对照返回了区间 → 权限成立，此时其他标的的空**才是**"从未 ST"的证据；
- **阳性对照也返回空** → 数据集不可用，**任何标的都不得被记为 never-ST**。

代码中的原文判定：

> "a security KNOWN to have been ST also returned empty, so the ST dataset is
> unavailable to this account; no security may be recorded as never-ST from
> this probe"

### 3.2 需要验证的区间性质（方法已实现）

区间起止日、开放区间、重叠、相邻转换、ST→\*ST、\*ST→ST、撤销风险警示、简称变更、
退市；以及五个板块 + 当前 ST + 历史 ST + 历史 \*ST + PT + 从未 ST + 已退市的覆盖。

样本需与交易所公告或 CNINFO 交叉核对。

## 4. 若 QMT ST 需要 VIP 权限

按任务要求的处置（已在代码与流程中固化）：

1. 报 `BLOCKED_BY_ENTITLEMENT`（而非静默失败）；
2. **不**把空字典当成"从未 ST"；
3. 继续走官方公告重建路径；
4. 保持严格 `FULL_UNIVERSE_DATA_READY` 定义不变；
5. **不**把 `st_intervals` 从强制 PIT 字段中移除。

## 5. 训练切片证书的条件

任务允许在**选定训练窗内每个 security-date 的 PIT 状态都已知**时，实现独立的
训练切片证书。

当前**不满足**该条件：沪市与北交所在整个历史上都没有带日期的 ST 登记，因此不存在
一个所有 PIT 状态已知的全宇宙窗口。故**未实现**该旁路，训练维持阻塞。

## 6. 未决

| 项 | 状态 |
| --- | --- |
| QMT ST 历史权限 | **未测量**（无 Windows 主机） |
| 沪市 ST 带日期登记 | 仍缺 |
| 北交所 ST 带日期登记 | 仍缺 |
| U0 严格 PIT | **仍被 st_intervals 阻塞** |
