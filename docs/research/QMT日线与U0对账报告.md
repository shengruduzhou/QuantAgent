# QMT 日线与 U0 对账报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **实现**：`src/quantagent/data/ashare/source_precedence.py`
- **产物**：`runtime/data/capabilities/qmt/reconciliation.json`

## 1. 结论

**未执行真实对账。** 本机为 Linux，无 Wine／xtquant／MiniQMT；QMT 为 Windows 客户端，其原生扩展仅有 Windows 版本。 没有 QMT 日线数据可供比对。

对账引擎与补丁治理**已实现并有 20 条测试覆盖**，等待 Windows 主机提供数据。

## 2. U0 保持日线真值源

明确的优先级（已编码为 `PRECEDENCE`）：

```
已验证的 U0 源
  → 仅在 U0 缺失或可证明有误时，用 QMT 打补丁
    → 公开数据源兜底
```

理由：U0 面板是 **17,829,080 行 / 5,892 只 / 1990-12-19..2026-07-24**，已通过
identity / provider / coverage / quality 四道闸门（BAR 判定 `U0_BAR_READY`）。
QMT 历史更短且权限因账户而异，整体替换等于用未验证换已验证。

## 3. 对账字段与容差

| 字段 | 容差 | 理由 |
| --- | --- | --- |
| open/high/low/close/preclose | 相对 1e-4（1bp） | 远松于供应商舍入，远紧于任何真实错误 |
| volume | 相对 1e-3 **或** 绝对 100 股 | 4000 万股上差 1 股是舍入 |
| amount | 相对 5e-3 | 供应商对成交额的舍入比成交量更粗 |

## 4. 三类不同性质的不一致（本引擎的核心设计）

把它们混为一谈会导致错误的修复动作，因此各有独立判定：

### 4.1 `MISMATCH_UNIT` —— 手 / 股 单位错误

当 QMT/U0 的 volume 比值接近 **100**（容差 0.01）时判定为单位不一致，而非数据不一致。

**处置：`REJECTED_UNIT_MISMATCH`，绝不打补丁。** 单位错误是某个适配器的 schema 缺陷，
按值修补等于把缺陷固化进面板。

### 4.2 `MISMATCH_ADJUSTMENT` —— 复权口径不一致

当 OHLC 四个字段**以同一比例**偏离（比值标准差 / 均值 < 1e-3）时，判定为复权口径
分歧（raw vs qfq vs hfq），而非坏行情。

**处置：`REJECTED_U0_AUTHORITATIVE`。** U0 声明了自己的复权口径，QMT 应当对齐，
而不是把两种口径混合。

### 4.3 `MISMATCH_VALUE` —— 真正的数值分歧

**处置：`REJECTED_U0_AUTHORITATIVE`。** 理由写在代码里：

> "U0 passed identity/provider/coverage/quality gates; a second opinion alone
> does not overturn it"

**"另一个源说得不一样"不构成"U0 错了"的证据。**

### 4.4 `MISSING_IN_U0` —— 唯一会被批准的补丁

U0 该单元格无值时，QMT 填补真实空缺，判 `APPROVED`。这不置换任何已有值。

### 4.5 `MISSING_IN_QMT`

QMT 历史较短属**预期**，不算缺陷，不产生补丁记录。

## 5. 补丁的完整溯源（绝不静默覆盖）

每一条被考虑的修改都产生 `PatchRecord`：

```
symbol / trade_date / field_name
old_provider / new_provider
old_value / new_value
reason / validation
old_hash / new_hash
decision / decided_at
```

**被拒绝的补丁同样留在账本里**（`ledger["records"]`）。只列出已应用变更的审计
无法показ"考虑过但拒绝了什么"。已应用的行会打上
`patch_provenance = "close:u0_verified->qmt_xtdata"` 标记。

## 6. 其他对账项

重复行（U0 侧与 QMT 侧分别统计）、缺失日期、停牌日表示、上市边界、退市边界，
均在 `reconcile()` 输出的 `presence_counts` / `duplicate_rows` / `outcome_counts` 中。

需产出**逐板块与逐供应商**的不一致表——待有数据后生成。

## 7. 未决

- QMT 日线数据：**未取得**
- 逐板块不一致表：**未生成**
- 已批准补丁数：**0**（无数据可比）
