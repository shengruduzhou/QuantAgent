# CI 运行时夹具依赖修复报告

- **生成时间**：2026-07-29（UTC）
- **源提交**：`17ada19996fc5f1fceae06bac60ff5ab05d4a922`
- **分支**：`agent/local-paper-full-universe-workstation`
- **基线 main**：`619f657430e4e6dbafeb9562ad407113d5761118`（PR #28 已合并，PR #28 head = `75e9d53`）
- **环境**：Ubuntu 20.04.6 / Python 3.12.2（干净 venv）

## 1. 根因

`tests/test_h032c_pit_closure.py` 调用 `load_master()`，该函数无条件读取：

```
runtime/reports/h028/track_a/historical_security_master.parquet
```

此文件被 `.gitignore` 屏蔽，**只存在于研究主机**，CI 的干净检出中不存在。

## 2. 实测复现（先复现再修）

把该产物临时移走后运行：

```
FAILED tests/test_h032c_pit_closure.py::test_supplemental_additions_actually_reach_the_backfill_master
FAILED tests/test_h032c_pit_closure.py::test_supplemental_union_dedupes_and_frozen_master_wins
2 failed, 6 passed
```

**实际是 2 个测试失败，不是 1 个。** 任务描述只提到后者；前者也失败，因为它的
skip 守卫只检查了 supplemental 文件，**没有检查 master 文件**。

## 3. 采用的修法：重构函数接受显式输入

任务给出四个候选，本次选择**第四个**（重构为接受显式输入），理由是被测的核心
是"并集语义"，而语义本身与磁盘上的具体产物无关。

- 新增纯函数 `union_master(master, supplemental)`——只对 DataFrame 运算；
- `load_master(master_path=None, supplemental_path=None)`——路径可注入，默认仍为
  生产路径，生产调用方无需改动。

**未采用**的做法（任务明令禁止，本次均未使用）：
CI 中跳过、标记 xfail、提交大体积 runtime Parquet、CI 下载生产数据、放宽断言、
针对 GitHub Actions 特判。

## 4. 替换后的测试覆盖（明显强于原版）

原测试只有一条断言（`duplicated().sum() == 0`）。新测试类 `TestSupplementalUnion`
覆盖任务要求的全部语义：

| 语义 | 测试 |
| --- | --- |
| 新增标的被并入 | `test_new_symbols_are_added` |
| **重复证券行** | `test_no_duplicate_security_rows` |
| supplemental 内部重复 | `test_supplemental_internal_duplicates_are_collapsed` |
| **冲突的主表身份** | `test_frozen_master_wins_on_conflicting_identity` |
| **冻结主表优先级** | 同上 |
| **非预期覆盖** | `test_listing_and_delisting_metadata_are_not_overwritten` |
| **缺失上市/退市元数据** | `test_missing_listing_metadata_in_supplemental_is_preserved_as_missing` |
| schema 不得被拓宽 | `test_supplemental_cannot_widen_the_schema` |
| 无主键文件被忽略 | `test_supplemental_without_symbol_column_is_ignored` |
| 路径可注入（本次修复本身） | `test_load_master_reads_injected_paths` |

## 5. 关键验证：用变异测试证明这些断言真的能抓到问题

一个不会失败的测试没有信息量。因此对 `union_master` 逐条植入缺陷：

| 植入的缺陷 | 触发的失败 |
| --- | --- |
| 去掉"冻结主表优先"的过滤 | `test_no_duplicate_security_rows`、`test_frozen_master_wins_on_conflicting_identity`、`test_listing_and_delisting_metadata_are_not_overwritten` |
| 去掉 supplemental 内部去重 | `test_supplemental_internal_duplicates_are_collapsed` |
| 允许 schema 拓宽 | `test_supplemental_cannot_widen_the_schema` |

**过程中发现我自己写的测试有缺陷**：第一次变异只触发了重复行那一条，因为两条
优先级断言用了 `.iloc[0]`——即使追加了重复行，第一行仍是冻结主表的行，断言照样
通过。已改为**先断言只有一行**，再断言字段。修正后三条断言全部触发。

## 6. 顺带修掉的一个潜在缺陷

原实现用 `except Exception: pass` 吞掉 supplemental 读取异常，把"文件损坏"变成
"没有增补"——一个**静默变小的宇宙且无任何信号**。现改为打印 WARNING 并有测试
`test_corrupt_supplemental_is_reported_not_swallowed` 覆盖。

## 7. 结果

| 场景 | 结果 |
| --- | --- |
| **无 runtime 产物（模拟 CI）** | **18 passed, 1 skipped**（跳过的是本就依赖真实宇宙的那条，守卫已补齐 master 检查） |
| 有 runtime 产物（研究主机） | **19 passed** |
| 真实 `load_master()` 回归 | 5,890 行，0 重复——无行为回归 |

## 8. 干净环境 CI 命令实测

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[test]"
python -m compileall -q src services scripts   # exit 0
python -m pytest tests/ -q --junitxml=pytest-results.xml
```

前端：

```bash
npm run typecheck   # exit 0
npm test -- --run   # 14 files / 36 tests passed
npm run build:vite  # ✓ built in 5.11s
```

## 9. 未决

**GitHub Actions 尚未运行。** 按任务要求，在新的 Actions run 对最新 PR 提交变绿
之前，**不得声称 CI 成功**。本报告只证明本地干净环境通过。
