# CI 失败根因与修复报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **分支**：`agent/qmt-data-closure-ci-recovery`
- **环境**：Ubuntu 20.04.6 / Python 3.12.2（干净 venv）/ CI runner ubuntu-latest + Python 3.12

## 1. 根因（一句话）

`src/quantagent/data/ashare/http.py` 在**模块顶层无条件** `import requests`，
而 `pyproject.toml` 的核心依赖里**没有声明 requests**。

开发机因为其他包间接装了 requests，所以本地全绿；CI 的干净环境在 **pytest 收集阶段**
直接崩溃：

```
ModuleNotFoundError: No module named 'requests'
```

**这不是策略或模型测试失败**，而是依赖声明与真实运行需求不一致。

## 2. 为什么必须改声明，而不是改 CI 命令

任务明确要求不得只在 CI 命令里补装 requests。原因是根本性的：

- 依赖元数据的作用就是描述**代码真正需要什么**；
- 在 CI 里补装只会让 CI 变绿，任何用 `pip install quantagent` 的下游安装仍然会坏；
- 而且下一个未声明的依赖会以同样的方式再次出现。

## 3. 采用的方案

在**核心依赖**中加入有界版本：

```toml
"requests>=2.31,<3",
```

放核心而非可选，理由是 `ashare/http.py` 是所有公开 A 股数据源共用的**带限速与
重试的 HTTP 传输层**，没有它整个数据底座连 import 都做不到——这是无条件生产依赖，
不是可选能力。

**未采用**的两个备选及理由：

| 备选 | 未采用原因 |
| --- | --- |
| 全面迁移到 httpx | 改动面远大于收益；现有传输层已实测调优（per-host 限速、501 视为限流），重写会引入新风险 |
| 改为惰性导入 + 能力门 | 会把一个**真实的硬依赖**伪装成可选能力，掩盖真相 |

## 4. 全树审计（不止修报错的那一个）

只修被报出的符号是不够的。用 AST 解析 `src/` 与 `services/` 下**每个模块的
顶层导入**，与 pyproject 声明集（核心 + 全部 extras）比对：

| 模块 | 声明状态 |
| --- | --- |
| numpy / pandas / polars / pyarrow / scipy / pydantic / pyyaml / typer / tabulate | ✅ 核心已声明 |
| fastapi / uvicorn / httpx | ✅ `[test]` extra 已声明 |
| **requests** | ❌ **未声明 → 本次修复** |
| tickflow / xtquant / MetaTrader5 | 供应商 SDK，非公开 PyPI，已确认为**惰性导入** |

**结论：requests 是唯一未声明的第三方运行时依赖。**

## 5. 回归测试：`tests/test_declared_dependencies.py`

分两层，理由是快慢各有不可替代的作用。

### 5.1 静态审计（每次 CI 都跑，毫秒级）

要求 `src/`、`services/` 中**每一个模块顶层的第三方导入**都在 pyproject 中有声明。

选择"模块顶层"作为边界是刻意的：函数内或 `try/except ImportError` 内的导入是
**可选能力**，代码本就应当优雅降级；而顶层导入是**硬需求**，必须出现在元数据里。

### 5.2 关键验证：这个门真的能抓到原始 bug 吗

一个从不报警的检查器没有信息量。因此实测把 requests 声明**删掉**后重跑：

```
AssertionError: these modules are imported unconditionally by production code
but are not declared in pyproject.toml, so a clean environment will fail during
collection: {'requests': ['src/quantagent/data/ashare/http.py']}
```

**它精确复现了 CI 的失败，并点名了模块与文件。** 恢复声明后测试通过。

另有两条防止门本身失效的测试：

- `test_audit_actually_detects_a_missing_declaration`：喂入一个必然未声明的模块，
  断言被检出；同时断言声明集非空（防止解析失败导致"空集恒过"）；
- `test_vendor_sdks_are_imported_defensively`：防止 `VENDOR_SDKS` 白名单被当成
  逃生舱口——名单内的 SDK 一旦出现在模块顶层即失败。

### 5.3 干净环境实测（可选开关，分钟级）

`QUANTAGENT_CLEAN_ENV_TEST=1` 时真实创建 venv、只装 `.[test]`、导入生产包并跑收集。
这是**唯一能证明"没有从开发环境继承依赖"**的测试，因为只有它不在开发环境里跑。
默认跳过是因为它要装一整套依赖。

## 6. 干净环境实测结果（本地执行 CI 的确切命令）

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[test]"
python -m compileall -q src services scripts
python -m pytest tests/ -q --junitxml=pytest-results.xml
```

| 步骤 | 结果 |
| --- | --- |
| venv 创建 | Python 3.12.2 |
| `pip install -e ".[test]"` | 成功，**装入 requests-2.34.2** |
| `compileall` | exit 0 |
| **收集（此前崩溃的步骤）** | **成功，收集到 1681 个测试** |
| 收集错误数 | **0** |

### 6.1 收集数差异核查（1702 开发机 vs 1681 干净环境）

差额 21 个，逐条比对测试 ID 后确认**全部**是 torch / gymnasium 相关：

- `tests/test_gpu_smoke.py`（`pytest.importorskip("torch")`）
- `tests/test_v7_deep_gpu.py`
- `tests/test_ft_transformer_multi_date_step.py`（`pytest.importorskip("torch")`）
- `tests/rl/*`（`pytest.importorskip("gymnasium")`）

torch 在 `[training]` extra 而非 `[test]`，属**预期的可选依赖跳过**，且经显式
`importorskip` 守卫，**收集错误为 0**——不是被吞掉的失败。

这与任务要求一致：普通 runner 不应强制 GPU 测试。

## 7. CI 结构改动

```yaml
- name: Declared-dependency audit
  run: python -m pytest tests/test_declared_dependencies.py -q
- name: Verify collection in the installed environment
  run: python -m pytest tests/ -q --collect-only > /dev/null
```

放在完整套件**之前**，这样未声明依赖会以它本来的面目被报出，而不是在 1700 个
测试之后变成一次崩溃。

新增 `qmt-contract` job：在 Linux runner 上只跑 QMT 的 mock 与契约测试，并显式
输出 `NOT_RUN_PLATFORM`，**不把未执行的能力报成通过**。

## 8. 未决

- **GitHub Actions 尚未运行**：本报告的绿灯仅指**本地干净环境**。CI 是否通过必须
  以 Actions 的实际结果为准，在其变绿之前不得声称 CI 成功。
