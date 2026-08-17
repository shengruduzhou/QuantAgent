# Round 21 — R5 测试专家实测报告 / Measured System Health

角色 / Role: R5 测试专家（只读审计，不改 `src/`、`apps/`，不 commit）
日期 / Date: 2026-08-18
基线 / Baseline commit: `b56ae57`（工作区 clean）
运行环境 / Runtime: `AI_quant_venv/bin/python3` = CPython 3.12.2 (conda-forge)，pytest 9.0.3，node v22.6.0 / npm 10.8.2
写入纪律 / Discipline: 本文件**增量写入**；证据缺失记 `unknown`，永不记 `pass`。

---

## 0. 环境事实 / Environment facts

| 项 | 实测值 |
|---|---|
| Python | 3.12.2 (conda-forge, main, Feb 16 2024) |
| pytest | 9.0.3 |
| `pytest-timeout` | **未安装**（`ModuleNotFoundError: No module named 'pytest_timeout'`）⇒ 全量跑**去掉** `--timeout`；无法对单测做超时熔断 |
| pytest 配置 | `pyproject.toml [tool.pytest.ini_options]`：`pythonpath=["src","."]`，`testpaths=["tests"]` |
| 测试文件数 | `find tests -name "test_*.py" \| wc -l` = **359** |
| node / npm | v22.6.0 / 10.8.2 |

> 注：`pytest-timeout` 缺失本身是一条**测试基础设施缺口**——任务书要求 `--timeout=600`，
> 但该参数在本环境下直接让 pytest 以 usage error 退出（首次尝试即复现）。
> 复现：`AI_quant_venv/bin/python3 -m pytest tests/ -q --timeout=600 -p no:cacheprovider`
> → `error: unrecognized arguments: --timeout=600`。

---

## 1. 后端全量 pytest / Full backend suite

命令 / Command:

```bash
AI_quant_venv/bin/python3 -m pytest tests/ -q -rf -p no:cacheprovider --durations=25
```

状态：**RUNNING**（结果见下节，落盘后回填）

---

## 2. 导入健康 / Import health

### 2.1 `compileall`

```bash
AI_quant_venv/bin/python3 -m compileall -q src
```

**退出码 0，零输出** ⇒ `src/` 下 539 个 `.py` 全部字节码编译通过。

### 2.2 逐模块 import 冒烟 / Per-module import smoke

脚本：对 `src/quantagent/**` 与 `services/**` 的**每一个**模块做 `importlib.import_module`。

```
IMPORT_SMOKE modules=597 ok=595 failed=2
```

**失败 2 项（同一根因）：**

| 模块 | 错误 |
|---|---|
| `quantagent.clean_room` | `ModuleNotFoundError: No module named 'quantagent.clean_room.engine'` |
| `quantagent.clean_room.risk` | 同上（父包 `__init__` 触发） |

#### DEF-R5-01（P0，实测）：`clean_room` 整包不可导入，且零测试零调用

- 位置：`src/quantagent/clean_room/__init__.py:27`
  ```python
  from quantagent.clean_room.engine import CleanRoomResult, run_backtest
  ```
- 事实：`src/quantagent/clean_room/` 目录下只有 `__init__.py` / `dataset.py` / `risk.py`，
  **`engine.py` 不存在**。
- 影响面：包内**三个**模块全部不可导入（`dataset` 在干净进程里同样失败——
  上面 smoke 只报 2 条是因为 `dataset` 在父包失败前已进 `sys.modules`，
  单独验证：
  ```bash
  AI_quant_venv/bin/python3 -c "import sys;sys.path.insert(0,'src');import quantagent.clean_room.dataset"
  # ModuleNotFoundError: No module named 'quantagent.clean_room.engine'
  ```
  ）
- **为什么没被发现**：全仓 `grep -rn clean_room` 命中**仅 3 行，全在它自己的 `__init__.py` 里**。
  没有任何测试、CLI、脚本、文档引用它 ⇒ pytest 全量跑不会碰到它，
  `compileall` 也发现不了（语法没错，是缺文件）。
- 严重性判断：`__init__.py` 的 docstring 声明这是**为逃离已测缺陷而建的 clean-room 回测**
  （明确点名 `backtest/engine.py` 的 interior-bar NAV 缺陷 "STILL OPEN"，
  Sharpe +1.4768 vs 诚实值 −7.10）。也就是说**本轮 charter 里最关键的一条遗留项，
  它的替代实现处于"提交了 2/3 个文件、整包 import 即崩、且无人引用"的状态**。
- 最小复现：`AI_quant_venv/bin/python3 -c "import quantagent.clean_room"`（PYTHONPATH=src）
- git 历史：`git log --oneline -- src/quantagent/clean_room/` → 仅 `581d9c1 fix dataset`。

#### 观察（非缺陷）：`src/quantagent/cli/v7_train.py` 带 UTF-8 BOM

AST 静态扫描时报 `invalid non-printable character U+FEFF (line 1)`。
CPython 的源码 tokenizer 会剥离 BOM，故**运行时可正常 import**（已在 smoke 中确认 ok）。
但任何用 `ast.parse(open(...).read())` 的自研审计/lint 工具会在该文件上静默失效。
记为 `观察`，不进 P0 队列。

---

## 3. 依赖声明审计 / Declared-dependency audit

方法：AST 扫描 `src/` 全部 539 个 `.py`，提取**模块顶层**（非函数内、非 try 内）import 的
第三方顶层包名，对照 `pyproject.toml` 的 `dependencies` + `optional-dependencies`。

### 3.1 P0：未声明却在模块级 import

**（无）** —— 历史上的 `requests` 缺声明（导致 clean-env CI 收集期崩溃）已修复并在
`pyproject.toml:17-22` 留下了注释说明；`psutil` 同样已提升为 core（`:24-26`）。

### 3.2 P1：未声明但在惰性/try 内 import（degrade-gracefully 路径）

以下包**不在 `pyproject.toml` 任何 extra 里**，只在函数体或 `try:` 内导入。
按 AGENTS.md「Optional dependencies must degrade gracefully」这是允许的形态，
但意味着**这些能力在任何 `pip install quantagent[all]` 的环境里都不可用**，
且 `[all]` extra 名不副实：

| 包 | 文件数 | 代表位置 |
|---|---|---|
| `lightgbm` | 6 | `src/quantagent/research/intraday_dot_factor_combo.py` |
| `xgboost` | 4 | 同上 |
| `xtquant` | 3 | `src/quantagent/data/providers/qmt_gateway.py` |
| `gymnasium` | 2 | `src/quantagent/rl/pit_portfolio_env.py` |
| `tickflow` | 2 | `src/quantagent/data/ashare/sources.py` |
| `stable_baselines3` | 1 | `src/quantagent/rl/train_ppo.py` |
| `optuna` | 1 | `src/quantagent/optimization/optuna_search.py` |
| `deap` | 1 | `src/quantagent/optimization/factor_evolution.py` |
| `catboost` / `joblib` | 1 | `src/quantagent/training/do_t_models.py` |
| `matplotlib` | 1 | `src/quantagent/training/diagnostics.py` |
| `dotenv` | 1 | `src/quantagent/agents/llm_skill_client.py` |
| `MetaTrader5` | 1 | `src/quantagent/data/providers/mt5_capability.py` |

其中 **`lightgbm` 值得单独记**：`AGENTS.md:68` 把
`train-alpha-v7 --model lightgbm` 列为「real LightGBM, fail-loud if missing」的
**真实数据命令**，但 lightgbm 不在任何 extra 中 ⇒ 文档承诺的命令在标准安装下必然 fail-loud。
（fail-loud 本身正确，问题是**声明与文档不一致**。）

`MetaTrader5` / `xtquant` 属 Windows-only，不可声明，符合既有裁决（见 memory：QMT 仅 Windows）。

### 3.3 `tests/test_declared_dependencies.py`

见第 1 节全量结果（该文件包含在 `tests/` 全量跑内）。

---
