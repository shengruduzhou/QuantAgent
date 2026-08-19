# Round 23 — R9 代码治理：执行删除提案 / Code Governance: Executing the Deletion Proposals

- 日期 / Date: 2026-08-19
- 基线 / Baseline: `main` @ `b313153`
- 分支 / Branch: `agent/round23-deadcode`（独立 worktree）
- 角色 / Role: R9（本轮**可写**：执行 Round 21 §B 已论证的删除与归档）
- 状态 / Status: 进行中（增量落盘）

## 0. 本轮纪律 / Discipline

- **禁用 `git stash`**：`refs/stash` 是仓库级共享的，Round 22 已实测它会弹出其他
  worktree agent 的工作。本轮所有临时保存一律用文件拷贝。
- **不触碰** `src/quantagent/factors/**`、`src/quantagent/data/**`、
  `src/quantagent/rl/**`（主角色与 R11 正在改）。
- 每个删除都附 `AGENTS.md:38` 要求的零引用 grep 命令与其输出。
- **分批删除，每批跑一次全量 pytest**，全绿才提交下一批；最后一次必须是全量绿。
- 删除**不得**改变任何现有测试的断言。

## 0.1 环境说明 / Environment note（影响可复现性）

本 worktree **没有** `AI_quant_venv/`（它只存在于主检出 `/home/shanhefu/QuantAgent/`），
因此本轮所有命令用绝对路径 `/home/shanhefu/QuantAgent/AI_quant_venv/bin/python3`。
`pyproject.toml:93` 的 `pythonpath = ["src", "."]` 以 rootdir 为基准，所以 pytest
导入的是**本 worktree 的 `src/`**，不是主检出的 editable 安装。已实测确认：

```
$ PYTHONPATH=<worktree>/src .../AI_quant_venv/bin/python3 -c "import quantagent; print(quantagent.__file__)"
<worktree>/src/quantagent/__init__.py
```

## 1. 零引用复核 / Zero-reference re-verification

复核脚本（本轮实际使用的那一条命令；`$REL` = `src/quantagent/` 下的模块路径，
`$STEM` = 文件名，`$DOTTED` = 点分模块名，额外附上该模块的**每一个顶层导出符号**）：

```bash
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=AI_quant_venv \
     --exclude-dir=node_modules --exclude-dir=runtime --exclude-dir=rd-agent \
     --exclude-dir=.claude \
     -e "$DOTTED" -e "\b$STEM\b" -e "\b<每个顶层符号>\b" <worktree-root> \
  | grep -v "^src/quantagent/$REL\.py:" \
  | grep -v "^src/quantagent\.egg-info/SOURCES\.txt:"
```

两条排除的理由：
- `src/quantagent.egg-info/SOURCES.txt` 是 setuptools 构建产物，不是引用。
- 命中自身定义行不是引用。

下表中「命中」一栏**不计** `docs/audits/round21/09_debug.md`（那是提出删除提案的
报告本身，把提案文档算成引用会让任何提案永远无法执行），但计入所有其他文档。

### 1.1 逐文件裁决 / Per-file verdict

| # | 文件 | 行数 | 外部命中 | 复核结论 |
|---|---|---|---|---|
| 1 | `src/quantagent/agents/agent_router.py` | 84 | 2（**均为其他模块 docstring 里的散文提及**） | **删除**（并清理那 2 处散文） |
| 2 | `src/quantagent/factors/pipeline_v6.py` | 44 | 0 | **本轮不动**：`factors/**` 是主角色/R11 的领地 |
| 3 | `src/quantagent/themes/policy_universe_builder.py` | 210 | 0 | **删除** |
| 4 | `src/quantagent/training/ablation_runner.py` | 16 | 0 | **删除** |
| 5 | `src/quantagent/quant_math/hmm_regime.py` | 157 | 0 | **删除** |
| 6 | `src/quantagent/strategy/rule_signals.py` | 61 | 0 | **删除** |
| 7 | `src/quantagent/quant_math/hrp.py` | 115 | 0 | **删除** |
| 8 | `src/quantagent/quant_math/realized_vol.py` | 79 | 0 | **删除** |
| 9 | `src/quantagent/fundamental/dupont.py` | 70 | 1（`docs/audits/round21/03_factor.md:74`） | **保留**（复核不通过，见 1.3） |
| 10 | `src/quantagent/strategy/weight_adapter.py` | 142 | 0 | **删除** |
| 11 | `src/quantagent/cli/fuyao_research.py` | 152 | 0 | **删除**（CLI 未注册，见 1.2） |
| 12 | `src/quantagent/training/composite_loss.py` | 110 | 0 | **删除** |
| 13 | `src/quantagent/quant_math/factor_attribution.py` | 110 | 1（`v7/agent_contracts.py:222`） | **保留**（复核确认，见 1.4） |
| 14 | `src/quantagent/portfolio/sector_etf_allocator.py` | 49 | 1（`v7/agent_contracts.py:199`） | **保留**（复核确认，见 1.4） |
| 15 | `src/quantagent/training/oom_safe_trainer.py` | 84 | **0（Round 21 说 5，本轮更正）** | **保留**（遵主角色指令，但 Round 21 的保留理由已被推翻，见 1.5） |

**本轮实际删除 #1、#3、#4、#5、#6、#7、#8、#10、#11、#12 共 10 个文件。**

零引用原始输出（每条都是「除自身与构建产物外无命中」）：

```
$ verify.sh themes/policy_universe_builder PolicyUniverseConfig build_policy_universe build_for_today
--- grep (imports / CLI / tests / README / AGENTS / docs), self excluded ---
--- END (empty above == zero external references) ---

$ verify.sh training/ablation_runner AblationResult summarize_ablations
--- grep ... ---
--- END (empty above == zero external references) ---

$ verify.sh quant_math/hmm_regime HMMConfig HMMState fit_gaussian_hmm \
      posterior_state_probabilities label_states_to_regimes hmm_regime_alpha_multiplier
--- grep ... ---
--- END (empty above == zero external references) ---

$ verify.sh strategy/rule_signals add_short_horizon_rule_signals
--- grep ... ---
--- END (empty above == zero external references) ---

$ verify.sh quant_math/hrp correlation_distance quasi_diagonalization hrp_weights herc_weights
--- grep ... ---
--- END (empty above == zero external references) ---

$ verify.sh quant_math/realized_vol parkinson garman_klass rogers_satchell yang_zhang \
      add_realized_vol_features
--- grep ... ---
--- END (empty above == zero external references) ---

$ verify.sh strategy/weight_adapter horizon_alpha short_signal_to_weight long_score_to_weight \
      combine_short_long_weights combine_sleeve_target_weights apply_lot_liquidity_constraints
--- grep ... ---
--- END (empty above == zero external references) ---

$ verify.sh cli/fuyao_research run_fuyao_research_backtest
--- grep ... ---
--- END (empty above == zero external references) ---

$ verify.sh training/composite_loss CompositeLossWeights v4_composite_loss
--- grep ... ---
--- END (empty above == zero external references) ---
```

（上面每条的完整输出里唯一被 `grep -v` 之外保留的行都是
`docs/audits/round21/09_debug.md` 的提案表格行本身，已按 §1 说明排除。）

### 1.2 `cli/fuyao_research.py`：CLI 未注册的复核（重跑，非引用 Round 21 快照）

```
$ PYTHONPATH=<worktree>/src .../python3 -c "
from quantagent.cli import app
names=[c.name for c in app.registered_commands]
print('registered commands:', len(names))
print('fuyao commands:', [n for n in names if 'fuyao' in str(n)])
print('run-fuyao-research-backtest registered:', 'run-fuyao-research-backtest' in names)"
registered commands: 109
fuyao commands: ['fetch-fuyao-daily', 'fetch-fuyao-capability', 'fetch-fuyao-market-dump', 'audit-fuyao-coverage', 'sync-fuyao-all']
run-fuyao-research-backtest registered: False
```

`quantagent/cli/__init__.py` 从未 import 该模块 ⇒ 那 152 行 `typer.Option` 实现
在任何入口都跑不到。**删除**。

### 1.3 `fundamental/dupont.py`：复核**不通过**，保留

```
$ verify.sh fundamental/dupont WACCProvider DuPontResult dupont_decomposition
docs/audits/round21/03_factor.md:74:| **基本面 / Fundamental** | ... | `fundamental/{dupont,peg,statements,market_valuation}.py` | 存在，但走 PIT join 而非 registry |
```

`docs/audits/round21/03_factor.md` 是**版本受控的 docs**，`AGENTS.md:38` 明文
「被 docs 引用即不可删」。本轮指令也要求先与 R3 对齐。⇒ **保留，记为待定**。
若将来要删，前置条件是 R3 先确认该文件不在基本面特征来源清单里并改掉那一行。

### 1.4 `quant_math/factor_attribution.py` / `portfolio/sector_etf_allocator.py`：保留（复核确认 Round 21 结论）

```
$ verify.sh quant_math/factor_attribution
src/quantagent/v7/agent_contracts.py:222:        existing_extension_points=("src/quantagent/backtest/engine.py", "src/quantagent/quant_math/factor_attribution.py"),

$ verify.sh portfolio/sector_etf_allocator
src/quantagent/v7/agent_contracts.py:199:            "src/quantagent/portfolio/sector_etf_allocator.py",
```

两处都不是 import，而是 agent 契约里的**路径字符串**。删除文件会让契约声明的
扩展点指向不存在的路径 —— 这正是本仓反复出现的「文档描述的能力实际不存在」
故障模式。**保留**。

### 1.5 ⚠️ `training/oom_safe_trainer.py`：Round 21 的保留理由**已被本轮推翻**，但仍按指令保留

Round 21 记它有 5 处命中，全在 `reports/code_audit/{README,module_map,
project_structure,training_backtest_flow}.md`，据此判「被 docs 引用 ⇒ 不得删」。
本轮复核发现 **`reports/` 根本不在版本控制里**：

```
$ git ls-files reports/ | head
(空)
$ grep -n "reports" .gitignore
30:reports/
31:!/reports/
32:/reports/*
33:!/reports/v6/
34:/reports/v6/*
35:!/reports/v6/v6_readiness_report.md
36:!/reports/v6/v6_readiness_report.json
$ ls .claude/worktrees/agent-a711464a5f23b5298/reports/
ls: cannot access '.../reports/': No such file or directory      # worktree 检出里压根没有这个目录
```

那四份 `.md` 是**主检出本地的未跟踪生成物**，不是仓库 docs。所以
`oom_safe_trainer.py` 在版本控制口径下是**真正的零引用**（`verify.sh` 输出为空）。

**本轮仍不删**，因为主角色的指令明文「不得删」。但请主角色注意：
Round 21 给出的保留依据（「被 docs 引用」）**不成立**，它成立的前提是把
gitignore 掉的本地产物当成仓库文档。真正的问题不变且更严重：
那四份代码地图把 `oom_safe_trainer` 写成「GPU 训练入口」的组成部分，
而它**零引用** ⇒ **文档描述了一个没有接线的能力**。请在下一轮裁定：
接线、或改文档、或删除。

### 1.6 派生候选：`agents/agent_reliability.py`（**因删除 agent_router 而变成零引用**）

`agent_router.py` 是 `AgentReliability` 的**唯一**消费者：

```
$ verify.sh agents/agent_reliability AgentReliability
src/quantagent/agents/agent_router.py:9:from quantagent.agents.agent_reliability import AgentReliability
src/quantagent/agents/agent_router.py:28:        reliability: AgentReliability | None = None,
src/quantagent/agents/agent_router.py:33:        self.reliability = reliability or AgentReliability()
```

只删 `agent_router.py` 会**当场制造**一个新的零引用模块 —— 这正是用户所说的
「屎山」的生成机制。因此本轮把它作为**派生候选**一并删除（两个文件构成一个
2 节点的死子图，不是单文件）。

**同族但保留**：`agents/views_schema.py` 里的 `AgentView` 与 `write_audit_jsonl`
在删除 `agent_router.py` 后也变成零引用，但同文件的 `EvidenceRecord` 仍被
`ashare_specialists.py:9` 与 `flow_agent.py:8` 使用，**文件必须保留**。
对活模块做符号级手术超出本轮范围，**记为 Round 24 待办**（证据已在此）。

另：全仓唯一的 Black-Litterman 相关模块是 `quant_math/signal_fusion.py`，它
**不消费** `AgentView`。所以 `AgentRouter`（evidence → BL views）不是"尚未接线的
设计缝"，而是一条真正的死路。

### 1.7 删除的连带修正（不改任何测试断言）

`src/quantagent/agents/ashare_specialists.py:145` 的 docstring 原文
「Convert specialist signals into the AgentRouter evidence contract.」在删除后会
指向一个不存在的类。已改为指向真正的契约（`views_schema.EvidenceRecord`），
并注明 Round 23 的删除。**没有改动任何测试断言，也没有改动任何可执行语句。**

删除后全仓悬挂引用复扫（唯一命中是上面那条新写的说明文字本身）：

```
$ grep -rIn "AgentRouter|agent_router|agent_reliability|AgentReliability|policy_universe_builder|ablation_runner|summarize_ablations|AblationResult|PolicyUniverseConfig|build_policy_universe" \
      --include=*.py --include=*.md --include=*.json --include=*.toml --include=*.cfg . \
  | grep -v "docs/audits/round2[13]/"
src/quantagent/agents/ashare_specialists.py:148:    former ``AgentRouter`` consumer was removed as proven-zero-reference dead
```

## 2. 分批删除与全量 pytest / Batched deletion & full-suite pytest

命令：`/home/shanhefu/QuantAgent/AI_quant_venv/bin/python3 -m pytest tests/ -q -p no:cacheprovider`

| 批 | 内容 | passed | skipped | 耗时 |
|---|---|---|---|---|
| B-0 基线 | `main@b313153`，未改动 | **3507** | 43 | 438.06s |
| B-1 删除 | `agents/agent_router.py`、`agents/agent_reliability.py`、`themes/policy_universe_builder.py`、`training/ablation_runner.py`（−374 行） | **3507** | 43 | 590.21s |
