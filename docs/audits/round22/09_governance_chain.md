# Round 22 · R9 代码治理与可靠性 · 治理审计哈希链修复

分支 `agent/round22-governance-chain` · 基线 `5957011` · 2026-08-19

范围限制：只改 `src/quantagent/governance/**` 与 `tests/governance/**`。

| 项 | 结论 |
| --- | --- |
| A-04（哈希链并发分叉，P1） | **已修复**，修复前后均有实测输出 |
| 链完整性主动检测 | **已加**（`verify()` 报孤儿数、`require_intact()` 抛错） |
| DEF-017 回归风险 | **未重新引入**，已用测试钉住 |
| A-05（决策协议零生产调用点） | **复核仍然成立**，未删除 |

---

## 1. 缺陷复现：修复前实测

复现脚本已重建为回归测试
`tests/governance/test_audit_chain_concurrency.py::test_concurrent_appends_leave_one_unbroken_chain`。

固定参数（避免 flaky）：**4 进程 × 40 条 = 160**，`fork` 启动方式，
**每一次 append 之前都过一次 `mp.Barrier(4)`**——上一轮的脚本靠调度运气撞车，
这里把撞车做成确定性的，四个写者必然在同一时刻读到同一条链尾。

用 `git stash push -- src/quantagent/governance/audit.py` 把修复移走、
测试留下，连续四次运行（修复前源码 + 修复后测试）：

```
E AssertionError: lines=160 distinct_sequences=105 duplicate_sequences=35 forked_prev_hash=35 reachable_from_genesis=0 orphaned=160
E AssertionError: lines=160 distinct_sequences=43  duplicate_sequences=40 forked_prev_hash=40 reachable_from_genesis=0 orphaned=160
E AssertionError: lines=160 distinct_sequences=41  duplicate_sequences=40 forked_prev_hash=40 reachable_from_genesis=0 orphaned=160
E AssertionError: lines=160 distinct_sequences=48  duplicate_sequences=41 forked_prev_hash=41 reachable_from_genesis=0 orphaned=160
E AssertionError: lines=160 distinct_sequences=40  duplicate_sequences=40 forked_prev_hash=40 reachable_from_genesis=0 orphaned=160
1 failed, 1 passed
```

比上一轮报告的数字**更糟**：round-21 的脚本还能从创世哈希走到 1 条，
这里 `reachable_from_genesis=0`——因为 barrier 让四个进程在**第一条**记录上就分叉，
`prev_hash == GENESIS_HASH` 的候选有 4 个，任何按链遍历的验证器在第 0 步就没有唯一后继。
**160 条治理记录全部孤儿，写入方全程没有收到任何错误。**

## 2. 修复后实测

```
$ AI_quant_venv/bin/python3 -m pytest tests/governance -q -p no:cacheprovider
53 passed in 0.6s     （连续 5 次运行，5 次全绿）

$ AI_quant_venv/bin/python3 -m pytest tests/governance tests/test_governance_protocol.py -q -p no:cacheprovider
82 passed in 0.67s    （既有的 tests/test_governance_protocol.py 未被本次改动破坏）
```

同一测试修复后断言的量：`lines=160 / distinct_sequences=160 / duplicate_sequences=0 /
forked_prev_hash=0 / reachable_from_genesis=160 / orphaned=0`，
且 `Counter(actor) == {agent0..agent3: 40}`——**没有任何一个写者的分支被静默丢弃**。

### stash 对照汇总

| 测试 | 修复前 | 修复后 | 是否鉴别性 |
| --- | --- | --- | --- |
| `test_concurrent_appends_leave_one_unbroken_chain` | FAIL（数字见上） | PASS | **是** |
| `test_interleaved_short_lived_writers_extend_one_chain` | PASS | PASS | 否（见下） |
| `test_audit_chain_integrity.py`（23 项） | 整模块 ImportError | 23 PASS | **是** |

第二个测试**修复前后都通过**，如实记在这里而不是充数：它是顺序场景
（每条记录由一个新进程写完即 `os._exit`），修复前本来就不会撞车。
它的作用是**约束修法的形状**——把 head 缓存进内存、只加线程锁的"修法"会让它挂——
但它自己不构成 A-04 的证据。按 DEF-030 的教训，鉴别性测试与形状约束测试必须分开标注。

`test_audit_chain_integrity.py` 修复前是整模块 `ImportError`
（`AuditChainCorruption` 不存在），所以刻意与并发复现分成两个文件：
否则 import 失败会盖掉上面那组实测数字。

## 3. 修法说明

`src/quantagent/governance/audit.py`。三个问题分别对应三处改动，**全部复用仓库既有模式**。

### 3.1 并发互斥：复用 `paper/execution_journal.py` 的锁

`AuditLog._exclusive_file_lock()` 是从
`src/quantagent/paper/execution_journal.py:228-256` 逐段搬过来的：
POSIX 走 `fcntl.flock(LOCK_EX)`，Windows 走 `msvcrt.locking`，两者都没有时
**抛错而不是无锁写**（fail-closed）。

锁放在 **sidecar `<log>.lock`** 而不是日志文件本身，理由与 execution_journal 相同：
日志每次只以追加模式打开几微秒，锁一个反复重开的句柄在间隙里毫无保护。

上一轮报告点名的对比事实——本仓另外 4 个 append-only 写者
（`paper/pending_signal.py`、`paper/account_identity.py`、`paper/execution_journal.py`、
`execution/parent_child.py`）**都** import 了 `fcntl`/`msvcrt`，
唯独带哈希链的这个没有——现在这个不对称消失了。

### 3.2 链尾必须在锁内重读

```python
with self._exclusive_file_lock():
    last = self._tail_entry()      # 锁内重读；锁外抓的链尾就是那条过期 head
    ...
```

这是分叉的直接成因：`sequence` 和 `prev_hash` 都由链尾派生，
锁外读到的链尾在拿到锁时可能已经被别的进程推进过。

### 3.3 持久化与 DEF-017

写入改为 `write → flush → os.fsync`（`os.replace` 原子写不适用：
这是 append-only 日志，不是整文件覆盖；`kill_switch.py:150-156` 的 tmp+replace
模式针对的是**整个状态文件重写**，用在这里会把历史整份重写，正好违背 append-only）。

`fsync` 失败时**latch 关闭**（新增 `AuditWriteUnavailable`），语义与
`domain/ledger.py:54-70` 在 DEF-017 之后采用的完全一致，docstring 也照抄了它的推理：

> fsync 抛 EIO 时字节可能已经交给 OS、磁盘满可能写了半行、只读挂载可能一行没写。
> 此时**没有人知道链尾是什么**，在上面继续 append 就是 DEF-017 的形状。

**没有重新引入 DEF-017**：本次的"锁内重读链尾"与 DEF-017 禁止的"重新同步"不是同一件事。

- 重读链尾解决的是**别的进程推进了链**（跨进程新鲜度）；
- DEF-017 禁止的是**本进程在自己的持久化失败之后，把磁盘上来路不明的字节当成可信链尾**。

两者的分界就是 latch：latch 一旦关闭，本实例**不再重读、不再 append**，
直接抛 `AuditWriteUnavailable`。恢复路径与 ledger 相同——新开一个实例
（等价于重启）诚实地重读文件，从**实际存在的内容**继续。
`TestDurableWriteLatch::test_failed_durable_write_latches_the_log_closed`
把这三步都钉住了：原 OSError 上抛 → 同实例再 append 抛 `AuditWriteUnavailable` →
新实例可以继续且 `require_intact()` 通过。

### 3.4 顺带修掉的 O(n²)

A-04 第 3 点：`last_entry()` 遍历全文件，n 次 append = O(n²)，
而 docstring 自己写着"the log grows without bound"。
改为从文件末尾按 8 KiB 块**倒着 seek** 取最后一行
（换行符 0x0A 不会出现在 UTF-8 多字节序列内部，`json.dumps` 也会转义串内换行，
所以按 `b"\n"` 切分总是切在记录边界上）。

实测（4000 条 append，每条约 500 B）：

```
before : 4000 appends 25.481s  (6.370 ms/append)
after  : 4000 appends  2.564s  (0.641 ms/append)     ← 且这一版还多做了 fsync 与加锁
```

`TestTailReading::test_seeked_tail_matches_a_full_scan` 用
`count ∈ {0,1,2,3,250}` 逐一比对 seek 结果与全量扫描结果，
另有一项覆盖非 ASCII payload（中文）+ 末尾空行。

## 4. 链完整性检测的覆盖情况

沿用既有 `verify()`（未新造第二套入口），做两处加强：

1. **`verify()` 现在报数**：新增 `entries_total` / `reachable` / `orphaned`。
   原来只返回 `valid: False`，"159 条孤儿"这个量在返回值里根本不存在。
   （`checked` 保留，既有测试依赖它。）
2. **新增 `require_intact()`**：`verify()` 的抛错版本，
   异常信息直接写明 `"3 of 6 records are reachable from genesis, 3 are orphaned"`。
   A-04 的失败形状不是"损坏没被检测到"，而是**没有人问**；
   把日志当证据用的调用方应当调这个。

`tests/governance/test_audit_chain_integrity.py`（23 项）覆盖：

| 场景 | 断言 |
| --- | --- |
| 完整链 | `valid=True`、`orphaned=0`、head ≠ genesis |
| 空日志 | 完整且 head = genesis（不是"未知"也不是"损坏"） |
| 删首条 | `reachable=0`、`orphaned=4`、抛错 |
| 删中间一条 | `reachable=2`、`orphaned=2`、抛错信息含 `2 of 4` |
| 末行被截半（torn line） | 报 `not a readable audit entry`、`reachable=2` |
| **末尾截断（无锚点）** | **`valid=True`——如实记录：裸链走查看不见** |
| 末尾截断（带 `expected_head`） | 抛错，信息含 `removed from the end` |
| 篡改中间一条 | 报 `does not match`、`reachable=2`、`orphaned=3` |
| 篡改并重算该条自身哈希 | 报 `does not chain to its predecessor`、`orphaned=2` |
| 交换两条顺序 | 抛错 |
| **A-04 的分叉形状** | `entries_total=6 / reachable=3 / orphaned=3`，抛错信息含孤儿数 |
| 链尾被改后再 append | 抛 `AuditChainCorruption`，且**一个字节都不写** |

### 一处如实声明的局限

**从末尾截断的链，裸走查检测不出来**，因为文件里没有任何东西记录"文件应该多长"，
截断后剩下的前缀在内部是完美的。这不是本次修法的疏漏，是哈希链的固有性质。
处理方式是**量出来并写进 docstring 与测试**，而不是让它悄悄躺着：
`require_intact(expected_head=...)` 接受调用方从上一次读取记下的 head 作为外部锚点；
不传锚点时会把被截断的日志报成 intact，测试
`test_tail_truncation_needs_an_external_head_anchor` 用一条带说明的断言把这个行为钉死
（"if this ever starts failing the docstring on require_intact is stale"）。

### `append` 内的检查边界

`append` 在锁内只校验**链尾自身**的哈希（O(1)），不做全链走查——
每次 append 都全链走查会把 O(n²) 原样请回来。
它能拦住"链尾被改 / 被撕裂"（正是它即将挂上去的那一条），
更早的断裂归 `require_intact()` 管。这条边界写在 `_tail_entry()` 的 docstring 里。

## 5. A-05 复核（只读，未做任何改动）

上一轮结论：`quantagent.governance` 的决策协议（`protocol` / `agents` / `envelopes` /
`audit`，33 kB）**零生产调用点**。本轮用原 grep 原样复跑：

```
$ grep -rn "quantagent\.governance" --include=*.py . | grep -v src/quantagent/governance/
scripts/verify_pr_isolated_audit.py:23:from quantagent.governance.github_audit_gate import (
tests/test_governance_protocol.py:15,16,17,25
tests/test_declared_dependencies.py:230
tests/governance/*  （含本轮新增的两个文件）
```

**结论：A-05 仍然成立。** 非测试命中只有一条
`scripts/verify_pr_isolated_audit.py`，它 import 的是 `github_audit_gate`
（进而 `isolated_production_audit`），**不是**决策协议。

需要精确一点的表述：由于 `governance/__init__.py:15` 无条件
`from quantagent.governance import agents, audit, envelopes, isolated_production_audit, protocol`，
上面那条 script 的 import 会让四个模块被**加载**。
所以准确说法是「**import 可达，但调用点为零**」：
`DecisionProtocol` 的构造与 `.decide()` 在 `src/`、`services/`、`scripts/`、`apps/`
里一次都没有出现；`AuditLog(...)` 的实例化同理，只出现在测试中。
（`src/quantagent/cli/governance.py` 名字相近但无关——它 import 的是
`execution.live_model_trust_v2_execution_policy` 与 `training.feature_contract`，
与本包没有关系。）

**未删除，也不建议删除**：AGENTS.md 要求删除前先证明零引用，而"零引用"本身正是
这里要报告的事实；删或不删是主角色的决定。与 R2 的 `RiskGate` /
`paper.RiskEngine` 并列进"待接线清单"。

顺带说明本轮修复与 A-05 的关系：**A-04 至今没有在生产上造成损失，原因就是 A-05**——
没有人在生产里写这份日志。所以本次修复的价值是**接线之前先把地基修好**，
而不是"修掉了一条正在流血的伤口"。这一点不应该在汇报里被夸大。

## 6. 事故记录：`git stash` 在共享仓库的多 worktree 下会串台

做修复前/后对照时用了 `git stash push` 移走修复。**`refs/stash` 是仓库级的、
所有 worktree 共享**，与分支无关。第二次 `git stash pop` popped 出来的
不是我自己的 stash，而是另一个角色 agent（RL）压进去的
`src/quantagent/rl/pit_portfolio_env.py`（106 行新增），
同时我自己的 `audit.py` 修复被丢弃、`git stash list` 变空。

处置：

1. 立刻把误取的 RL 改动存成补丁
   `<scratchpad>/RECOVERED_rl_pit_portfolio_env.patch`（170 行）；
2. 用 `git stash push -m "RETURNED-BY-R9: ..."` **原样归还**到 `refs/stash`，
   让 RL 角色能按常规 `git stash list` / `pop` 找回；
3. 重建自己的 `audit.py` 修复并复跑全部测试（82 passed），随即 commit 落盘。

**给后续轮次的规则：worktree-隔离的 agent 一律禁止使用 `git stash`。**
需要临时移走改动就 `cp` 到 scratchpad 再 `cp` 回来，或者用
`git stash create` + `git stash store`（不进共享栈也要小心），
最稳妥是直接 `cp`。本轮之后的对照实验全部改用文件级备份完成。

## 7. 交付物

| 文件 | 变更 |
| --- | --- |
| `src/quantagent/governance/audit.py` | flock 互斥、锁内重读链尾、fsync + DEF-017 latch、`verify()` 报孤儿数、新增 `require_intact()`、seek 取链尾 |
| `tests/governance/test_audit_chain_concurrency.py` | 新增，A-04 复现转成闸门（2 项） |
| `tests/governance/test_audit_chain_integrity.py` | 新增，链完整性检测覆盖（23 项） |

commit `4003038` on `agent/round22-governance-chain`。
