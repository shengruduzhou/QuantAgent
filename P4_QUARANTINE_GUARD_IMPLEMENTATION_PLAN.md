# P4_QUARANTINE_GUARD_IMPLEMENTATION_PLAN — 隔离守卫实施计划（Phase 2.5，未实施）

> 目标：让被烧 holdout（2025-09-01→2026-05-18）在**代码层面**不可被随手评测。
> 状态：**计划完毕、等待放行后实施**。预计 diff：~90 行新增（含测试），改动现有文件 ~15 行。

## 1. 修改/新增文件清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `configs/quarantined_windows.json` | 新增 | 机器可读隔离窗登记表（见 §2） |
| `src/quantagent/backtest/quarantine.py` | 新增（~40 行） | 纯函数守卫：`load_quarantine()`, `check_window(start, end) -> Violation|None`, `log_access(reason, argv, window)`；无重依赖，可被任何脚本 import |
| `scripts/baseline_protocol.py` | 修改（~12 行） | 在 **`evaluate()` 函数体开头**（不是只在 argparse main）插入守卫 + `--allow-quarantined` 参数透传；panel 的 `end+10d` 缓冲在未放行时**钳制到隔离窗前一交易日** |
| `tests/test_quarantine_guard.py` | 新增（~35 行） | 纯函数单测（无数据依赖）+ 消息文本断言 |

守卫放在 `evaluate()` 内是关键：`ensemble_weight_search.py`、`factor_combo_search.py` 等都是 `import baseline_protocol as bp; bp.evaluate(...)` 直连调用，只挂 argparse 层会被完全绕过。

## 2. 隔离窗定义（`configs/quarantined_windows.json`）

```json
{
  "windows": [
    {
      "start": "2025-09-01",
      "end": "2026-05-18",
      "reason": "burned final holdout — >=35 direct evals + 5 selection decisions",
      "evidence": "HOLDOUT_CONTAMINATION_AUDIT.md / HOLDOUT_ARTIFACT_CENSUS.csv",
      "declared": "2026-07-03"
    }
  ],
  "log_path": "runtime/state/holdout_access_log.jsonl"
}
```

- `end=2026-05-18` = silver panel 当前最大交易日（审计实测），亦即"已被搜索污染的最后日期"。
- **FRESH 窗（2026-05-19+）不进此表**：它由 `EVALUATION_PROTOCOL_V2.md` §2 的预注册制管理（首读 ≥120 交易日），本守卫只管"禁止旧窗再评测"。若担心误用，后续可加第二条 window `type=preregistered_only`（Phase 6 再议，本次不做，保持 diff 最小）。

## 3. 默认行为（fail-closed）

评测窗 `[start, end]`（`end=None` 视为 +∞）与任一隔离窗**相交** ⇒ 在读任何数据之前：

```
QUARANTINE VIOLATION: requested window 2025-09-01..2026-05-15 intersects
quarantined holdout 2025-09-01..2026-05-18
  reason  : burned final holdout — >=35 direct evals + 5 selection decisions
  evidence: HOLDOUT_CONTAMINATION_AUDIT.md
This window must not be used for evaluation or selection.
To proceed for diagnostics only, pass:  --allow-quarantined "<justification>"
(access will be logged to runtime/state/holdout_access_log.jsonl)
```

退出码 **3**（区别于普通错误 1/2，方便脚本捕获）。

## 4. 覆写行为

- `--allow-quarantined "<非空理由>"`：放行一次；写 JSONL：`{ts, git_hash, argv, requested_window, quarantine_hit, reason}` 到 `runtime/state/holdout_access_log.jsonl`（append-only）。
- 空字符串理由 = 拒绝（必须写人话）。
- 库调用路径（`bp.evaluate(...)` 直连）：新增关键字参数 `allow_quarantined: str | None = None`，语义同 CLI。
- 放行后产出的任何 json/metrics 会被 stamp `"quarantine_override": {"reason": ...}` 字段，下游一眼可辨。

## 5. panel 缓冲钳制（防隐性泄漏）

现状：`evaluate()` 读 panel 到 `end+10d`，末尾 delay-1 成交可落入 2025-09-01+（Phase 2.5 重放时实测确认此机制）。
patch：未放行时 `panel_end = min(end+10d, quarantine_start − 1d)`；放行时维持原状。对 `end ≤ 2025-08-31` 的正常评测，影响 = 末尾 1 个执行日被截（Phase 2.5 已测：27 候选 CAGR 平均偏移 −0.5pp、秩一致性 0.992）——在文档与 CHANGELOG 注明。

## 6. 测试与冒烟（实施时按序执行）

1. **单测**（无数据）：`check_window` 的相交/包含/边界（start=end=2025-09-01）/end=None/多窗；理由为空拒绝；消息文本含 "QUARANTINE VIOLATION" 与 evidence 路径。
2. **导入检查**：`python -c "import baseline_protocol"` 与 `python -m compileall scripts src`。
3. **期望失败冒烟**：`baseline_protocol.py --predictions <既有小 preds> --start 2025-09-01 --end 2025-09-30` ⇒ 退出码 3，零数据读取（守卫先于 read_parquet）。
4. **期望通过冒烟**：同命令 + `--start 2025-06-02 --end 2025-08-31` ⇒ 正常运行且 panel 末日 ≤ 2025-08-29（钳制生效）。
5. **覆写冒烟**：`--allow-quarantined "audit replay"` ⇒ 运行 + log 行出现 + 输出 json 带 override 字段。
6. **间接调用冒烟**：`python -c` 直接调 `bp.evaluate(..., start='2025-09-01')` ⇒ raise（证明 import 路径也被守住）。

## 7. 防未来脚本误用

- 守卫在 `evaluate()` 内 ⇒ 现有 search 脚本（ensemble_weight_search / factor_combo_search / regime_strategy_search / topk 类）**自动继承**，无需逐个改。
- 残余绕过面（记录在案，Phase 3 处理）：直接调 `run_strict_backtest_v8` / `simulate_ashare_target_weights` 的 ~30 个 stage/研究脚本。计划：Phase 3 依赖图确认后，在 `run_strict_backtest_v8` 加**软警告**（stderr 一行 + 不阻断，避免破坏既有研究脚本），并把 P7（搜索脚本 `--test-start` 默认值改必填）一并落地。
- `AGENTS.md` 增补一行硬规则指向 `configs/quarantined_windows.json`（Phase 7 文档统一时做）。

## 8. 不做的事

- 不改 strict 引擎语义、不动 PIT/schema 锁、不删除任何历史 artifact、不追溯改写已有 metrics 文件。
