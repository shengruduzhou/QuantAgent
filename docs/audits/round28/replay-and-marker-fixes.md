# Round 28 — 证据重放与审计 framing 修复

日期 / Date: 2026-09-19。修复基线为 main
`124f5b017cbd65d41451f69a5d6b20921c4395f6`。
保持 **RESEARCH / NOT LIVE READY；RL NOT ENABLED**。

## 已复现问题 / Confirmed findings

- [#151](https://github.com/shengruduzhou/QuantAgent/issues/151)：独立 R9 的增量
  finding 由主角色重新复现。满足四岗的 trusted approvals 中，一条评论同时包含
  合法 marker 和 payload 为 `not-json` 的第二个 marker，旧解析器仍返回
  `passed=True`、`malformed_comment_ids=()`。根因是正则只统计符合 object
  形状的块，第二个畸形块被漏掉。
- [#152](https://github.com/shengruduzhou/QuantAgent/issues/152)：独立 R3 的增量
  finding 由主角色重新复现。从 active snapshot 连续调用公开 pure transition
  API，失败证据 `A, B, A` 导致 `degraded/1 → degraded/2 → retired/3`。
  只有两个独立观察，不应满足三次退役条件。已有 `FactorLifecycleLedger.observe`
  的历史去重正确，未发现仓库内其它非测试 direct caller；不扩大影响描述。

## 实现与兼容 / Changes and compatibility

- 审计评论先统计 marker 起始边界，再验证唯一完整块并解码 JSON。
  畸形、空、非 object 或未闭合的额外块不能隐藏；author association 仍先于
  内容解析，未授权公开评论不产生 approval，也不能 veto。
- Snapshot 新增默认空的 `seen_evidence_digests`，规范化为有序去重 tuple，
  保留 pure transition 消费过的证据。`ledger.latest` 和 `replay_lifecycle`
  按 factor/version 从已有 transition 重建历史，旧 JSONL schema/hash 不变。
  Severe semantic violation 仍优先 quarantine；新因子仍最多进入 shadow。
- 历史 snapshot 若只保存 last digest，无法凭空恢复更早历史；继续运行前应从
  ledger 或完整 transition replay 恢复。新 snapshot 的 JSON round-trip 保留历史。
  证据历史空间随独立 digest 数线性增长，不宣称无限历史的常量空间去重。

## 主角色验证 / Implementation validation

使用 synthetic unit fixtures 验证程序行为，没有读取真实因子、冻结 holdout、
训练模型或调用实盘。以下定向命令获得 **57 passed，2 warnings**：

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/governance/test_github_audit_gate.py \
  tests/governance/test_isolated_audit_workflow_contract.py \
  tests/governance/test_code_debug_auditor_role.py \
  tests/governance/test_post_change_ai_quant_audit.py \
  tests/factors/test_lifecycle_state_machine.py \
  tests/factors/test_lifecycle_promotion_guards.py
```

两个 warning 来自已有 factor report 的 label horizon 推断；没有用它们给出收益
或生产验收结论。`git diff --check` 通过。CI 和修改后独立审批以本 PR 的实际
head 记录为准；上述增量 finding 不是最终 approve，也不代表完整 11 岗验收完成。

## 持续保留的限制 / Outstanding evidence

真实 Qlib bundle 单位/字段校准、官方 calendar 完整性、Windows MiniQMT 与
实际浏览器验收仍未完成。预览服务不可达后，沙箱外启动请求被自动审批拒绝，
没有可声称通过的 browser DOM/截图证据。未读取或调参冻结 holdout。

2026-09-18 核对有 76 个远端分支。PR #145/#146/#148/#150 的分支 tip 等于
各自已合并 PR head，head tree 与 squash tree 相同，squash commit 均为 main
祖先。它们是已核对的清理候选；连接器未提供 delete-ref、gh 不可用，**没有删除**。
这一核对不代表其它分支也可删除。

## 一手资料 / Primary references

- [Python re](https://docs.python.org/3/library/re.html)：正则匹配只返回符合模式的
  子串；不能把 object 形状匹配次数当成 framing 数量。
- [Python json](https://docs.python.org/3/library/json.html)：JSON decoder 接受多种
  顶层值，因此解码后仍须显式检查 audit payload 为 object。
