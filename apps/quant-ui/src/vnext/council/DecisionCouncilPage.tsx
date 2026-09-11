import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  CheckCircle,
  Gavel,
  Prohibit,
  Question,
  ShieldCheck,
  Warning,
} from "@phosphor-icons/react";
import { apiPost } from "../../api/client";
import type {
  CouncilFinding,
  CouncilOverride,
  CouncilReview,
  CouncilRoster,
  FusionRunSummary,
} from "../../api/types";
import { useApi } from "../../hooks/useApi";
import {
  ActionableState,
  TruthNotice,
  WorkbenchHeader,
  WorkbenchMetricStrip,
  WorkbenchPanel,
} from "../workbench/InstitutionalWorkbench";

const VERDICT_META: Record<
  CouncilFinding["verdict"],
  { label: string; tone: "success" | "warning" | "danger" | "control"; icon: typeof CheckCircle }
> = {
  pass: { label: "通过", tone: "success", icon: CheckCircle },
  warn: { label: "保留意见", tone: "warning", icon: Warning },
  blocked: { label: "否决", tone: "danger", icon: Prohibit },
  unknown: { label: "证据不足", tone: "control", icon: Question },
};

const DECISION_META: Record<string, { label: string; tone: "success" | "warning" | "danger" }> = {
  PROMOTABLE: { label: "可进入人工 Gate", tone: "success" },
  PROMOTABLE_WITH_WARNINGS: { label: "有保留意见", tone: "warning" },
  INSUFFICIENT_EVIDENCE: { label: "证据不足", tone: "warning" },
  BLOCKED: { label: "被否决", tone: "danger" },
};

const COMPANY_PHASES = [
  { label: "01 数据准入", roleIds: ["data_acquisition", "data_quality", "microstructure"] },
  { label: "02 研究验证", roleIds: ["factor_integrity", "model_validation", "fusion_search"] },
  { label: "03 组合落地", roleIds: ["portfolio_risk", "execution_realism"] },
  { label: "04 独立裁决", roleIds: ["challenger", "compliance", "governance"] },
] as const;

function effectiveVerdict(finding?: CouncilFinding): CouncilFinding["verdict"] {
  if (!finding) return "unknown";
  if (finding.verdict === "unknown" && finding.override?.verdict !== "blocked") return "unknown";
  return finding.override?.verdict ?? finding.verdict;
}

export function DecisionCouncilPage(): JSX.Element {
  const [selectedRunId, setSelectedRunId] = useState("");
  const [openOverrideRole, setOpenOverrideRole] = useState("");
  const [overrideVerdict, setOverrideVerdict] = useState<"pass" | "warn" | "blocked">("warn");
  const [overrideReason, setOverrideReason] = useState("");
  const [overrideAuthor, setOverrideAuthor] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const roster = useApi<CouncilRoster>(["council-roster"], "/council/roster");
  const runs = useApi<FusionRunSummary[]>(["fusion-runs"], "/fusion/runs");
  const runList = useMemo(
    () => (Array.isArray(runs.data?.data) ? runs.data.data : []),
    [runs.data],
  );
  const effectiveRunId = selectedRunId || runList[0]?.id || "";
  const review = useApi<CouncilReview>(
    ["council-review", effectiveRunId],
    effectiveRunId ? `/council/review/fusion/${effectiveRunId}` : null,
  );

  const data = review.data?.data;
  useEffect(() => {
    setOpenOverrideRole("");
    setOverrideReason("");
    setError("");
  }, [effectiveRunId, data?.subject.contentHash, data?.subject.candidateId, data?.policyFingerprint]);
  const findings = data?.findings ?? [];
  const roles = roster.data?.data.roles ?? [];
  const thresholds = roster.data?.data.thresholds;
  const decision = data?.decision;
  const expectedRoleIds = COMPANY_PHASES.flatMap((phase) => [...phase.roleIds]);
  const contractIssue = useMemo(() => {
    const rosterData = roster.data?.data;
    if (!rosterData || !data) return "";
    const roleIds = rosterData.roles.map((item) => item.id);
    const findingIds = data.findings.map((item) => item.roleId);
    if (rosterData.protocolVersion !== 2 || data.protocolVersion !== 2) {
      return "Council protocolVersion 不是 v2；当前裁决不可用于晋级。";
    }
    if (rosterData.policyFingerprint !== data.policyFingerprint) {
      return "Roster 与 review 的 policy fingerprint 不一致；请刷新后重试。";
    }
    if (
      roleIds.length !== expectedRoleIds.length
      || new Set(roleIds).size !== roleIds.length
      || roleIds.some((item, index) => item !== expectedRoleIds[index])
    ) {
      return "Council v2 必须按固定顺序返回完整 11 岗；当前 roster 缺失、重复或漂移。";
    }
    if (
      findingIds.length !== expectedRoleIds.length
      || new Set(findingIds).size !== findingIds.length
      || findingIds.some((item, index) => item !== expectedRoleIds[index])
    ) {
      return "Review 未返回完整 11 岗裁决；缺失不能被静默过滤。";
    }
    return "";
  }, [data, roster.data]);
  const decisionMeta = decision && !contractIssue ? DECISION_META[decision.state] : undefined;

  const counts = useMemo(() => {
    const effective = findings.map(effectiveVerdict);
    return {
      pass: effective.filter((item) => item === "pass").length,
      warn: effective.filter((item) => item === "warn").length,
      blocked: effective.filter((item) => item === "blocked").length,
      unknown: effective.filter((item) => item === "unknown").length,
    };
  }, [findings]);

  const submitOverride = async (finding: CouncilFinding): Promise<void> => {
    if (!data || contractIssue) return;
    if (finding.verdict === "unknown" && overrideVerdict !== "blocked") {
      setError("证据不足不能改判为通过或保留意见；请先补充证据并重新审议。");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await apiPost<CouncilOverride>("/council/overrides", {
        subjectType: "fusion_run",
        subjectId: effectiveRunId,
        subjectContentHash: data.subject.contentHash,
        candidateId: data.subject.candidateId,
        roleId: finding.roleId,
        findingHash: finding.findingHash,
        originalVerdict: finding.verdict,
        verdict: overrideVerdict,
        reason: overrideReason,
        author: overrideAuthor,
        protocolVersion: data.protocolVersion,
        policyFingerprint: data.policyFingerprint,
      });
      setOpenOverrideRole("");
      setOverrideReason("");
      await review.refetch();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "推翻记录提交失败");
    } finally {
      setSubmitting(false);
    }
  };

  const roleLabel = (roleId: string): string =>
    roles.find((item) => item.id === roleId)?.label ?? roleId;

  return (
    <div className="page institutional-workbench council-page">
      <WorkbenchHeader
        eyebrow="ATLAS L5 / DECISION COUNCIL"
        title="多 Agent 决策议事会"
        description="每个角色只在自身职责域内审查，只消费结构化证据，并只能否决自己域内的问题。人工可以推翻任一角色，但推翻会写入不可删除的审计日志。"
        asOf={contractIssue ? "协议漂移" : decision ? decisionMeta?.label ?? decision.state : "等待审查对象"}
        context="证据缺失记为 unknown，不记为通过"
      />

      <WorkbenchMetricStrip
        metrics={[
          {
            label: "议事会结论",
            value: decisionMeta?.label ?? "—",
            detail: decision?.summary ?? "选择一个搜索产物后生成",
            tone: !decisionMeta ? "neutral" : decisionMeta.tone === "danger" ? "danger" : decisionMeta.tone === "warning" ? "warning" : "positive",
            icon: Gavel,
          },
          { label: "通过", value: String(counts.pass), detail: `${roles.length} 个角色`, tone: counts.pass ? "positive" : "neutral", icon: CheckCircle },
          { label: "保留意见", value: String(counts.warn), detail: "不阻塞，但需复核", tone: "warning", icon: Warning },
          { label: "否决", value: String(counts.blocked), detail: "阻塞晋级，不阻塞研究", tone: counts.blocked ? "danger" : "neutral", icon: Prohibit },
          { label: "证据不足", value: String(counts.unknown), detail: "缺证据，不计为通过", tone: counts.unknown ? "warning" : "neutral", icon: Question },
          {
            label: "已被推翻",
            value: String(decision?.overriddenRoles.length ?? 0),
            detail: "全部记入审计日志",
            tone: decision?.overriddenRoles.length ? "ai" : "neutral",
            icon: ShieldCheck,
          },
        ]}
      />
      <span className="sr-only" role="status" aria-live="polite">
        {decision ? `议事会结论：${decisionMeta?.label ?? decision.state}。${decision.summary}` : "议事会等待审查对象。"}
      </span>

      <WorkbenchPanel
        eyebrow="COMPANY REVIEW CHAIN"
        title={contractIssue || !roles.length ? "Council 协议不可用" : `${roles.length} 个角色共同协商`}
        meta={data ? `Council v${data.protocolVersion} · ${data.policyFingerprint.slice(0, 12)}` : "等待协议证据"}
      >
        {contractIssue ? <TruthNotice tone="warning">{contractIssue}</TruthNotice> : null}
        <ol className="council-company-flow" aria-label="公司共同决策流程">
          {COMPANY_PHASES.map((phase, phaseIndex) => {
            const phaseRoles = phase.roleIds
              .map((roleId) => roles.find((role) => role.id === roleId))
              .filter((role): role is NonNullable<typeof role> => Boolean(role));
            if (!phaseRoles.length) return null;
            return (
              <li key={phase.label}>
                <strong>{phase.label}</strong>
                <div>
                  {phaseRoles.map((role) => {
                    const finding = findings.find((item) => item.roleId === role.id);
                    const effective = effectiveVerdict(finding);
                    const meta = VERDICT_META[effective];
                    return (
                      <span key={role.id} className="atlas-chip" data-tone={meta.tone}>
                        {role.label} · {meta.label}
                      </span>
                    );
                  })}
                </div>
                {phaseIndex < COMPANY_PHASES.length - 1 ? <ArrowRight aria-hidden="true" size={14} /> : null}
              </li>
            );
          })}
        </ol>
        <TruthNotice tone="warning">
          “共同协商”不是多数票：unknown 会阻止晋级但不阻止继续研究；CIO 必须汇总前十岗，不能绕过人工 Gate；liveEligible 永远为 false。
        </TruthNotice>
      </WorkbenchPanel>

      <section className="atlas-split">
        <div className="atlas-stack">
          <WorkbenchPanel
            eyebrow="ROLE VERDICTS"
            title="角色裁决"
            meta={data ? `对象 ${data.subject.candidateLabel ?? data.subject.id}` : "无对象"}
          >
            {findings.length ? (
              <div className="council-findings">
                {findings.map((finding) => {
                  const effective = effectiveVerdict(finding);
                  const meta = VERDICT_META[effective as CouncilFinding["verdict"]];
                  const VerdictIcon = meta.icon;
                  const role = roles.find((item) => item.id === finding.roleId);
                  return (
                    <article
                      key={finding.roleId}
                      className="atlas-surface council-finding"
                      data-rail={meta.tone === "control" ? "warning" : meta.tone}
                    >
                      <header>
                        <div>
                          <span className="atlas-eyebrow">{role?.vetoScope ?? "review"}</span>
                          <strong>
                            <span aria-hidden="true">
                              {String(findings.findIndex((item) => item.roleId === finding.roleId) + 1).padStart(2, "0")}
                              /{String(roles.length).padStart(2, "0")} ·{" "}
                            </span>
                            <span>{roleLabel(finding.roleId)}</span>
                          </strong>
                          <small>{role?.domain}</small>
                        </div>
                        <span className="atlas-chip" data-tone={meta.tone}>
                          <VerdictIcon size={11} weight="fill" />
                          {meta.label}
                        </span>
                      </header>
                      <p className="council-headline">{finding.headline}</p>
                      <p className="council-detail">{finding.detail}</p>

                      <details className="council-evidence">
                        <summary>查看该裁决使用的证据</summary>
                        <dl>
                          {Object.entries(finding.evidence).map(([key, value]) => (
                            <div key={key}>
                              <dt>{key}</dt>
                              <dd className="mono">
                                {value === null || value === undefined
                                  ? "null"
                                  : typeof value === "object"
                                    ? JSON.stringify(value)
                                    : String(value)}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      </details>

                      {finding.override ? (
                        <div className="council-override-record">
                          <span className="atlas-chip" data-tone="agent">人工推翻</span>
                          <div>
                            <strong>
                              {VERDICT_META[finding.override.replacedVerdict].label} →{" "}
                              {VERDICT_META[finding.override.verdict].label}
                            </strong>
                            <small>
                              {finding.override.author} · {finding.override.recordedAt}
                            </small>
                            <small>{finding.override.reason}</small>
                          </div>
                        </div>
                      ) : null}

                      <footer>
                        <span>下一步：{finding.nextAction}</span>
                        <button
                          type="button"
                          className="atlas-action"
                          onClick={() => {
                            setOverrideReason("");
                            setError("");
                            setOverrideVerdict(finding.verdict === "unknown" ? "blocked" : "warn");
                            setOpenOverrideRole(
                              openOverrideRole === finding.roleId ? "" : finding.roleId,
                            );
                          }}
                          aria-expanded={openOverrideRole === finding.roleId}
                        >
                          <Gavel size={12} />人工推翻
                        </button>
                      </footer>

                      {openOverrideRole === finding.roleId ? (
                        <form
                          className="council-override-form"
                          onSubmit={(event) => {
                            event.preventDefault();
                            void submitOverride(finding);
                          }}
                        >
                          <TruthNotice tone="warning">
                            推翻不会删除原裁决：原裁决与推翻记录会并列保存，并写入审计日志。
                            {finding.verdict === "unknown" ? "证据不足仅可改判为否决；请先补充证据并重新审议。" : ""}
                          </TruthNotice>
                          <label className="atlas-field">
                            <span>改判为</span>
                            <select
                              value={overrideVerdict}
                              onChange={(event) =>
                                setOverrideVerdict(event.target.value as "pass" | "warn" | "blocked")}
                            >
                              <option value="pass" disabled={finding.verdict === "unknown"}>通过</option>
                              <option value="warn" disabled={finding.verdict === "unknown"}>保留意见</option>
                              <option value="blocked">否决</option>
                            </select>
                          </label>
                          <label className="atlas-field">
                            <span>决策人</span>
                            <input
                              value={overrideAuthor}
                              onChange={(event) => setOverrideAuthor(event.target.value)}
                              required
                            />
                          </label>
                          <label className="atlas-field">
                            <span>理由（至少 8 个字符）</span>
                            <textarea
                              rows={3}
                              value={overrideReason}
                              onChange={(event) => setOverrideReason(event.target.value)}
                              required
                              minLength={8}
                            />
                          </label>
                          <div className="atlas-row">
                            <button
                              type="submit"
                              className="atlas-action"
                              data-variant="primary"
                              disabled={submitting || Boolean(contractIssue) || overrideReason.trim().length < 8 || !overrideAuthor.trim()}
                            >
                              {submitting ? "记录中" : "记录推翻"}
                            </button>
                            <button
                              type="button"
                              className="atlas-action"
                              onClick={() => setOpenOverrideRole("")}
                            >
                              取消
                            </button>
                          </div>
                          {error ? <p className="council-error" role="alert">{error}</p> : null}
                        </form>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            ) : (
              <ActionableState
                title={runList.length ? "正在读取审查对象" : "没有可审查的研究产物"}
                detail="议事会审查 Runtime 中已完成的因子融合搜索。先在因子融合工场启动一次搜索，产物写入后此处会自动出现。"
                icon={Gavel}
              />
            )}
          </WorkbenchPanel>
        </div>

        <aside className="atlas-stack">
          <WorkbenchPanel eyebrow="SUBJECT" title="审查对象" meta={`${runList.length} 个可选`}>
            {runList.length ? (
              <ul className="foundry-run-list">
                {runList.map((run) => (
                  <li key={run.id}>
                    <button
                      type="button"
                      className={run.id === effectiveRunId ? "selected" : ""}
                      onClick={() => setSelectedRunId(run.id)}
                      aria-pressed={run.id === effectiveRunId}
                    >
                      <strong>{run.name}</strong>
                      <small>{run.nTrials ?? "?"} 次试验 · 前沿 {run.frontierSize}</small>
                      <small className="mono">{run.contentHash ?? "no hash"}</small>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <ActionableState title="Runtime 中没有产物" detail="搜索完成后会自动出现在此。" compact />
            )}
          </WorkbenchPanel>

          <WorkbenchPanel eyebrow="PROMOTION BARS" title="晋级阈值" meta="全部会被实际检查">
            {thresholds ? (
              <table className="atlas-grid">
                <tbody>
                  {Object.entries(thresholds).map(([key, value]) => (
                    <tr key={key}>
                      <td>{key}</td>
                      <td className="num">{String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <ActionableState title="阈值不可用" detail="Quant API 未连接。" compact />
            )}
            {roster.data?.data.protocol ? (
              <TruthNotice>
                Council v{roster.data.data.protocolVersion} · {roster.data.data.policyFingerprint.slice(0, 12)} · {roster.data.data.protocol}
                {" "}{roster.data.data.migration}
              </TruthNotice>
            ) : null}
          </WorkbenchPanel>

          <WorkbenchPanel
            eyebrow="AUDIT LOG"
            title="推翻审计"
            meta={`${data?.overrides.length ?? 0} 条记录`}
          >
            {data?.overrides.length ? (
              <ul className="council-audit">
                {[...data.overrides].reverse().map((item, index) => (
                  <li key={`${item.recordedAt}-${index}`}>
                    <div className="atlas-row">
                      <span className="atlas-chip" data-tone="agent">{roleLabel(item.roleId)}</span>
                      <span className="atlas-chip" data-tone={VERDICT_META[item.verdict].tone}>
                        {VERDICT_META[item.verdict].label}
                      </span>
                    </div>
                    <small className="mono">{item.author} · {item.recordedAt}</small>
                    <small className="mono">
                      {item.effective ? "effective" : item.scopeStatus ?? "historical"}
                    </small>
                    <p>{item.reason}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <ActionableState
                title="尚无人工推翻"
                detail="所有裁决均为角色自动生成。任何人工推翻都会在此留下不可删除的记录。"
                compact
              />
            )}
          </WorkbenchPanel>
        </aside>
      </section>
    </div>
  );
}
