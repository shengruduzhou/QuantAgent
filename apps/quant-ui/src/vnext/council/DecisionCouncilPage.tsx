import { useMemo, useState } from "react";
import {
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
  const findings = data?.findings ?? [];
  const roles = roster.data?.data.roles ?? [];
  const thresholds = roster.data?.data.thresholds;
  const decision = data?.decision;
  const decisionMeta = decision ? DECISION_META[decision.state] : undefined;

  const counts = useMemo(() => {
    const effective = findings.map((item) => item.override?.verdict ?? item.verdict);
    return {
      pass: effective.filter((item) => item === "pass").length,
      warn: effective.filter((item) => item === "warn").length,
      blocked: effective.filter((item) => item === "blocked").length,
      unknown: effective.filter((item) => item === "unknown").length,
    };
  }, [findings]);

  const submitOverride = async (roleId: string): Promise<void> => {
    setSubmitting(true);
    setError("");
    try {
      await apiPost<CouncilOverride>("/council/overrides", {
        subjectType: "fusion_run",
        subjectId: effectiveRunId,
        roleId,
        verdict: overrideVerdict,
        reason: overrideReason,
        author: overrideAuthor,
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
        asOf={decision ? decisionMeta?.label ?? decision.state : "等待审查对象"}
        context="证据缺失记为 unknown，不记为通过"
      />

      <WorkbenchMetricStrip
        metrics={[
          {
            label: "议事会结论",
            value: decisionMeta?.label ?? "—",
            detail: decision?.summary ?? "选择一个搜索产物后生成",
            tone: decisionMeta?.tone === "danger" ? "danger" : decisionMeta?.tone === "warning" ? "warning" : "positive",
            icon: Gavel,
          },
          { label: "通过", value: String(counts.pass), detail: `${roles.length} 个角色`, tone: "positive", icon: CheckCircle },
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
                  const effective = finding.override?.verdict ?? finding.verdict;
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
                          <strong>{roleLabel(finding.roleId)}</strong>
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
                          onClick={() =>
                            setOpenOverrideRole(
                              openOverrideRole === finding.roleId ? "" : finding.roleId,
                            )}
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
                            void submitOverride(finding.roleId);
                          }}
                        >
                          <TruthNotice tone="warning">
                            推翻不会删除原裁决：原裁决与推翻记录会并列保存，并写入审计日志。
                          </TruthNotice>
                          <label className="atlas-field">
                            <span>改判为</span>
                            <select
                              value={overrideVerdict}
                              onChange={(event) =>
                                setOverrideVerdict(event.target.value as "pass" | "warn" | "blocked")}
                            >
                              <option value="pass">通过</option>
                              <option value="warn">保留意见</option>
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
                              disabled={submitting || overrideReason.trim().length < 8 || !overrideAuthor.trim()}
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
              <TruthNotice>{roster.data.data.protocol}</TruthNotice>
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
