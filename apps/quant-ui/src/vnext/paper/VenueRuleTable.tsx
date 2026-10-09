import type { VenueRuleRow } from "./venueRules";

const STATE_LABEL: Record<VenueRuleRow["state"], { text: string; tone: string }> = {
  normal: { text: "WITHIN LIMIT", tone: "primary" },
  warning: { text: "≥80% OF LIMIT", tone: "warning" },
  breach: { text: "BREACH", tone: "danger" },
  unmeasured: { text: "NOT MEASURED", tone: "warning" },
  per_order: { text: "PER ORDER", tone: "control" },
};

const ENFORCEMENT_LABEL: Record<VenueRuleRow["enforcement"], { text: string; tone: string }> = {
  enforced: { text: "ENFORCED", tone: "success" },
  not_attached: { text: "NO ENGINE", tone: "danger" },
  unknown: { text: "UNKNOWN", tone: "warning" },
};

/**
 * Paper-venue limits with unit, enforcement point and the paper account's
 * like-for-like current value. Shared by the Risk page and the dashboard so
 * the two never disagree on what a limit is or what it is compared against.
 */
export function VenueRuleTable({ rows, compact = false }: { rows: VenueRuleRow[]; compact?: boolean }): JSX.Element {
  if (!rows.length) {
    return <div className="atlas-empty"><strong>没有风险规则</strong><p>/api/risk/rules 未返回 paper venue 的限额；不显示任何默认阈值。下一步：检查 Quant API 是否能导入 quantagent.paper.risk。</p></div>;
  }
  return (
    <div className="atlas-scroll-x">
      <table className={`atlas-grid venue-rule-table ${compact ? "compact" : ""}`.trim()} aria-label="Paper venue 风险规则">
        <thead>
          <tr>
            <th scope="col">Rule</th>
            <th scope="col" className="num">Limit</th>
            <th scope="col">Unit · enforced at</th>
            <th scope="col">Paper account now</th>
            <th scope="col">State</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const state = STATE_LABEL[row.state];
            const enforcement = ENFORCEMENT_LABEL[row.enforcement];
            return (
              <tr key={row.id} data-state={row.state}>
                <td>
                  <strong className="venue-rule-name">{row.name}</strong>
                  {compact ? null : <small className="venue-rule-desc">{row.description}</small>}
                </td>
                <td className="num">
                  <span className="venue-rule-limit">{row.threshold}</span>
                  <small className="venue-rule-desc">{row.thresholdSource === "venue" ? "venue config" : row.thresholdSource === "code default" ? "code default" : "—"}</small>
                </td>
                <td>
                  <span>{row.unitLabel}</span>
                  <small className="venue-rule-desc">{row.enforcedAtLabel}</small>
                </td>
                <td>
                  {row.current.kind === "measured" ? (
                    <>
                      <span className="mono">{row.current.text}{row.current.ratio !== null ? <em className="venue-rule-ratio"> · {Math.round(row.current.ratio * 100)}% of limit</em> : null}</span>
                      <small className="venue-rule-desc">{row.current.basis}</small>
                    </>
                  ) : row.current.kind === "per_order" ? (
                    <span className="venue-rule-muted">{row.current.text}</span>
                  ) : (
                    <>
                      <span className="venue-rule-unmeasured">not measured for the paper account</span>
                      <small className="venue-rule-desc" title={row.current.reason ?? undefined}>{row.current.text}{row.current.reason ? ` (${row.current.reason.split(":")[0]})` : ""}</small>
                    </>
                  )}
                </td>
                <td>
                  <span className="venue-rule-chips">
                    <span className="atlas-chip" data-tone={state.tone}>{state.text}</span>
                    <span className="atlas-chip" data-tone={enforcement.tone} title={row.codeLocation ?? undefined}>{enforcement.text}</span>
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
