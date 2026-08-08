import { StateView } from "../../components/StateView";
import { useApi } from "../../hooks/useApi";
import "./ProductionReadinessPanel.css";

interface ReadinessDimension {
  key: string;
  label: string;
  state: string;
  severity: string;
  reasons: string[];
  evidence: Record<string, unknown>;
}

interface ProductionReadinessStatus {
  schemaVersion: number;
  generatedAt: string;
  aggregateTradingReady: null;
  aggregateStateSemantics: string;
  cards: ReadinessDimension[];
}

const DIMENSIONS = [
  ["modelTrust", "Model Trust"],
  ["brokerQuery", "Broker Query Readiness"],
  ["targetRisk", "Target Risk"],
  ["orderRisk", "Order Risk"],
  ["killSwitch", "KillSwitch"],
  ["reconciliation", "Reconciliation"],
  ["productArming", "Product Arming"],
  ["hostCertification", "Host / Platform Certification"],
] as const;

const SAFE_EVIDENCE_KEYS = new Set([
  "modelId",
  "certificateStatus",
  "trustClass",
  "requiredMetricSemantics",
  "observedMetricSemantics",
  "manifest",
  "certificate",
  "asOf",
  "validUntil",
  "queryOnly",
  "semantics",
  "stateFile",
  "updatedAt",
  "liveTradingAvailable",
  "writeApis",
  "brokerOrExchangeIntegration",
  "platform",
  "qmtApi",
  "accountQuery",
  "portableContractIsNotHostCertification",
  "unexplainedDifferences",
]);

function evidenceText(evidence: Record<string, unknown>): string[] {
  return Object.entries(evidence)
    .filter(([key]) => SAFE_EVIDENCE_KEYS.has(key))
    .slice(0, 6)
    .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : String(value ?? "—")}`);
}

function cardClass(severity: string): string {
  if (severity === "blocked") return "production-readiness-card is-blocked";
  if (severity === "ok") return "production-readiness-card is-ok";
  return "production-readiness-card is-info";
}

export function ProductionReadinessPanel(): JSX.Element {
  const query = useApi<ProductionReadinessStatus>(
    ["production-readiness"],
    "/governance/production-readiness",
    undefined,
    { refetchInterval: 15_000, staleTime: 5_000 },
  );

  if (query.isLoading) {
    return <StateView state="loading" detail="正在读取生产就绪机器证据。" />;
  }

  const data = query.data?.data;
  if (query.isError || !data || !Array.isArray(data.cards)) {
    return (
      <section className="production-readiness-shell" aria-label="生产就绪真值">
        <StateView
          state="unavailable"
          title="生产就绪证据不可用"
          detail="未取得 /api/governance/production-readiness；缺失证据不会被推断为安全。"
        />
      </section>
    );
  }

  const cardMap = new Map(data.cards.map((card) => [card.key, card]));

  return (
    <section className="production-readiness-shell" aria-label="生产就绪真值">
      <header className="production-readiness-header">
        <div>
          <p className="production-readiness-eyebrow">OPERATOR / PRODUCTION TRUTH</p>
          <h2>生产就绪独立证据</h2>
          <p>
            八个维度分别读取机器证据；不存在的证书保持 UNKNOWN / NOT CERTIFIED，风险接线也不等于本次授权通过。
          </p>
        </div>
        <div className="production-readiness-asof">
          <span>schema v{data.schemaVersion}</span>
          <strong>{data.generatedAt || "—"}</strong>
        </div>
      </header>

      <div className="production-readiness-grid">
        {DIMENSIONS.map(([key, fallbackLabel]) => {
          const card = cardMap.get(key);
          const reasons = card?.reasons ?? ["machine_evidence_missing"];
          const evidence = card ? evidenceText(card.evidence ?? {}) : [];
          return (
            <article className={cardClass(card?.severity ?? "blocked")} data-testid={`readiness-${key}`} key={key}>
              <div className="production-readiness-card-head">
                <h3>{card?.label ?? fallbackLabel}</h3>
                <span className="production-readiness-state">{card?.state ?? "UNKNOWN"}</span>
              </div>
              <p className="production-readiness-card-rule">
                {key === "targetRisk" || key === "orderRisk"
                  ? "WIRED 仅证明强制路径存在，不代表当前组合或订单已经通过。"
                  : "状态只来自当前机器证据，缺失不会自动提升。"}
              </p>
              {reasons.length > 0 ? (
                <ul className="production-readiness-reasons">
                  {reasons.slice(0, 4).map((reason) => <li key={reason}>{reason}</li>)}
                </ul>
              ) : (
                <p className="production-readiness-clear">没有阻塞原因。</p>
              )}
              {evidence.length > 0 && (
                <dl className="production-readiness-evidence">
                  {evidence.map((entry) => {
                    const separator = entry.indexOf(":");
                    return (
                      <div key={entry}>
                        <dt>{entry.slice(0, separator)}</dt>
                        <dd>{entry.slice(separator + 1).trim()}</dd>
                      </div>
                    );
                  })}
                </dl>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
