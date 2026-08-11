import {
  CheckCircle,
  ClockCounterClockwise,
  Database,
  Fingerprint,
  ShieldCheck,
  ShieldWarning,
  WarningCircle,
} from "@phosphor-icons/react";
import { useApi } from "../../hooks/useApi";
import { formatNumber } from "../../utils/format";
import {
  ActionableState,
  TruthNotice,
  WorkbenchPanel,
} from "../workbench/InstitutionalWorkbench";

export interface PaperExecutionRecord {
  sequence: number;
  payloadSha256: string;
  signalDate: string;
  executionDate: string;
  status: string;
  recordedAt: string;
  recordSha256: string;
  details: Record<string, unknown>;
}

export interface PaperExecutionEvidence {
  journal: {
    state: "valid" | "invalid" | "unavailable";
    verified: boolean;
    path: string;
    recordCount: number;
    terminalCount: number;
    unresolvedCount: number;
    reason?: string | null;
  };
  accountIdentity: {
    state: "valid" | "invalid" | "unavailable";
    verified: boolean;
    path: string;
    accountInstanceId?: string | null;
    portfolioId?: string | null;
    initialCashCny?: string | null;
    payloadSha256?: string | null;
    reason?: string | null;
  };
  canonicalPrefix: {
    state: "valid" | "invalid" | "unavailable";
    verified: boolean;
    boundTerminalCount: number;
    legacyUnboundTerminalCount: number;
    latestTerminalBound: boolean;
    reason?: string | null;
  };
  summary: {
    attention: "ok" | "warning" | "critical" | "pending" | "unavailable";
    latestStatus?: string | null;
    latestSignalDate?: string | null;
    latestExecutionDate?: string | null;
    latestRecordedAt?: string | null;
    calendarAssurance: string;
    shadowAcceptanceCalendarEligible: boolean;
    productionPretradeRiskCertified: boolean;
    accountIdentityVerified: boolean;
    latestTerminalCanonicalPrefixBound: boolean;
    riskScope?: string | null;
    sessionClosed?: boolean | null;
    orderCount?: number | null;
    fillCount?: number | null;
    navBefore?: number | null;
    navAfter?: number | null;
    statusCounts: Record<string, number>;
  };
  records: PaperExecutionRecord[];
  operatorTruth: {
    paperExecutionEvidence: boolean;
    accountIdentityVerified: boolean;
    canonicalExecutionPrefixCertified: boolean;
    productionLiveCertified: boolean;
    authoritativeCalendarCertified: boolean;
    message: string;
  };
}

const STATUS_LABEL: Record<string, string> = {
  execution_started: "执行已启动 / 尚未闭合",
  execution_observed: "Paper 执行已观察",
  execution_blocked: "执行被风控/市场条件阻止",
  missed_execution_session: "错过精确下一交易会话",
  execution_indeterminate: "执行结果不确定 / 账户冻结",
};

const STATUS_TONE: Record<string, string> = {
  execution_started: "pending",
  execution_observed: "success",
  execution_blocked: "warning",
  missed_execution_session: "warning",
  execution_indeterminate: "danger",
};

function statusLabel(status?: string | null): string {
  if (!status) return "暂无连续执行证据";
  return STATUS_LABEL[status] ?? status;
}

function yesNo(value: boolean): string {
  return value ? "YES" : "NO";
}

function navDelta(before?: number | null, after?: number | null): string {
  if (before == null || after == null) return "—";
  return formatNumber(after - before);
}

function recordedAt(value?: string | null): string {
  if (!value) return "—";
  return value.replace("T", " ").slice(0, 19);
}

function shortHash(value?: string | null, length = 12): string {
  if (!value) return "—";
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

export function PaperExecutionEvidencePanel(): JSX.Element {
  const query = useApi<PaperExecutionEvidence>(
    ["paper-execution-evidence"],
    "/paper/execution-evidence",
    { limit: 12 },
    { refetchInterval: 15_000, staleTime: 5_000 },
  );

  if (query.isLoading) {
    return (
      <WorkbenchPanel eyebrow="CONTINUOUS PAPER ACCOUNT" title="连续 Paper 执行证据" meta="loading verified journal">
        <div role="status" aria-live="polite">
          <ActionableState
            compact
            icon={ClockCounterClockwise}
            title="正在验证执行证据链"
            detail="读取 canonical paper runtime；尚未把加载状态解释成健康或异常。"
          />
        </div>
      </WorkbenchPanel>
    );
  }

  if (query.isError || !query.data?.data) {
    return (
      <WorkbenchPanel eyebrow="CONTINUOUS PAPER ACCOUNT" title="连续 Paper 执行证据" meta="API unavailable">
        <div role="alert">
          <ActionableState
            compact
            tone="danger"
            icon={ShieldWarning}
            title="执行证据 API 不可用"
            detail={query.error?.message ?? "无法取得 paper execution evidence；不推断任何执行或实盘状态。"}
            primary={{ label: "重新读取", onClick: () => void query.refetch() }}
          />
        </div>
      </WorkbenchPanel>
    );
  }

  const evidence = query.data.data;
  const unresolved = evidence.journal.unresolvedCount > 0;
  const identityInvalid = evidence.accountIdentity.state === "invalid";
  const prefixInvalid = evidence.canonicalPrefix.state === "invalid";
  const critical = evidence.journal.state === "invalid"
    || evidence.summary.attention === "critical"
    || identityInvalid
    || prefixInvalid;
  const liveRegionRole = critical ? "alert" : "status";
  const liveCertified = evidence.operatorTruth.productionLiveCertified;
  const calendarCertified = evidence.operatorTruth.authoritativeCalendarCertified;
  const identityVerified = evidence.operatorTruth.accountIdentityVerified;
  const prefixCertified = evidence.operatorTruth.canonicalExecutionPrefixCertified;
  const latestStatus = evidence.summary.latestStatus;
  const latestRecords = evidence.records.slice(0, 6);
  const leadTitle = identityInvalid
    ? "Paper 账户身份校验失败"
    : prefixInvalid
      ? "Canonical 执行前缀校验失败"
      : evidence.journal.state === "invalid"
        ? "执行证据链校验失败"
        : unresolved
          ? "存在未闭合执行尝试"
          : evidence.journal.state === "unavailable"
            ? "尚无连续执行 journal"
            : statusLabel(latestStatus);
  const leadMessage = unresolved && evidence.journal.state === "valid"
    ? `${evidence.journal.unresolvedCount} 个 execution_started 尚无 terminal outcome；禁止把最新成功记录解释为账户整体健康。`
    : identityInvalid
      ? evidence.accountIdentity.reason ?? "account_identity.json 无法验证；按 fail-closed 处理。"
      : prefixInvalid
        ? evidence.canonicalPrefix.reason ?? "terminal receipt 无法绑定 canonical ledger prefix。"
        : evidence.operatorTruth.message;

  return (
    <WorkbenchPanel
      eyebrow="CONTINUOUS PAPER ACCOUNT / VERIFIED EVIDENCE"
      title="连续 Paper 执行证据"
      meta={`${evidence.journal.path} · ${evidence.journal.verified ? "hash-chain verified" : evidence.journal.state}`}
      className="paper-execution-panel"
      actions={<button type="button" className="paper-evidence-refresh" onClick={() => void query.refetch()}>刷新证据</button>}
    >
      <div className={`paper-evidence-live tone-${critical ? "danger" : evidence.summary.attention}`} role={liveRegionRole} aria-live={critical ? "assertive" : "polite"}>
        <div className="paper-evidence-lead">
          {critical ? <ShieldWarning size={24} weight="duotone" /> : evidence.journal.verified ? <CheckCircle size={24} weight="duotone" /> : <Database size={24} weight="duotone" />}
          <div>
            <strong>{leadTitle}</strong>
            <span>{leadMessage}</span>
          </div>
          <span className={`paper-evidence-status status-${STATUS_TONE[latestStatus ?? ""] ?? "neutral"}`}>{latestStatus ?? evidence.journal.state}</span>
        </div>
      </div>

      <div className="paper-evidence-cert-grid" aria-label="Paper、账户与实盘证据隔离">
        <article>
          <span>Paper execution evidence</span>
          <strong>{yesNo(evidence.operatorTruth.paperExecutionEvidence)}</strong>
          <small>journal hash-chain</small>
        </article>
        <article className={identityVerified ? "certified" : "locked"}>
          <span>Account identity verified</span>
          <strong>{yesNo(identityVerified)}</strong>
          <small>{evidence.accountIdentity.portfolioId ?? evidence.accountIdentity.state}</small>
        </article>
        <article className={prefixCertified ? "certified" : "locked"}>
          <span>Canonical terminal prefix</span>
          <strong>{yesNo(prefixCertified)}</strong>
          <small>{evidence.canonicalPrefix.boundTerminalCount} bound · {evidence.canonicalPrefix.legacyUnboundTerminalCount} legacy</small>
        </article>
        <article className={liveCertified ? "certified" : "locked"}>
          <span>Production pre-trade certified</span>
          <strong>{yesNo(liveCertified)}</strong>
          <small>{evidence.summary.riskScope ?? "no production risk certificate"}</small>
        </article>
        <article className={calendarCertified ? "certified" : "locked"}>
          <span>Authoritative calendar certified</span>
          <strong>{yesNo(calendarCertified)}</strong>
          <small>{evidence.summary.calendarAssurance}</small>
        </article>
        <article className={unresolved ? "locked" : ""}>
          <span>Unresolved attempts</span>
          <strong>{evidence.journal.unresolvedCount}</strong>
          <small>{evidence.journal.recordCount} journal records</small>
        </article>
      </div>

      <div className={`paper-evidence-identity ${identityVerified ? "verified" : "unverified"}`} aria-label="Paper account identity evidence">
        <Fingerprint size={18} weight="duotone" aria-hidden="true" />
        <span><small>Portfolio</small><strong>{evidence.accountIdentity.portfolioId ?? "—"}</strong></span>
        <span><small>Account instance</small><code title={evidence.accountIdentity.accountInstanceId ?? undefined}>{shortHash(evidence.accountIdentity.accountInstanceId, 16)}</code></span>
        <span><small>Initial cash CNY</small><strong>{evidence.accountIdentity.initialCashCny ?? "—"}</strong></span>
        <span><small>Identity SHA-256</small><code title={evidence.accountIdentity.payloadSha256 ?? undefined}>{shortHash(evidence.accountIdentity.payloadSha256, 16)}</code></span>
        {identityVerified ? <ShieldCheck size={16} weight="fill" aria-label="identity verified" /> : <ShieldWarning size={16} weight="fill" aria-label="identity unavailable or invalid" />}
      </div>

      {!liveCertified || !calendarCertified ? (
        <TruthNotice tone="warning">
          Paper/shadow 观察结果、账户 identity、canonical prefix 均不等于实盘认证。当前界面不会因为 execution_observed、正收益或 session_closed 自动显示“可实盘”。
        </TruthNotice>
      ) : null}

      {evidence.journal.state === "unavailable" ? (
        <ActionableState
          compact
          tone="warning"
          icon={Database}
          title="尚无连续执行 journal"
          detail={evidence.journal.reason ?? "等待 continuous paper consumer 写入 canonical runtime。"}
        />
      ) : evidence.journal.state === "invalid" ? (
        <ActionableState
          compact
          tone="danger"
          icon={ShieldWarning}
          title="执行证据无法验证"
          detail={evidence.journal.reason ?? evidence.canonicalPrefix.reason ?? evidence.accountIdentity.reason ?? "journal / identity / canonical prefix 无法验证；按 fail-closed 处理。"}
        />
      ) : (
        <>
          <div className="paper-evidence-metrics">
            <span><small>Signal date</small><strong>{evidence.summary.latestSignalDate ?? "—"}</strong></span>
            <span><small>Execution date</small><strong>{evidence.summary.latestExecutionDate ?? "—"}</strong></span>
            <span><small>Orders / fills</small><strong>{evidence.summary.orderCount ?? "—"} / {evidence.summary.fillCount ?? "—"}</strong></span>
            <span><small>NAV Δ</small><strong>{navDelta(evidence.summary.navBefore, evidence.summary.navAfter)}</strong></span>
            <span><small>Session closed</small><strong>{evidence.summary.sessionClosed == null ? "—" : yesNo(evidence.summary.sessionClosed)}</strong></span>
            <span><small>Recorded</small><strong>{recordedAt(evidence.summary.latestRecordedAt)}</strong></span>
          </div>

          <div className="paper-evidence-history">
            <header><strong>最近证据记录</strong><span>latest first · immutable journal projection</span></header>
            {latestRecords.length ? (
              <div className="paper-evidence-records">
                {latestRecords.map((record) => (
                  <article key={`${record.sequence}-${record.recordSha256}`}>
                    <span className={`paper-evidence-dot status-${STATUS_TONE[record.status] ?? "neutral"}`} aria-hidden="true" />
                    <div><strong>{statusLabel(record.status)}</strong><small>{record.signalDate} → {record.executionDate}</small></div>
                    <code>#{record.sequence} · {record.recordSha256.slice(0, 10)}</code>
                  </article>
                ))}
              </div>
            ) : <div className="paper-evidence-empty"><WarningCircle size={16} />暂无可信记录</div>}
          </div>
        </>
      )}
    </WorkbenchPanel>
  );
}

export { navDelta, shortHash, statusLabel };
