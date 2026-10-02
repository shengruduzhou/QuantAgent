import { ArrowRight, Fingerprint, LockKey, ShieldCheck, ShieldWarning, WarningCircle } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { describeReason, type PaperAccount, type PaperRiskState } from "../../api/paperAccount";
import { usePaperAccount } from "../../hooks/usePaperAccount";
import { formatNumber, formatPercent } from "../../utils/format";

/**
 * Paper Account & Risk — the HTTP paper venue's own account and the venue
 * risk engine's measurements of it (`GET /api/paper/account`).
 *
 * This is the only surface that answers "which account is this, and is the
 * kill switch tripped?". It deliberately shows nothing from a backtest: the
 * venue's drawdown is measured from *its* all-time peak, and its limits are
 * the ones `RiskLimits` enforces, read from the same response. Every null the
 * producer sends is rendered as 未测量 with the producer's reason — never as
 * 0%, "clear" or "pass".
 */

type Tone = "neutral" | "primary" | "success" | "warning" | "danger";

const INDUSTRY_MODE: Record<string, { label: string; tone: Tone; detail: string }> = {
  measured_with_map: { label: "MEASURED", tone: "primary", detail: "行业权重按映射逐单测量" },
  refuse_unmeasured: { label: "REFUSING UNMEASURED", tone: "warning", detail: "没有行业映射：每笔买入以 industry_unmeasured 拒绝" },
  opt_out: { label: "OPT-OUT", tone: "warning", detail: "限额 ≥ 100%：行业集中度不受限制" },
};

export function shortHash(value: string | null | undefined, length = 12): string | null {
  if (!value) return null;
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

function runtimeRelative(path: string): string {
  const index = path.lastIndexOf("/runtime/");
  return index >= 0 ? path.slice(index + 1) : path;
}

function reasonCode(reason: string): string {
  const head = reason.split(":")[0].trim();
  return head.length > 48 ? `${head.slice(0, 48)}…` : head;
}

/** The single rendering of "the venue did not measure this". */
export function Unmeasured({ reason, label = "未测量", codeOnly = false }: { reason?: string | null; label?: string; codeOnly?: boolean }): JSX.Element {
  const text = describeReason(reason);
  return (
    <span className="paper-risk-unmeasured" title={reason ?? "producer gave no reason"}>
      <b>{label}</b>
      {codeOnly && reason && text ? null : <small>{text ?? reason ?? "生产者未给出原因"}</small>}
      {reason && text ? <code>{reasonCode(reason)}</code> : null}
    </span>
  );
}

function Meter({ value, limit }: { value: number; limit: number | null | undefined }): JSX.Element | null {
  if (limit === null || limit === undefined || !(limit > 0)) return null;
  const ratio = value / limit;
  const tone: Tone = ratio >= 1 ? "danger" : ratio >= 0.8 ? "warning" : "primary";
  return (
    <span className="atlas-meter" data-tone={tone === "primary" ? undefined : tone} role="meter" aria-valuemin={0} aria-valuemax={limit} aria-valuenow={value} aria-label={`${formatPercent(ratio, 0)} of limit used`}>
      <i style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }} />
    </span>
  );
}

function Measure({ label, children, foot }: { label: string; children: ReactNode; foot?: ReactNode }): JSX.Element {
  return (
    <article className="paper-risk-measure">
      <span className="atlas-eyebrow">{label}</span>
      <div className="paper-risk-measure-body">{children}</div>
      {foot ? <small className="paper-risk-foot">{foot}</small> : null}
    </article>
  );
}

function Chip({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }): JSX.Element {
  return <span className="atlas-chip" data-tone={tone === "neutral" ? undefined : tone} title={title}>{children}</span>;
}

function sessionLoss(risk: PaperRiskState): { value: number | null; reason: string | null } {
  if (risk.nav === null) return { value: null, reason: risk.reasons.nav ?? "nav_unavailable" };
  if (risk.sessionStartEquity === null || !(risk.sessionStartEquity > 0)) {
    return { value: null, reason: risk.reasons.sessionStartEquity ?? "no_session_start_valuation" };
  }
  return { value: Math.max(0, 1 - risk.nav / risk.sessionStartEquity), reason: null };
}

function killSwitchView(risk: PaperRiskState): { label: string; tone: Tone; detail: string } {
  const kill = risk.killSwitch;
  if (kill.active) {
    return {
      label: `KILL ACTIVE · ${kill.scope ?? "UNKNOWN SCOPE"}`,
      tone: "danger",
      detail: kill.reduceOnly ? "reduce-only：买入被拒，卖出放行" : "非 reduce-only：该范围内全部委托被拒",
    };
  }
  if (!risk.riskEngineAttached) {
    return { label: "KILL UNARMED", tone: "danger", detail: "风险引擎未接入：回撤与单日亏损从不评估，kill switch 无法触发" };
  }
  return { label: "KILL ARMED", tone: "success", detail: "已接入，未触发；在成交后与开盘时评估回撤和单日亏损" };
}

/** Kill-switch chip for page headers, read from the same cached account query. */
export function PaperKillSwitchChip(): JSX.Element {
  const query = usePaperAccount();
  const risk = query.data?.data?.riskState;
  if (query.isLoading) return <Chip tone="neutral">KILL …</Chip>;
  if (!risk) return <Chip tone="warning" title={query.error?.message ?? "paper account unavailable"}>KILL UNKNOWN</Chip>;
  const view = killSwitchView(risk);
  return <Chip tone={view.tone} title={view.detail}>{view.tone === "danger" ? <ShieldWarning size={11} weight="bold" /> : <ShieldCheck size={11} weight="bold" />}{view.label}</Chip>;
}

export function paperAccountTone(account: PaperAccount): Tone {
  const risk = account.riskState;
  if (risk.killSwitch.active || !risk.riskEngineAttached || risk.unpriceableSymbols.length) return "danger";
  if (account.accountIdentity.reasons.accountInstanceId?.startsWith("identity_record_unreadable")) return "danger";
  if (!account.accountIdentity.accountInstanceId || risk.industryLimit?.mode !== "measured_with_map" || !account.writable) return "warning";
  return "primary";
}

export function PaperAccountRiskView({ account, variant = "overview" }: { account: PaperAccount; variant?: "overview" | "execution" }): JSX.Element {
  const identity = account.accountIdentity;
  const risk = account.riskState;
  const limits = risk.limits;
  const kill = killSwitchView(risk);
  const loss = sessionLoss(risk);
  const live = account.mode?.live_trading_available;
  const industry = risk.industryLimit;
  const industryMode = industry ? INDUSTRY_MODE[industry.mode] ?? { label: industry.mode.toUpperCase(), tone: "warning" as Tone, detail: "未知模式" } : null;
  const positionCount = Object.keys(account.positions ?? {}).length;
  const tone = paperAccountTone(account);
  const turnover = risk.sessionTurnover;
  const check = risk.lastPortfolioCheck;

  return (
    <section className="atlas-surface paper-risk-card" data-rail={tone === "primary" ? "" : tone} aria-label="Paper Account & Risk" data-variant={variant}>
      <header className="paper-risk-head">
        <div>
          <span className="atlas-eyebrow">PAPER ACCOUNT · VENUE RISK STATE · /api/paper/account</span>
          <h2>Paper Account &amp; Risk</h2>
          <p>{variant === "overview"
            ? "Paper venue 自己的账本与风险引擎读数；下方的回测卡片是研究产物，不是这个账户。"
            : "下单 venue 的账户身份与风险引擎读数；执行证据（下方）来自另一份连续执行 journal。"}</p>
        </div>
        <div className="paper-risk-chips" aria-label="账户安全状态">
          <Chip tone="neutral" title={account.mode?.banner}>{account.mode?.mode ?? "MODE UNKNOWN"}</Chip>
          {live === false ? <Chip tone="success" title={account.mode?.banner}><LockKey size={11} weight="bold" />LIVE DISABLED</Chip>
            : live === true ? <Chip tone="danger"><WarningCircle size={11} weight="bold" />LIVE AVAILABLE</Chip>
              : <Chip tone="warning">LIVE UNKNOWN</Chip>}
          <Chip tone={kill.tone} title={kill.detail}>{kill.tone === "danger" ? <ShieldWarning size={11} weight="bold" /> : <ShieldCheck size={11} weight="bold" />}{kill.label}</Chip>
          {account.writable ? null : <Chip tone="warning" title={account.writerLockError ?? undefined}>READ-ONLY</Chip>}
        </div>
      </header>

      <dl className="paper-risk-identity" aria-label="Paper account identity">
        <Fingerprint size={18} weight="duotone" aria-hidden="true" />
        <div><dt>Portfolio</dt><dd className="mono">{identity.portfolioId}</dd></div>
        <div>
          <dt>Account instance</dt>
          <dd>{identity.accountInstanceId
            ? <code title={identity.accountInstanceId}>{shortHash(identity.accountInstanceId, 16)}</code>
            : <Unmeasured label="未记录" reason={identity.reasons.accountInstanceId} />}</dd>
        </div>
        <div>
          <dt>Identity SHA-256</dt>
          <dd>{identity.identitySha256
            ? <code title={identity.identitySha256}>{shortHash(identity.identitySha256)}</code>
            : <Unmeasured label="未记录" reason={identity.reasons.identitySha256} codeOnly={identity.reasons.identitySha256 === identity.reasons.accountInstanceId} />}</dd>
        </div>
        <div>
          <dt>Initial cash CNY</dt>
          <dd className="mono">{formatNumber(identity.initialCash)}</dd>
          <small>{identity.accountInstanceId ? "identity record" : "process configuration"}</small>
        </div>
        <div className="paper-risk-path"><dt>Identity file</dt><dd><code title={identity.identityPath}>{runtimeRelative(identity.identityPath)}</code></dd></div>
      </dl>

      <div className="paper-risk-grid">
        <Measure label="NAV · venue marks" foot={`${risk.valuationMarks.count} marks · ${risk.valuationMarks.source}`}>
          {risk.nav !== null ? <strong className="atlas-figure small">{formatNumber(risk.nav)}</strong> : <Unmeasured reason={risk.reasons.nav} />}
        </Measure>

        <Measure
          label="Drawdown from all-time peak"
          foot={risk.peakEquity !== null ? `peak ${formatNumber(risk.peakEquity)} CNY` : <Unmeasured label="peak 未测量" reason={risk.reasons.peakEquity} />}
        >
          {risk.drawdownFromPeak !== null ? (
            <>
              <span className="paper-risk-vs"><strong className="atlas-figure small">{formatPercent(risk.drawdownFromPeak)}</strong><span>/ limit {limits?.max_drawdown != null ? formatPercent(limits.max_drawdown, 0) : "未配置"}</span></span>
              <Meter value={risk.drawdownFromPeak} limit={limits?.max_drawdown} />
            </>
          ) : <Unmeasured reason={risk.reasons.drawdownFromPeak ?? risk.reasons.riskEngine} />}
        </Measure>

        <Measure
          label="Session loss · vs opening equity"
          foot={risk.sessionStartEquity !== null ? `session open ${formatNumber(risk.sessionStartEquity)} CNY${limits?.max_daily_loss != null ? ` · cap ${formatNumber(limits.max_daily_loss)} CNY` : ""}` : "derived: 1 − NAV ÷ session-open equity"}
        >
          {loss.value !== null ? (
            <>
              <span className="paper-risk-vs"><strong className="atlas-figure small">{formatPercent(loss.value)}</strong><span>/ limit {limits?.max_daily_loss_fraction != null ? formatPercent(limits.max_daily_loss_fraction, 0) : "未配置"}</span></span>
              <Meter value={loss.value} limit={limits?.max_daily_loss_fraction} />
            </>
          ) : <Unmeasured reason={risk.riskEngineAttached ? loss.reason : risk.reasons.riskEngine} />}
        </Measure>

        <Measure
          label="Session turnover · of NAV"
          foot={turnover ? `${turnover.session} · notional ${formatNumber(turnover.notional)} CNY` : undefined}
        >
          {turnover && turnover.fractionOfNav !== null ? (
            <>
              <span className="paper-risk-vs"><strong className="atlas-figure small">{formatPercent(turnover.fractionOfNav)}</strong><span>/ limit {formatPercent(turnover.limit, 0)}</span></span>
              <Meter value={turnover.fractionOfNav} limit={turnover.limit} />
            </>
          ) : <Unmeasured reason={turnover ? risk.reasons.nav ?? "nav_unavailable" : risk.reasons.sessionTurnover ?? risk.reasons.riskEngine} />}
        </Measure>

        <Measure label="Kill switch" foot={kill.detail}>
          <span className={`paper-risk-state tone-${kill.tone}`}>{kill.label}</span>
          {risk.killSwitch.active ? (
            <dl className="paper-risk-kill">
              <div><dt>Scope</dt><dd>{risk.killSwitch.scope ?? "—"}</dd></div>
              <div><dt>Since</dt><dd className="mono">{risk.killSwitch.triggeredAt?.replace("T", " ").slice(0, 19) ?? "未记录"}</dd></div>
              <div><dt>Reduce-only</dt><dd>{risk.killSwitch.reduceOnly ? "YES" : "NO"}</dd></div>
              <div className="wide"><dt>Reason</dt><dd>{risk.killSwitch.reason ?? "未记录原因"}</dd></div>
              {risk.killSwitch.switches.length > 1 ? <div className="wide"><dt>Switches</dt><dd>{risk.killSwitch.switches.map((item) => `${item.scope}${item.key ? `:${item.key}` : ""}`).join(" · ")}</dd></div> : null}
            </dl>
          ) : !risk.riskEngineAttached ? <Unmeasured label="无引擎" reason={risk.reasons.riskEngine} /> : null}
        </Measure>

        <Measure
          label="Industry limit"
          foot={industry ? `limit ${formatPercent(industry.limit, 0)} · map ${industry.mapLoaded ? `${industry.mapSymbolCount} symbols` : "not loaded"}${industry.source ? ` · ${industry.source}` : ""}` : undefined}
        >
          {industry && industryMode ? (
            <>
              <span className={`paper-risk-state tone-${industryMode.tone}`}>{industryMode.label}</span>
              <small className="paper-risk-detail">{industry.error ? `${industryMode.detail}（${industry.error}）` : industryMode.detail}</small>
            </>
          ) : <Unmeasured label="未施加" reason={risk.reasons.industryLimit ?? risk.reasons.riskEngine} />}
        </Measure>
      </div>

      <footer className="paper-risk-footer">
        <span>
          <b>Unpriceable symbols</b>
          {risk.unpriceableSymbols.length
            ? <span className="paper-risk-state tone-danger">{risk.unpriceableSymbols.join(", ")} — NAV 与组合限额不可测</span>
            : <span>{positionCount ? `none · ${positionCount} positions all marked` : "none · no open positions"}</span>}
        </span>
        <span>
          <b>Last portfolio check</b>
          {check
            ? <span className={`paper-risk-state tone-${check.approved ? "primary" : "danger"}`}>{check.verdict}{check.failed_checks.length ? ` · ${check.failed_checks.join(", ")}` : ""} · {check.decided_at.replace("T", " ").slice(0, 19)}</span>
            : <Unmeasured label="未运行" reason={risk.reasons.lastPortfolioCheck ?? risk.reasons.riskEngine} />}
        </span>
        {variant === "overview" ? <Link className="paper-risk-link" to="/t-plus-one">T+1 执行与证据 <ArrowRight size={13} /></Link> : <Link className="paper-risk-link" to="/risk">Risk Manager <ArrowRight size={13} /></Link>}
      </footer>
    </section>
  );
}

export function PaperAccountRiskCard({ variant = "overview" }: { variant?: "overview" | "execution" }): JSX.Element {
  const query = usePaperAccount();
  if (query.isLoading) {
    return (
      <section className="atlas-surface paper-risk-card" aria-label="Paper Account & Risk" aria-busy="true">
        <header className="paper-risk-head"><div><span className="atlas-eyebrow">PAPER ACCOUNT · VENUE RISK STATE</span><h2>Paper Account &amp; Risk</h2><p role="status">正在读取 paper venue 账户与风险引擎状态；加载中不解释为健康或异常。</p></div></header>
      </section>
    );
  }
  const account = query.data?.data;
  if (query.isError || !account || !account.riskState || !account.accountIdentity) {
    return (
      <section className="atlas-surface paper-risk-card" data-rail="danger" aria-label="Paper Account & Risk">
        <header className="paper-risk-head">
          <div>
            <span className="atlas-eyebrow">PAPER ACCOUNT · VENUE RISK STATE</span>
            <h2>Paper Account &amp; Risk</h2>
            <p role="alert">无法读取 /api/paper/account：{query.error?.message ?? "响应缺少 accountIdentity / riskState"}。账户身份、回撤与 kill switch 状态均为未知——不按“未触发”处理。</p>
          </div>
          <div className="paper-risk-chips"><Chip tone="warning">KILL UNKNOWN</Chip>{query.isError ? <button type="button" className="atlas-action" onClick={() => void query.refetch()}>重新读取</button> : null}</div>
        </header>
      </section>
    );
  }
  return <PaperAccountRiskView account={account} variant={variant} />;
}
