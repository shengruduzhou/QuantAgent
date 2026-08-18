import { useMemo } from "react";
import {
  CheckCircle,
  Clock,
  Database,
  GitBranch,
  Lock,
  ShieldCheck,
  Stack,
  WarningCircle,
} from "@phosphor-icons/react";
import { StateView } from "../../components/StateView";
import { useApi } from "../../hooks/useApi";
import {
  ActionableState,
  TruthNotice,
  WorkbenchHeader,
  WorkbenchMetricStrip,
  WorkbenchPanel,
  type WorkbenchMetric,
} from "../workbench/InstitutionalWorkbench";

interface ShadowStatus {
  status: string;
  reason?: string;
  decision?: string;
  validDays?: number;
  requiredDays?: number;
  validDates?: string[];
  excludedDates?: Array<{ date: string; reason: string }>;
  nextExpectedValidDate?: string | null;
  ledgerChainValid?: boolean;
  ledgerRecordsTotal?: number;
  fidelityCertificatePasses?: boolean;
  fidelityCertificateHash?: string | null;
  unblindOrNonRoutineAccesses?: number;
  certificateWritten?: boolean;
}

interface S4Status {
  status: string;
  reason?: string;
  decision?: string;
  exactReproduction?: boolean;
  deterministic?: boolean;
  archivedInputsComplete?: boolean;
  refitCutoffsReplayed?: number;
  semanticsChanged?: boolean;
  freshAccess?: boolean;
  reverified?: boolean;
  codeOrTraceHashChanged?: boolean | null;
}

interface BoardCoverage {
  covered?: number;
  total?: number;
}

interface U0Status {
  status: string;
  reason?: string;
  dataReadinessState?: string;
  trainingPermitted?: boolean;
  gatePass?: Record<string, boolean>;
  missingEvidence?: string[];
  evidenceSources?: Record<string, unknown>;
  coverageByBoard?: Record<string, BoardCoverage>;
  coverageByStatus?: Record<string, BoardCoverage>;
  boardsAbsent?: string[];
  coveredSecurities?: number;
  masterSecurities?: number;
  coverageShare?: number;
  notYetAcquired?: number;
  identity?: {
    securities?: number;
    bseCurrent920?: number;
    bseLegacyCodes?: number;
    delistedInMaster?: number;
    symbolNormalisation?: string;
  };
  provider?: {
    servingProvidersByFamily?: Record<string, string[]>;
    familiesWithoutProvider?: string[];
    fallbackProvidersExercised?: boolean;
    fallbackSymbolsServed?: number;
    environmentBlockers?: Array<{ provider?: string; detail?: string }>;
  };
  quality?: {
    verdicts?: Record<string, string>;
    failures?: string[];
    notRun?: string[];
    adjustmentMethod?: string;
    volumeUnit?: string;
    amountUnit?: string;
    amountCoverage?: number;
  };
  pitFieldAvailability?: Record<string, string>;
  blockedPitFields?: string[];
  suspensionCoverageWindow?: string[] | null;
  panel?: {
    sha256?: string;
    rows?: number;
    symbols?: number;
    dateRange?: (string | null)[];
    sessionGapsSuspended?: number;
    sessionGapsUnexplained?: number;
    sessionGapsProviderTruncated?: number;
    ohlcViolationsQuarantined?: number;
    servingProviderCounts?: Record<string, number>;
  };
}

interface AshareFoundationStatus {
  status: string;
  reason?: string;
  capability?: {
    probes?: number;
    supportedProbes?: number;
    providersWithAnySupport?: string[];
    servingProvidersByFamily?: Record<string, string[]>;
    familiesWithoutAnyProvider?: string[];
    blockers?: Array<{ provider?: string; dataset_family?: string; status?: string; detail?: string }>;
    environment?: { platform?: string; egress?: string };
  };
  securityMaster?: {
    securities?: number;
    byBoard?: Record<string, number>;
    byStatus?: Record<string, number>;
    currentStNames?: number;
    delistingDateCoverage?: number;
  };
  intraday?: {
    frequencyMinutes?: number;
    symbolsWithBars?: number;
    rows?: number;
    symbolSessions?: number;
    servingProviders?: Record<string, number>;
    depthLimitation?: string;
  };
  adjustmentForensics?: {
    results?: Array<{ label?: string; events_tested?: number; sign_agreement?: number; verdict?: string }>;
  };
  validation?: {
    panelRows?: number;
    panelSymbols?: number;
    dateRange?: string[];
    verdicts?: Record<string, number>;
  };
}

interface U0BarPitStatus {
  status: string;
  barReadiness?: {
    decision?: string;
    gatePass?: Record<string, boolean>;
    coveredByBoard?: Record<string, number>;
    boardsAbsent?: string[];
    fetchableBacklog?: number;
    panelSha256?: string;
  };
  strictPitReadiness?: {
    decision?: string;
    trainingPermitted?: boolean;
    blockedPitFields?: string[];
  };
  pitSourceAudit?: Record<string, string>;
  tickflowBenchmark?: {
    sdkVersion?: string;
    count10000Works?: boolean;
    batchEntitled?: boolean;
    measuredRatePerMin?: number;
    recommendedPath?: string;
    old100BarCause?: string;
  };
  bseIdentity?: {
    decision?: string;
    authoritativeCount?: number;
    masterCount?: number;
    truePlaceholders?: string[];
    missingFromMaster?: string[];
  };
  pitMetadataSourcing?: {
    closedFields?: string[];
    blockedFields?: string[];
    delistingDatesSourced?: number;
  };
  tickflowEntitlement?: Record<string, unknown>;
  reconciliation?: {
    supplementalAdditions?: number;
    supplementalSymbols?: string[];
    dualIdentityCollisions?: number;
    starCovered?: number;
    starTotal?: number;
  };
}

interface LineageStatus {
  status: string;
  reason?: string;
  headCommit?: string;
  originMainCommit?: string;
  headEqualsOriginMain?: boolean;
  h030RemotelyRecoverable?: boolean;
  overlappingFiles?: string[];
  expectedConflictAreas?: string[];
  integrationBranch?: string;
}

interface GovernedCommand {
  commandId: string;
  type: string;
  requiresNetwork: boolean;
  parameters: string[];
}

interface GovernanceStatus {
  shadow: ShadowStatus;
  s4: S4Status;
  u0: U0Status;
  u0BarPit?: U0BarPitStatus;
  ashareFoundation?: AshareFoundationStatus;
  lineage: LineageStatus;
  governedCommands: GovernedCommand[];
  blinding: string;
}

function boolTone(value: boolean | undefined): "positive" | "danger" | "neutral" {
  if (value === true) return "positive";
  if (value === false) return "danger";
  return "neutral";
}

function yesNo(value: boolean | undefined): string {
  if (value === true) return "PASS";
  if (value === false) return "FAIL";
  return "—";
}

/**
 * Audit counts must never fall back to `0`.
 *
 * A governance panel that prints "越权/解密访问 0" for a field the status
 * artifact never reported reads as "audited and clean" — which is exactly the
 * DEF-023 shape (never audited recorded as clean). Absent stays absent.
 */
function auditCount(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "未测量";
  return value.toLocaleString();
}

/** Same rule for a measured `covered / total` pair. */
function auditRatio(
  covered: number | null | undefined,
  total: number | null | undefined,
  separator = " / ",
): string {
  if (typeof covered !== "number" || Number.isNaN(covered)) return "未测量";
  if (typeof total !== "number" || Number.isNaN(total)) return "未测量";
  return `${covered}${separator}${total}`;
}

export function GovernancePage(): JSX.Element {
  const query = useApi<GovernanceStatus>(["governance"], "/governance/status", undefined, {
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  const data = query.data?.data;

  const metrics = useMemo<WorkbenchMetric[]>(() => {
    if (!data) return [];
    const shadow = data.shadow;
    const s4 = data.s4;
    const u0 = data.u0;
    const lineage = data.lineage;
    return [
      {
        label: "影子有效交易日",
        value: shadow.status === "ready" ? `${shadow.validDays ?? 0} / ${shadow.requiredDays ?? 7}` : "不可用",
        detail: shadow.decision ?? shadow.reason ?? "shadow registry",
        tone: shadow.certificateWritten ? "positive" : "info",
        icon: Clock,
      },
      {
        label: "Track-F 账本链",
        value: shadow.ledgerChainValid === true ? "VALID" : shadow.ledgerChainValid === false ? "BROKEN" : "—",
        detail: `${auditCount(shadow.ledgerRecordsTotal)} 条记录 · 越权访问 ${auditCount(shadow.unblindOrNonRoutineAccesses)}`,
        tone: boolTone(shadow.ledgerChainValid),
        icon: Lock,
      },
      {
        label: "保真证书",
        value: shadow.fidelityCertificateHash ?? "—",
        detail: shadow.fidelityCertificatePasses ? "passes" : "not passing",
        tone: boolTone(shadow.fidelityCertificatePasses),
        icon: ShieldCheck,
      },
      {
        label: "S4 批量重放",
        value: s4.status === "ready" ? (s4.decision ?? "—") : "不可用",
        detail: s4.status === "ready" ? `${auditCount(s4.refitCutoffsReplayed)} 个 cutoff · 确定性 ${yesNo(s4.deterministic)}` : (s4.reason ?? ""),
        tone: s4.decision === "S4_BATCH_REPLAY_READY" ? "positive" : "warning",
        icon: CheckCircle,
      },
      {
        label: "U0 数据就绪",
        value: u0.status === "ready" ? (u0.dataReadinessState ?? "—") : "不可用",
        detail: u0.status === "ready" ? `训练许可 ${yesNo(u0.trainingPermitted)}` : (u0.reason ?? ""),
        tone: u0.trainingPermitted ? "positive" : "danger",
        icon: Database,
      },
      {
        label: "分支血缘",
        value: lineage.headEqualsOriginMain ? "HEAD=origin/main" : "分叉",
        detail: `冲突区 ${lineage.expectedConflictAreas?.length ?? 0} · H-030 可远程恢复 ${yesNo(lineage.h030RemotelyRecoverable)}`,
        tone: lineage.headEqualsOriginMain ? "positive" : "warning",
        icon: GitBranch,
      },
    ];
  }, [data]);

  if (query.isLoading) {
    return <StateView state="loading" detail="正在读取运营治理清单。" />;
  }
  if (query.isError || !data) {
    return (
      <StateView
        state="error"
        title="治理状态不可用"
        detail="无法读取 /api/governance/status；确认 quant_api 正在运行且 runtime manifests 已生成。"
      />
    );
  }

  const { shadow, s4, u0, lineage, governedCommands } = data;
  const trainingReady = u0.trainingPermitted === true;

  return (
    <div className="iw-workbench governance-workbench">
      <WorkbenchHeader
        eyebrow="OPERATIONS / GOVERNANCE"
        title="运营治理总览"
        description="H-031 运营治理面板：影子测试进度、S4 批量重放、U0 全宇宙数据就绪与分支血缘。只展示存在性与关卡级字段，绝不展示候选级表现。"
        asOf={shadow.certificateWritten ? "FROZEN_BLIND_PAPER_ACTIVE" : (shadow.decision ?? "runtime")}
        context="existence + gate level only"
      />
      <WorkbenchMetricStrip metrics={metrics} />

      <TruthNotice tone="warning">
        本面板不解密、不读取、不展示任何候选级净值、收益、回撤或夏普指标。所有数字均为存在性计数或关卡布尔值。
      </TruthNotice>

      <WorkbenchPanel eyebrow="TRACK F" title="盲化前向影子测试" meta={shadow.decision ?? "unavailable"}>
        {shadow.status !== "ready" ? (
          <ActionableState
            title="影子注册表尚未生成"
            detail={shadow.reason ?? "运行 validate-shadow-days 治理命令以生成注册表。"}
            icon={WarningCircle}
            tone="warning"
            compact
          />
        ) : (
          <div className="governance-grid">
            <dl className="governance-facts">
              <div><dt>有效交易日</dt><dd>{shadow.validDays} / {shadow.requiredDays}</dd></div>
              <div><dt>有效日期</dt><dd>{shadow.validDates?.length ? shadow.validDates.join(", ") : "—"}</dd></div>
              <div><dt>下一个预期有效日</dt><dd>{shadow.nextExpectedValidDate ?? "—"}</dd></div>
              <div><dt>账本链</dt><dd>{shadow.ledgerChainValid ? "VALID" : "BROKEN"}（{shadow.ledgerRecordsTotal} 条）</dd></div>
              <div><dt>越权/解密访问</dt><dd>{auditCount(shadow.unblindOrNonRoutineAccesses)}</dd></div>
              <div><dt>证书</dt><dd>{shadow.certificateWritten ? "已签发" : "累积中（未早签）"}</dd></div>
            </dl>
            <div className="governance-excluded">
              <h3>被排除的日期</h3>
              {shadow.excludedDates?.length ? (
                <ul>
                  {shadow.excludedDates.map((item) => (
                    <li key={item.date}><strong>{item.date}</strong><span>{item.reason}</span></li>
                  ))}
                </ul>
              ) : (
                <p>无被排除日期。</p>
              )}
            </div>
          </div>
        )}
      </WorkbenchPanel>

      <WorkbenchPanel eyebrow="TRACK S4" title="冻结 S4 批量重放证书" meta={s4.decision ?? "unavailable"}>
        {s4.status !== "ready" ? (
          <ActionableState title="S4 证书缺失" detail={s4.reason ?? "运行 certify-s4-batch-replay。"} icon={WarningCircle} tone="warning" compact />
        ) : (
          <dl className="governance-facts">
            <div><dt>判定</dt><dd>{s4.decision}</dd></div>
            <div><dt>逐 cutoff 精确复现</dt><dd>{yesNo(s4.exactReproduction)}</dd></div>
            <div><dt>双跑确定性</dt><dd>{yesNo(s4.deterministic)}</dd></div>
            <div><dt>归档输入完整</dt><dd>{yesNo(s4.archivedInputsComplete)}</dd></div>
            <div><dt>重放 refit cutoff 数</dt><dd>{auditCount(s4.refitCutoffsReplayed)}</dd></div>
            <div><dt>语义变化 / FRESH 访问</dt><dd>{yesNo(s4.semanticsChanged)} / {yesNo(s4.freshAccess)}</dd></div>
            <div><dt>代码或 trace 哈希变化</dt><dd>{s4.codeOrTraceHashChanged === null ? "—" : yesNo(s4.codeOrTraceHashChanged)}</dd></div>
          </dl>
        )}
      </WorkbenchPanel>

      <WorkbenchPanel eyebrow="TRACK U0" title="全宇宙数据就绪" meta={u0.dataReadinessState ?? "unavailable"}>
        {u0.status !== "ready" ? (
          <ActionableState title="U0 就绪证书缺失" detail={u0.reason ?? "运行 audit-u0-full-universe。"} icon={WarningCircle} tone="warning" compact />
        ) : (
          <div className="governance-grid">
            <dl className="governance-facts">
              <div><dt>数据就绪状态</dt><dd>{u0.dataReadinessState}</dd></div>
              <div><dt>训练许可</dt><dd>{yesNo(u0.trainingPermitted)}</dd></div>
              <div><dt>关卡</dt><dd>{Object.entries(u0.gatePass ?? {}).map(([g, ok]) => `${g}:${ok ? "PASS" : "FAIL"}`).join(" · ")}</dd></div>
              <div><dt>覆盖证券</dt><dd>{u0.coveredSecurities ?? "—"} / {u0.masterSecurities ?? "—"}{u0.coverageShare != null ? `（${(u0.coverageShare * 100).toFixed(1)}%）` : ""}</dd></div>
              <div><dt>尚未采集</dt><dd>{u0.notYetAcquired ?? "—"}</dd></div>
              <div><dt>缺席板块</dt><dd>{u0.boardsAbsent?.length ? u0.boardsAbsent.join(", ") : "无"}</dd></div>
              <div><dt>退市覆盖（生存者偏差）</dt><dd>{u0.coverageByStatus?.delisted ? `${auditRatio(u0.coverageByStatus.delisted.covered, u0.coverageByStatus.delisted.total, "/")} 有行情` : "未测量"}</dd></div>
              <div><dt>复权口径</dt><dd>{u0.quality?.adjustmentMethod ?? "—"}</dd></div>
              <div><dt>单位</dt><dd>成交量 {u0.quality?.volumeUnit ?? "—"} · 成交额 {u0.quality?.amountUnit ?? "—"}{u0.quality?.amountCoverage != null ? `（成交额覆盖 ${(u0.quality.amountCoverage * 100).toFixed(1)}%）` : ""}</dd></div>
              <div><dt>校验失败项</dt><dd>{u0.quality?.failures?.length ? u0.quality.failures.join(", ") : "无"}</dd></div>
              <div><dt>未执行校验</dt><dd>{u0.quality?.notRun?.length ? u0.quality.notRun.join(", ") : "无"}</dd></div>
              <div><dt>回退供应商已实测</dt><dd>{yesNo(u0.provider?.fallbackProvidersExercised)}（{auditCount(u0.provider?.fallbackSymbolsServed)} 票）</dd></div>
              <div><dt>缺口分类（停牌 / 供应商截断 / 无法解释）</dt><dd>{u0.panel?.sessionGapsSuspended ?? "—"} / {u0.panel?.sessionGapsProviderTruncated ?? "—"} / {u0.panel?.sessionGapsUnexplained ?? "—"}</dd></div>
              <div><dt>OHLC 矛盾行（已隔离）</dt><dd>{u0.panel?.ohlcViolationsQuarantined ?? "—"}</dd></div>
              <div><dt>面板</dt><dd>{u0.panel?.rows?.toLocaleString() ?? "—"} 行 · {u0.panel?.symbols ?? "—"} 票 · {u0.panel?.dateRange?.[0] ?? "—"} → {u0.panel?.dateRange?.[1] ?? "—"}</dd></div>
              <div><dt>缺失证据</dt><dd>{u0.missingEvidence?.length ? u0.missingEvidence.join(", ") : "无"}</dd></div>
            </dl>
            <div className="governance-boards">
              <h3>板块覆盖（已覆盖 / 总数）</h3>
              <ul>
                {Object.entries(u0.coverageByBoard ?? {}).map(([board, row]) => (
                  <li key={board}><strong>{board}</strong><span>{auditRatio(row?.covered, row?.total)}</span></li>
                ))}
              </ul>
              <h3>PIT 执行字段</h3>
              <ul className="governance-pit">
                {Object.entries(u0.pitFieldAvailability ?? {}).map(([field, state]) => (
                  <li key={field} className={String(state).includes("BLOCKED") ? "blocked" : ""}>
                    <strong>{field}</strong><span>{state}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
        <div className="governance-train-gate">
          <button type="button" className="iw-primary-action" disabled={!trainingReady} aria-disabled={!trainingReady}>
            全宇宙训练
          </button>
          {!trainingReady ? (
            <small>
              该控制保持禁用，直至选择一份已验证的 FULL_UNIVERSE_DATA_READY manifest（当前：{u0.dataReadinessState ?? "未知"}）。
            </small>
          ) : (
            <small>已验证 FULL_UNIVERSE_DATA_READY；下一代盲测时钟须在候选冻结后另行启动。</small>
          )}
        </div>
      </WorkbenchPanel>

      {data.ashareFoundation && data.ashareFoundation.status === "ready" ? (
        <WorkbenchPanel
          eyebrow="A股数据底座"
          title="供应商能力 / 授权矩阵与采集溯源"
          meta={`${auditRatio(data.ashareFoundation.capability?.supportedProbes, data.ashareFoundation.capability?.probes, "/")} 实网探针 SUPPORTED`}
        >
          <div className="governance-grid">
            <dl className="governance-facts">
              <div><dt>运行环境出网</dt><dd>{data.ashareFoundation.capability?.environment?.egress ?? "—"}</dd></div>
              <div><dt>有实测支持的供应商</dt><dd>{data.ashareFoundation.capability?.providersWithAnySupport?.join(", ") || "—"}</dd></div>
              <div><dt>证券主表</dt><dd>{data.ashareFoundation.securityMaster?.securities ?? "—"} 只 · 退市 {data.ashareFoundation.securityMaster?.byStatus?.delisted ?? "—"} · 当前 ST {data.ashareFoundation.securityMaster?.currentStNames ?? "—"}</dd></div>
              <div><dt>分钟数据</dt><dd>{data.ashareFoundation.intraday ? `${auditCount(data.ashareFoundation.intraday.symbolsWithBars)} 票 · ${auditCount(data.ashareFoundation.intraday.rows)} 根 ${data.ashareFoundation.intraday.frequencyMinutes ?? "?"}分钟条` : "未采集"}</dd></div>
              <div><dt>分钟深度限制</dt><dd>{data.ashareFoundation.intraday?.depthLimitation ?? "—"}</dd></div>
              <div><dt>校验结论</dt><dd>{Object.entries(data.ashareFoundation.validation?.verdicts ?? {}).map(([k, v]) => `${k}:${v}`).join(" · ") || "—"}</dd></div>
            </dl>
            <div className="governance-boards">
              <h3>各数据族的可用供应商（实测）</h3>
              <ul className="governance-pit">
                {Object.entries(data.ashareFoundation.capability?.servingProvidersByFamily ?? {}).map(([family, providers]) => (
                  <li key={family} className={providers?.length ? "" : "blocked"}>
                    <strong>{family}</strong><span>{providers?.length ? providers.join(", ") : "无"}</span>
                  </li>
                ))}
              </ul>
              <h3>外部阻塞（授权 / 环境）</h3>
              <ul className="governance-pit">
                {(data.ashareFoundation.capability?.blockers ?? []).slice(0, 12).map((blocker, index) => (
                  <li key={`${blocker.provider}-${blocker.dataset_family}-${index}`} className="blocked">
                    <strong>{blocker.provider} · {blocker.dataset_family}</strong><span>{blocker.status}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          {data.ashareFoundation.adjustmentForensics?.results?.length ? (
            <>
              <h3>复权口径取证（除权日因子回放）</h3>
              <ul className="governance-pit">
                {data.ashareFoundation.adjustmentForensics.results.map((row) => (
                  <li key={row.label} className={row.verdict === "RAW" ? "" : "blocked"}>
                    <strong>{row.label}</strong>
                    <span>{row.verdict} · 符号一致率 {row.sign_agreement ?? "未测量"}（{auditCount(row.events_tested)} 事件）</span>
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </WorkbenchPanel>
      ) : null}

      {data.u0BarPit && data.u0BarPit.status === "ready" ? (
        <WorkbenchPanel
          eyebrow="U0 · H-032B"
          title="条数据就绪 vs 严格 PIT 就绪（分列）"
          meta={data.u0BarPit.tickflowBenchmark?.count10000Works ? "TickFlow count=10000 native" : "TickFlow"}
        >
          <div className="governance-grid">
            <dl className="governance-facts">
              <div><dt>条数据就绪（bar）</dt><dd>{data.u0BarPit.barReadiness?.decision ?? "—"}</dd></div>
              <div><dt>bar 关卡</dt><dd>{Object.entries(data.u0BarPit.barReadiness?.gatePass ?? {}).map(([g, ok]) => `${g}:${ok ? "PASS" : "FAIL"}`).join(" · ") || "—"}</dd></div>
              <div><dt>严格 PIT 就绪</dt><dd>{data.u0BarPit.strictPitReadiness?.decision ?? "—"}</dd></div>
              <div><dt>训练许可（PIT 门）</dt><dd>{yesNo(data.u0BarPit.strictPitReadiness?.trainingPermitted)}</dd></div>
              <div><dt>PIT 阻塞字段</dt><dd>{data.u0BarPit.strictPitReadiness?.blockedPitFields?.length ? data.u0BarPit.strictPitReadiness.blockedPitFields.join(", ") : "无"}</dd></div>
              <div><dt>bar 缺席板块</dt><dd>{data.u0BarPit.barReadiness?.boardsAbsent?.length ? data.u0BarPit.barReadiness.boardsAbsent.join(", ") : "无"}</dd></div>
            </dl>
            <div className="governance-boards">
              <h3>TickFlow 能力基准</h3>
              <dl className="governance-facts">
                <div><dt>SDK</dt><dd>{data.u0BarPit.tickflowBenchmark?.sdkVersion ?? "—"}</dd></div>
                <div><dt>count=10000</dt><dd>{yesNo(data.u0BarPit.tickflowBenchmark?.count10000Works)}</dd></div>
                <div><dt>batch 授权</dt><dd>{yesNo(data.u0BarPit.tickflowBenchmark?.batchEntitled)}</dd></div>
                <div><dt>实测限速</dt><dd>{data.u0BarPit.tickflowBenchmark?.measuredRatePerMin ?? "—"} req/min</dd></div>
              </dl>
              <h3>BSE 身份</h3>
              <dl className="governance-facts">
                <div><dt>判定</dt><dd>{data.u0BarPit.bseIdentity?.decision ?? "—"}</dd></div>
                <div><dt>权威/master</dt><dd>{data.u0BarPit.bseIdentity?.authoritativeCount ?? "—"} / {data.u0BarPit.bseIdentity?.masterCount ?? "—"}</dd></div>
                <div><dt>真占位码</dt><dd>{data.u0BarPit.bseIdentity?.truePlaceholders?.length ? data.u0BarPit.bseIdentity.truePlaceholders.join(",") : "0（无占位）"}</dd></div>
              </dl>
            </div>
          </div>
          <dl className="governance-facts">
            <div><dt>PIT 已闭合字段</dt><dd>{data.u0BarPit.pitMetadataSourcing?.closedFields?.length ? data.u0BarPit.pitMetadataSourcing.closedFields.join(", ") : "—"}</dd></div>
            <div><dt>退市日已取</dt><dd>{data.u0BarPit.pitMetadataSourcing?.delistingDatesSourced ?? "—"}</dd></div>
            <div><dt>PIT 仍阻塞</dt><dd>{data.u0BarPit.pitMetadataSourcing?.blockedFields?.length ? data.u0BarPit.pitMetadataSourcing.blockedFields.join(", ") : "无"}</dd></div>
            <div><dt>身份补录 / 双身份冲突</dt><dd>{data.u0BarPit.reconciliation?.supplementalAdditions ?? "—"} / {data.u0BarPit.reconciliation?.dualIdentityCollisions ?? "—"}</dd></div>
            <div><dt>STAR 覆盖</dt><dd>{data.u0BarPit.reconciliation?.starCovered ?? "—"} / {data.u0BarPit.reconciliation?.starTotal ?? "—"}</dd></div>
          </dl>
          <ul className="governance-pit">
            {Object.entries(data.u0BarPit.pitSourceAudit ?? {}).map(([field, state]) => (
              <li key={field} className={String(state).includes("REQUIRED") || String(state).includes("BLOCKED") ? "blocked" : ""}>
                <strong>{field}</strong><span>{state}</span>
              </li>
            ))}
          </ul>
          <TruthNotice tone="info">
            条数据就绪仅解锁 smoke 测试（数据集构建 / 特征物化 / 内存基准 / CLI 校验）；正式训练需严格 PIT 就绪 = FULL_UNIVERSE_DATA_READY。
          </TruthNotice>
        </WorkbenchPanel>
      ) : null}

      <WorkbenchPanel eyebrow="TRACK I" title="分支与架构血缘" meta={lineage.integrationBranch ?? "unavailable"}>
        {lineage.status !== "ready" ? (
          <ActionableState title="血缘报告缺失" detail={lineage.reason ?? "生成 runtime/reports/h031/branch_lineage.json。"} icon={WarningCircle} tone="warning" compact />
        ) : (
          <dl className="governance-facts">
            <div><dt>HEAD</dt><dd>{lineage.headCommit?.slice(0, 12) ?? "—"}</dd></div>
            <div><dt>origin/main</dt><dd>{lineage.originMainCommit?.slice(0, 12) ?? "—"}</dd></div>
            <div><dt>HEAD = origin/main</dt><dd>{yesNo(lineage.headEqualsOriginMain)}</dd></div>
            <div><dt>H-030 可远程恢复</dt><dd>{yesNo(lineage.h030RemotelyRecoverable)}</dd></div>
            <div><dt>重叠文件</dt><dd>{lineage.overlappingFiles?.length ? lineage.overlappingFiles.join(", ") : "无"}</dd></div>
            <div><dt>预期冲突区</dt><dd>{lineage.expectedConflictAreas?.length ? lineage.expectedConflictAreas.join(", ") : "无"}</dd></div>
          </dl>
        )}
      </WorkbenchPanel>

      <WorkbenchPanel eyebrow="GOVERNED COMMANDS" title="受治理运营命令" meta={`${governedCommands.length} 个已登记 · 无自由 shell`}>
        <ul className="governance-commands">
          {governedCommands.map((command) => (
            <li key={command.commandId}>
              <strong>{command.commandId}</strong>
              <span className="governance-command-type">{command.type}</span>
              {command.requiresNetwork ? <span className="governance-network"><Stack size={12} weight="duotone" /> 需显式网络确认</span> : <span className="governance-local">本地</span>}
              <small>{command.parameters.length ? command.parameters.join(", ") : "无参数"}</small>
            </li>
          ))}
        </ul>
        <TruthNotice tone="info">
          所有命令经 allowlisted JobRunner 提交（受限 Runtime 路径、可取消、可审计）。取消经任务中心 POST /api/jobs/&#123;id&#125;/cancel。
        </TruthNotice>
      </WorkbenchPanel>
    </div>
  );
}
