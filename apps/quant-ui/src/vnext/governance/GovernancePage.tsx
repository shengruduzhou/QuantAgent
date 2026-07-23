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

interface U0Status {
  status: string;
  reason?: string;
  dataReadinessState?: string;
  trainingPermitted?: boolean;
  gatePass?: Record<string, boolean>;
  coverageByBoard?: Record<string, number>;
  boardsAbsent?: string[];
  blockedByData?: number;
  coverageBacklogFetchable?: number;
  retryClassCounts?: Record<string, number>;
  providerFailures?: number;
  pitGate?: Record<string, string>;
  pitFieldAvailability?: Record<string, string>;
  survivorshipBias?: {
    delisted_total?: number;
    delisted_with_bar_history?: number;
    delisted_with_delisting_date?: number;
    delisted_fraction_of_master?: number;
  };
  starBseProbe?: Record<string, string>;
  coveredBarHistory?: number;
  backfill?: {
    masterSecurities?: number;
    panelSymbols?: number;
    missingSymbols?: number;
    stagedBackfillFiles?: number;
  } | null;
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
        detail: `${shadow.ledgerRecordsTotal ?? 0} 条记录 · 越权访问 ${shadow.unblindOrNonRoutineAccesses ?? 0}`,
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
        detail: s4.status === "ready" ? `${s4.refitCutoffsReplayed ?? 0} 个 cutoff · 确定性 ${yesNo(s4.deterministic)}` : (s4.reason ?? ""),
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
              <div><dt>越权/解密访问</dt><dd>{shadow.unblindOrNonRoutineAccesses ?? 0}</dd></div>
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
            <div><dt>重放 refit cutoff 数</dt><dd>{s4.refitCutoffsReplayed ?? 0}</dd></div>
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
              <div><dt>覆盖行情（票数）</dt><dd>{u0.coveredBarHistory ?? "—"}</dd></div>
              <div><dt>BLOCKED_BY_DATA</dt><dd>{u0.blockedByData ?? "—"}</dd></div>
              <div><dt>可取回未回填（backlog）</dt><dd>{u0.coverageBacklogFetchable ?? "—"}</dd></div>
              <div><dt>供应商空响应</dt><dd>{u0.providerFailures ?? "—"}</dd></div>
              <div><dt>缺席板块</dt><dd>{u0.boardsAbsent?.length ? u0.boardsAbsent.join(", ") : "无"}</dd></div>
              <div><dt>退市生存者偏差</dt><dd>{u0.survivorshipBias?.delisted_total != null ? `${u0.survivorshipBias.delisted_with_bar_history ?? 0}/${u0.survivorshipBias.delisted_total} 有行情, ${u0.survivorshipBias.delisted_with_delisting_date ?? 0} 有退市日` : "—"}</dd></div>
              <div><dt>STAR/BSE 探针</dt><dd>{u0.starBseProbe && Object.keys(u0.starBseProbe).length ? Object.entries(u0.starBseProbe).map(([b, d]) => `${b}:${d}`).join(" · ") : "—"}</dd></div>
              <div><dt>回填进度（staged）</dt><dd>{u0.backfill?.stagedBackfillFiles ?? "—"} · 缺 {u0.backfill?.missingSymbols ?? "—"}</dd></div>
            </dl>
            <div className="governance-boards">
              <h3>已覆盖板块</h3>
              <ul>
                {Object.entries(u0.coverageByBoard ?? {}).map(([board, count]) => (
                  <li key={board}><strong>{board}</strong><span>{count}</span></li>
                ))}
              </ul>
              <h3>PIT 执行字段</h3>
              <ul className="governance-pit">
                {Object.entries(u0.pitGate ?? {}).map(([field, state]) => (
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
