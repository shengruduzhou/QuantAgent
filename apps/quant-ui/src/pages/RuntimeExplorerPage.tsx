import { useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  ArrowClockwise,
  Broom,
  Check,
  Database,
  File,
  MagnifyingGlass,
  ShieldCheck,
  Trash,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { useSearchParams } from "react-router-dom";
import { apiPost } from "../api/client";
import type {
  CleanupResult,
  Page,
  RuntimeArtifact,
  RuntimeCleanupAnalysis,
} from "../api/types";
import { useApi } from "../hooks/useApi";
import { MetricCard } from "../components/MetricCard";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatBytes, formatDate } from "../utils/format";

const kinds = ["", "backtest", "model", "prediction", "factor", "selection", "risk", "do_t", "log", "dataset", "report"];

type RuntimeTab = "artifacts" | "cleanup";

export function RuntimeExplorerPage(): JSX.Element {
  const [searchParams] = useSearchParams();
  const [tab, setTab] = useState<RuntimeTab>("artifacts");
  const [kind, setKind] = useState("");
  const [query, setQuery] = useState(searchParams.get("query") ?? "");
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState("");
  const [refresh, setRefresh] = useState(false);
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [cleanupError, setCleanupError] = useState("");
  const [cleanupResult, setCleanupResult] = useState<CleanupResult | null>(null);
  const deferredQuery = useDeferredValue(query);
  const artifacts = useApi<Page<RuntimeArtifact>>(
    ["runtime-index", kind, deferredQuery, page, refresh],
    "/system/runtime-index",
    { kind, query: deferredQuery, page, pageSize: 100, refresh },
  );
  const preview = useApi<Record<string, unknown>>(
    ["runtime-preview", selectedId],
    selectedId ? `/system/runtime-index/${selectedId}/preview` : null,
    { limit: 100 },
  );
  const cleanup = useApi<RuntimeCleanupAnalysis>(
    ["runtime-cleanup", refresh],
    tab === "cleanup" ? "/system/runtime-cleanup" : null,
  );
  const data = artifacts.data?.data;
  const selectedArtifact = useMemo(
    () => data?.items.find((artifact) => artifact.id === selectedId) ?? null,
    [data?.items, selectedId],
  );

  useEffect(() => {
    const defaults = cleanup.data?.data.candidates.filter((item) => item.safeDefault).map((item) => item.id);
    if (defaults?.length && !selectedCandidates.length && !cleanupResult) {
      setSelectedCandidates(defaults);
    }
  }, [cleanup.data?.data.candidates, cleanupResult, selectedCandidates.length]);

  const selectedSize = useMemo(() => (
    (cleanup.data?.data.candidates ?? [])
      .filter((item) => selectedCandidates.includes(item.id))
      .reduce((sum, item) => sum + item.sizeBytes, 0)
  ), [cleanup.data?.data.candidates, selectedCandidates]);

  const runCleanup = async (): Promise<void> => {
    setCleanupBusy(true);
    setCleanupError("");
    try {
      const result = await apiPost<CleanupResult>("/system/runtime-cleanup", {
        candidateIds: selectedCandidates,
        confirmation: "DELETE",
      });
      setCleanupResult(result.data);
      setSelectedCandidates([]);
      setConfirmOpen(false);
      setRefresh((value) => !value);
      await artifacts.refetch();
      await cleanup.refetch();
    } catch (error) {
      setCleanupError(error instanceof Error ? error.message : "cleanup failed");
    } finally {
      setCleanupBusy(false);
    }
  };

  return (
    <div className="page runtime-page">
      <section className="runtime-topline">
        <div>
          <span className="page-kicker">RUNTIME OPERATIONS</span>
          <h2>Artifact Explorer & Safe Cleanup</h2>
          <p>统一查看真实产物、解析能力、空间占用和可审计删除候选。</p>
        </div>
        <div className="runtime-tabs">
          <button className={tab === "artifacts" ? "active" : ""} onClick={() => setTab("artifacts")}><Database size={16} /> 产物索引</button>
          <button className={tab === "cleanup" ? "active" : ""} onClick={() => setTab("cleanup")}><Broom size={16} /> 安全清理</button>
        </div>
      </section>

      {tab === "artifacts" ? (
        <>
          <section className="workbench-toolbar runtime-toolbar">
            <label className="runtime-search">
              <span>Artifact 搜索</span>
              <div><MagnifyingGlass size={16} /><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="path / run / model / symbol" /></div>
            </label>
            <label>
              <span>类型</span>
              <select value={kind} onChange={(event) => { setKind(event.target.value); setPage(1); }}>
                {kinds.map((item) => <option value={item} key={item}>{item || "全部类型"}</option>)}
              </select>
            </label>
            <button className="secondary-button" onClick={() => setRefresh((value) => !value)}><ArrowClockwise size={16} /> 重建索引</button>
            <div className="truth-note"><File size={17} /><span>大文件只读取 schema/metadata；preview 有严格行列上限。</span></div>
          </section>

          <section className="runtime-grid">
            <Panel title="Runtime Artifact Index" eyebrow={`${data?.total ?? 0} matched · page ${data?.page ?? 1}`} className="runtime-table-panel">
              {data?.items.length ? (
                <>
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead><tr><th>类型</th><th>文件</th><th>Run / Horizon</th><th>Trust / Validation</th><th className="numeric">大小</th><th>来源时间</th><th>状态</th></tr></thead>
                      <tbody>
                        {data.items.map((artifact) => (
                          <tr
                            key={artifact.id}
                            className={selectedId === artifact.id ? "row-selected" : ""}
                            onClick={() => setSelectedId(artifact.id)}
                            tabIndex={0}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") setSelectedId(artifact.id);
                            }}
                          >
                            <td><span className={`artifact-kind kind-${artifact.kind}`}>{artifact.kind}</span></td>
                            <td className="artifact-path"><strong>{artifact.name}</strong><span>{artifact.path}</span></td>
                            <td><strong>{artifact.runId ?? "—"}</strong><span>{artifact.horizon ?? artifact.extension}</span></td>
                            <td className="artifact-trust-cell">
                              <StatusBadge status={trustBadgeStatus(artifact.trustClass)} label={artifact.trustClass} />
                              <span>{artifact.validationStatus}</span>
                            </td>
                            <td className="numeric mono">{formatBytes(artifact.sizeBytes)}</td>
                            <td className="mono"><strong>{formatDate(artifact.sourceTime ?? artifact.modifiedAt)}</strong><span>{artifact.sourceTime ? "manifest" : "filesystem"}</span></td>
                            <td><StatusBadge status={artifact.status} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="pagination">
                    <button disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>上一页</button>
                    <span>{page} / {Math.max(1, Math.ceil(data.total / data.pageSize))}</span>
                    <button disabled={!data.hasNext} onClick={() => setPage((value) => value + 1)}>下一页</button>
                  </div>
                </>
              ) : <StateView state={artifacts.isLoading ? "loading" : artifacts.isError ? "error" : "empty"} detail={artifacts.error?.message} />}
            </Panel>

            <Panel title="Artifact Contract & Preview" eyebrow="Manifest-aware · fail-closed capabilities" className="runtime-preview-panel">
              {selectedArtifact ? (
                <section className="artifact-contract">
                  <div className="artifact-contract-badges">
                    <StatusBadge status={trustBadgeStatus(selectedArtifact.trustClass)} label={selectedArtifact.trustClass} />
                    <StatusBadge status={validationBadgeStatus(selectedArtifact.validationStatus)} label={selectedArtifact.validationStatus} />
                    <StatusBadge status={selectedArtifact.freshnessStatus} label={selectedArtifact.freshnessStatus} />
                  </div>
                  <dl>
                    <div><dt>Schema</dt><dd>{selectedArtifact.schemaVersion ?? "undeclared"}</dd></div>
                    <div><dt>Manifest</dt><dd>{selectedArtifact.manifestPath ?? "none"}</dd></div>
                    <div><dt>Source time</dt><dd>{formatDate(selectedArtifact.sourceTime ?? selectedArtifact.modifiedAt)}</dd></div>
                    <div><dt>Capabilities</dt><dd>{selectedArtifact.capabilities.join(" · ") || "metadata"}</dd></div>
                  </dl>
                  {selectedArtifact.issues.length ? (
                    <div className="artifact-contract-issues">
                      {selectedArtifact.issues.map((issue) => <span key={`${issue.code}-${issue.path ?? ""}`}><WarningCircle size={13} />{issue.code}: {issue.message}</span>)}
                    </div>
                  ) : null}
                </section>
              ) : null}
              {selectedId ? (
                preview.isLoading ? <StateView state="loading" /> :
                  preview.data ? <pre className="json-view">{JSON.stringify(preview.data.data, null, 2)}</pre> :
                    <StateView state="error" detail={preview.error?.message} />
              ) : <StateView state="empty" detail="点击左侧 artifact 查看安全预览。" />}
            </Panel>
          </section>
        </>
      ) : (
        <section className="cleanup-page">
          <div className="metric-grid metric-grid-4 cleanup-metrics">
            <MetricCard label="Runtime 总体积" value={formatBytes(cleanup.data?.data.runtimeSizeBytes)} icon={Database} />
            <MetricCard label="可清理候选" value={formatBytes(cleanup.data?.data.candidateSizeBytes)} detail={`${cleanup.data?.data.candidates.length ?? 0} candidate groups`} icon={Trash} />
            <MetricCard label="默认安全项" value={formatBytes(cleanup.data?.data.safeDefaultSizeBytes)} tone="positive" detail="test / smoke / superseded UI captures" icon={ShieldCheck} />
            <MetricCard label="已选择" value={formatBytes(selectedSize)} tone={selectedSize > 0 ? "warning" : "neutral"} detail={`${selectedCandidates.length} groups`} icon={Check} />
          </div>

          <div className="cleanup-layout">
            <Panel title="清理候选" eyebrow="每项均由 backend 重新验证路径与保护规则" className="cleanup-candidates-panel">
              {cleanup.isLoading ? <StateView state="loading" /> : cleanup.data?.data.candidates.length ? (
                <div className="cleanup-list">
                  {cleanup.data.data.candidates.map((candidate) => {
                    const checked = selectedCandidates.includes(candidate.id);
                    return (
                      <button
                        key={candidate.id}
                        className={`${checked ? "selected" : ""} ${candidate.requiresExplicit ? "cleanup-review" : ""}`}
                        onClick={() => setSelectedCandidates((current) =>
                          checked ? current.filter((id) => id !== candidate.id) : [...current, candidate.id],
                        )}
                      >
                        <span className={`cleanup-check ${checked ? "checked" : ""}`}>{checked ? <Check size={13} weight="bold" /> : null}</span>
                        <span className="cleanup-copy">
                          <strong>{candidate.label}</strong>
                          <small>{candidate.reason}</small>
                          <em>{candidate.paths.slice(0, 2).join(" · ")}{candidate.paths.length > 2 ? ` · +${candidate.paths.length - 2}` : ""}</em>
                        </span>
                        <span className="cleanup-size"><strong>{formatBytes(candidate.sizeBytes)}</strong><small>{candidate.itemCount} files</small></span>
                        <StatusBadge
                          status={candidate.safeDefault ? "ready" : candidate.requiresExplicit ? "warning" : "partial"}
                          label={candidate.safeDefault ? "默认安全" : candidate.requiresExplicit ? "需人工复核" : "可重建"}
                        />
                      </button>
                    );
                  })}
                </div>
              ) : <StateView state="empty" detail="当前没有可识别清理候选。" />}
              <div className="cleanup-actions">
                <div><ShieldCheck size={18} /><span>canonical raw/silver/manifests 与主模型 registry 永久受保护。</span></div>
                <button className="danger-button" disabled={!selectedCandidates.length} onClick={() => setConfirmOpen(true)}><Trash size={16} /> 删除已选 · {formatBytes(selectedSize)}</button>
              </div>
            </Panel>

            <Panel title="保护范围" eyebrow="Hard backend guardrails" className="cleanup-protected-panel">
              <div className="protected-list">
                {(cleanup.data?.data.protected ?? []).map((path) => <div key={path}><ShieldCheck size={16} /><code>{path}</code></div>)}
              </div>
              {cleanupResult ? (
                <div className="cleanup-result">
                  <Check size={20} />
                  <div>
                    <strong>清理完成 · 释放 {formatBytes(cleanupResult.freedBytes)}</strong>
                    <span>{cleanupResult.auditPath}</span>
                  </div>
                </div>
              ) : null}
              {cleanupError ? <div className="cleanup-error"><WarningCircle size={18} />{cleanupError}</div> : null}
            </Panel>
          </div>
        </section>
      )}

      {confirmOpen ? (
        <div className="modal-overlay" role="presentation" onMouseDown={() => setConfirmOpen(false)}>
          <section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-label="确认清理 runtime" onMouseDown={(event) => event.stopPropagation()}>
            <header><div><WarningCircle size={22} /><span><strong>确认删除 runtime 产物</strong><small>此操作只作用于当前选中的 backend-approved candidates。</small></span></div><button onClick={() => setConfirmOpen(false)}><X size={17} /></button></header>
            <div className="confirm-summary">
              <strong>{selectedCandidates.length} groups</strong>
              <span>{formatBytes(selectedSize)}</span>
            </div>
            <p>删除后会写入 cleanup audit，并自动重建 runtime index。大型训练集候选可能影响历史复现，请确认选择范围。</p>
            <footer>
              <button className="secondary-button" onClick={() => setConfirmOpen(false)}>取消</button>
              <button className="danger-button" disabled={cleanupBusy} onClick={runCleanup}>{cleanupBusy ? "正在清理…" : "确认删除"}</button>
            </footer>
          </section>
        </div>
      ) : null}
    </div>
  );
}

function trustBadgeStatus(trustClass: RuntimeArtifact["trustClass"]): string {
  if (trustClass === "production_ready") return "ready";
  if (trustClass === "contaminated") return "error";
  if (trustClass === "paper_only" || trustClass === "research_only") return "warning";
  return "unavailable";
}

function validationBadgeStatus(status: RuntimeArtifact["validationStatus"]): string {
  if (status === "verified") return "ready";
  if (status === "invalid") return "error";
  return status === "declared" ? "partial" : "unavailable";
}
