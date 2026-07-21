import { useEffect, useMemo, useState } from "react";
import { Broom, Check, Database, ShieldCheck, Trash, WarningCircle, X } from "@phosphor-icons/react";
import { apiPost } from "../../api/client";
import type { CleanupResult, RuntimeCleanupAnalysis } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { formatBytes } from "../../utils/format";
import { MetricCard } from "../MetricCard";
import { Panel } from "../Panel";
import { StateView } from "../StateView";
import { StatusBadge } from "../StatusBadge";

interface RuntimeCleanupWorkspaceProps {
  refreshToken: boolean;
  onChanged: () => void;
}

export function RuntimeCleanupWorkspace({ refreshToken, onChanged }: RuntimeCleanupWorkspaceProps): JSX.Element {
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [cleanupError, setCleanupError] = useState("");
  const [cleanupResult, setCleanupResult] = useState<CleanupResult | null>(null);
  const cleanup = useApi<RuntimeCleanupAnalysis>(["runtime-cleanup", refreshToken], "/system/runtime-cleanup");

  useEffect(() => {
    const defaults = cleanup.data?.data.candidates.filter((item) => item.safeDefault).map((item) => item.id);
    if (defaults?.length && !selectedCandidates.length && !cleanupResult) setSelectedCandidates(defaults);
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
      await cleanup.refetch();
      onChanged();
    } catch (error) {
      setCleanupError(error instanceof Error ? error.message : "cleanup failed");
    } finally {
      setCleanupBusy(false);
    }
  };

  return (
    <section className="cleanup-page">
      <div className="metric-grid metric-grid-4 cleanup-metrics">
        <MetricCard label="Runtime 总体积" value={formatBytes(cleanup.data?.data.runtimeSizeBytes)} icon={Database} />
        <MetricCard label="可清理候选" value={formatBytes(cleanup.data?.data.candidateSizeBytes)} detail={`${cleanup.data?.data.candidates.length ?? 0} candidate groups`} icon={Trash} />
        <MetricCard label="默认安全项" value={formatBytes(cleanup.data?.data.safeDefaultSizeBytes)} tone="positive" detail="test / smoke / superseded UI captures" icon={ShieldCheck} />
        <MetricCard label="已选择" value={formatBytes(selectedSize)} tone={selectedSize > 0 ? "warning" : "neutral"} detail={`${selectedCandidates.length} groups`} icon={Check} />
      </div>

      <div className="cleanup-layout">
        <Panel title="清理候选" eyebrow="backend re-validates paths and protection rules" className="cleanup-candidates-panel">
          {cleanup.isLoading ? <StateView state="loading" /> : cleanup.data?.data.candidates.length ? (
            <div className="cleanup-list">
              {cleanup.data.data.candidates.map((candidate) => {
                const checked = selectedCandidates.includes(candidate.id);
                return (
                  <button
                    key={candidate.id}
                    className={`${checked ? "selected" : ""} ${candidate.requiresExplicit ? "cleanup-review" : ""}`}
                    onClick={() => setSelectedCandidates((current) => checked ? current.filter((id) => id !== candidate.id) : [...current, candidate.id])}
                  >
                    <span className={`cleanup-check ${checked ? "checked" : ""}`}>{checked ? <Check size={13} weight="bold" /> : null}</span>
                    <span className="cleanup-copy"><strong>{candidate.label}</strong><small>{candidate.reason}</small><em>{candidate.paths.slice(0, 2).join(" · ")}{candidate.paths.length > 2 ? ` · +${candidate.paths.length - 2}` : ""}</em></span>
                    <span className="cleanup-size"><strong>{formatBytes(candidate.sizeBytes)}</strong><small>{candidate.itemCount} files</small></span>
                    <StatusBadge status={candidate.safeDefault ? "ready" : candidate.requiresExplicit ? "warning" : "partial"} label={candidate.safeDefault ? "默认安全" : candidate.requiresExplicit ? "需人工复核" : "可重建"} />
                  </button>
                );
              })}
            </div>
          ) : <StateView state="empty" detail="当前没有可识别清理候选。" />}
          <div className="cleanup-actions">
            <div><Broom size={18} /><span>canonical raw/silver/manifests 与主模型 registry 永久受保护。</span></div>
            <button className="danger-button" disabled={!selectedCandidates.length} onClick={() => setConfirmOpen(true)}><Trash size={16} /> 删除已选 · {formatBytes(selectedSize)}</button>
          </div>
        </Panel>

        <Panel title="保护范围" eyebrow="Hard backend guardrails" className="cleanup-protected-panel">
          <div className="protected-list">{(cleanup.data?.data.protected ?? []).map((path) => <div key={path}><ShieldCheck size={16} /><code>{path}</code></div>)}</div>
          {cleanupResult ? <div className="cleanup-result"><Check size={20} /><div><strong>清理完成 · 释放 {formatBytes(cleanupResult.freedBytes)}</strong><span>{cleanupResult.auditPath}</span></div></div> : null}
          {cleanupError ? <div className="cleanup-error"><WarningCircle size={18} />{cleanupError}</div> : null}
        </Panel>
      </div>

      {confirmOpen ? (
        <div className="modal-overlay" role="presentation" onMouseDown={() => setConfirmOpen(false)}>
          <section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-label="确认清理 runtime" onMouseDown={(event) => event.stopPropagation()}>
            <header><div><WarningCircle size={22} /><span><strong>确认删除 runtime 产物</strong><small>只作用于当前 backend-approved candidates。</small></span></div><button onClick={() => setConfirmOpen(false)}><X size={17} /></button></header>
            <div className="confirm-summary"><strong>{selectedCandidates.length} groups</strong><span>{formatBytes(selectedSize)}</span></div>
            <p>删除后会写入 cleanup audit 并重建索引。大型训练集候选可能影响历史复现，请确认选择范围。</p>
            <footer><button className="secondary-button" onClick={() => setConfirmOpen(false)}>取消</button><button className="danger-button" disabled={cleanupBusy} onClick={runCleanup}>{cleanupBusy ? "正在清理…" : "确认删除"}</button></footer>
          </section>
        </div>
      ) : null}
    </section>
  );
}
