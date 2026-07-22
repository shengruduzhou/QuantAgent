import { useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  ArrowClockwise,
  ArrowRight,
  Broom,
  CirclesThreePlus,
  Database,
  File,
  FlowArrow,
  HardDrives,
  MagnifyingGlass,
  ShieldCheck,
  Stack,
  WarningCircle,
} from "@phosphor-icons/react";
import { useNavigate, useSearchParams } from "react-router-dom";
import type { Page, RuntimeArtifact, RuntimeCatalog, RuntimeLineage, RuntimeRunSummary } from "../api/types";
import { DataManagerWorkspace } from "../components/runtime_manager/DataManagerWorkspace";
import { RuntimeCleanupWorkspace } from "../components/runtime_manager/RuntimeCleanupWorkspace";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { useApi } from "../hooks/useApi";
import { formatBytes, formatDate, formatNumber } from "../utils/format";

const kinds = ["", "backtest", "model", "prediction", "target_weights", "factor", "selection", "risk", "do_t", "log", "dataset", "report", "manifest", "unknown"];
const trustClasses = ["", "production_ready", "paper_only", "research_only", "contaminated", "unclassified"];
const validations = ["", "verified", "declared", "unverified", "invalid"];
const capabilities = ["", "preview", "research_display", "production_display", "paper_execution", "audit_replay"];

type RuntimeTab = "catalog" | "data" | "runs" | "lineage" | "cleanup";

function requestedRuntimeTab(value: string | null): RuntimeTab | null {
  return value === "catalog" || value === "data" || value === "runs" || value === "lineage" || value === "cleanup" ? value : null;
}

export function RuntimeExplorerPage(): JSX.Element {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const requestedView = searchParams.get("view");
  const [tab, setTab] = useState<RuntimeTab>(() => requestedRuntimeTab(requestedView) ?? "catalog");
  const [kind, setKind] = useState("");
  const [trustClass, setTrustClass] = useState("");
  const [validationStatus, setValidationStatus] = useState("");
  const [capability, setCapability] = useState("");
  const [runId, setRunId] = useState(searchParams.get("runId") ?? "");
  const [query, setQuery] = useState(searchParams.get("query") ?? "");
  const [sortBy, setSortBy] = useState("modifiedAt");
  const [sortDirection, setSortDirection] = useState("desc");
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState("");
  const [refreshToken, setRefreshToken] = useState(false);
  const deferredQuery = useDeferredValue(query);

  useEffect(() => {
    const next = requestedRuntimeTab(requestedView);
    if (next) setTab(next);
  }, [requestedView]);

  const catalog = useApi<RuntimeCatalog>(["runtime-catalog", refreshToken], "/system/runtime-catalog", { refresh: refreshToken });
  const artifacts = useApi<Page<RuntimeArtifact>>(
    ["runtime-index", kind, trustClass, validationStatus, capability, runId, deferredQuery, sortBy, sortDirection, page, refreshToken],
    "/system/runtime-index",
    {
      kind,
      trustClass,
      validationStatus,
      capability,
      runId,
      query: deferredQuery,
      sortBy,
      sortDirection,
      page,
      pageSize: 100,
      refresh: refreshToken,
    },
  );
  const preview = useApi<Record<string, unknown>>(
    ["runtime-preview", selectedId],
    selectedId && tab === "catalog" ? `/system/runtime-index/${selectedId}/preview` : null,
    { limit: 100 },
  );
  const lineage = useApi<RuntimeLineage>(
    ["runtime-lineage", selectedId],
    selectedId && tab === "lineage" ? `/system/runtime-index/${selectedId}/lineage` : null,
  );
  const data = artifacts.data?.data;
  const summary = catalog.data?.data.summary;
  const selectedArtifact = useMemo(
    () => data?.items.find((artifact) => artifact.id === selectedId) ?? lineage.data?.data.artifact ?? null,
    [data?.items, lineage.data?.data.artifact, selectedId],
  );

  const resetPage = (): void => setPage(1);
  const rebuildIndex = (): void => setRefreshToken((value) => !value);
  const openRun = (run: RuntimeRunSummary): void => {
    setRunId(run.id);
    setPage(1);
    setTab("catalog");
  };
  const selectForLineage = (artifact: RuntimeArtifact): void => {
    setSelectedId(artifact.id);
    setTab("lineage");
  };
  const openModule = (artifact: RuntimeArtifact): void => {
    const destinations: Record<string, string> = {
      backtest: "/backtests", model: "/models", factor: "/factors", selection: "/selection",
      risk: "/risk", do_t: "/t-plus-one", report: "/reports",
    };
    const destination = destinations[artifact.kind];
    if (destination) navigate(`${destination}${artifact.runId ? `?runId=${encodeURIComponent(artifact.runId)}` : ""}`);
  };

  return (
    <div className="page runtime-page runtime-manager-page">
      <section className="runtime-commandbar">
        <div>
          <span className="page-kicker">RUNTIME / DATA MANAGER</span>
          <h2>Data Ops · Artifact Catalog · Runs · Lineage</h2>
          <p>唯一 RuntimeIndexer 投影；manifest 声明优先，未声明关系不推断。</p>
        </div>
        <div className="runtime-tabs terminal-segments">
          <button className={tab === "data" ? "active" : ""} onClick={() => setTab("data")}><HardDrives size={15} /> Data Ops</button>
          <button className={tab === "catalog" ? "active" : ""} onClick={() => setTab("catalog")}><Database size={15} /> Catalog</button>
          <button className={tab === "runs" ? "active" : ""} onClick={() => setTab("runs")}><Stack size={15} /> Runs</button>
          <button className={tab === "lineage" ? "active" : ""} onClick={() => setTab("lineage")}><FlowArrow size={15} /> Lineage</button>
          <button className={tab === "cleanup" ? "active" : ""} onClick={() => setTab("cleanup")}><Broom size={15} /> Cleanup</button>
        </div>
      </section>

      <section className="runtime-stat-strip" aria-label="Runtime catalog summary">
        <RuntimeStat label="Artifacts" value={summary?.artifactCount.toLocaleString() ?? "—"} detail={formatBytes(summary?.totalSizeBytes)} />
        <RuntimeStat label="Runs" value={summary?.runCount.toLocaleString() ?? "—"} detail="declared path groups" />
        <RuntimeStat label="Manifest" value={summary ? `${(summary.manifestCoverage * 100).toFixed(1)}%` : "—"} detail="contract coverage" />
        <RuntimeStat label="Verified" value={(summary?.byValidation.verified ?? 0).toLocaleString()} detail="hash / declared checks" tone="positive" />
        <RuntimeStat label="Invalid" value={(summary?.byValidation.invalid ?? 0).toLocaleString()} detail="fail-closed" tone={(summary?.byValidation.invalid ?? 0) ? "negative" : "neutral"} />
        <RuntimeStat label="Contaminated" value={(summary?.byTrust.contaminated ?? 0).toLocaleString()} detail="not production-ready" tone={(summary?.byTrust.contaminated ?? 0) ? "warning" : "neutral"} />
        <RuntimeStat label="Indexed" value={summary?.indexedAt ? formatDate(summary.indexedAt) : "—"} detail={catalog.data?.data.roots?.join(" · ") || "runtime unavailable"} />
      </section>

      {tab === "data" ? <DataManagerWorkspace /> : null}

      {tab === "catalog" ? (
        <>
          <section className="runtime-filterbar">
            <label className="runtime-search"><span>Search</span><div><MagnifyingGlass size={15} /><input value={query} onChange={(event) => { setQuery(event.target.value); resetPage(); }} placeholder="path / run / model / symbol" /></div></label>
            <FilterSelect label="Kind" value={kind} onChange={(value) => { setKind(value); resetPage(); }} values={kinds} />
            <FilterSelect label="Trust" value={trustClass} onChange={(value) => { setTrustClass(value); resetPage(); }} values={trustClasses} />
            <FilterSelect label="Validation" value={validationStatus} onChange={(value) => { setValidationStatus(value); resetPage(); }} values={validations} />
            <FilterSelect label="Capability" value={capability} onChange={(value) => { setCapability(value); resetPage(); }} values={capabilities} />
            <label><span>Run</span><input value={runId} onChange={(event) => { setRunId(event.target.value); resetPage(); }} placeholder="all runs" /></label>
            <label><span>Sort</span><select value={sortBy} onChange={(event) => setSortBy(event.target.value)}><option value="modifiedAt">Modified</option><option value="sizeBytes">Size</option><option value="name">Name</option><option value="kind">Kind</option><option value="trustClass">Trust</option></select></label>
            <button className="sort-direction" onClick={() => setSortDirection((value) => value === "desc" ? "asc" : "desc")}>{sortDirection.toUpperCase()}</button>
            <button className="secondary-button" onClick={rebuildIndex}><ArrowClockwise size={15} /> Reindex</button>
          </section>

          <section className="runtime-workbench-grid">
            <Panel title="Artifact Catalog" eyebrow={`${data?.total ?? 0} matched · page ${data?.page ?? 1}`} className="runtime-catalog-panel">
              {data?.items.length ? (
                <>
                  <div className="table-scroll runtime-catalog-table"><table className="data-table"><thead><tr><th>Kind</th><th>Artifact / Path</th><th>Run / Range</th><th>Trust</th><th>Validation</th><th>Quality</th><th className="numeric">Rows / Size</th><th>Time</th></tr></thead>
                    <tbody>{data.items.map((artifact) => (
                      <tr key={artifact.id} className={selectedId === artifact.id ? "row-selected" : ""} onClick={() => setSelectedId(artifact.id)} tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelectedId(artifact.id); }}>
                        <td><span className={`artifact-kind kind-${artifact.kind}`}>{artifact.kind}</span></td>
                        <td className="artifact-path"><strong>{artifact.name}</strong><span>{artifact.path}</span></td>
                        <td><strong>{artifact.runId ?? "—"}</strong><span>{artifact.dateStart || artifact.dateEnd ? `${artifact.dateStart ?? "?"} → ${artifact.dateEnd ?? "?"}` : artifact.horizon ?? artifact.extension}</span></td>
                        <td><StatusBadge status={trustBadgeStatus(artifact.trustClass)} label={artifact.trustClass} /></td>
                        <td><StatusBadge status={validationBadgeStatus(artifact.validationStatus)} label={artifact.validationStatus} /></td>
                        <td>{artifact.qualityStatus ?? "undeclared"}</td>
                        <td className="numeric mono"><strong>{artifact.rows != null ? formatNumber(artifact.rows, 0) : "—"}</strong><span>{formatBytes(artifact.sizeBytes)}</span></td>
                        <td className="mono"><strong>{formatDate(artifact.dataAsOf ?? artifact.sourceTime ?? artifact.modifiedAt)}</strong><span>{artifact.dataAsOf ? "data as-of" : artifact.sourceTime ? "manifest" : "filesystem"}</span></td>
                      </tr>
                    ))}</tbody>
                  </table></div>
                  <div className="pagination"><button disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>上一页</button><span>{page} / {Math.max(1, Math.ceil(data.total / data.pageSize))}</span><button disabled={!data.hasNext} onClick={() => setPage((value) => value + 1)}>下一页</button></div>
                </>
              ) : <StateView state={artifacts.isLoading ? "loading" : artifacts.isError ? "error" : "empty"} detail={artifacts.error?.message} />}
            </Panel>

            <Panel title="Artifact Inspector" eyebrow="contract · preview · operations" className="runtime-inspector-panel">
              {selectedArtifact ? (
                <>
                  <section className="artifact-inspector-head"><File size={18} /><div><strong>{selectedArtifact.name}</strong><code>{selectedArtifact.path}</code></div></section>
                  <div className="artifact-contract-badges"><StatusBadge status={trustBadgeStatus(selectedArtifact.trustClass)} label={selectedArtifact.trustClass} /><StatusBadge status={validationBadgeStatus(selectedArtifact.validationStatus)} label={selectedArtifact.validationStatus} /><StatusBadge status={selectedArtifact.freshnessStatus} label={selectedArtifact.freshnessStatus} /></div>
                  <dl className="artifact-metadata-grid">
                    <div><dt>Schema</dt><dd>{selectedArtifact.schemaVersion ?? "undeclared"}</dd></div><div><dt>Declared type</dt><dd>{selectedArtifact.declaredKind ?? "undeclared"} · {selectedArtifact.kindSource ?? "legacy"}</dd></div>
                    <div><dt>Producer</dt><dd>{selectedArtifact.producer ?? "undeclared"}</dd></div><div><dt>Quality</dt><dd>{selectedArtifact.qualityStatus ?? "undeclared"}</dd></div>
                    <div><dt>Manifest / Run source</dt><dd>{selectedArtifact.manifestPath ?? "none"} · {selectedArtifact.runIdSource ?? "none"}</dd></div><div><dt>Source / as-of</dt><dd>{selectedArtifact.dataAsOf ?? selectedArtifact.sourceTime ?? "filesystem mtime only"}</dd></div>
                  </dl>
                  <div className="capability-list">{selectedArtifact.capabilities.map((item) => <span key={item}>{item}</span>)}</div>
                  {selectedArtifact.issues.length ? <div className="artifact-contract-issues">{selectedArtifact.issues.map((issue) => <span key={`${issue.code}-${issue.path ?? ""}`}><WarningCircle size={13} />{issue.code}: {issue.message}</span>)}</div> : null}
                  <div className="inspector-actions"><button onClick={() => selectForLineage(selectedArtifact)}><FlowArrow size={14} /> Lineage</button>{selectedArtifact.runId ? <button onClick={() => { setRunId(selectedArtifact.runId ?? ""); resetPage(); }}><Stack size={14} /> Filter run</button> : null}{["backtest", "model", "factor", "selection", "risk", "do_t", "report"].includes(selectedArtifact.kind) ? <button onClick={() => openModule(selectedArtifact)}><ArrowRight size={14} /> Open module</button> : null}</div>
                  <div className="safe-preview">{preview.isLoading ? <StateView state="loading" /> : preview.data ? <pre className="json-view">{JSON.stringify(preview.data.data, null, 2)}</pre> : <StateView state={preview.isError ? "error" : "unavailable"} detail={preview.error?.message ?? "该 artifact 没有安全预览能力。"} />}</div>
                </>
              ) : <StateView state="empty" detail="选择 catalog 中的 artifact 查看契约、预览和操作。" />}
            </Panel>
          </section>
        </>
      ) : null}

      {tab === "runs" ? <RunCatalog runs={catalog.data?.data.runs ?? []} loading={catalog.isLoading} error={catalog.error?.message} onOpen={openRun} /> : null}
      {tab === "lineage" ? <LineageWorkbench selected={selectedArtifact} lineage={lineage.data?.data} loading={lineage.isLoading} error={lineage.error?.message} onSelect={selectForLineage} onCatalog={() => setTab("catalog")} /> : null}
      {tab === "cleanup" ? <RuntimeCleanupWorkspace refreshToken={refreshToken} onChanged={rebuildIndex} /> : null}
    </div>
  );
}

function RuntimeStat({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail: string; tone?: string }): JSX.Element {
  return <div className={`runtime-stat tone-${tone}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function FilterSelect({ label, value, values, onChange }: { label: string; value: string; values: string[]; onChange: (value: string) => void }): JSX.Element {
  return <label><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{values.map((item) => <option value={item} key={item}>{item || `All ${label.toLowerCase()}`}</option>)}</select></label>;
}

function RunCatalog({ runs, loading, error, onOpen }: { runs: RuntimeRunSummary[]; loading: boolean; error?: string; onOpen: (run: RuntimeRunSummary) => void }): JSX.Element {
  return <Panel title="Run Catalog" eyebrow={`${runs.length} indexed run groups · derived from the existing RuntimeIndexer`} className="runtime-runs-panel">{runs.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Run ID</th><th>Artifacts</th><th>Kinds</th><th>Trust / Validation</th><th>Capabilities</th><th>Data range</th><th>Updated</th><th>Issues</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id} onClick={() => onOpen(run)} tabIndex={0}><td className="mono"><strong>{run.id}</strong><span>{formatBytes(run.totalSizeBytes)}</span></td><td>{run.artifactCount}</td><td>{run.kinds.join(" · ")}</td><td><strong>{run.trustClasses.join(" · ")}</strong><span>{run.validationStatuses.join(" · ")}</span></td><td>{run.capabilities.join(" · ")}</td><td className="mono">{run.dateStart || run.dateEnd ? `${run.dateStart ?? "?"} → ${run.dateEnd ?? "?"}` : "undeclared"}</td><td>{formatDate(run.latestModifiedAt)}</td><td><StatusBadge status={run.issueCount ? "warning" : "ready"} label={String(run.issueCount)} /></td></tr>)}</tbody></table></div> : <StateView state={loading ? "loading" : error ? "error" : "empty"} detail={error ?? "没有可识别 runId 的 artifacts。"} />}</Panel>;
}

function LineageWorkbench({ selected, lineage, loading, error, onSelect, onCatalog }: { selected: RuntimeArtifact | null; lineage?: RuntimeLineage; loading: boolean; error?: string; onSelect: (artifact: RuntimeArtifact) => void; onCatalog: () => void }): JSX.Element {
  if (!selected) return <Panel title="Artifact Lineage" eyebrow="manifest-declared relations only"><StateView state="empty" detail="先在 Catalog 中选择 artifact；系统不会根据相似文件名伪造 lineage。" /><div className="lineage-empty-action"><button className="secondary-button" onClick={onCatalog}>返回 Catalog</button></div></Panel>;
  if (loading) return <StateView state="loading" />;
  if (error || !lineage) return <StateView state="error" detail={error ?? "lineage unavailable"} />;
  return <section className="lineage-workbench"><div className="lineage-status"><FlowArrow size={16} /><strong>{lineage.status.toUpperCase()}</strong><span>Only explicit, safe repository-relative references are resolved.</span></div><div className="lineage-columns"><LineageColumn title="UPSTREAM" empty="No upstream declared" items={lineage.upstream.map((edge) => edge.artifact ?? edge.reference)} onSelect={onSelect} /><div className="lineage-focus"><CirclesThreePlus size={23} /><span>{lineage.artifact.kind}</span><strong>{lineage.artifact.name}</strong><code>{lineage.artifact.path}</code><StatusBadge status={trustBadgeStatus(lineage.artifact.trustClass)} label={lineage.artifact.trustClass} /></div><LineageColumn title="DOWNSTREAM" empty="No indexed downstream" items={lineage.downstream} onSelect={onSelect} /></div>{lineage.issues.length ? <div className="artifact-contract-issues">{lineage.issues.map((issue) => <span key={issue.code}><WarningCircle size={13} />{issue.message}</span>)}</div> : null}</section>;
}

function LineageColumn({ title, empty, items, onSelect }: { title: string; empty: string; items: Array<RuntimeArtifact | string>; onSelect: (artifact: RuntimeArtifact) => void }): JSX.Element {
  return <section className="lineage-column"><header>{title}</header>{items.length ? items.map((item) => typeof item === "string" ? <div className="lineage-unresolved" key={item}><WarningCircle size={14} /><span><strong>Unresolved reference</strong><code>{item}</code></span></div> : <button key={item.id} onClick={() => onSelect(item)}><span className={`artifact-kind kind-${item.kind}`}>{item.kind}</span><strong>{item.name}</strong><code>{item.path}</code></button>) : <div className="lineage-none">{empty}</div>}</section>;
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
