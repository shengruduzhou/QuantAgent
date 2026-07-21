import { useMemo, useState } from "react";
import { ArrowsClockwise, MagnifyingGlass } from "@phosphor-icons/react";
import { useApi } from "../hooks/useApi";
import type {
  VnpyParityCapability,
  VnpyParityStatus,
  VnpyParityView,
} from "../api/parityTypes";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";

const STATUS_LABELS: Record<VnpyParityStatus, string> = {
  not_audited: "未审计",
  missing: "缺失",
  planned: "已规划",
  in_progress: "进行中",
  partial: "部分实现",
  implemented: "已实现",
  verified: "已验证",
  blocked: "阻塞",
  not_applicable: "不适用",
};

function listOrDash(items: string[]): string {
  return items.length ? items.join(" · ") : "—";
}

function statusClass(status: VnpyParityStatus): string {
  return `parity-status parity-status-${status.replaceAll("_", "-")}`;
}

function CapabilityInspector({ capability }: { capability?: VnpyParityCapability }): JSX.Element {
  if (!capability) return <StateView state="empty" detail="选择一项能力查看来源、差距和下一步。" />;

  return (
    <div className="parity-inspector-content">
      <div className="parity-inspector-title">
        <div>
          <span className="mono">{capability.id}</span>
          <h2>{capability.name}</h2>
        </div>
        <span className={statusClass(capability.status)}>{STATUS_LABELS[capability.status]}</span>
      </div>

      <section>
        <h3>VN.PY SOURCE</h3>
        <dl>
          <div><dt>Repository</dt><dd><a href={`https://github.com/${capability.source.repo}`} target="_blank" rel="noreferrer">{capability.source.repo}</a></dd></div>
          <div><dt>Module</dt><dd className="mono">{capability.source.module}</dd></div>
          <div><dt>Version</dt><dd>{capability.source.version}{capability.source.commit ? ` · ${capability.source.commit}` : ""}</dd></div>
        </dl>
        <p>{capability.description}</p>
      </section>

      <section>
        <h3>QUANTAGENT MAPPING</h3>
        <dl>
          <div><dt>Modules</dt><dd>{listOrDash(capability.quantagent.modules)}</dd></div>
          <div><dt>API</dt><dd>{listOrDash(capability.quantagent.api)}</dd></div>
          <div><dt>Events</dt><dd>{listOrDash(capability.quantagent.events)}</dd></div>
          <div><dt>Artifacts</dt><dd>{listOrDash(capability.quantagent.artifacts)}</dd></div>
          <div><dt>Frontend</dt><dd>{listOrDash(capability.quantagent.frontend)}</dd></div>
        </dl>
      </section>

      <section className="parity-inspector-gap">
        <h3>CURRENT GAP</h3>
        <p>{capability.gap}</p>
      </section>

      <section>
        <h3>ADOPTION DECISION</h3>
        <p>{capability.adoption}</p>
      </section>

      <section className="parity-next-action">
        <h3>NEXT ACTION</h3>
        <p>{capability.nextAction}</p>
      </section>

      <section>
        <h3>VERIFICATION</h3>
        <dl>
          <div><dt>Tests</dt><dd>{listOrDash(capability.tests)}</dd></div>
          <div><dt>Evidence</dt><dd>{listOrDash(capability.evidence)}</dd></div>
          <div><dt>Limitations</dt><dd>{listOrDash(capability.limitations)}</dd></div>
        </dl>
      </section>
    </div>
  );
}

export function VnpyParityPage(): JSX.Element {
  const [category, setCategory] = useState("");
  const [status, setStatus] = useState<VnpyParityStatus | "">("");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const parity = useApi<VnpyParityView>(
    ["vnpy-parity", category, status, query],
    "/system/vnpy-parity",
    {
      category: category || null,
      status: status || null,
      query: query || null,
    },
    { staleTime: 30_000 },
  );

  const data = parity.data?.data;
  const selected = useMemo(
    () => data?.capabilities.find((item) => item.id === selectedId) ?? data?.capabilities[0],
    [data?.capabilities, selectedId],
  );
  const completion = data ? `${(data.summary.completionRatio * 100).toFixed(1)}%` : "—";

  if (parity.isLoading) return <StateView state="loading" />;
  if (parity.isError) return <StateView state="error" detail={parity.error.message} />;
  if (parity.data?.status === "error") {
    return <StateView state="error" detail={parity.data.issues[0]?.message ?? "VN.PY 能力注册表校验失败。"} />;
  }

  return (
    <div className="page parity-page">
      <section className="parity-commandbar">
        <div>
          <span className="mono">GOVERNANCE / VN.PY PARITY</span>
          <h1>Capability Registry</h1>
          <p>页面存在不等于能力完成；只有后端、API、UI、真实状态变化、实时反馈、测试和浏览器证据完整时才可标记 verified。</p>
        </div>
        <div className="parity-baseline">
          <span>Authority</span>
          <strong>{data?.sourceBaseline.repo ?? "vnpy/vnpy"}</strong>
          <small>release {data?.sourceBaseline.release ?? "—"} · {data?.sourceBaseline.commit ?? "—"}</small>
        </div>
        <button type="button" className="secondary-button" onClick={() => void parity.refetch()}>
          <ArrowsClockwise size={15} /> 重新校验
        </button>
      </section>

      <section className="parity-filterbar" aria-label="能力筛选">
        <label>
          <span>分类</span>
          <select value={category} onChange={(event) => setCategory(event.target.value)}>
            <option value="">全部分类</option>
            {(data?.categories ?? []).map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label>
          <span>状态</span>
          <select value={status} onChange={(event) => setStatus(event.target.value as VnpyParityStatus | "")}>
            <option value="">全部状态</option>
            {(data?.statuses ?? []).map((item) => <option key={item} value={item}>{STATUS_LABELS[item]}</option>)}
          </select>
        </label>
        <label className="parity-search">
          <span>搜索</span>
          <div><MagnifyingGlass size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="能力、模块、API、差距或下一步" /></div>
        </label>
        <div className="parity-registry-meta">
          <span>Registry</span>
          <strong>{data?.registryVersion ?? "—"}</strong>
          <small>{data?.completeness ?? "unavailable"} · {data?.generatedAt?.slice(0, 16) ?? "—"}</small>
        </div>
      </section>

      <section className="parity-stat-strip">
        <div><span>当前结果</span><strong>{data?.summary.total ?? 0}</strong><small>capabilities</small></div>
        <div><span>加权完成度</span><strong>{completion}</strong><small>verified 权重最高</small></div>
        <div><span>已验证</span><strong>{data?.summary.verified ?? 0}</strong><small>严格门禁</small></div>
        <div><span>待处理</span><strong>{data?.summary.actionable ?? 0}</strong><small>不含 not_applicable</small></div>
        <div className="parity-status-counts">
          <span>状态分布</span>
          <p>{Object.entries(data?.summary.byStatus ?? {}).map(([key, value]) => <b key={key}>{STATUS_LABELS[key as VnpyParityStatus] ?? key} {value}</b>)}</p>
        </div>
      </section>

      <div className="parity-workspace">
        <Panel
          title="能力对齐矩阵"
          eyebrow={`${data?.capabilities.length ?? 0} rows · machine-readable single source`}
          className="parity-table-panel"
        >
          {data?.capabilities.length ? (
            <div className="parity-table-wrap">
              <table className="data-table parity-table">
                <thead>
                  <tr><th>Category</th><th>Capability</th><th>Status</th><th>QuantAgent mapping</th><th>Current gap</th><th>Next action</th></tr>
                </thead>
                <tbody>
                  {data.capabilities.map((item) => (
                    <tr
                      key={item.id}
                      className={selected?.id === item.id ? "selected" : ""}
                      onClick={() => setSelectedId(item.id)}
                    >
                      <td className="mono">{item.category}</td>
                      <td>
                        <button type="button" className="parity-row-button" onClick={() => setSelectedId(item.id)}>
                          <strong>{item.name}</strong><small className="mono">{item.id}</small>
                        </button>
                      </td>
                      <td><span className={statusClass(item.status)}>{STATUS_LABELS[item.status]}</span></td>
                      <td>{item.quantagent.modules[0] ?? item.quantagent.frontend[0] ?? "—"}</td>
                      <td>{item.gap}</td>
                      <td>{item.nextAction}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <StateView state="empty" detail="当前筛选条件没有能力项；注册表本身不会使用 mock 数据填充。" />}
        </Panel>

        <aside className="parity-inspector">
          <CapabilityInspector capability={selected} />
        </aside>
      </div>

      <section className="parity-governance-grid">
        <Panel title="Verified 门禁" eyebrow="全部满足才允许标记 verified">
          <ol>{(data?.verificationPolicy.verifiedRequires ?? []).map((item) => <li key={item}>{item}</li>)}</ol>
        </Panel>
        <Panel title="已知覆盖缺口" eyebrow="显式记录，不静默遗漏">
          <ul>{(data?.knownCoverageGaps ?? []).map((item) => <li key={item}>{item}</li>)}</ul>
        </Panel>
      </section>
    </div>
  );
}
