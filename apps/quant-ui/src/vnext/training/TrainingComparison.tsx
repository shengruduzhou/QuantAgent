import { ArrowsLeftRight, X } from "@phosphor-icons/react";
import type { ModelComparison } from "../../api/types";
import { formatNumber } from "../../utils/format";

interface TrainingComparisonProps {
  ids: string[];
  comparison?: ModelComparison;
  loading: boolean;
  clear: () => void;
}

export function TrainingComparison({ ids, comparison, loading, clear }: TrainingComparisonProps): JSX.Element | null {
  if (ids.length < 2) return null;
  const models = comparison?.models ?? [];
  const metricKeys = (comparison?.metricKeys ?? []).slice(0, 8);

  return (
    <section className="vnext-training-comparison" aria-label="训练模型对比">
      <header>
        <div><ArrowsLeftRight size={16} /><span><strong>MODEL COMPARISON</strong><small>{ids.length} selected · independent persisted metrics</small></span></div>
        <button type="button" onClick={clear}><X size={14} /> Clear compare</button>
      </header>
      {loading ? <p>正在读取模型对比 artifact…</p> : models.length ? (
        <div className="vnext-training-comparison-table">
          <table>
            <thead><tr><th>Metric</th>{models.map((model) => <th key={model.id}><strong>{model.version ?? model.id}</strong><small>{model.verdict ?? model.status}</small></th>)}</tr></thead>
            <tbody>
              {metricKeys.map((metric) => <tr key={metric}><th>{metric}</th>{models.map((model) => <td key={model.id}>{formatNumber(model.metrics[metric], 4)}</td>)}</tr>)}
            </tbody>
          </table>
        </div>
      ) : <p>所选模型没有可比较的共同 persisted metrics。</p>}
    </section>
  );
}
