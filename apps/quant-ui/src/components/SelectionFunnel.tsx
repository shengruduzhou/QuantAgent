interface FunnelStage {
  stage: string;
  count: number | null;
  reason?: string | null;
}

interface SelectionFunnelProps {
  stages: FunnelStage[];
}

export function SelectionFunnel({ stages }: SelectionFunnelProps): JSX.Element {
  const maximum = Math.max(...stages.map((stage) => stage.count ?? 0), 1);
  return (
    <div className="funnel-list">
      {stages.map((stage, index) => {
        const count = stage.count ?? 0;
        const previous = index === 0 ? count : stages[index - 1]?.count ?? count;
        const rejection = previous > 0 ? Math.max(0, 1 - count / previous) : 0;
        return (
          <div className="funnel-row" key={stage.stage}>
            <div className="funnel-index">{index + 1}</div>
            <div className="funnel-stage">
              <strong>{stage.stage}</strong>
              <span>{stage.reason ?? "已持久化 gate"}</span>
            </div>
            <div className="funnel-bar">
              <i style={{ width: `${Math.max(8, (count / maximum) * 100)}%` }} />
            </div>
            <strong className="mono">{count.toLocaleString()}</strong>
            <span className="tone-negative">{index === 0 ? "—" : `${(rejection * 100).toFixed(1)}%`}</span>
          </div>
        );
      })}
    </div>
  );
}
