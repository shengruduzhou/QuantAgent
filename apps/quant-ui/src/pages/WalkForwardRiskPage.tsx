import { useEffect, useState } from "react";

/**
 * Walk-forward risk-control results.
 *
 * Two rules this page follows deliberately, because the backend models three
 * states and a UI that collapses them defeats the point:
 *
 *  1. A missing metric renders as 暂无, never as 0. Zero is the BEST possible
 *     reading on drawdown, so showing it for an absent measurement inverts the
 *     meaning.
 *  2. Windows the runner flagged `low_breadth` are shown, but visually marked
 *     and excluded from the aggregate — a cross-sectional rank over <300 names
 *     is not measuring what the metric name suggests.
 *
 * A-share colour convention: red = up, green = down.
 */

interface WindowResult {
  window: number;
  train: string;
  test_year: number;
  confidence: "ok" | "low_breadth";
  median_symbols: number;
  risk_on: Record<string, number | string | null>;
  risk_off: Record<string, number | string | null>;
}

interface Payload {
  runId: string | null;
  state: string | null;
  windowsTotal: number | null;
  windowsOkBreadth: number | null;
  breadthFloor: number | null;
  aggregate: Record<string, Record<string, number | null>> | null;
  note: string | null;
  results: WindowResult[];
}

const API = "/api/walkforward-risk";

function pct(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "暂无";
  return `${(v * 100).toFixed(1)}%`;
}
function num(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "暂无";
  return v.toFixed(2);
}
/** red = up, green = down (A-share convention). */
function tone(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "wf-muted";
  return v >= 0 ? "wf-up" : "wf-down";
}

export function WalkForwardRiskPage(): JSX.Element {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const r = await fetch(`${API}/results`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = (await r.json()) as Payload;
        if (alive) { setData(j); setError(null); }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
    };
    poll();
    // Keep polling while a run is in flight so the page updates live.
    const id = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (error) return <div className="wf-page"><p className="wf-error">无法读取回测结果：{error}</p></div>;
  if (!data) return <div className="wf-page"><p>加载中…</p></div>;

  const agg = data.aggregate ?? {};
  const on = agg.risk_on ?? {};
  const off = agg.risk_off ?? {};

  return (
    <div className="wf-page">
      <header className="wf-header">
        <h1>滚动窗口风控回测</h1>
        <p>
          训练 3 年 → 回测第 4 年，逐年滑动，全覆盖。
          状态：<strong>{data.state ?? "暂无"}</strong>
          {" · "}窗口 {data.windowsTotal ?? "暂无"}
          {" · "}达到样本宽度门槛 {data.windowsOkBreadth ?? "暂无"}
          （≥{data.breadthFloor ?? "?"} 只）
        </p>
      </header>

      <section className="wf-agg">
        <h2>汇总（仅统计达标窗口）</h2>
        <table className="wf-table">
          <thead>
            <tr><th>指标</th><th>风控开启</th><th>风控关闭（对照）</th></tr>
          </thead>
          <tbody>
            {[
              ["年度收益", "total_return", pct],
              ["最大回撤", "max_drawdown", pct],
              ["年化波动", "vol_annual", pct],
              ["夏普", "sharpe", num],
              ["卡玛", "calmar", num],
            ].map(([label, key, fmt]) => {
              const f = fmt as (v: unknown) => string;
              const k = key as string;
              return (
                <tr key={k}>
                  <td>{label as string}</td>
                  <td className={tone(on[k])}>{f(on[k])}</td>
                  <td className={tone(off[k])}>{f(off[k])}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {data.note && <p className="wf-note">{data.note}</p>}
      </section>

      <section>
        <h2>逐窗口结果</h2>
        <table className="wf-table">
          <thead>
            <tr>
              <th>回测年</th><th>训练窗口</th><th>标的数</th><th>可信度</th>
              <th>风控开·收益</th><th>风控开·回撤</th>
              <th>风控关·收益</th><th>风控关·回撤</th>
            </tr>
          </thead>
          <tbody>
            {data.results.map((r) => (
              <tr key={r.window} className={r.confidence === "low_breadth" ? "wf-lowbreadth" : ""}>
                <td>{r.test_year}</td>
                <td>{r.train}</td>
                <td>{r.median_symbols}</td>
                <td>{r.confidence === "ok" ? "达标" : "样本过少"}</td>
                <td className={tone(r.risk_on.total_return)}>{pct(r.risk_on.total_return)}</td>
                <td className={tone(r.risk_on.max_drawdown)}>{pct(r.risk_on.max_drawdown)}</td>
                <td className={tone(r.risk_off.total_return)}>{pct(r.risk_off.total_return)}</td>
                <td className={tone(r.risk_off.max_drawdown)}>{pct(r.risk_off.max_drawdown)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

export default WalkForwardRiskPage;
