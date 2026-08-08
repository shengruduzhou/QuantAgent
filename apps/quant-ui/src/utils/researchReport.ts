export interface ResearchReportSection {
  title: string;
  rows: Array<[string, string]>;
}

export interface ResearchReportChartSeries {
  name: string;
  values: Array<number | null | undefined>;
}

export interface ResearchReportChart {
  title: string;
  labels: string[];
  series: ResearchReportChartSeries[];
  valueFormat?: "number" | "percent";
}

export interface OfflineResearchReport {
  title: string;
  generatedAt: string;
  subtitle: string;
  sections: ResearchReportSection[];
  provenance: Array<[string, string]>;
  limitations: string[];
  charts?: ResearchReportChart[];
}

function escapeHtml(value: unknown): string {
  return String(value ?? "—")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function finiteValues(chart: ResearchReportChart): number[] {
  return chart.series.flatMap((series) => series.values)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function renderChart(chart: ResearchReportChart, chartIndex: number): string {
  const width = 920;
  const height = 280;
  const left = 62;
  const right = 24;
  const top = 30;
  const bottom = 56;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const values = finiteValues(chart);
  if (!values.length || chart.labels.length < 2) {
    return `<section class="panel wide"><h2>${escapeHtml(chart.title)}</h2><p class="empty">暂无可绘制的持久化数值。</p></section>`;
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const pad = Math.max(Math.abs(min) * 0.05, 0.01);
    min -= pad;
    max += pad;
  }
  if (min > 0) min = 0;
  if (max < 0) max = 0;
  const span = max - min || 1;
  const xAt = (index: number): number => left + plotWidth * index / Math.max(1, chart.labels.length - 1);
  const yAt = (value: number): number => top + plotHeight * (max - value) / span;
  const zeroY = yAt(0);
  const format = (value: number): string => chart.valueFormat === "percent"
    ? `${(value * 100).toFixed(1)}%`
    : Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 3 });

  const grid = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const value = max - span * ratio;
    const y = top + plotHeight * ratio;
    return `<line x1="${left}" y1="${y.toFixed(2)}" x2="${width - right}" y2="${y.toFixed(2)}" class="gridline"/><text x="${left - 8}" y="${(y + 4).toFixed(2)}" text-anchor="end" class="axis-label">${escapeHtml(format(value))}</text>`;
  }).join("");

  const seriesSvg = chart.series.map((series, seriesIndex) => {
    const points = series.values.map((value, index) => {
      if (typeof value !== "number" || !Number.isFinite(value)) return "";
      return `${xAt(index).toFixed(2)},${yAt(value).toFixed(2)}`;
    }).filter(Boolean).join(" ");
    if (!points) return "";
    return `<polyline points="${points}" class="series series-${seriesIndex % 6}" vector-effect="non-scaling-stroke"/>`;
  }).join("");

  const xLabels = chart.labels.map((label, index) => {
    const show = chart.labels.length <= 12 || index === 0 || index === chart.labels.length - 1 || index % Math.ceil(chart.labels.length / 8) === 0;
    if (!show) return "";
    return `<text x="${xAt(index).toFixed(2)}" y="${height - 24}" text-anchor="middle" class="axis-label">${escapeHtml(label.slice(0, 18))}</text>`;
  }).join("");

  const legend = chart.series.map((series, index) => `<span><i class="legend-dot series-bg-${index % 6}"></i>${escapeHtml(series.name)}</span>`).join("");
  return `<section class="panel wide chart-panel" data-chart-index="${chartIndex}"><div class="chart-head"><h2>${escapeHtml(chart.title)}</h2><div class="legend">${legend}</div></div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(chart.title)}"><line x1="${left}" y1="${zeroY.toFixed(2)}" x2="${width - right}" y2="${zeroY.toFixed(2)}" class="zero-line"/>${grid}${seriesSvg}${xLabels}</svg></section>`;
}

export function buildOfflineResearchHtml(report: OfflineResearchReport): string {
  const sections = report.sections.map((section) => `
    <section class="panel">
      <h2>${escapeHtml(section.title)}</h2>
      <table><tbody>${section.rows.map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`).join("")}</tbody></table>
    </section>`).join("");
  const charts = (report.charts ?? []).map(renderChart).join("");
  const provenance = report.provenance.map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td><code>${escapeHtml(value)}</code></td></tr>`).join("");
  const limitations = report.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const chartData = JSON.stringify(report.charts ?? []).replaceAll("<", "\\u003c");

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(report.title)}</title><style>
:root{color-scheme:dark;background:#0b0f14;color:#e8edf3;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;background:#0b0f14}.wrap{max-width:1180px;margin:0 auto;padding:34px}.hero{border:1px solid #27313d;background:#111820;padding:28px;margin-bottom:18px}.kicker{font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;color:#95a3b5;text-transform:uppercase}.hero h1{font-size:30px;margin:10px 0}.hero p{color:#aeb9c6;margin:0}.meta{display:flex;gap:18px;flex-wrap:wrap;margin-top:18px;font-size:12px;color:#95a3b5}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{border:1px solid #27313d;background:#111820;padding:20px}.panel h2{font-size:16px;margin:0 0 14px}table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #202936;font-size:13px}th{width:36%;color:#95a3b5;font-weight:500}td{color:#e8edf3}code{white-space:pre-wrap;word-break:break-all;color:#9cc7ff}.wide{grid-column:1/-1}.warning{border-left:3px solid #d9a441}li{margin:7px 0;color:#c7d0da}.foot{margin-top:20px;color:#7f8b99;font-size:11px}.chart-panel{overflow:hidden}.chart-head{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.legend{display:flex;gap:12px;flex-wrap:wrap;color:#aeb9c6;font-size:11px}.legend span{display:inline-flex;align-items:center;gap:5px}.legend-dot{width:8px;height:8px;border-radius:999px;display:inline-block}svg{display:block;width:100%;height:auto}.gridline{stroke:#27313d;stroke-width:1}.zero-line{stroke:#617083;stroke-width:1}.axis-label{fill:#8795a6;font:10px ui-monospace,SFMono-Regular,Menlo,monospace}.series{fill:none;stroke-width:2}.series-0{stroke:#67b7ff}.series-1{stroke:#8bd49c}.series-2{stroke:#e4ba72}.series-3{stroke:#d58cff}.series-4{stroke:#ff8b86}.series-5{stroke:#7ad7d0}.series-bg-0{background:#67b7ff}.series-bg-1{background:#8bd49c}.series-bg-2{background:#e4ba72}.series-bg-3{background:#d58cff}.series-bg-4{background:#ff8b86}.series-bg-5{background:#7ad7d0}.empty{color:#8795a6;font-size:12px}.raw-data{display:none}
@media(max-width:760px){.wrap{padding:16px}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.chart-head{display:grid}}
@media print{body,.hero,.panel{background:#fff;color:#111}.hero,.panel{border-color:#ccc}.hero p,.meta,th,li{color:#444}td{color:#111}.gridline{stroke:#ddd}.axis-label{fill:#555}}
</style></head><body><main class="wrap"><header class="hero"><div class="kicker">QuantAgent · governed research artifact</div><h1>${escapeHtml(report.title)}</h1><p>${escapeHtml(report.subtitle)}</p><div class="meta"><span>生成时间 ${escapeHtml(report.generatedAt)}</span><span>真实/模拟状态以各 artifact provenance 为准</span><span>非投资建议</span></div></header><div class="grid">${charts}${sections}<section class="panel wide"><h2>数据来源与口径</h2><table><tbody>${provenance}</tbody></table></section><section class="panel wide warning"><h2>限制与未通过项</h2><ul>${limitations}</ul></section></div><script type="application/json" id="quantagent-chart-data" class="raw-data">${chartData}</script><footer class="foot">本文件为离线单文件 HTML；图表、样式与数据均内嵌。不包含 API Key，不执行网络请求，也不会把 unavailable 数据替换为模拟值。</footer></main></body></html>`;
}

export function downloadOfflineResearchHtml(filename: string, report: OfflineResearchReport): void {
  const html = buildOfflineResearchHtml(report);
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
