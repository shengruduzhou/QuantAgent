export interface ResearchReportSection {
  title: string;
  rows: Array<[string, string]>;
}

export interface OfflineResearchReport {
  title: string;
  generatedAt: string;
  subtitle: string;
  sections: ResearchReportSection[];
  provenance: Array<[string, string]>;
  limitations: string[];
}

function escapeHtml(value: unknown): string {
  return String(value ?? "—")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function buildOfflineResearchHtml(report: OfflineResearchReport): string {
  const sections = report.sections.map((section) => `
    <section class="panel">
      <h2>${escapeHtml(section.title)}</h2>
      <table><tbody>${section.rows.map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`).join("")}</tbody></table>
    </section>`).join("");
  const provenance = report.provenance.map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td><code>${escapeHtml(value)}</code></td></tr>`).join("");
  const limitations = report.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(report.title)}</title><style>
:root{color-scheme:dark;background:#0b0f14;color:#e8edf3;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;background:#0b0f14}.wrap{max-width:1180px;margin:0 auto;padding:34px}.hero{border:1px solid #27313d;background:#111820;padding:28px;margin-bottom:18px}.kicker{font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;color:#95a3b5;text-transform:uppercase}.hero h1{font-size:30px;margin:10px 0}.hero p{color:#aeb9c6;margin:0}.meta{display:flex;gap:18px;flex-wrap:wrap;margin-top:18px;font-size:12px;color:#95a3b5}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{border:1px solid #27313d;background:#111820;padding:20px}.panel h2{font-size:16px;margin:0 0 14px}table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #202936;font-size:13px}th{width:36%;color:#95a3b5;font-weight:500}td{color:#e8edf3}code{white-space:pre-wrap;word-break:break-all;color:#9cc7ff}.wide{grid-column:1/-1}.warning{border-left:3px solid #d9a441}li{margin:7px 0;color:#c7d0da}.foot{margin-top:20px;color:#7f8b99;font-size:11px}@media(max-width:760px){.wrap{padding:16px}.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
@media print{body,.hero,.panel{background:#fff;color:#111}.hero,.panel{border-color:#ccc}.hero p,.meta,th,li{color:#444}td{color:#111}}
</style></head><body><main class="wrap"><header class="hero"><div class="kicker">QuantAgent · governed research artifact</div><h1>${escapeHtml(report.title)}</h1><p>${escapeHtml(report.subtitle)}</p><div class="meta"><span>生成时间 ${escapeHtml(report.generatedAt)}</span><span>真实/模拟状态以各 artifact provenance 为准</span><span>非投资建议</span></div></header><div class="grid">${sections}<section class="panel wide"><h2>数据来源与口径</h2><table><tbody>${provenance}</tbody></table></section><section class="panel wide warning"><h2>限制与未通过项</h2><ul>${limitations}</ul></section></div><footer class="foot">本文件为离线单文件 HTML；不包含 API Key，不执行网络请求，也不会把 unavailable 数据替换为模拟值。</footer></main></body></html>`;
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
