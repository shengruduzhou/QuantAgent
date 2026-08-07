import { DownloadSimple, FileHtml, FileText, ShieldCheck, TrendUp, WarningCircle } from "@phosphor-icons/react";
import type { BacktestSummary, RiskOverview, SystemOverview } from "../api/types";
import { downloadJson } from "../api/client";
import { useApi } from "../hooks/useApi";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { formatNumber, formatPercent } from "../utils/format";

function downloadHtml(name: string, report: Record<string, unknown>): void {
  const payload = JSON.stringify(report).replace(/</g, "\\u003c");
  const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuantAgent Research Report</title><style>body{font:14px system-ui;margin:0;background:#0b0f14;color:#e6edf3}main{max-width:1100px;margin:auto;padding:28px}.card{background:#151b23;border:1px solid #30363d;border-radius:12px;padding:18px;margin:12px 0}pre{white-space:pre-wrap;word-break:break-word;color:#c9d1d9}small{color:#8b949e}</style></head><body><main><small>QuantAgent · Offline Research Report · 非投资建议</small><h1>Research Evidence Report</h1><div class="card"><p>该文件由已持久化 runtime artifact 生成；缺失字段不会由前端推断或模拟补齐。</p></div><div class="card"><pre id="payload"></pre></div></main><script>const data=${payload};document.getElementById('payload').textContent=JSON.stringify(data,null,2);</script></body></html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" }); const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click(); URL.revokeObjectURL(url);
}

export function ReportsPage(): JSX.Element {
  const overview = useApi<SystemOverview>(["reports-overview"], "/system/overview");
  const backtests = useApi<BacktestSummary[]>(["reports-backtests"], "/backtests");
  const risk = useApi<RiskOverview>(["reports-risk"], "/risk/overview");
  const data = overview.data?.data; const latest = data?.latestBacktest;
  if (overview.isLoading) return <StateView state="loading" />;
  if (!data) return <StateView state="empty" />;
  const report: Record<string, unknown> = {
    generatedFrom: "QuantAgent persisted runtime artifacts", generatedAt: new Date().toISOString(), latestBacktest: latest,
    risk: risk.data?.data, model: data.latestModel, selection: data.latestSelection,
    evidenceCompleteness: { backtest: Boolean(latest), risk: Boolean(risk.data?.data), model: Boolean(data.latestModel), selection: Boolean(data.latestSelection) },
    limitations: ["No live orders are generated.", "Missing per-trade factor attribution remains unavailable.", "Independent factor trades are only shown when persisted."],
  };
  return <div className="page institutional-workbench reports-page">
    <section className="report-hero panel"><div><span className="report-kicker"><FileText size={17} /> QuantAgent Research Brief</span><h2>{latest?.name ?? "当前量化研究状态"}</h2><p>基于真实 runtime artifact 自动汇总。用于研究复盘、模型诊断与风险讨论，不构成投资建议。</p></div><div className="report-export-actions"><button className="secondary-button" onClick={() => downloadJson("quantagent-research-report.json", report)}><DownloadSimple size={16}/> JSON 数据</button><button className="primary-button" onClick={() => downloadHtml("quantagent-research-report.html", report)}><FileHtml size={16}/> 离线 HTML</button></div></section>
    <section className="report-grid">
      <Panel title="策略逻辑摘要" eyebrow="Strategy Summary"><ReportParagraph icon={TrendUp} title="收益与执行">当前主回测总收益为 {formatPercent(latest?.totalReturn)}，年化收益 {formatPercent(latest?.annualReturn)}，Sharpe {formatNumber(latest?.sharpe)}，最大回撤 {formatPercent(latest?.maxDrawdown)}。交易与收益字段仅来自标准 order blotter 与 NAV artifact。</ReportParagraph><ReportParagraph icon={ShieldCheck} title="安全边界">QuantAgent UI 只消费 target weights、回测、模型和风险产物。Agent 与 optimizer 不直接生成真实订单，凭据不进入浏览器。</ReportParagraph></Panel>
      <Panel title="证据完整度" eyebrow="Evidence Completeness"><ul className="report-list"><li><strong>Backtest</strong><span>{latest ? "已持久化" : "缺失"}</span></li><li><strong>Risk</strong><span>{risk.data?.data ? "已持久化" : "缺失"}</span></li><li><strong>Model</strong><span>{data.latestModel ? "已持久化" : "缺失"}</span></li><li><strong>Selection</strong><span>{data.latestSelection ? "已持久化" : "缺失"}</span></li></ul></Panel>
      <Panel title="收益来源归因" eyebrow="Return Attribution"><ul className="report-list"><li><strong>模型版本</strong><span>{data.latestModel?.version ?? "暂无 persisted version"}</span></li><li><strong>因子版本</strong><span>{latest?.factorVersion ?? "暂无 run metadata"}</span></li><li><strong>研究股票池</strong><span>{data.stockPoolCount ?? 0} names / {data.candidateCount ?? 0} candidates</span></li><li><strong>T+1 做 T 贡献</strong><span>{formatPercent(latest?.tContribution)}</span></li><li><strong>总成本</strong><span>{formatNumber(latest?.totalCost)}</span></li></ul></Panel>
      <Panel title="风险来源归因" eyebrow="Risk Attribution"><ul className="report-list"><li><strong>单票最大亏损</strong><span>{formatNumber(risk.data?.data.maxSingleStockLoss)}</span></li><li><strong>单日最大亏损</strong><span>{formatPercent(risk.data?.data.maxDailyLoss)}</span></li><li><strong>连续亏损天数</strong><span>{risk.data?.data.consecutiveLossDays ?? "暂无"}</span></li><li><strong>流动性风险</strong><span>{formatPercent(risk.data?.data.liquidityRisk)}</span></li><li><strong>跌停风险</strong><span>{formatPercent(risk.data?.data.limitDownRisk)}</span></li></ul></Panel>
      <Panel title="当前缺陷" eyebrow="Known Limitations"><div className="limitation-list"><ReportParagraph icon={WarningCircle} title="逐笔解释缺口">没有逐笔 signal reason / factor contribution 的 artifact 时，UI 显示“暂无数据”，不会从结果反推原因。</ReportParagraph><ReportParagraph icon={WarningCircle} title="单因子成交缺口">只保存 IC/ICIR 而没有独立成交 artifact 的因子不会显示伪造买卖点。</ReportParagraph></div></Panel>
      <Panel title="实验目录" eyebrow={`${backtests.data?.data.length ?? 0} backtests`} className="report-experiment-list"><div className="report-run-list">{(backtests.data?.data ?? []).slice(0,12).map((run)=><div key={run.id}><span><strong>{run.name}</strong><small>{run.horizon ?? "research"}</small></span><span className={(run.annualReturn ?? 0)>=0?"tone-positive":"tone-negative"}>{formatPercent(run.annualReturn)}</span><code>{run.path}</code></div>)}</div></Panel>
    </section>
  </div>;
}

function ReportParagraph({ icon: Icon, title, children }: { icon: typeof FileText; title: string; children: React.ReactNode; }): JSX.Element { return <div className="report-paragraph"><Icon size={20} weight="duotone" /><div><strong>{title}</strong><p>{children}</p></div></div>; }
