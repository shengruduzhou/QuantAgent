import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { ArrowSquareOut, DownloadSimple, Flask, Graph, ShieldCheck } from "@phosphor-icons/react";
import { Link, useSearchParams } from "react-router-dom";
import { EChart } from "../../components/EChart";
import { StateView } from "../../components/StateView";
import { StatusBadge } from "../../components/StatusBadge";
import { useApi } from "../../hooks/useApi";
import { useVNextChartPalette } from "../theme";

interface Playbook { id: string; title: string; kind: string; status: string; route: string; evidence: string[]; }
interface Catalog { count: number; items: Playbook[]; source: string; rule: string; }
type Payload = Record<string, unknown>;

function finite(value: unknown): number | null { return typeof value === "number" && Number.isFinite(value) ? value : null; }
function pct(value: unknown): string { const n = finite(value); return n === null ? "—" : `${n >= 0 ? "+" : ""}${(n * 100).toFixed(2)}%`; }
function num(value: unknown, digits = 2): string { const n = finite(value); return n === null ? "—" : n.toLocaleString("zh-CN", { maximumFractionDigits: digits }); }
function recordOf(value: unknown): Record<string, unknown> | undefined { return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined; }
function latestAnomalyReason(row: Record<string, unknown>): string {
  const events = Array.isArray(row.events) ? row.events.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object")) : [];
  const event = events[0]; return String(event?.reason ?? event?.anomaly_reason ?? event?.reason_desc ?? "—");
}
function rowsOf(payload: Payload | undefined, key = "rows"): Array<Record<string, unknown>> {
  const value = payload?.[key]; return Array.isArray(value) ? value.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object")) : [];
}
function downloadOfflineReport(playbook: Playbook, payload: Payload): void {
  const escaped = JSON.stringify(payload).replace(/</g, "\\u003c");
  const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${playbook.id} ${playbook.title}</title><style>body{font:14px system-ui;margin:0;background:#0d1117;color:#e6edf3}main{max-width:1180px;margin:auto;padding:28px}.card{border:1px solid #30363d;border-radius:14px;padding:18px;margin:14px 0;background:#161b22}pre{white-space:pre-wrap;word-break:break-word;color:#c9d1d9}small{color:#8b949e}h1{font-size:24px}</style></head><body><main><small>QuantAgent · Fuyao Research Playbook</small><h1>${playbook.id} · ${playbook.title}</h1><div class="card"><strong>数据与口径</strong><p>本报告由 QuantAgent 服务端真实数据/持久化研究结果生成；缺失项不会使用模拟金融数据填补。非投资建议。</p></div><div class="card"><pre id="out"></pre></div></main><script>const data=${escaped};document.getElementById('out').textContent=JSON.stringify(data,null,2);</script></body></html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" }); const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `quantagent-playbook-${playbook.id}.html`; anchor.click(); URL.revokeObjectURL(url);
}

export function MarketPlaybooksPage(): JSX.Element {
  const palette = useVNextChartPalette(); const [params, setParams] = useSearchParams();
  const catalog = useApi<Catalog>(["market-playbooks"], "/market/playbooks", undefined, { staleTime: 10 * 60_000 });
  const items = catalog.data?.data?.items ?? []; const requested = params.get("id") ?? "10";
  const [symbol, setSymbol] = useState(params.get("symbol") ?? "600519.SH"); const [benchmark, setBenchmark] = useState(params.get("benchmark") ?? "000300.SH"); const [indexSymbol, setIndexSymbol] = useState(params.get("indexSymbol") ?? "881101.TI"); const [costBps, setCostBps] = useState(8);
  const selected = items.find((item) => item.id === requested) ?? items[0]; const computed = Boolean(selected);
  const run = useApi<Payload>(["market-playbook-run", selected?.id, symbol, benchmark, indexSymbol, costBps], computed && selected ? `/market/playbooks/${selected.id}` : null, { symbol, benchmark, indexSymbol, costBps }, { staleTime: 60_000 });
  const payload = run.data?.data;

  const option = useMemo<EChartsOption | null>(() => {
    if (!selected || !payload) return null;
    if (selected.id === "01" || selected.id === "03") {
      const bars = rowsOf(payload,"bars"); return { animation:false, tooltip:{trigger:"axis"}, grid:{left:55,right:24,top:24,bottom:52}, xAxis:{type:"category",data:bars.map(r=>String(r.datetime??r.trade_date??"")),axisLabel:{hideOverlap:true}}, yAxis:{type:"value",scale:true}, dataZoom:[{type:"inside"},{type:"slider",height:18,bottom:8}], series:[{type:"line",showSymbol:false,data:bars.map(r=>r.close),lineStyle:{color:palette.primary,width:2}}] };
    }
    if (selected.id === "02") {
      const rows = rowsOf(payload); return { animation:false, tooltip:{trigger:"axis"}, legend:{data:["营收增长","净利率","现金转化","负债率"]}, grid:{left:55,right:30,top:38,bottom:45}, xAxis:{type:"category",data:rows.map(r=>String(r.fiscalYear??""))}, yAxis:{type:"value",axisLabel:{formatter:(value:number)=>`${(value*100).toFixed(0)}%`}}, series:[{name:"营收增长",type:"bar",data:rows.map(r=>r.revenueGrowth),itemStyle:{color:palette.primary}},{name:"净利率",type:"line",data:rows.map(r=>r.netMargin),lineStyle:{color:palette.series[1]}},{name:"现金转化",type:"line",data:rows.map(r=>r.cashConversion),lineStyle:{color:palette.series[2]}},{name:"负债率",type:"line",data:rows.map(r=>r.debtToAssets),lineStyle:{color:palette.series[3]}}] };
    }
    if (selected.id === "06") {
      const rows = rowsOf(payload); return { animation:false, tooltip:{trigger:"axis"}, legend:{data:["20日收益","20日成交额"]}, grid:{left:60,right:60,top:38,bottom:58}, xAxis:{type:"category",data:rows.slice(0,40).map(r=>String(r.symbol??"")),axisLabel:{rotate:45}}, yAxis:[{type:"value",axisLabel:{formatter:(value:number)=>`${(value*100).toFixed(0)}%`}},{type:"value",position:"right"}], series:[{name:"20日收益",type:"bar",data:rows.slice(0,40).map(r=>r.return20d),itemStyle:{color:palette.primary}},{name:"20日成交额",type:"line",yAxisIndex:1,showSymbol:false,data:rows.slice(0,40).map(r=>r.turnover20d),lineStyle:{color:palette.series[1]}}] };
    }
    if (selected.id === "09") {
      const rows = rowsOf(payload); return { animation:false, tooltip:{trigger:"axis"}, legend:{data:["5日强度","20日强度","加速度"]}, grid:{left:100,right:30,top:38,bottom:45}, xAxis:{type:"value",axisLabel:{formatter:(value:number)=>`${(value*100).toFixed(0)}%`}}, yAxis:{type:"category",data:rows.map(r=>String(r.name??r.symbol??"")),axisLabel:{width:90,overflow:"truncate"}}, series:[{name:"5日强度",type:"bar",data:rows.map(r=>r.return5d),itemStyle:{color:palette.primary}},{name:"20日强度",type:"bar",data:rows.map(r=>r.return20d),itemStyle:{color:palette.series[1]}},{name:"加速度",type:"scatter",data:rows.map(r=>[r.acceleration,String(r.name??r.symbol??"")]),itemStyle:{color:palette.series[2]}}] };
    }
    if (selected.id === "11") {
      const rows = rowsOf(payload); return { animation:false, tooltip:{trigger:"axis"}, legend:{data:["个股","基准","热榜名次"]}, grid:{left:55,right:55,top:35,bottom:45}, xAxis:{type:"category",data:rows.map(r=>String(r.date??"")),axisLabel:{hideOverlap:true}}, yAxis:[{type:"value",scale:true,name:"指数化价格"},{type:"value",inverse:true,name:"热榜名次"}], dataZoom:[{type:"inside"}], series:[{name:"个股",type:"line",showSymbol:false,data:rows.map(r=>r.stockIndex),lineStyle:{color:palette.primary}},{name:"基准",type:"line",showSymbol:false,data:rows.map(r=>r.benchmarkIndex),lineStyle:{color:palette.series[1]}},{name:"热榜名次",type:"line",yAxisIndex:1,showSymbol:false,data:rows.map(r=>r.rank),lineStyle:{color:palette.series[2]}}] };
    }
    if (["13","14","15"].includes(selected.id)) {
      const rows = rowsOf(payload); let nav = 1; const navSeries = rows.map((r) => { if (typeof r.nav === "number") return r.nav; const ret = finite(r.netReturn) ?? 0; nav *= 1 + ret; return nav; }); return { animation:false, tooltip:{trigger:"axis"}, grid:{left:55,right:24,top:24,bottom:45}, xAxis:{type:"category",data:rows.map(r=>String(r.date??"")),axisLabel:{hideOverlap:true}}, yAxis:{type:"value",scale:true,name:"净值"}, dataZoom:[{type:"inside"}], series:[{type:"line",showSymbol:false,data:navSeries,lineStyle:{color:palette.primary,width:2},areaStyle:{opacity:0.08}}] };
    }
    if (selected.id === "12") { const rows = rowsOf(payload,"ladder"); return { animation:false, tooltip:{trigger:"axis"}, xAxis:{type:"category",data:rows.map(r=>String(r.date??""))}, yAxis:{type:"value",minInterval:1,name:"最高板"}, series:[{type:"bar",data:rows.map(r=>r.maxBoard),itemStyle:{color:palette.primary}}] }; }
    if (selected.id === "10") { const rows = rowsOf(payload); return { animation:false, tooltip:{trigger:"axis"}, legend:{data:["经营现金流","FCF proxy","净利润"]}, xAxis:{type:"category",data:rows.map(r=>String(r.fiscalYear??""))}, yAxis:{type:"value",scale:true}, series:[{name:"经营现金流",type:"bar",data:rows.map(r=>r.operatingCashFlow),itemStyle:{color:palette.primary}},{name:"FCF proxy",type:"line",data:rows.map(r=>r.freeCashFlowProxy),lineStyle:{color:palette.series[1]}},{name:"净利润",type:"line",data:rows.map(r=>r.netProfit),lineStyle:{color:palette.series[2]}}] }; }
    if (selected.id === "16") { const nodes = Array.isArray(payload.nodes) ? payload.nodes as Array<Record<string, unknown>> : []; const links = Array.isArray(payload.links) ? payload.links as Array<Record<string, unknown>> : []; return { animation:false, tooltip:{}, series:[{type:"graph",layout:"force",roam:true,label:{show:true},force:{repulsion:160,edgeLength:[50,130]},data:nodes.map(n=>({name:String(n.id??""),value:n.value,symbolSize:Math.max(12,Math.min(42,12+Math.log10(Math.abs(finite(n.value)??0)+1)*2))})),links:links.map(l=>({source:String(l.source??""),target:String(l.target??""),value:l.value}))}] }; }
    return null;
  }, [payload, selected, palette]);

  if (catalog.isLoading) return <StateView state="loading" detail="正在加载 Fuyao 16 个研究模板契约。" />;
  if (!selected) return <StateView state="empty" detail="没有可用研究模板。" />;
  const metrics = (payload?.metrics && typeof payload.metrics === "object") ? payload.metrics as Record<string, unknown> : {};

  return <div className="page institutional-workbench playbooks-page">
    <section className="playbook-hero panel"><div><span className="playbook-kicker"><Flask size={17}/> Fuyao × QuantAgent Research Playbooks</span><h2>16 个数据看板、研究与回测工作台</h2><p>功能与证据链对齐官方示例，视觉与治理保持 QuantAgent。真实数据只由服务端读取 API Key；PIT、T+1、成本、基准与缺失状态显式展示。</p></div><StatusBadge tone="neutral">{items.length}/16 registered</StatusBadge></section>
    <section className="playbook-layout"><aside className="playbook-nav panel">{items.map((item)=><button key={item.id} className={item.id===selected.id?"active":""} onClick={()=>{const next=new URLSearchParams(params);next.set("id",item.id);setParams(next)}}><b>{item.id}</b><span>{item.title}<small>{item.kind} · {item.status}</small></span></button>)}</aside>
      <main className="playbook-main"><section className="panel playbook-toolbar"><div><span className="eyebrow">PLAYBOOK {selected.id}</span><h3>{selected.title}</h3><p>{selected.evidence.join(" · ")}</p></div><div className="report-export-actions">{payload ? <button className="secondary-button" onClick={()=>downloadOfflineReport(selected,payload)}><DownloadSimple size={16}/>离线 HTML 报告</button> : null}{selected.route !== `/market-playbooks?id=${selected.id}` ? <Link className="primary-button" to={selected.route}>打开原生工作台 <ArrowSquareOut size={16}/></Link> : null}</div></section>
        <section className="panel playbook-controls"><label>{selected.id==="05"?"自选股（逗号分隔）":"股票代码"}<input value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())}/></label><label>基准指数<input value={benchmark} onChange={e=>setBenchmark(e.target.value.toUpperCase())}/></label><label>行业/概念指数<input value={indexSymbol} onChange={e=>setIndexSymbol(e.target.value.toUpperCase())}/></label><label>成本 bps<input type="number" min="0" max="500" value={costBps} onChange={e=>setCostBps(Number(e.target.value)||0)}/></label></section>
        {run.isLoading ? <StateView state="loading" detail="正在按真实数据重新计算研究工作台。" /> : !payload ? <StateView state="unavailable" detail={run.error instanceof Error ? run.error.message : "上游数据或权限不可用；不会用模拟金融数据补齐。"}/> : <>
          <section className="playbook-metrics">{Object.entries(metrics).slice(0,6).map(([key,value])=><div className="panel" key={key}><small>{key}</small><strong>{key.toLowerCase().includes("return")||key.toLowerCase().includes("drawdown")||key.toLowerCase().includes("rate")?pct(value):num(value,3)}</strong></div>)}{selected.id==="11"?<><div className="panel"><small>Spearman</small><strong>{num(payload.spearman,3)}</strong></div><div className="panel"><small>样本数</small><strong>{num(payload.sample,0)}</strong></div></>:null}{selected.id==="12"?<><div className="panel"><small>涨停数量</small><strong>{num(payload.limitUpCount,0)}</strong></div><div className="panel"><small>最高板</small><strong>{num(payload.maxBoard,0)}</strong></div></>:null}</section>
          {option ? <section className="panel playbook-chart"><EChart option={option} height={410}/></section> : null}
          {selected.id==="05" ? <section className="panel playbook-table"><table><thead><tr><th>标的</th><th>最新价</th><th>涨跌幅</th><th>异动数</th><th>最新原因</th></tr></thead><tbody>{rowsOf(payload).map((row)=>{const snapshot=recordOf(row.snapshot);return <tr key={String(row.symbol)}><td>{String(row.symbol??"")}</td><td>{num(snapshot?.last_price)}</td><td>{num(snapshot?.price_change_ratio_pct)}%</td><td>{num(row.eventCount,0)}</td><td>{latestAnomalyReason(row)}</td></tr>})}</tbody></table></section> : null}
          {selected.id==="09" ? <section className="panel playbook-table"><table><thead><tr><th>行业</th><th>5日</th><th>20日</th><th>加速度</th><th>当前上涨占比</th></tr></thead><tbody>{rowsOf(payload).map((row)=><tr key={String(row.symbol)}><td>{String(row.name??row.symbol??"")}</td><td>{pct(row.return5d)}</td><td>{pct(row.return20d)}</td><td>{pct(row.acceleration)}</td><td>{pct(row.advancingShare)}</td></tr>)}</tbody></table></section> : null}
          <section className="playbook-evidence panel"><h4><ShieldCheck size={18}/> 证据与口径</h4><pre>{JSON.stringify({provenance:payload.provenance,assumptions:payload.assumptions,notes:payload.notes,pitKey:payload.pitKey,periodKey:payload.periodKey},null,2)}</pre></section>{selected.id==="16"?<div className="playbook-caption"><Graph size={16}/> 拓扑仅穿透所选指数当前成分；不会把当前成分回填为历史成分。</div>:null}
        </>}
      </main>
    </section>
  </div>;
}
