import { useState } from "react";
import { ArrowRight, Database, FileHtml, Pulse, ShieldCheck, TrendUp, WarningCircle } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { StateView } from "../../components/StateView";
import { StatusBadge } from "../../components/StatusBadge";
import { useApi } from "../../hooks/useApi";
import { formatNumber, formatPercent } from "../../utils/format";

interface DataGroup {
  id: string;
  title: string;
  capabilities: string[];
  quantagent: string[];
}

interface BestPractice {
  id: string;
  slug: string;
  title: string;
  category: string;
  quantagent_path: string;
  endpoints: string[];
  outputs: string[];
  contract: string[];
  boundaries: string[];
}

interface BestPracticePayload {
  source: string;
  count: number;
  dataGroups: DataGroup[];
  items: BestPractice[];
  outputContract: {
    offlineHtml: boolean;
    showDataTime: boolean;
    showMode: boolean;
    showSourceEndpoint: boolean;
    showCalculationBasis: boolean;
    showNonInvestmentAdvice: boolean;
    browserApiKey: boolean;
    unavailableDataPolicy: string;
  };
}

interface HeatRadar {
  current: Record<string, { item?: Array<Record<string, unknown>> } | null>;
  top3RankTrends: Record<string, { item?: Array<Record<string, unknown>> } | null>;
  issues: Array<{ message?: string }>;
  window: { start: string; end: string; days: number };
  boundary: string;
}

interface CashflowQualityRow {
  periodEndMs: number | null;
  reportDateMs: number | null;
  cashConversion: number | null;
  freeCashFlow: number | null;
  freeCashFlowMargin: number | null;
  accrualRatio: number | null;
  receivablePressure: number | null;
  netCashRatio: number | null;
  fieldCompleteness: number;
  missingFields: string[];
}

interface CashflowQuality {
  symbol: string;
  rows: CashflowQualityRow[];
  formulas: Record<string, string>;
  boundary: string;
}

interface AttentionPrice {
  symbol: string;
  benchmark: string;
  rows: Array<Record<string, unknown>>;
  spearman: number | null;
  sampleSize: number;
  rankAxis?: string;
  boundary?: string;
}

const CATEGORY_LABELS: Record<string, string> = {
  market: "行情 / 横截面",
  financial: "财务 / PIT",
  special: "特色盘面",
  backtest: "严格回测",
};

const IMPLEMENTATION: Record<string, { label: string; status: string; detail: string }> = {
  "01": { label: "已实装", status: "ready", detail: "单股 K 线 / MA / 成交额 / 回撤 / 估值" },
  "02": { label: "已实装", status: "ready", detail: "PIT 三表与财务体检" },
  "03": { label: "已实装", status: "ready", detail: "指数历史 + 当前成分截面" },
  "04": { label: "已实装", status: "ready", detail: "涨停池 + 连板天梯" },
  "05": { label: "部分", status: "partial", detail: "当前异动已接；自选股联动仍复用通用工作站" },
  "06": { label: "已实装", status: "ready", detail: "全市场 dump / Parquet / Data Lab" },
  "07": { label: "已实装", status: "ready", detail: "day/hour 热股 + 飙升 + Top3 30日排名趋势" },
  "08": { label: "已实装", status: "ready", detail: "all / 机构 / 游资三视图" },
  "09": { label: "部分", status: "partial", detail: "单行业历史/当前成分已接；全行业 5/20/60 强度矩阵待专属聚合" },
  "10": { label: "已实装", status: "ready", detail: "官方现金流质量公式 + 字段完整度 + 披露日" },
  "11": { label: "已实装", status: "ready", detail: "排名 / 前复权股价 / 沪深300日轴 + 同期 Spearman" },
  "12": { label: "部分", status: "partial", detail: "当前涨停脉冲已接；历史封单留存序列待专属归档" },
  "13": { label: "治理已接", status: "partial", detail: "T+1/成本/复权/晋级门已锁；专属突破诊断器仍需 artifact" },
  "14": { label: "治理已接", status: "partial", detail: "T+1/现金状态契约已锁；专属状态泳道执行器仍需 artifact" },
  "15": { label: "治理已接", status: "partial", detail: "T+1/RankIC/成本契约已锁；专属形成×持有敏感性仍需 artifact" },
  "16": { label: "部分", status: "partial", detail: "龙虎榜日级数据已接；跨日概念拓扑受历史概念映射能力约束" },
};

export function FuyaoBestPracticesPage(): JSX.Element {
  const [draftSymbol, setDraftSymbol] = useState("600519.SH");
  const [symbol, setSymbol] = useState("600519.SH");
  const query = useApi<BestPracticePayload>(["fuyao-best-practices"], "/market/best-practices", undefined, { staleTime: 60 * 60_000 });
  const heat = useApi<HeatRadar>(["fuyao-heat-radar"], "/market/heat-radar?trendDays=30", undefined, { staleTime: 60_000 });
  const cashflow = useApi<CashflowQuality>(["fuyao-cashflow-quality", symbol], `/market/stocks/${encodeURIComponent(symbol)}/cashflow-quality?limit=5`, undefined, { staleTime: 5 * 60_000 });
  const attention = useApi<AttentionPrice>(["fuyao-attention-price", symbol], `/market/stocks/${encodeURIComponent(symbol)}/attention-price?days=90&benchmark=000300.SH`, undefined, { staleTime: 5 * 60_000 });
  const data = query.data?.data;

  if (query.isLoading) return <StateView state="loading" detail="加载 Fuyao 产品契约。" />;
  if (!data) return <StateView state="unavailable" title="Fuyao 产品契约不可用" detail="无法读取 /api/market/best-practices。" />;

  const heatData = heat.data?.data;
  const cashData = cashflow.data?.data;
  const cashLatest = cashData?.rows?.[0];
  const attentionData = attention.data?.data;

  const applySymbol = (): void => {
    const normalized = draftSymbol.trim().toUpperCase();
    if (/^\d{6}\.(SH|SZ|BJ)$/.test(normalized)) setSymbol(normalized);
  };

  return (
    <div className="page institutional-workbench market-intelligence-page">
      <header className="mi-hero">
        <div>
          <div className="mi-eyebrow"><Database size={16} /> Fuyao / Financial-API · Best Practices Parity</div>
          <h1>Fuyao 全场景研究工场</h1>
          <p>不是示例链接清单。16 个官方最佳实践被拆成“真实数据能力、可执行分析、页面产物与边界”四层；只有真正接通的能力显示已实装，其余保持部分覆盖或上游边界。</p>
        </div>
        <div className="mi-hero-meta">
          <StatusBadge status="ready" label={`${data.count}/16 contract registered`} />
          <span>真实数据走服务端 Fuyao</span>
          <span>unavailable 不造 mock</span>
        </div>
      </header>

      <section className="mi-metric-grid">
        {data.dataGroups.map((group) => (
          <div className="mi-metric" key={group.id}>
            <span>{group.title}</span>
            <strong>{group.capabilities.length}</strong>
            <small>{group.capabilities.join(" · ")}</small>
          </div>
        ))}
      </section>

      <section className="mi-panel mi-span-2">
        <header><div><strong>页面 / 报告统一产物契约</strong><span>所有场景必须遵守，而不是每个页面各写一套口径</span></div><FileHtml size={18} /></header>
        <div className="capability-grid">
          <Contract label="离线单文件 HTML" ready={data.outputContract.offlineHtml} />
          <Contract label="显著数据时间" ready={data.outputContract.showDataTime} />
          <Contract label="真实 / 模拟模式" ready={data.outputContract.showMode} />
          <Contract label="来源 endpoint" ready={data.outputContract.showSourceEndpoint} />
          <Contract label="计算口径" ready={data.outputContract.showCalculationBasis} />
          <Contract label="非投资建议" ready={data.outputContract.showNonInvestmentAdvice} />
          <Contract label="浏览器不含 API Key" ready={!data.outputContract.browserApiKey} />
          <div><span>缺失数据</span><strong>{data.outputContract.unavailableDataPolicy}</strong></div>
        </div>
      </section>

      <section className="mi-panel mi-span-2">
        <header><div><strong>已实装 Fuyao 分析工具</strong><span>参数只有点击“应用”后才会重新取数，避免输入过程中连续请求</span></div><Pulse size={18} /></header>
        <div className="market-search-shell">
          <input value={draftSymbol} onChange={(event) => setDraftSymbol(event.target.value)} onKeyDown={(event) => event.key === "Enter" && applySymbol()} aria-label="Fuyao 分析股票代码" />
          <button className="primary-button" type="button" onClick={applySymbol}>应用 {symbol}</button>
        </div>
        <div className="mi-metric-grid">
          <LiveMetric
            title="07 · 热度雷达"
            status={heat.data?.status ?? (heat.isLoading ? "loading" : "unavailable")}
            value={heatData ? `${countItems(heatData.current.hotDay)} / ${countItems(heatData.current.hotHour)}` : "—"}
            detail={heatData ? `24h/小时热股 · Top3趋势 ${Object.keys(heatData.top3RankTrends ?? {}).length} 只 · ${heatData.window.start} → ${heatData.window.end}` : "等待真实 Fuyao 数据"}
          />
          <LiveMetric
            title="10 · 现金流质量"
            status={cashflow.data?.status ?? (cashflow.isLoading ? "loading" : "unavailable")}
            value={cashLatest ? formatPercent(cashLatest.freeCashFlowMargin) : "—"}
            detail={cashLatest ? `FCF率 · 现金转化 ${formatNumber(cashLatest.cashConversion)} · 完整度 ${formatPercent(cashLatest.fieldCompleteness)}` : `${symbol} 暂无可用年度三表`}
          />
          <LiveMetric
            title="11 · 热榜—股价"
            status={attention.data?.status ?? (attention.isLoading ? "loading" : "unavailable")}
            value={attentionData?.spearman == null ? "—" : formatNumber(attentionData.spearman)}
            detail={attentionData ? `同期 Spearman · n=${attentionData.sampleSize} · benchmark ${attentionData.benchmark}` : `${symbol} 暂无对齐样本`}
          />
        </div>
        {heatData?.boundary ? <div className="backtest-context-note"><WarningCircle size={14} /> {heatData.boundary}</div> : null}
        {cashData?.boundary ? <div className="backtest-context-note"><ShieldCheck size={14} /> {cashData.boundary}</div> : null}
        {attentionData?.boundary ? <div className="backtest-context-note"><TrendUp size={14} /> {attentionData.boundary}</div> : null}
      </section>

      <section className="mi-coverage-grid">
        {data.items.map((item) => {
          const implementation = IMPLEMENTATION[item.id] ?? { label: "契约已登记", status: "partial", detail: "需要进一步验证" };
          return (
            <article className="mi-panel" key={item.id}>
              <header>
                <div><strong>{item.id} · {item.title}</strong><span>{CATEGORY_LABELS[item.category] ?? item.category}</span></div>
                <StatusBadge status={implementation.status} label={implementation.label} />
              </header>
              <div className="backtest-context-note">{implementation.detail}</div>
              <div className="fuyao-contract-block">
                <b>必须输出</b><p>{item.outputs.join(" · ")}</p>
                <b>计算 / 执行契约</b><ul>{item.contract.map((line) => <li key={line}>{line}</li>)}</ul>
                <b>边界</b><ul>{item.boundaries.map((line) => <li key={line}><WarningCircle size={13} /> {line}</li>)}</ul>
                <details><summary>数据 endpoints ({item.endpoints.length})</summary><div className="mi-source-strip">{item.endpoints.map((endpoint) => <code key={endpoint}>{endpoint}</code>)}</div></details>
              </div>
              <Link className="primary-button" to={item.quantagent_path}>进入 QuantAgent 实际工作站 <ArrowRight size={14} /></Link>
            </article>
          );
        })}
      </section>

      <section className="mi-source-strip">
        <ShieldCheck size={16} /><strong>QuantAgent 额外治理：</strong>
        <span>PIT / available_at</span><span>purged walk-forward + embargo</span><span>explicit benchmark</span><span>PBO / DSR / SPA</span><span>final holdout</span><span>T+1 close-signal execution</span>
      </section>
    </div>
  );
}

function Contract({ label, ready }: { label: string; ready: boolean }): JSX.Element {
  return <div><span>{label}</span><StatusBadge status={ready ? "ready" : "blocked"} label={ready ? "required" : "blocked"} /></div>;
}

function LiveMetric({ title, status, value, detail }: { title: string; status: string; value: string; detail: string }): JSX.Element {
  return <div className="mi-metric"><span>{title}</span><strong>{value}</strong><small>{detail}</small><StatusBadge status={status} /></div>;
}

function countItems(panel: { item?: Array<Record<string, unknown>> } | null | undefined): number {
  return Array.isArray(panel?.item) ? panel.item.length : 0;
}
