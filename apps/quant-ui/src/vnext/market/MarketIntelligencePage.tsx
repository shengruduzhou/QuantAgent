import { useEffect, useMemo, useState, type FormEvent } from "react";
import type { EChartsOption } from "echarts";
import { Buildings, ChartBar, Coins, Database, Fire, Pulse, TrendUp } from "@phosphor-icons/react";
import { Link, useSearchParams } from "react-router-dom";
import { EChart } from "../../components/EChart";
import { StateView } from "../../components/StateView";
import { StatusBadge } from "../../components/StatusBadge";
import { useApi } from "../../hooks/useApi";
import { formatCompact } from "../../utils/format";
import { useVNextChartPalette } from "../theme";

interface HotItem {
  thscode?: string;
  ticker?: string;
  name?: string;
  rank?: number;
  heat?: string | number;
  rank_change?: number | null;
  rank_trend?: string;
}

interface DragonItem {
  thscode?: string;
  ticker?: string;
  name?: string;
  net_value?: number;
  org_net_value?: number;
  hot_money_net_value?: number;
  hot_rank?: number;
  range_days?: number;
}

interface LadderStock { thscode?: string; ticker?: string; name?: string; board_num?: number; }
interface LadderRow { date?: string; boards?: Record<string, LadderStock[]>; }
interface IntelligencePayload {
  timestamp?: number;
  item?: HotItem[] | LadderRow[];
  stock_items?: DragonItem[];
  hot_money_items?: Array<Record<string, unknown>>;
  stock_count?: number;
  count?: number;
  trade_date?: string;
  window?: { length?: number; date_list?: string[] };
}
interface MarketIntelligenceData {
  source: string;
  panels: {
    hotDay?: IntelligencePayload | null;
    hotHour?: IntelligencePayload | null;
    skyrocketDay?: IntelligencePayload | null;
    skyrocketHour?: IntelligencePayload | null;
    dragonAll?: IntelligencePayload | null;
    dragonOrg?: IntelligencePayload | null;
    dragonHotMoney?: IntelligencePayload | null;
    limitLadder?: IntelligencePayload | null;
  };
  issues: Array<{ panel: string; endpoint: string; message: string }>;
  provenance: Record<string, string>;
}

interface IndexCatalogItem { thscode: string; name: string; }
interface IndexCatalogData { tag: string; timestamp?: number; items: IndexCatalogItem[]; source: string; endpoint: string; }
interface IndexSnapshot { symbol: string; name?: string | null; lastPrice?: number | null; changePercent?: number | null; amount?: number | null; }
interface IndexBar { datetime: string; symbol: string; open: number; high: number; low: number; close: number; volume?: number | null; amount?: number | null; }
interface IndexOverviewData { symbol: string; source: string; bars: IndexBar[]; constituentCount: number; constituents: Array<Record<string, unknown>>; snapshots: IndexSnapshot[]; provenance: Record<string, string>; }

interface FinancialHealthData {
  symbol: string;
  source: string;
  period: string;
  statements: {
    income: Array<Record<string, unknown>>;
    balance: Array<Record<string, unknown>>;
    cashflow: Array<Record<string, unknown>>;
  };
  provenance: Record<string, string>;
  pitKey: string;
  periodKey: string;
}

type ViewKey = "pulse" | "sector" | "financial" | "coverage";
type IndexTag = "industry" | "cn_concept";

const COVERAGE = [
  ["01", "单股行情与趋势速览", "原生", "/stock-replay"],
  ["02", "单股财务体检", "原生", "financial"],
  ["03", "概念板块联动", "原生", "sector"],
  ["04", "涨停池与连板天梯", "行情脉冲", "pulse"],
  ["05", "自选股当日异动监控", "capability 可接", "pulse"],
  ["06", "本地全市场趋势研究", "Data Lab", "/runtime?view=data"],
  ["07", "市场热度与飙升雷达", "原生", "pulse"],
  ["08", "龙虎榜机构与游资观察", "原生", "pulse"],
  ["09", "行业强度作战矩阵", "原生", "sector"],
  ["10", "现金流质量稽核台", "财务视图", "financial"],
  ["11", "热榜—股价关系观察台", "单股联动", "/stock-replay"],
  ["12", "涨停情绪市场脉冲屏", "行情脉冲", "pulse"],
  ["13", "价格成交量突破回测台", "Backtester", "/backtests"],
  ["14", "时间序列动量回测台", "Backtester", "/backtests"],
  ["15", "短期反转回测实验室", "Backtester", "/backtests"],
  ["16", "龙虎榜资金流向拓扑台", "龙虎榜", "pulse"],
] as const;

function panelItems<T>(payload: IntelligencePayload | null | undefined): T[] {
  return Array.isArray(payload?.item) ? payload.item as T[] : [];
}

function money(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? formatCompact(value) : "—";
}

function percentPoints(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}%` : "—";
}

function dateFromMs(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? new Date(value).toLocaleDateString("zh-CN") : "—";
}

function readNumber(row: Record<string, unknown> | undefined, keys: string[]): number | null {
  if (!row) return null;
  for (const key of keys) {
    const value = row[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }): JSX.Element {
  return <div className="mi-metric"><span>{label}</span><strong>{value}</strong>{detail ? <small>{detail}</small> : null}</div>;
}

function DataTable({ headers, rows }: { headers: string[]; rows: Array<Array<string | number>> }): JSX.Element {
  return (
    <div className="mi-table-wrap">
      <table className="mi-table"><thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead>
        <tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={`${rowIndex}-${cellIndex}`}>{cell}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

export function MarketIntelligencePage(): JSX.Element {
  const palette = useVNextChartPalette();
  const [searchParams] = useSearchParams();
  const initialSymbol = searchParams.get("symbol")?.toUpperCase() ?? "600519.SH";
  const [view, setView] = useState<ViewKey>("pulse");
  const [indexTag, setIndexTag] = useState<IndexTag>("industry");
  const [selectedIndex, setSelectedIndex] = useState("");
  const [financialInput, setFinancialInput] = useState(initialSymbol);
  const [financialSymbol, setFinancialSymbol] = useState(initialSymbol);

  const intelligence = useApi<MarketIntelligenceData>(["market-intelligence"], "/market/intelligence", undefined, { staleTime: 30_000 });
  const catalog = useApi<IndexCatalogData>(["market-index-catalog", indexTag], "/market/indexes", { tag: indexTag }, { staleTime: 10 * 60_000 });
  const indexOverview = useApi<IndexOverviewData>(["market-index-overview", selectedIndex], selectedIndex ? `/market/indexes/${encodeURIComponent(selectedIndex)}/overview` : null, { calendarDays: 180 }, { staleTime: 60_000 });
  const financial = useApi<FinancialHealthData>(["market-financial-health", financialSymbol], financialSymbol ? `/market/stocks/${encodeURIComponent(financialSymbol)}/financial-health` : null, { limit: 5 }, { staleTime: 5 * 60_000 });

  useEffect(() => {
    const items = catalog.data?.data?.items ?? [];
    if (!items.length) return;
    if (!selectedIndex || !items.some((item) => item.thscode === selectedIndex)) setSelectedIndex(items[0].thscode);
  }, [catalog.data?.data?.items, selectedIndex]);

  const panels = intelligence.data?.data?.panels;
  const hotRows = panelItems<HotItem>(panels?.hotDay).slice(0, 12);
  const skyrocketRows = panelItems<HotItem>(panels?.skyrocketDay).slice(0, 12);
  const dragonRows = (panels?.dragonAll?.stock_items ?? []).slice().sort((a, b) => Math.abs(b.net_value ?? 0) - Math.abs(a.net_value ?? 0)).slice(0, 12);
  const ladderRows = panelItems<LadderRow>(panels?.limitLadder);

  const ladderSeries = useMemo(() => ladderRows.slice().reverse().map((row) => {
    const stocks = Object.values(row.boards ?? {}).flat();
    return { date: row.date ?? "—", maxBoard: stocks.reduce((max, stock) => Math.max(max, stock.board_num ?? 0), 0), count: stocks.length };
  }), [ladderRows]);

  const hotOption = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 80, right: 24, top: 18, bottom: 28 },
    xAxis: { type: "value", name: "排名变化" },
    yAxis: { type: "category", inverse: true, data: hotRows.slice(0, 10).map((item) => item.name ?? item.ticker ?? item.thscode ?? "—") },
    series: [{ type: "bar", data: hotRows.slice(0, 10).map((item) => item.rank_change ?? 0), itemStyle: { color: palette.primary } }],
  }), [hotRows, palette.primary]);

  const dragonOption = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => money(value) },
    grid: { left: 84, right: 28, top: 18, bottom: 28 },
    xAxis: { type: "value", name: "净额" },
    yAxis: { type: "category", inverse: true, data: dragonRows.slice(0, 10).map((item) => item.name ?? item.ticker ?? item.thscode ?? "—") },
    series: [{ type: "bar", data: dragonRows.slice(0, 10).map((item) => item.net_value ?? 0), itemStyle: { color: palette.series[1] } }],
  }), [dragonRows, palette.series]);

  const ladderOption = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 48, right: 24, top: 18, bottom: 42 },
    xAxis: { type: "category", data: ladderSeries.map((item) => item.date), axisLabel: { hideOverlap: true } },
    yAxis: { type: "value", minInterval: 1, name: "最高板" },
    series: [{ type: "line", data: ladderSeries.map((item) => item.maxBoard), smooth: false, showSymbol: true, lineStyle: { color: palette.primary }, itemStyle: { color: palette.primary } }],
  }), [ladderSeries, palette.primary]);

  const indexData = indexOverview.data?.data;
  const indexLineOption = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 52, right: 22, top: 18, bottom: 38 },
    xAxis: { type: "category", data: (indexData?.bars ?? []).map((bar) => bar.datetime.slice(0, 10)), axisLabel: { hideOverlap: true } },
    yAxis: { type: "value", scale: true },
    dataZoom: [{ type: "inside" }],
    series: [{ type: "line", showSymbol: false, data: (indexData?.bars ?? []).map((bar) => bar.close), lineStyle: { color: palette.primary, width: 1.5 } }],
  }), [indexData?.bars, palette.primary]);

  const breadth = useMemo(() => {
    const rows = indexData?.snapshots ?? [];
    const up = rows.filter((row) => (row.changePercent ?? 0) > 0).length;
    const down = rows.filter((row) => (row.changePercent ?? 0) < 0).length;
    const flat = rows.length - up - down;
    return { up, down, flat, sample: rows.length };
  }, [indexData?.snapshots]);

  const constituentOption = useMemo<EChartsOption>(() => {
    const rows = (indexData?.snapshots ?? []).slice().sort((a, b) => (b.changePercent ?? -Infinity) - (a.changePercent ?? -Infinity)).slice(0, 15);
    return {
      animation: false,
      tooltip: { trigger: "axis", valueFormatter: (value: unknown) => percentPoints(value) },
      grid: { left: 92, right: 24, top: 18, bottom: 28 },
      xAxis: { type: "value", name: "涨跌幅 %" },
      yAxis: { type: "category", inverse: true, data: rows.map((row) => row.name ?? row.symbol) },
      series: [{ type: "bar", data: rows.map((row) => row.changePercent ?? 0), itemStyle: { color: palette.series[2] } }],
    };
  }, [indexData?.snapshots, palette.series]);

  const financialData = financial.data?.data;
  const income = financialData?.statements.income ?? [];
  const balance = financialData?.statements.balance ?? [];
  const cashflow = financialData?.statements.cashflow ?? [];
  const financialRows = useMemo(() => income.slice().reverse().map((row) => {
    const year = row.fiscal_year ?? dateFromMs(row.period_end_ms);
    const samePeriod = (rows: Array<Record<string, unknown>>) => rows.find((candidate) => candidate.period_end_ms === row.period_end_ms) ?? rows.find((candidate) => candidate.fiscal_year === row.fiscal_year);
    const balanceRow = samePeriod(balance);
    const cashRow = samePeriod(cashflow);
    return {
      year: String(year),
      revenue: readNumber(row, ["operating_revenue", "total_operating_revenue", "total_revenue", "revenue"]),
      profit: readNumber(row, ["net_profit_parent_company", "net_profit_attributable_to_parent", "net_profit"]),
      operatingCash: readNumber(cashRow, ["act_cash_flow_net", "net_cash_flow_operating"]),
      assets: readNumber(balanceRow, ["total_assets"]),
      liabilities: readNumber(balanceRow, ["total_liabilities"]),
      reportDate: dateFromMs(row.report_date_ms),
    };
  }), [balance, cashflow, income]);

  const financialOption = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => money(value) },
    legend: { data: ["营收", "净利润", "经营现金流"] },
    grid: { left: 70, right: 24, top: 38, bottom: 34 },
    xAxis: { type: "category", data: financialRows.map((row) => row.year) },
    yAxis: { type: "value", axisLabel: { formatter: (value: number) => formatCompact(value) } },
    series: [
      { name: "营收", type: "bar", data: financialRows.map((row) => row.revenue), itemStyle: { color: palette.series[0] } },
      { name: "净利润", type: "line", data: financialRows.map((row) => row.profit), lineStyle: { color: palette.series[2] }, itemStyle: { color: palette.series[2] } },
      { name: "经营现金流", type: "line", data: financialRows.map((row) => row.operatingCash), lineStyle: { color: palette.series[3] }, itemStyle: { color: palette.series[3] } },
    ],
  }), [financialRows, palette.series]);

  const submitFinancial = (event: FormEvent): void => {
    event.preventDefault();
    const value = financialInput.trim().toUpperCase();
    if (value) setFinancialSymbol(value);
  };

  const setCoverageTarget = (target: string): void => {
    if (target === "pulse" || target === "sector" || target === "financial" || target === "coverage") setView(target);
  };

  const panelStatus = intelligence.data?.status ?? "loading";
  const indexItems = catalog.data?.data?.items ?? [];
  const latestFinancial = financialRows.at(-1);

  return (
    <div className="page institutional-workbench market-intelligence-page">
      <header className="mi-hero">
        <div><div className="mi-eyebrow"><Pulse size={16} /> Financial-API Reference · QuantAgent Native</div><h1>市场数据情报台</h1><p>同花顺 Fuyao 真实数据负责市场观察；QuantAgent 负责因子、模型、回测、风控与证据链。两层分离，不把榜单直接解释成交易信号。</p></div>
        <div className="mi-hero-meta"><StatusBadge status={panelStatus} label={panelStatus === "ready" ? "Fuyao live" : panelStatus} /><span>API Key 仅服务端读取</span><span>缺失 capability 不造 mock</span></div>
      </header>

      <nav className="mi-tabs" aria-label="市场数据情报视图">
        <button className={view === "pulse" ? "active" : ""} onClick={() => setView("pulse")}><Fire size={15} /> 市场脉冲</button>
        <button className={view === "sector" ? "active" : ""} onClick={() => setView("sector")}><Buildings size={15} /> 行业 / 概念</button>
        <button className={view === "financial" ? "active" : ""} onClick={() => setView("financial")}><Coins size={15} /> 财务体检</button>
        <button className={view === "coverage" ? "active" : ""} onClick={() => setView("coverage")}><Database size={15} /> 16 场景映射</button>
      </nav>

      {view === "pulse" ? <>
        <section className="mi-metric-grid">
          <Metric label="24h 热股" value={String(panelItems<HotItem>(panels?.hotDay).length)} detail="hot-stock-list · day" />
          <Metric label="日榜飙升" value={String(skyrocketRows.length)} detail="skyrocket-list · day" />
          <Metric label="龙虎榜股票" value={String(panels?.dragonAll?.stock_count ?? dragonRows.length)} detail={panels?.dragonAll?.trade_date ?? "latest available"} />
          <Metric label="连板窗口" value={`${panels?.limitLadder?.window?.length ?? ladderRows.length} 日`} detail="limit-up-ladder" />
        </section>
        {intelligence.data?.data?.issues?.length ? <div className="mi-issues">{intelligence.data.data.issues.map((issue) => <span key={issue.panel}>{issue.panel}: {issue.message}</span>)}</div> : null}
        <section className="mi-grid-2">
          <article className="mi-panel"><header><div><strong>热度排名变化</strong><span>24h 热股 Top10</span></div><Pulse size={18} /></header>{hotRows.length ? <EChart option={hotOption} className="mi-chart" ariaLabel="热股排名变化" /> : <StateView state={intelligence.isLoading ? "loading" : "empty"} detail="热股榜暂不可用。" />}</article>
          <article className="mi-panel"><header><div><strong>龙虎榜净额结构</strong><span>按绝对净额排序</span></div><ChartBar size={18} /></header>{dragonRows.length ? <EChart option={dragonOption} className="mi-chart" ariaLabel="龙虎榜净额" /> : <StateView state={intelligence.isLoading ? "loading" : "empty"} detail="龙虎榜暂不可用。" />}</article>
          <article className="mi-panel"><header><div><strong>30 日连板高度</strong><span>仅使用天梯返回的有限样本</span></div><TrendUp size={18} /></header>{ladderSeries.length ? <EChart option={ladderOption} className="mi-chart" ariaLabel="连板最高板高度" /> : <StateView state={intelligence.isLoading ? "loading" : "empty"} detail="连板天梯暂不可用。" />}</article>
          <article className="mi-panel"><header><div><strong>热股 / 飙升榜</strong><span>不同语义分栏，不合成为单一评分</span></div><Fire size={18} /></header><div className="mi-ranked-columns"><div><h3>热股</h3>{hotRows.slice(0, 8).map((item) => <div className="mi-ranked-row" key={`hot-${item.thscode}`}><b>{item.rank ?? "—"}</b><span>{item.name ?? item.thscode}</span><small>{item.rank_change == null ? "—" : `${item.rank_change > 0 ? "+" : ""}${item.rank_change}`}</small></div>)}</div><div><h3>飙升</h3>{skyrocketRows.slice(0, 8).map((item) => <div className="mi-ranked-row" key={`sky-${item.thscode}`}><b>{item.rank ?? "—"}</b><span>{item.name ?? item.thscode}</span><small>{item.rank_change == null ? "—" : `${item.rank_change > 0 ? "+" : ""}${item.rank_change}`}</small></div>)}</div></div></article>
        </section>
        <section className="mi-source-strip"><span>hot-stock-list / skyrocket-list</span><span>dragon-tiger-list: all · org · hot_money</span><span>limit-up-ladder: 30 trading days</span><strong>观察数据 ≠ 交易指令</strong></section>
      </> : null}

      {view === "sector" ? <>
        <section className="mi-control-bar"><label>目录<select value={indexTag} onChange={(event) => { setIndexTag(event.target.value as IndexTag); setSelectedIndex(""); }}><option value="industry">同花顺行业</option><option value="cn_concept">A 股概念</option></select></label><label>指数 / 板块<select value={selectedIndex} onChange={(event) => setSelectedIndex(event.target.value)}>{indexItems.map((item) => <option value={item.thscode} key={item.thscode}>{item.name} · {item.thscode}</option>)}</select></label><span>当前成分 ≠ 历史成分；成分等权涨跌不是指数贡献。</span></section>
        <section className="mi-metric-grid"><Metric label="当前成分" value={String(indexData?.constituentCount ?? 0)} detail={selectedIndex || "—"} /><Metric label="快照样本" value={String(breadth.sample)} detail="最多展示前 80 个成分快照" /><Metric label="上涨 / 下跌" value={`${breadth.up} / ${breadth.down}`} detail={`平盘 ${breadth.flat}`} /><Metric label="目录规模" value={String(indexItems.length)} detail={indexTag} /></section>
        <section className="mi-grid-2"><article className="mi-panel"><header><div><strong>指数 / 板块历史走势</strong><span>Fuyao index historical · 日线</span></div><TrendUp size={18} /></header>{indexData?.bars?.length ? <EChart option={indexLineOption} className="mi-chart mi-chart-tall" ariaLabel="行业或概念指数历史走势" /> : <StateView state={indexOverview.isLoading ? "loading" : "empty"} detail="选择板块后展示指数历史。" />}</article><article className="mi-panel"><header><div><strong>当前成分涨跌分布</strong><span>横截面快照，不推断历史归属</span></div><Buildings size={18} /></header>{indexData?.snapshots?.length ? <EChart option={constituentOption} className="mi-chart mi-chart-tall" ariaLabel="当前成分涨跌分布" /> : <StateView state={indexOverview.isLoading ? "loading" : "empty"} detail="当前没有可用成分快照。" />}</article></section>
        {indexData?.snapshots?.length ? <article className="mi-panel"><header><div><strong>成分穿透</strong><span>名称 / 代码 / 涨跌 / 成交额</span></div></header><DataTable headers={["标的", "代码", "涨跌幅", "成交额"]} rows={indexData.snapshots.slice(0, 30).map((row) => [row.name ?? "—", row.symbol, percentPoints(row.changePercent), money(row.amount)])} /></article> : null}
      </> : null}

      {view === "financial" ? <>
        <section className="mi-control-bar"><form className="mi-financial-form" onSubmit={submitFinancial}><label>完整 thscode<input value={financialInput} onChange={(event) => setFinancialInput(event.target.value)} placeholder="600519.SH" /></label><button className="primary-button" type="submit">加载财务</button></form><span>年报最近 5 期；以 report_date_ms 控制披露时点，不以报告期结束日提前使用。</span></section>
        <section className="mi-metric-grid"><Metric label="最近报告期" value={latestFinancial?.year ?? "—"} detail={`披露 ${latestFinancial?.reportDate ?? "—"}`} /><Metric label="营业收入" value={money(latestFinancial?.revenue)} detail="annual" /><Metric label="净利润" value={money(latestFinancial?.profit)} detail="annual" /><Metric label="经营现金流" value={money(latestFinancial?.operatingCash)} detail="annual" /></section>
        <section className="mi-grid-2"><article className="mi-panel mi-span-2"><header><div><strong>收入 / 利润 / 经营现金流</strong><span>同一报告期对齐三张报表</span></div><Coins size={18} /></header>{financialRows.length ? <EChart option={financialOption} className="mi-chart mi-chart-tall" ariaLabel="财务健康趋势" /> : <StateView state={financial.isLoading ? "loading" : "empty"} detail="当前财务报表暂无可展示数据。" />}</article></section>
        {financialRows.length ? <article className="mi-panel"><header><div><strong>报告期核对</strong><span>PIT: report_date_ms · Period: period_end_ms</span></div></header><DataTable headers={["年度", "披露日", "营收", "净利润", "经营现金流", "总资产", "总负债"]} rows={financialRows.map((row) => [row.year, row.reportDate, money(row.revenue), money(row.profit), money(row.operatingCash), money(row.assets), money(row.liabilities)])} /></article> : null}
      </> : null}

      {view === "coverage" ? <section className="mi-coverage-grid">{COVERAGE.map(([id, title, status, target]) => {
        const externalRoute = target.startsWith("/");
        return <article className="mi-coverage-card" key={id}><span>{id}</span><div><strong>{title}</strong><small>{status}</small></div>{externalRoute ? <Link to={target}>打开</Link> : <button type="button" onClick={() => setCoverageTarget(target)}>查看</button>}</article>;
      })}</section> : null}

      <footer className="mi-footer"><span>参考 Financial-API 16 个 inspirations 的信息架构，但不复制品牌与静态示例数据。</span><span>真实数据只从 Fuyao 后端 API / QuantAgent persisted artifacts 获取；无权限、无数据、非交易日均原样呈现。</span></footer>
    </div>
  );
}
