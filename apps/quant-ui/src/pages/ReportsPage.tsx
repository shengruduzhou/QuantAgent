import { DownloadSimple, FileHtml, FileText, ShieldCheck, TrendUp, WarningCircle } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";
import type { BacktestSummary, RiskOverview, SystemOverview } from "../api/types";
import { downloadJson } from "../api/client";
import { useApi } from "../hooks/useApi";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { UNMEASURED_TITLE, formatNumber, formatPercent, toneClass } from "../utils/format";
import { downloadOfflineResearchHtml, type OfflineResearchReport } from "../utils/researchReport";

const PROMOTION_POLICY = [
  ["PBO", "≤ 0.25", "组合/模型搜索的数据挖掘风险上限"],
  ["DSR probability", "≥ 0.95", "Deflated Sharpe 必须通过多重试验修正"],
  ["SPA p-value", "≤ 0.05", "相对明确基准的优越性需通过数据挖掘修正"],
  ["Benchmark", "显式指定", "不得用空 benchmark 声称超额收益"],
  ["PIT", "必须通过", "所有输入 available_at 不晚于决策时点"],
  ["Final holdout", "仅验收一次", "模型/参数选择不得读取最终留出集"],
  ["Close signal", "T+1 执行", "收盘生成的信号不得在同一收盘价成交"],
] as const;

export function ReportsPage(): JSX.Element {
  const navigate = useNavigate();
  const overview = useApi<SystemOverview>(["reports-overview"], "/system/overview");
  const backtests = useApi<BacktestSummary[]>(["reports-backtests"], "/backtests");
  const risk = useApi<RiskOverview>(["reports-risk"], "/risk/overview");
  const data = overview.data?.data;
  const latest = data?.latestBacktest;

  if (overview.isLoading) return <StateView state="loading" />;
  if (!data) return <StateView state="empty" />;

  const limitations = [
    "若 runtime 未持久化 promotion_gate.json，则 PBO / DSR / SPA 视为未提供证据，不得标记 production eligible。",
    "缺少逐笔 signal reason、factor contribution 或 cashAfter 的历史实验保持 unavailable，不从结果反推原因。",
    "只有 daily return、没有 minute fill price/quantity 的 T+1 历史记录必须继续标记 daily-only。",
    "当前指数成分只能表示当前归属；在官方历史成分/权重能力可用前不得回填到历史日期。",
    "榜单、异动、龙虎榜与热度是 observation data；本报告不会直接把它们转换成交易建议。",
  ];

  const report = {
    generatedFrom: "QuantAgent persisted runtime artifacts",
    generatedAt: new Date().toISOString(),
    latestBacktest: latest,
    risk: risk.data?.data,
    model: data.latestModel,
    selection: data.latestSelection,
    promotionPolicy: PROMOTION_POLICY,
    limitations,
  };

  const offlineReport: OfflineResearchReport = {
    title: latest?.name ?? "QuantAgent 当前量化研究状态",
    generatedAt: report.generatedAt,
    subtitle: "基于持久化 runtime artifact 的离线研究报告。缺失证据保持缺失，不构成投资建议。",
    sections: [
      {
        title: "收益与回撤",
        rows: [
          ["总收益", formatPercent(latest?.totalReturn)],
          ["年化收益", formatPercent(latest?.annualReturn)],
          ["Sharpe", formatNumber(latest?.sharpe)],
          ["最大回撤", formatPercent(latest?.maxDrawdown)],
          ["Calmar", formatNumber(latest?.calmar)],
          ["换手率", formatPercent(latest?.turnover)],
          ["总成本", formatNumber(latest?.totalCost)],
        ],
      },
      {
        title: "风险摘要",
        rows: [
          ["单票最大亏损", formatNumber(risk.data?.data.maxSingleStockLoss)],
          ["单日最大亏损", formatPercent(risk.data?.data.maxDailyLoss)],
          ["连续亏损天数", String(risk.data?.data.consecutiveLossDays ?? "—")],
          ["流动性风险", formatPercent(risk.data?.data.liquidityRisk)],
          ["跌停风险", formatPercent(risk.data?.data.limitDownRisk)],
        ],
      },
      {
        title: "研究身份",
        rows: [
          ["模型版本", data.latestModel?.version ?? "—"],
          ["因子版本", latest?.factorVersion ?? "—"],
          ["研究股票池", `${data.stockPoolCount ?? 0} names`],
          ["候选数", String(data.candidateCount ?? 0)],
          ["Artifact", latest?.path ?? "—"],
        ],
      },
      {
        title: "生产晋级契约",
        rows: PROMOTION_POLICY.map(([name, threshold, detail]) => [name, `${threshold} · ${detail}`]),
      },
    ],
    provenance: [
      ["系统状态", "/api/system/overview → persisted runtime artifacts"],
      ["回测目录", "/api/backtests → persisted backtest artifacts"],
      ["风险摘要", "/api/risk/overview → persisted risk artifacts"],
      ["Fuyao 行情/财务/特色数据", "/api/market/* → server-side HITHINK_FINANCE_API_KEY"],
      ["晋级证据", "runtime/.../promotion_gate.json"],
    ],
    limitations,
  };

  return (
    <div className="page institutional-workbench reports-page">
      <section className="report-hero panel">
        <div>
          <span className="report-kicker"><FileText size={17} /> QuantAgent Evidence & Research Report</span>
          <h2>{latest?.name ?? "当前量化研究状态"}</h2>
          <p>报告只汇总已持久化证据。研究偏好、榜单热度或漂亮净值不能替代 PIT、基准、PBO/DSR/SPA 与最终留出集验收。</p>
        </div>
        <div className="report-actions">
          <button onClick={() => downloadJson("quantagent-research-report.json", report)}><DownloadSimple size={16} /> JSON 证据包</button>
          <button className="primary-button" onClick={() => downloadOfflineResearchHtml("quantagent-research-report.html", offlineReport)}><FileHtml size={16} /> 离线 HTML 报告</button>
        </div>
      </section>

      <section className="report-grid">
        <Panel title="策略逻辑摘要" eyebrow="Strategy Summary">
          <ReportParagraph icon={TrendUp} title="收益与执行">
            当前主回测总收益为 {formatPercent(latest?.totalReturn)}，年化收益 {formatPercent(latest?.annualReturn)}，
            Sharpe {formatNumber(latest?.sharpe)}，最大回撤 {formatPercent(latest?.maxDrawdown)}。
            指标只来自已识别 artifact；收盘信号的合规执行口径必须至少为 T+1。
          </ReportParagraph>
          <ReportParagraph icon={ShieldCheck} title="安全边界">
            QuantAgent 将研究候选与生产资格分离。缺少 benchmark、PIT、最终 holdout 或统计闸门证据时，
            即使候选位于 Pareto 前沿，也只能保留为 research candidate。
          </ReportParagraph>
        </Panel>

        <Panel title="生产晋级契约" eyebrow="Research Integrity Gate">
          <div className="report-run-list">
            {PROMOTION_POLICY.map(([name, threshold, detail]) => (
              <div key={name}>
                <span><strong>{name}</strong><small>{detail}</small></span>
                <code>{threshold}</code>
              </div>
            ))}
          </div>
          <div className="backtest-context-note">实际通过/阻塞状态以每次研究运行生成的 promotion_gate.json 为准；本页不以缺失字段推断“通过”。</div>
        </Panel>

        <Panel title="收益来源归因" eyebrow="Return Attribution">
          <ul className="report-list">
            <li><strong>模型版本</strong><span>{data.latestModel?.version ?? "暂无 persisted version"}</span></li>
            <li><strong>因子版本</strong><span>{latest?.factorVersion ?? "暂无 run metadata"}</span></li>
            <li><strong>研究股票池</strong><span>{data.stockPoolCount ?? 0} names / {data.candidateCount ?? 0} candidates</span></li>
            <li><strong>T+1 做 T 贡献</strong><span>{formatPercent(latest?.tContribution)}</span></li>
            <li><strong>总成本</strong><span>{formatNumber(latest?.totalCost)}</span></li>
          </ul>
        </Panel>

        <Panel title="风险来源归因" eyebrow="Risk Attribution">
          <ul className="report-list">
            <li><strong>单票最大亏损</strong><span>{formatNumber(risk.data?.data.maxSingleStockLoss)}</span></li>
            <li><strong>单日最大亏损</strong><span>{formatPercent(risk.data?.data.maxDailyLoss)}</span></li>
            <li><strong>连续亏损天数</strong><span>{risk.data?.data.consecutiveLossDays ?? "暂无"}</span></li>
            <li><strong>流动性风险</strong><span>{formatPercent(risk.data?.data.liquidityRisk)}</span></li>
            <li><strong>跌停风险</strong><span>{formatPercent(risk.data?.data.limitDownRisk)}</span></li>
          </ul>
        </Panel>

        <Panel title="Fuyao 页面产物契约" eyebrow="Financial-API Best Practices">
          <div className="limitation-list">
            <ReportParagraph icon={FileHtml} title="可离线 HTML">
              报告可直接导出单文件 HTML；CSS 与报告数据内联，不包含 API Key，不依赖页面重新联网取数。
            </ReportParagraph>
            <ReportParagraph icon={ShieldCheck} title="来源与口径">
              导出物显式保留数据来源、artifact/endpoint、生成时间、真实/模拟边界与非投资建议；unavailable 不使用 mock 填充。
            </ReportParagraph>
            <ReportParagraph icon={TrendUp} title="回测专属诊断">
              价格突破、时间序列动量、短期反转等策略的形成期、T+1 执行、成本、状态与敏感性应由对应 backtest artifact 提供；没有 artifact 时不在报告层伪造。
            </ReportParagraph>
          </div>
          <div className="report-actions"><button onClick={() => navigate("/market-intelligence")}>打开 Fuyao 市场情报</button><button onClick={() => navigate("/backtests")}>打开回测工作站</button></div>
        </Panel>

        <Panel title="当前限制 / 阻塞项" eyebrow="Fail-closed Limitations">
          <div className="limitation-list">
            {limitations.map((item) => <ReportParagraph key={item} icon={WarningCircle} title="需保留证据"><>{item}</></ReportParagraph>)}
          </div>
        </Panel>

        <Panel title="实验目录" eyebrow={`${backtests.data?.data.length ?? 0} backtests`} className="report-experiment-list">
          <div className="report-run-list">
            {(backtests.data?.data ?? []).slice(0, 12).map((run) => (
              <div key={run.id}>
                <span><strong>{run.name}</strong><small>{run.horizon ?? "research"}</small></span>
                <span className={toneClass(run.annualReturn)} title={run.annualReturn == null ? UNMEASURED_TITLE : undefined}>{formatPercent(run.annualReturn)}</span>
                <code>{run.path}</code>
              </div>
            ))}
          </div>
        </Panel>
      </section>
    </div>
  );
}

function ReportParagraph({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof FileText;
  title: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div className="report-paragraph">
      <Icon size={20} weight="duotone" />
      <div><strong>{title}</strong><p>{children}</p></div>
    </div>
  );
}
