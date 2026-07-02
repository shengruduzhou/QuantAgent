import { DownloadSimple, FileText, ShieldCheck, TrendUp, WarningCircle } from "@phosphor-icons/react";
import type { BacktestSummary, RiskOverview, SystemOverview } from "../api/types";
import { downloadJson } from "../api/client";
import { useApi } from "../hooks/useApi";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { formatNumber, formatPercent } from "../utils/format";

export function ReportsPage(): JSX.Element {
  const overview = useApi<SystemOverview>(["reports-overview"], "/system/overview");
  const backtests = useApi<BacktestSummary[]>(["reports-backtests"], "/backtests");
  const risk = useApi<RiskOverview>(["reports-risk"], "/risk/overview");
  const data = overview.data?.data;
  const latest = data?.latestBacktest;

  if (overview.isLoading) return <StateView state="loading" />;
  if (!data) return <StateView state="empty" />;

  const report = {
    generatedFrom: "QuantAgent persisted runtime artifacts",
    latestBacktest: latest,
    risk: risk.data?.data,
    model: data.latestModel,
    selection: data.latestSelection,
    limitations: [
      "No live orders are generated.",
      "Missing per-trade factor attribution remains unavailable.",
      "Independent factor trades are only shown when persisted.",
    ],
  };

  return (
    <div className="page reports-page">
      <section className="report-hero panel">
        <div>
          <span className="report-kicker"><FileText size={17} /> QuantAgent Research Brief</span>
          <h2>{latest?.name ?? "当前量化研究状态"}</h2>
          <p>基于真实 runtime artifact 自动汇总。该报告用于研究复盘、模型诊断与风险讨论，不构成投资建议。</p>
        </div>
        <button className="primary-button" onClick={() => downloadJson("quantagent-research-report.json", report)}><DownloadSimple size={16} /> 导出报告数据</button>
      </section>

      <section className="report-grid">
        <Panel title="策略逻辑摘要" eyebrow="Strategy Summary">
          <ReportParagraph icon={TrendUp} title="收益与执行">
            当前主回测总收益为 {formatPercent(latest?.totalReturn)}，年化收益 {formatPercent(latest?.annualReturn)}，
            Sharpe {formatNumber(latest?.sharpe)}，最大回撤 {formatPercent(latest?.maxDrawdown)}。
            交易与收益字段仅来自已识别的标准 order blotter 与 NAV artifact。
          </ReportParagraph>
          <ReportParagraph icon={ShieldCheck} title="安全边界">
            QuantAgent UI 只消费 target weights、回测、模型和风险产物。Agent 与 optimizer 不生成订单，
            QMT live trading 未通过 UI 暴露，所有任务路径限制在项目 runtime。
          </ReportParagraph>
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

        <Panel title="当前缺陷" eyebrow="Known Limitations">
          <div className="limitation-list">
            <ReportParagraph icon={WarningCircle} title="逐笔解释缺口">
              多数 strict backtest order audit 没有逐笔 signal reason、factor contribution 与 cashAfter；
              UI 对这些字段显示“暂无数据”，不会从结果反推原因。
            </ReportParagraph>
            <ReportParagraph icon={WarningCircle} title="单因子交易缺口">
              当前 runtime 主要保存因子 IC/ICIR 与 judgment metrics。没有独立成交 artifact 的因子不会显示伪造买卖点。
            </ReportParagraph>
            <ReportParagraph icon={WarningCircle} title="T+1 数据粒度">
              部分历史做 T artifact 只有 daily return，没有 minute fill price/quantity；这类记录明确标记为 daily-only。
            </ReportParagraph>
          </div>
        </Panel>

        <Panel title="实验目录" eyebrow={`${backtests.data?.data.length ?? 0} backtests`} className="report-experiment-list">
          <div className="report-run-list">
            {(backtests.data?.data ?? []).slice(0, 12).map((run) => (
              <div key={run.id}>
                <span><strong>{run.name}</strong><small>{run.horizon ?? "research"}</small></span>
                <span className={(run.annualReturn ?? 0) >= 0 ? "tone-positive" : "tone-negative"}>{formatPercent(run.annualReturn)}</span>
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
