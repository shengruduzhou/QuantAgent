import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { FusionCandidate, FusionRunDetail } from "../../api/types";
import { EChart } from "../../components/EChart";
import { useVNextChartPalette } from "../theme";
import { TruthNotice, WorkbenchPanel } from "../workbench/InstitutionalWorkbench";

function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function ratio(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

const ROBUSTNESS_LABELS: Array<{
  key: keyof FusionCandidate["robustnessBreakdown"];
  label: string;
  explanation: string;
}> = [
  {
    key: "foldConsistency",
    label: "折间一致性",
    explanation: "每折都为正，还是被一折运气带起来的",
  },
  {
    key: "overfittingResistance",
    label: "抗过拟合",
    explanation: "1 − PBO；样本内冠军在样本外掉出中位数的概率",
  },
  {
    key: "deflatedSharpeProbability",
    label: "收缩后显著性",
    explanation: "按实际试验次数收缩后的 Sharpe 显著性概率",
  },
  {
    key: "regimeConsistency",
    label: "最差 regime 下限",
    explanation: "最差一折的 Sharpe 被压缩到 [0,1]",
  },
];

export function CandidateInspector({
  candidate,
  summary,
}: {
  candidate: FusionCandidate;
  summary: FusionRunDetail["summary"];
}): JSX.Element {
  const palette = useVNextChartPalette();

  const weightOption = useMemo<EChartsOption>(() => {
    const entries = Object.entries(candidate.weights)
      .filter(([, value]) => Math.abs(value) > 1e-6)
      .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]))
      .slice(0, 12)
      .reverse();
    return {
      animationDuration: 200,
      grid: { left: 4, right: 24, top: 6, bottom: 6, containLabel: true },
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const item = params as { name: string; value: number };
          return `${item.name}<br/>权重 ${ratio(item.value, 3)}`;
        },
      },
      xAxis: { type: "value", axisLabel: { fontSize: 9 }, splitLine: { show: true } },
      yAxis: {
        type: "category",
        data: entries.map(([name]) => name),
        axisLabel: { fontSize: 9 },
      },
      series: [
        {
          type: "bar",
          barWidth: 9,
          data: entries.map(([, value]) => ({
            value,
            itemStyle: { color: value >= 0 ? palette.primary : palette.warning },
          })),
        },
      ],
    };
  }, [candidate.weights, palette]);

  const folds = candidate.folds ?? [];

  return (
    <WorkbenchPanel
      eyebrow="CANDIDATE INSPECTOR"
      title={candidate.label}
      meta={`${candidate.scheme}${candidate.isControl ? " · 对照组" : " · 拟合方案"}`}
      className="foundry-inspector"
    >
      {candidate.isControl ? (
        <TruthNotice tone="warning">
          这是对照方案，不读取训练段。它出现在前沿上意味着拟合方案没有赢过一个不学习的基线——
          这本身就是结论，不是失败的运行。
        </TruthNotice>
      ) : null}

      <dl className="foundry-metrics">
        <div>
          <dt>超额收益</dt>
          <dd className={`atlas-figure small tone-${candidate.metrics.excessReturn >= 0 ? "positive" : "negative"}`}>
            {percent(candidate.metrics.excessReturn)}
          </dd>
          <small>基准 {percent(candidate.metrics.benchmarkAnnualReturn)}</small>
        </div>
        <div>
          <dt>年化收益</dt>
          <dd className="atlas-figure small">{percent(candidate.metrics.annualReturn)}</dd>
          <small>成本后 · 拖累 {percent(candidate.metrics.costDrag, 2)}</small>
        </div>
        <div>
          <dt>最大回撤</dt>
          <dd className="atlas-figure small">{percent(candidate.metrics.maxDrawdown)}</dd>
          <small>取最差一折，非折间平均</small>
        </div>
        <div>
          <dt>稳健性</dt>
          <dd className="atlas-figure small">{ratio(candidate.metrics.robustness)}</dd>
          <small>Sharpe {ratio(candidate.metrics.sharpe)} · Calmar {ratio(candidate.metrics.calmar)}</small>
        </div>
      </dl>

      <div className="foundry-robustness-breakdown">
        <span className="atlas-eyebrow">ROBUSTNESS EVIDENCE</span>
        {ROBUSTNESS_LABELS.map(({ key, label, explanation }) => {
          const value = candidate.robustnessBreakdown[key];
          const numeric = typeof value === "number" ? value : null;
          return (
            <div key={key} className="foundry-robustness-row">
              <div>
                <strong>{label}</strong>
                <small>{explanation}</small>
              </div>
              <span className="mono">{numeric === null ? "无估计" : ratio(numeric)}</span>
              <i
                className="atlas-meter"
                data-tone={numeric === null ? "warning" : numeric > 0.6 ? "success" : "warning"}
              >
                <i style={{ width: `${Math.round((numeric ?? 0) * 100)}%` }} />
              </i>
            </div>
          );
        })}
        {candidate.robustnessBreakdown.pbo === null ? (
          <TruthNotice tone="warning">
            本次搜索的时间切片不足以估计 PBO，抗过拟合项按“无证据”（0.5）计入，不按通过计入。
          </TruthNotice>
        ) : null}
      </div>

      <div className="foundry-weights">
        <span className="atlas-eyebrow">BLEND WEIGHTS · 折间坐标中位数</span>
        <EChart option={weightOption} className="foundry-weight-chart" ariaLabel="融合权重构成" />
      </div>

      {folds.length ? (
        <div className="atlas-scroll-x">
          <table className="atlas-grid foundry-folds">
            <thead>
              <tr>
                <th scope="col">折</th>
                <th scope="col">训练段</th>
                <th scope="col">样本外段</th>
                <th scope="col" className="num">超额</th>
              </tr>
            </thead>
            <tbody>
              {folds.map((fold) => {
                const excess = Number(fold.metrics?.excessReturn ?? Number.NaN);
                return (
                  <tr key={fold.foldIndex}>
                    <td className="mono">#{fold.foldIndex}</td>
                    <td className="mono">{fold.trainStart} → {fold.trainEnd}</td>
                    <td className="mono">{fold.testStart} → {fold.testEnd}</td>
                    <td className={`num tone-${excess >= 0 ? "positive" : "negative"}`}>
                      {percent(excess)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      <TruthNotice>
        全部指标都在样本外折上计算，训练段只用于产生权重。
        本次搜索共 {summary.nTrials ?? "?"} 次试验，成本 {summary.transactionCostBps ?? "?"} bps，
        基准口径 {summary.benchmarkMode ?? "未记录"}。
      </TruthNotice>
    </WorkbenchPanel>
  );
}
