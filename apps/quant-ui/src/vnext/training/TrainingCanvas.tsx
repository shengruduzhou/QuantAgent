import { useMemo, useState } from "react";
import { CircleNotch, Pulse } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import type { JobSummary, TrainingMetricPoint } from "../../api/types";
import { EChart } from "../../components/EChart";
import { StateView } from "../../components/StateView";
import { formatNumber } from "../../utils/format";
import { useVNextChartPalette } from "../theme";

type MetricView = "overview" | "loss" | "rankic";

function smooth(values: Array<number | null | undefined>, windowSize = 4): Array<number | null> {
  return values.map((value, index) => {
    if (value === null || value === undefined) return null;
    const window = values.slice(Math.max(0, index - windowSize + 1), index + 1).filter((item): item is number => typeof item === "number");
    return window.reduce((sum, item) => sum + item, 0) / window.length;
  });
}

export function TrainingCanvas({ points, job }: { points: TrainingMetricPoint[]; job?: JobSummary }): JSX.Element {
  const [smoothed, setSmoothed] = useState(true);
  const [metricView, setMetricView] = useState<MetricView>("overview");
  const palette = useVNextChartPalette();
  const option = useMemo<EChartsOption>(() => {
    const train = points.map((point) => point.loss);
    const validation = points.map((point) => point.validationLoss);
    const rankIc = points.map((point) => point.rankIc ?? point.metrics.rank_ic ?? point.metrics.rankic);
    const dual = metricView === "overview";
    const categories = points.map((point) => point.epoch);
    const showLoss = metricView !== "rankic";
    const showRankIc = metricView !== "loss";
    return {
      animationDuration: 180,
      axisPointer: { link: dual ? [{ xAxisIndex: "all" }] : [], label: { backgroundColor: palette.tooltipBorder } },
      tooltip: { trigger: "axis", backgroundColor: palette.tooltip, borderColor: palette.tooltipBorder, textStyle: { color: palette.tooltipText } },
      legend: { top: 2, data: [showLoss ? "Train loss" : "", showLoss ? "Validation loss" : "", showRankIc ? "RankIC" : ""].filter(Boolean), textStyle: { color: palette.text, fontSize: 11 } },
      grid: dual
        ? [{ left: 54, right: 24, top: 42, height: "52%" }, { left: 54, right: 24, top: "72%", height: "17%" }]
        : { left: 58, right: 24, top: 42, bottom: 48 },
      xAxis: dual
        ? [
          { type: "category", data: categories, boundaryGap: false, axisLabel: { show: false }, axisLine: { lineStyle: { color: palette.axis } } },
          { type: "category", gridIndex: 1, data: categories, boundaryGap: false, axisLabel: { color: palette.muted }, axisLine: { lineStyle: { color: palette.axis } } },
        ]
        : { type: "category", data: categories, boundaryGap: false, axisLabel: { color: palette.muted }, axisLine: { lineStyle: { color: palette.axis } } },
      yAxis: dual
        ? [
          { type: "value", scale: true, axisLabel: { color: palette.muted }, splitLine: { lineStyle: { color: palette.grid } } },
          { type: "value", gridIndex: 1, scale: true, axisLabel: { color: palette.muted }, splitLine: { lineStyle: { color: palette.grid } } },
        ]
        : { type: "value", scale: true, axisLabel: { color: palette.muted }, splitLine: { lineStyle: { color: palette.grid } } },
      dataZoom: [
        { type: "inside", xAxisIndex: dual ? [0, 1] : [0], zoomOnMouseWheel: true, moveOnMouseMove: true },
        { type: "slider", xAxisIndex: dual ? [0, 1] : [0], bottom: 3, height: 14, borderColor: palette.axis, backgroundColor: palette.slider, fillerColor: "rgba(79,140,255,.18)", dataBackground: { lineStyle: { color: palette.muted }, areaStyle: { color: palette.sliderData } }, selectedDataBackground: { areaStyle: { color: palette.sliderSelected } }, textStyle: { color: palette.muted } },
      ],
      series: [
        ...(showLoss ? [
          { name: "Train loss", type: "line" as const, data: smoothed ? smooth(train) : train, showSymbol: false, lineStyle: { color: "#5794ef", width: 1.8 }, markPoint: { symbolSize: 28, data: points.length ? [{ type: "min" as const, name: "Best" }] : [] } },
          { name: "Validation loss", type: "line" as const, data: smoothed ? smooth(validation) : validation, showSymbol: false, lineStyle: { color: "#d9a441", width: 1.6 } },
        ] : []),
        ...(showRankIc ? [{ name: "RankIC", type: "line" as const, xAxisIndex: dual ? 1 : 0, yAxisIndex: dual ? 1 : 0, data: smoothed ? smooth(rankIc) : rankIc, showSymbol: false, lineStyle: { color: "#30b99d", width: 1.5 }, areaStyle: { color: "rgba(48,185,157,.08)" } }] : []),
      ],
    };
  }, [metricView, palette, points, smoothed]);
  const latest = points.at(-1);
  const validationLosses = points.flatMap((point) => point.validationLoss == null ? [] : [point.validationLoss]);
  const bestValidation = validationLosses.length ? Math.min(...validationLosses) : null;
  const progress = job?.progress === null || job?.progress === undefined ? null : Math.max(0, Math.min(1, job.progress));
  const eta = estimateEta(job, progress);

  return (
    <section className="vnext-training-canvas">
      <header>
        <div><span>LIVE TRAINING CANVAS</span><h2>{job?.commandId ?? "Persisted training metrics"}</h2></div>
        <div className="vnext-training-header-actions">
          <nav className="vnext-training-views" aria-label="训练指标视图">
            {(["overview", "loss", "rankic"] as MetricView[]).map((item) => <button type="button" key={item} className={metricView === item ? "active" : ""} aria-pressed={metricView === item} onClick={() => setMetricView(item)}>{item === "rankic" ? "RankIC" : item}</button>)}
          </nav>
          <div className="vnext-training-controls"><button type="button" className={!smoothed ? "active" : ""} onClick={() => setSmoothed(false)}>Raw</button><button type="button" className={smoothed ? "active" : ""} onClick={() => setSmoothed(true)}>Smoothed</button></div>
        </div>
      </header>
      <div className={`vnext-run-pulse state-${job?.status ?? "unavailable"}`}>
        <span>{job?.status === "running" ? <CircleNotch size={15} className="spin" /> : <Pulse size={15} />}<strong>{job?.status?.toUpperCase() ?? "NO ACTIVE RUN"}</strong><small>{job?.message ?? "Persisted metrics inspection"}</small></span>
        <i aria-label={progress === null ? "训练进度不可用" : `训练进度 ${Math.round(progress * 100)}%`}><b style={{ width: `${(progress ?? 0) * 100}%` }} /></i>
        <em>{progress === null ? "PROGRESS UNAVAILABLE" : `${Math.round(progress * 100)}%`}</em>
        <span><small>ETA</small><strong>{eta}</strong></span>
      </div>
      <div className="vnext-live-metrics">
        <span><small>Epoch</small><strong>{latest?.epoch ?? "—"}</strong></span>
        <span><small>Current loss</small><strong>{formatNumber(latest?.loss, 5)}</strong></span>
        <span><small>Best validation</small><strong>{formatNumber(bestValidation, 5)}</strong></span>
        <span><small>RankIC</small><strong>{formatNumber(latest?.rankIc ?? latest?.metrics.rank_ic, 4)}</strong></span>
        <span><small>Learning rate</small><strong>{formatNumber(latest?.learningRate ?? latest?.metrics.learning_rate, 6)}</strong></span>
        <span><small>GPU memory</small><strong>{latest?.gpuMemory === undefined ? "UNAVAILABLE" : formatNumber(latest.gpuMemory)}</strong></span>
        <span><small>Throughput</small><strong>{latest?.samplesPerSecond === undefined ? "UNAVAILABLE" : `${formatNumber(latest.samplesPerSecond, 1)} samples/s`}</strong></span>
        <span><small>Gradient norm</small><strong>{formatNumber(latest?.gradientNorm ?? latest?.metrics.gradient_norm, 4)}</strong></span>
      </div>
      {points.length ? <EChart option={option} className="chart" style={{ height: 430 }} /> : <StateView state={job && ["queued", "running"].includes(job.status) ? "loading" : "empty"} detail="当前 run 尚未产生可解析 training-metrics artifact。任务状态仍来自 persisted Jobs。" />}
    </section>
  );
}

function estimateEta(job: JobSummary | undefined, progress: number | null): string {
  if (!job || job.status !== "running" || !job.startedAt || progress === null || progress <= 0 || progress >= 1) return "UNAVAILABLE";
  const startedAt = Date.parse(job.startedAt);
  if (!Number.isFinite(startedAt)) return "UNAVAILABLE";
  const elapsedSeconds = (Date.now() - startedAt) / 1_000;
  if (elapsedSeconds <= 0) return "UNAVAILABLE";
  const remainingSeconds = elapsedSeconds * ((1 - progress) / progress);
  if (!Number.isFinite(remainingSeconds)) return "UNAVAILABLE";
  if (remainingSeconds >= 3_600) return `${Math.floor(remainingSeconds / 3_600)}h ${Math.round((remainingSeconds % 3_600) / 60)}m`;
  if (remainingSeconds >= 60) return `${Math.round(remainingSeconds / 60)}m`;
  return `${Math.round(remainingSeconds)}s`;
}
