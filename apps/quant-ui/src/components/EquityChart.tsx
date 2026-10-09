import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { EquityPoint } from "../api/types";
import { useVNextChartPalette } from "../vnext/theme";
import { EChart } from "./EChart";

interface EquityChartProps {
  points: EquityPoint[];
  height?: number;
  showDrawdown?: boolean;
}

/** Drawdown is stored as a negative fraction (-0.0609); read it as a percent. */
export function formatDrawdownAxis(value: number): string {
  if (!Number.isFinite(value)) return "";
  return `${(value * 100).toFixed(Math.abs(value) < 0.1 && value !== 0 ? 1 : 0)}%`;
}

function formatTooltipValue(value: unknown, kind: "nav" | "drawdown"): string {
  const number = Array.isArray(value) ? value[value.length - 1] : value;
  if (typeof number !== "number" || !Number.isFinite(number)) return "未测量";
  return kind === "drawdown" ? `${(number * 100).toFixed(2)}%` : number.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
}

function formatAxisDate(value: string): string {
  const match = value.match(/^(\d{4}-\d{2}-\d{2})/);
  return match?.[1] ?? value;
}

export function EquityChart({
  points,
  height = 280,
  showDrawdown = true,
}: EquityChartProps): JSX.Element {
  const palette = useVNextChartPalette();
  const option = useMemo<EChartsOption>(() => {
    const hasBenchmark = points.some((point) => point.benchmarkNav !== null && point.benchmarkNav !== undefined);
    // Benchmark NAV arrives as an index (~1.0) while the portfolio NAV is CNY:
    // plotted raw it sat flat at 0 on the CNY axis and hid a benchmark that
    // beat the strategy (round-29 R6). Rebase it to the portfolio's first NAV.
    const anchor = points.find((point) => point.benchmarkNav != null && point.benchmarkNav !== 0
      && Number.isFinite(point.benchmarkNav) && Number.isFinite(point.nav));
    const benchmarkScale = anchor ? anchor.nav / (anchor.benchmarkNav as number) : null;
    const xAxisIndexes = showDrawdown ? [0, 1] : [0];
    return ({
    animation: false,
    backgroundColor: "transparent",
    grid: showDrawdown
      // The drawdown pane used to be 13% tall with 8 stacked tick labels;
      // it now gets ~22% and two ticks so its labels never overlap.
      ? [{ left: 64, right: 18, top: 34, height: "46%" }, { left: 64, right: 18, top: "64%", height: "21%" }]
      : { left: 64, right: 18, top: 34, bottom: 46 },
    legend: {
      show: hasBenchmark,
      top: 2,
      left: 60,
      itemWidth: 15,
      itemHeight: 2,
      textStyle: { color: palette.text, fontSize: 11 },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: palette.tooltip,
      borderColor: palette.tooltipBorder,
      borderWidth: 1,
      padding: [9, 11],
      textStyle: { color: palette.tooltipText, fontSize: 11 },
    },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    xAxis: showDrawdown
      ? [
          { type: "category", boundaryGap: false, data: points.map((point) => point.datetime), axisLabel: { show: false }, axisTick: { show: false }, axisLine: { lineStyle: { color: palette.axis } } },
          { type: "category", boundaryGap: false, gridIndex: 1, data: points.map((point) => point.datetime), axisLabel: { color: palette.muted, fontSize: 11, hideOverlap: true, formatter: formatAxisDate }, axisTick: { show: false }, axisLine: { lineStyle: { color: palette.axis } } },
        ]
      : { type: "category", boundaryGap: false, data: points.map((point) => point.datetime), axisLabel: { color: palette.muted, fontSize: 11, hideOverlap: true, formatter: formatAxisDate }, axisTick: { show: false }, axisLine: { lineStyle: { color: palette.axis } } },
    yAxis: showDrawdown
      ? [
          { type: "value", scale: true, axisLabel: { color: palette.muted, fontSize: 11 }, axisTick: { show: false }, axisLine: { show: false }, splitLine: { lineStyle: { color: palette.grid } } },
          { type: "value", gridIndex: 1, max: 0, splitNumber: 2, name: "回撤", nameLocation: "middle", nameGap: 46, nameTextStyle: { color: palette.muted, fontSize: 10 }, axisLabel: { color: palette.muted, fontSize: 10, hideOverlap: true, formatter: formatDrawdownAxis }, axisTick: { show: false }, axisLine: { show: false }, splitLine: { lineStyle: { color: palette.grid } } },
        ]
      : { type: "value", scale: true, axisLabel: { color: palette.muted, fontSize: 11 }, axisTick: { show: false }, axisLine: { show: false }, splitLine: { lineStyle: { color: palette.grid } } },
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: xAxisIndexes,
        filterMode: "none",
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        moveOnMouseWheel: false,
      },
      {
        type: "slider",
        xAxisIndex: xAxisIndexes,
        bottom: 4,
        height: 13,
        borderColor: palette.axis,
        backgroundColor: palette.slider,
        fillerColor: palette.sliderSelected,
        dataBackground: { lineStyle: { color: palette.muted }, areaStyle: { color: palette.sliderData } },
        selectedDataBackground: { lineStyle: { color: palette.primary }, areaStyle: { color: palette.sliderSelected } },
        handleStyle: { color: palette.primary, borderColor: palette.primary },
        textStyle: { color: palette.muted, fontSize: 9 },
      },
    ],
    series: [
      {
        name: "Portfolio NAV",
        type: "line",
        data: points.map((point) => point.nav),
        showSymbol: false,
        lineStyle: { color: palette.primary, width: 1.8 },
        areaStyle: { color: palette.primary, opacity: 0.06 },
        tooltip: { valueFormatter: (value: unknown) => formatTooltipValue(value, "nav") },
      },
      ...(hasBenchmark && benchmarkScale !== null
        ? [{
            name: "Benchmark (rebased to initial NAV)",
            type: "line" as const,
            data: points.map((point) => (point.benchmarkNav == null ? null : point.benchmarkNav * benchmarkScale)),
            showSymbol: false,
            lineStyle: { color: palette.muted, width: 1.2, type: "dashed" as const },
            tooltip: { valueFormatter: (value: unknown) => formatTooltipValue(value, "nav") },
          }]
        : []),
      ...(showDrawdown
        ? [{
            name: "Drawdown",
            type: "line" as const,
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: points.map((point) => point.drawdown ?? null),
            showSymbol: false,
            // A risk measure, so status danger — not market red, which means "up".
            lineStyle: { color: palette.danger, width: 1.2 },
            areaStyle: { color: palette.danger, opacity: 0.16 },
            tooltip: { valueFormatter: (value: unknown) => formatTooltipValue(value, "drawdown") },
          }]
        : []),
    ],
    });
  }, [palette, points, showDrawdown]);

  return <EChart option={option} className="chart equity-chart" style={{ height }} ariaLabel="组合净值与回撤交互图表；滚轮缩放，拖动平移" />;
}
