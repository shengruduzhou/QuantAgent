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
    const xAxisIndexes = showDrawdown ? [0, 1] : [0];
    return ({
    animation: false,
    backgroundColor: "transparent",
    grid: showDrawdown
      ? [{ left: 64, right: 18, top: 34, height: "53%" }, { left: 64, right: 18, top: "69%", height: "13%" }]
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
          { type: "value", gridIndex: 1, axisLabel: { color: palette.muted, fontSize: 10, formatter: "{value}" }, axisTick: { show: false }, axisLine: { show: false }, splitLine: { show: false } },
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
        fillerColor: "rgba(76, 141, 255, .18)",
        dataBackground: { lineStyle: { color: palette.muted }, areaStyle: { color: palette.sliderData } },
        selectedDataBackground: { lineStyle: { color: "#4c8dff" }, areaStyle: { color: palette.sliderSelected } },
        handleStyle: { color: "#75a9ff", borderColor: "#75a9ff" },
        textStyle: { color: palette.muted, fontSize: 9 },
      },
    ],
    series: [
      {
        name: "Portfolio NAV",
        type: "line",
        data: points.map((point) => point.nav),
        showSymbol: false,
        lineStyle: { color: "#2f83ff", width: 1.8 },
        areaStyle: { color: "rgba(47,131,255,.06)" },
      },
      ...(hasBenchmark
        ? [{
            name: "Benchmark NAV",
            type: "line" as const,
            data: points.map((point) => point.benchmarkNav ?? null),
            showSymbol: false,
            lineStyle: { color: "#6f8798", width: 1.2, type: "dashed" as const },
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
            lineStyle: { color: "#f05a5a", width: 1.2 },
            areaStyle: { color: "rgba(240,90,90,.16)" },
          }]
        : []),
    ],
    });
  }, [palette, points, showDrawdown]);

  return <EChart option={option} className="chart equity-chart" style={{ height }} ariaLabel="组合净值与回撤交互图表；滚轮缩放，拖动平移" />;
}
