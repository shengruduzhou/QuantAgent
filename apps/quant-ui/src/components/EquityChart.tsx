import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { EquityPoint } from "../api/types";
import { EChart } from "./EChart";

interface EquityChartProps {
  points: EquityPoint[];
  height?: number;
  showDrawdown?: boolean;
}

export function EquityChart({
  points,
  height = 280,
  showDrawdown = true,
}: EquityChartProps): JSX.Element {
  const option = useMemo<EChartsOption>(() => ({
    animation: false,
    backgroundColor: "transparent",
    grid: showDrawdown
      ? [{ left: 52, right: 18, top: 18, height: "60%" }, { left: 52, right: 18, top: "76%", height: "16%" }]
      : { left: 52, right: 18, top: 18, bottom: 32 },
    tooltip: { trigger: "axis", backgroundColor: "#0b1824", borderColor: "#27425a", textStyle: { color: "#d7e4ef" } },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    xAxis: showDrawdown
      ? [
          { type: "category", data: points.map((point) => point.datetime), axisLabel: { show: false }, axisLine: { lineStyle: { color: "#20364a" } } },
          { type: "category", gridIndex: 1, data: points.map((point) => point.datetime), axisLabel: { color: "#71879a", fontSize: 10 }, axisLine: { lineStyle: { color: "#20364a" } } },
        ]
      : { type: "category", data: points.map((point) => point.datetime), axisLabel: { color: "#71879a", fontSize: 10 }, axisLine: { lineStyle: { color: "#20364a" } } },
    yAxis: showDrawdown
      ? [
          { type: "value", scale: true, axisLabel: { color: "#71879a", fontSize: 10 }, splitLine: { lineStyle: { color: "#14283a" } } },
          { type: "value", gridIndex: 1, axisLabel: { color: "#71879a", fontSize: 10, formatter: "{value}" }, splitLine: { show: false } },
        ]
      : { type: "value", scale: true, axisLabel: { color: "#71879a", fontSize: 10 }, splitLine: { lineStyle: { color: "#14283a" } } },
    series: [
      {
        name: "Portfolio NAV",
        type: "line",
        data: points.map((point) => point.nav),
        showSymbol: false,
        lineStyle: { color: "#2f83ff", width: 1.8 },
        areaStyle: { color: "rgba(47,131,255,.08)" },
      },
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
  }), [points, showDrawdown]);

  return <EChart option={option} className="chart" key={height} />;
}
