import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { RiskOverview } from "../api/types";
import { EChart } from "./EChart";

interface RiskRadarProps {
  risk: RiskOverview;
}

function normalize(value: number | null | undefined, scale = 1): number {
  if (value === null || value === undefined) return 0;
  return Math.min(100, Math.abs(value) * scale * 100);
}

export function RiskRadar({ risk }: RiskRadarProps): JSX.Element {
  const option = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { backgroundColor: "#0b1824", borderColor: "#27425a", textStyle: { color: "#d7e4ef" } },
    radar: {
      center: ["50%", "52%"],
      radius: "66%",
      splitNumber: 4,
      indicator: [
        { name: "回撤", max: 100 },
        { name: "单票亏损", max: 100 },
        { name: "单日亏损", max: 100 },
        { name: "流动性", max: 100 },
        { name: "跌停", max: 100 },
        { name: "连续亏损", max: 100 },
      ],
      axisName: { color: "#9cb1c3", fontSize: 10 },
      splitArea: { areaStyle: { color: ["rgba(17,38,55,.16)", "rgba(17,38,55,.32)"] } },
      splitLine: { lineStyle: { color: "#294057" } },
      axisLine: { lineStyle: { color: "#294057" } },
    },
    series: [{
      type: "radar",
      data: [{
        value: [
          normalize(risk.maxDrawdown),
          normalize(risk.maxSingleStockLoss, 0.02),
          normalize(risk.maxDailyLoss, 4),
          normalize(risk.liquidityRisk),
          normalize(risk.limitDownRisk),
          Math.min(100, (risk.consecutiveLossDays ?? 0) * 10),
        ],
        name: "Risk exposure",
        lineStyle: { color: "#3f8cff", width: 1.8 },
        areaStyle: { color: "rgba(63,140,255,.18)" },
        itemStyle: { color: "#46d7bb" },
      }],
    }],
  }), [risk]);

  return <EChart option={option} className="chart" />;
}
