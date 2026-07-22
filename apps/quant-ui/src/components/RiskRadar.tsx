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
    tooltip: { backgroundColor: "#07131d", borderColor: "#31536a", textStyle: { color: "#e2edf4", fontSize: 11 } },
    radar: {
      center: ["50%", "51%"],
      radius: "62%",
      splitNumber: 5,
      indicator: [
        { name: "回撤", max: 100 },
        { name: "单票亏损", max: 100 },
        { name: "单日亏损", max: 100 },
        { name: "流动性", max: 100 },
        { name: "跌停", max: 100 },
        { name: "连续亏损", max: 100 },
      ],
      axisName: { color: "#91a8b8", fontSize: 11 },
      splitArea: { areaStyle: { color: ["rgba(9,24,35,.36)", "rgba(12,31,44,.18)"] } },
      splitLine: { lineStyle: { color: "#284354" } },
      axisLine: { lineStyle: { color: "#284354" } },
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
        lineStyle: { color: "#4c8dff", width: 2 },
        areaStyle: { color: "rgba(76,141,255,.15)" },
        itemStyle: { color: "#2bc6d6", borderColor: "#07131d", borderWidth: 2 },
      }],
    }],
  }), [risk]);

  return <EChart option={option} className="chart" ariaLabel="组合风险相对阈值雷达图" />;
}
