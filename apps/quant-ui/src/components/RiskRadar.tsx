import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { RiskOverview } from "../api/types";
import { EChart } from "./EChart";

interface RiskRadarProps {
  risk: RiskOverview;
}

/**
 * Scale a risk axis to 0-100, or return null when it was never measured.
 *
 * Returning 0 for a missing value is not a neutral default on this chart: 0 is
 * the *best possible* reading on every axis, so an unmeasured book renders as a
 * collapsed polygon that says "no risk exposure". `/api/risk/overview` returns
 * null for most axes on runs that carry no risk snapshot, so the picture a
 * reader saw was the inverse of the truth -- absent evidence drawn as safety.
 *
 * echarts renders null as a gap in the polygon, which reads as missing rather
 * than as zero, and the unmeasured axes are named explicitly below the chart.
 */
function normalize(value: number | null | undefined, scale = 1): number | null {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  return Math.min(100, Math.abs(value) * scale * 100);
}

const AXES = ["回撤", "单票亏损", "单日亏损", "流动性", "跌停", "连续亏损"] as const;

export function RiskRadar({ risk }: RiskRadarProps): JSX.Element {
  const values = useMemo(
    () => [
      normalize(risk.maxDrawdown),
      normalize(risk.maxSingleStockLoss, 0.02),
      normalize(risk.maxDailyLoss, 4),
      normalize(risk.liquidityRisk),
      normalize(risk.limitDownRisk),
      risk.consecutiveLossDays === null || risk.consecutiveLossDays === undefined
        ? null
        : Math.min(100, risk.consecutiveLossDays * 10),
    ],
    [risk],
  );

  const unmeasured = AXES.filter((_, index) => values[index] === null);

  const option = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: {
      backgroundColor: "#07131d",
      borderColor: "#31536a",
      textStyle: { color: "#e2edf4", fontSize: 11 },
    },
    radar: {
      center: ["50%", "51%"],
      radius: "62%",
      splitNumber: 5,
      indicator: AXES.map((name) => ({ name, max: 100 })),
      axisName: { color: "#91a8b8", fontSize: 11 },
      splitArea: { areaStyle: { color: ["rgba(9,24,35,.36)", "rgba(12,31,44,.18)"] } },
      splitLine: { lineStyle: { color: "#284354" } },
      axisLine: { lineStyle: { color: "#284354" } },
    },
    series: [{
      type: "radar",
      data: [{
        value: values,
        name: "Risk exposure",
        lineStyle: { color: "#4c8dff", width: 2 },
        areaStyle: { color: "rgba(76,141,255,.15)" },
        itemStyle: { color: "#2bc6d6", borderColor: "#07131d", borderWidth: 2 },
      }],
    }],
  }), [values]);

  return (
    <>
      <EChart option={option} className="chart" ariaLabel="组合风险相对阈值雷达图" />
      {unmeasured.length > 0 && (
        <p className="chart-note chart-note-unmeasured" role="note">
          未测量：{unmeasured.join("、")}
          {unmeasured.length === AXES.length ? "（本次运行没有风险快照）" : ""}
        </p>
      )}
    </>
  );
}
