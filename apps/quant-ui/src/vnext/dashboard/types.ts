export type DecisionView = "portfolio" | "model" | "backtest" | "market" | "risk" | "training";

export interface ActionQueueItem {
  id: string;
  severity: "critical" | "warning" | "info" | "success";
  entity: string;
  reason: string;
  timestamp: string;
  source: string;
  action: string;
  path: string;
}
