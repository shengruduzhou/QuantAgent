import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, test, vi } from "vitest";
import { ProductionReadinessPanel } from "./ProductionReadinessPanel";


afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const PAYLOAD = {
  schemaVersion: 1,
  generatedAt: "2026-08-08T12:00:00+00:00",
  aggregateTradingReady: null,
  aggregateStateSemantics: "intentionally_not_computed_show_all_dimensions",
  cards: [
    { key: "modelTrust", label: "Model Trust", state: "BLOCKED", severity: "blocked", reasons: ["certificate_missing"], evidence: { certificateStatus: "missing" } },
    { key: "brokerQuery", label: "Broker Query Readiness", state: "READY", severity: "ok", reasons: [], evidence: { queryOnly: true, validUntil: "2026-08-09T12:00:00+00:00" } },
    { key: "targetRisk", label: "Target Risk", state: "WIRED", severity: "info", reasons: ["wiring_presence_is_not_a_runtime_pass"], evidence: { semantics: "target_before_order_session_bound_v1" } },
    { key: "orderRisk", label: "Order Risk", state: "WIRED", severity: "info", reasons: ["wiring_presence_is_not_a_runtime_pass"], evidence: { semantics: "order_intent_gate_after_target_v1" } },
    { key: "killSwitch", label: "KillSwitch", state: "NOT_CONFIGURED", severity: "unknown", reasons: ["runtime_evidence_missing"], evidence: {} },
    { key: "reconciliation", label: "Reconciliation", state: "NOT_CERTIFIED", severity: "unknown", reasons: ["runtime_evidence_missing"], evidence: {} },
    { key: "productArming", label: "Product Arming", state: "NOT_ARMED", severity: "blocked", reasons: ["product_policy_live_disabled"], evidence: { liveTradingAvailable: false } },
    { key: "hostCertification", label: "Host / Platform Certification", state: "NOT_CERTIFIED", severity: "unknown", reasons: ["runtime_evidence_missing"], evidence: { portableContractIsNotHostCertification: true } },
  ],
};

function renderPanel(payload = PAYLOAD): void {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(
    JSON.stringify({ status: "ready", data: payload, issues: [] }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  )));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <ProductionReadinessPanel />
    </QueryClientProvider>,
  );
}

test("renders eight independent machine-truth dimensions without one green verdict", async () => {
  renderPanel();
  expect(await screen.findByLabelText("生产就绪真值")).toBeInTheDocument();
  const cards = DIMENSION_KEYS.map((key) => screen.getByTestId(`readiness-${key}`));
  expect(cards).toHaveLength(8);

  expect(screen.getByTestId("readiness-brokerQuery")).toHaveTextContent("READY");
  expect(screen.getByTestId("readiness-modelTrust")).toHaveTextContent("BLOCKED");
  expect(screen.getByTestId("readiness-productArming")).toHaveTextContent("NOT_ARMED");
  expect(screen.getByTestId("readiness-targetRisk")).toHaveTextContent("WIRED 仅证明强制路径存在");

  const text = document.body.textContent?.toLowerCase() ?? "";
  expect(text).not.toContain("aggregate trading ready");
  for (const banned of ["nav", "sharpe", "cagr", "drawdown", "calmar", "sortino", "pnl"]) {
    expect(new RegExp(`\\b${banned}\\b`).test(text)).toBe(false);
  }
});

test("missing one server card stays UNKNOWN instead of disappearing", async () => {
  renderPanel({ ...PAYLOAD, cards: PAYLOAD.cards.filter((card) => card.key !== "reconciliation") });
  expect(await screen.findByTestId("readiness-reconciliation")).toHaveTextContent("UNKNOWN");
  expect(screen.getByTestId("readiness-reconciliation")).toHaveTextContent("machine_evidence_missing");
});

const DIMENSION_KEYS = [
  "modelTrust",
  "brokerQuery",
  "targetRisk",
  "orderRisk",
  "killSwitch",
  "reconciliation",
  "productArming",
  "hostCertification",
] as const;
