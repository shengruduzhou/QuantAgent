import { fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, test, vi } from "vitest";
import { VnpyParityPage } from "./VnpyParityPage";

const capability = {
  id: "risk.risk_manager",
  category: "risk",
  name: "Plugin-based RiskManager",
  status: "partial",
  source: {
    repo: "vnpy/vnpy_riskmanager",
    module: "RiskEngine / rule plugins",
    version: "4.x",
    commit: null,
  },
  description: "Discoverable rules and interception logs.",
  quantagent: {
    modules: ["src/quantagent/risk/risk_gate.py"],
    api: ["/api/risk/rules"],
    events: [],
    artifacts: ["risk_events"],
    frontend: ["Risk Center"],
  },
  gap: "Per-rule configuration and realtime interception projection are incomplete.",
  adoption: "Keep RiskGate canonical and add a rule registry.",
  tests: ["RiskGate tests"],
  evidence: ["Risk Center"],
  limitations: ["configuration UI limited"],
  nextAction: "Implement typed risk rule registry.",
};

test("renders and inspects the machine-readable vn.py parity registry", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
    status: "ready",
    data: {
      schemaVersion: "quantagent.vnpy-parity.v1",
      registryVersion: "2026.07.22.1",
      title: "QuantAgent × VeighNa Capability Parity Registry",
      generatedAt: "2026-07-22T08:30:00+09:00",
      sourceBaseline: {
        repo: "vnpy/vnpy",
        release: "4.4.0",
        commit: "0e8e5ba",
        releaseDate: "2026-05-14",
        notes: [],
      },
      completeness: "partial",
      verificationPolicy: { verifiedRequires: ["real backend capability", "browser verification"] },
      knownCoverageGaps: ["exact extension commit pins"],
      categories: ["risk"],
      statuses: ["partial", "verified"],
      summary: {
        total: 1,
        byStatus: { partial: 1 },
        byCategory: { risk: 1 },
        verified: 0,
        actionable: 1,
        completionRatio: 0.5,
      },
      capabilities: [capability],
    },
    issues: [],
  }), { status: 200, headers: { "Content-Type": "application/json" } })));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <VnpyParityPage />
    </QueryClientProvider>,
  );

  expect((await screen.findAllByText("Plugin-based RiskManager")).length).toBeGreaterThan(0);
  expect(screen.getByText("50.0%")).toBeInTheDocument();
  fireEvent.click(screen.getAllByText("Plugin-based RiskManager")[0]);

  const inspector = screen.getByRole("complementary", { name: "能力详情" });
  expect(within(inspector).getByText(/Per-rule configuration/)).toBeInTheDocument();
  expect(within(inspector).getByText("Implement typed risk rule registry.")).toBeInTheDocument();
});
