/**
 * Integration check against a running QuantAgent API.
 *
 * Component tests with stubbed responses prove the page renders *some* shape;
 * they cannot catch the case where the backend's real payload has drifted from
 * that shape. This file renders the same components against a live API and
 * asserts on whatever it actually returns.
 *
 * It is skipped unless QUANT_UI_LIVE_API points at a reachable API, so the
 * normal suite stays hermetic:
 *
 *   QUANT_UI_LIVE_API=http://127.0.0.1:8000 npx vitest run --mode live \
 *     src/vnext/runs/ResearchRunsPage.live.test.tsx
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeAll, describe, expect, test, vi } from "vitest";
import { ResearchRunsPage } from "./ResearchRunsPage";

vi.mock("../../components/EChart", () => ({
  EChart: () => <div data-testid="nav-chart" />,
}));

const BASE = (import.meta.env?.QUANT_UI_LIVE_API as string | undefined)
  ?? (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.QUANT_UI_LIVE_API
  ?? "";
let reachable = false;

beforeAll(async () => {
  if (!BASE) return;
  try {
    const response = await fetch(`${BASE}/health`);
    reachable = response.ok;
  } catch {
    reachable = false;
  }
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe.runIf(BASE)("live API contract", () => {
  test("renders real runs, their conclusions and their evidence", async () => {
    expect(reachable, `API at ${BASE} is not reachable`).toBe(true);

    // The page builds relative URLs against window.location; point them at the
    // live server without changing the component under test.
    const nativeFetch = globalThis.fetch;
    vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input instanceof Request ? input.url : input);
      const rewritten = url.replace(/^https?:\/\/[^/]+/, BASE);
      return nativeFetch(rewritten, init);
    });
    vi.stubGlobal("EventSource", class {
      close(): void {}
      addEventListener(): void {}
    });

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><ResearchRunsPage /></QueryClientProvider>);

    await waitFor(
      () => expect(screen.getByRole("heading", { name: "研究运行" })).toBeInTheDocument(),
      { timeout: 15_000 },
    );

    // Whatever the runtime holds, the page must reach a definite state: either
    // real runs with a conclusion, or an empty state that names the next step.
    const runs = await fetch(`${BASE}/api/strategies/runs`).then((response) => response.json());
    if (!runs.data.length) {
      expect(screen.getByText("尚无研究运行")).toBeInTheDocument();
      return;
    }

    const detail = await fetch(`${BASE}/api/strategies/runs/${runs.data[0].runId}`)
      .then((response) => response.json());
    const conclusion = detail.data.result.conclusion;
    expect(conclusion, "a resolved run must carry a conclusion").toBeTruthy();

    await waitFor(
      () => expect(screen.getAllByText(conclusion.headline).length).toBeGreaterThan(0),
      { timeout: 15_000 },
    );

    // Reasons are the whole point of the conclusion: they must reach the DOM.
    for (const reason of conclusion.reasons.slice(0, 2)) {
      expect(screen.getAllByText(reason).length).toBeGreaterThan(0);
    }
  }, 60_000);
});
