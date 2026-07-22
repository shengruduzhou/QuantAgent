import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { SettingsPage } from "./SettingsPage";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("opens a fail-closed full-universe training template from the deep link", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = url.endsWith("/jobs")
      ? []
      : {
        runtime: { artifactCount: 10, totalSizeBytes: 1024, indexedAt: "2026-07-22T00:00:00Z" },
        riskStatus: "ready",
      };
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <MemoryRouter initialEntries={["/settings?job=train&universe=all"]}>
      <QueryClientProvider client={queryClient}>
        <SettingsPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );

  expect(await screen.findByText(/全宇宙训练 · ALL SYMBOLS IN DATASET/)).toBeInTheDocument();
  const editor = screen.getByRole("textbox") as HTMLTextAreaElement;
  const payload = JSON.parse(editor.value) as { commandId: string; parameters: Record<string, unknown> };

  expect(payload.commandId).toBe("train-v8-deep");
  expect(payload.parameters.horizon_class).toBe("short_5d");
  expect(payload.parameters.feature_policy).toBe("judgment");
  expect(payload.parameters).not.toHaveProperty("symbols");
  expect(payload.parameters).not.toHaveProperty("symbols_file");
  expect(screen.getByRole("button", { name: /确认并启动全宇宙训练/ })).toBeInTheDocument();
});
