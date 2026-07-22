import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { AppShell } from "./AppShell";

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  vi.unstubAllGlobals();
});

test("routes Help to the internal workstation guide", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
    status: "ready",
    data: {
      runtime: { artifactCount: 0, totalSizeBytes: 0, indexedAt: "2026-07-22T00:00:00Z" },
      riskStatus: "ready",
    },
    issues: [],
  }), { status: 200, headers: { "Content-Type": "application/json" } })));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <MemoryRouter initialEntries={["/"]}>
      <QueryClientProvider client={queryClient}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<div>首页内容</div>} />
            <Route path="help" element={<h1>产品内帮助页</h1>} />
          </Route>
        </Routes>
      </QueryClientProvider>
    </MemoryRouter>,
  );

  expect(await screen.findByText("首页内容")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "帮助" }));
  expect(await screen.findByRole("heading", { name: "产品内帮助页" })).toBeInTheDocument();
});
