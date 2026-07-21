import { act, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, test, vi } from "vitest";
import type { ApiResponse, EventEnvelope, JobSummary } from "../api/types";
import { GLOBAL_JOBS_QUERY_KEY, useJobEvents } from "./useJobEvents";


class MockWebSocket {
  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = 0;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string | URL) {
    this.url = String(url);
    MockWebSocket.instances.push(this);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  emit(value: EventEnvelope): void {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(value) }));
  }

  serverClose(): void {
    this.readyState = 3;
    this.onclose?.(new CloseEvent("close", { code: 1012, reason: "restart" }));
  }

  close(code = 1000, reason = ""): void {
    if (this.readyState === 3) return;
    this.readyState = 3;
    this.onclose?.(new CloseEvent("close", { code, reason }));
  }
}


function envelope(eventType: string, payload: Record<string, unknown>): EventEnvelope {
  return {
    schemaVersion: "quantagent.event.v1",
    eventId: "evt_fixture",
    eventType,
    topic: "jobs",
    occurredAt: "2026-07-22T00:00:00.000+00:00",
    source: "test",
    sequence: 1,
    payload,
  };
}


function Probe(): JSX.Element {
  const state = useJobEvents(true);
  return <div data-testid="connection">{state.status}</div>;
}


afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  MockWebSocket.instances = [];
});


test("job event stream applies snapshots and reconnects after disconnect", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const view = render(
    <QueryClientProvider client={queryClient}>
      <Probe />
    </QueryClientProvider>,
  );

  expect(MockWebSocket.instances).toHaveLength(1);
  expect(MockWebSocket.instances[0].url).toContain("/api/events/ws?topics=jobs");
  act(() => MockWebSocket.instances[0].open());
  expect(screen.getByTestId("connection")).toHaveTextContent("live");

  const job: JobSummary = {
    id: "job_fixture",
    type: "train",
    status: "running",
    commandId: "train-v8-deep",
    createdAt: "2026-07-22T00:00:00+00:00",
    outputPaths: [],
  };
  act(() => MockWebSocket.instances[0].emit(envelope("system.snapshot", { jobs: [job] })));
  const cached = queryClient.getQueryData<ApiResponse<JobSummary[]>>(GLOBAL_JOBS_QUERY_KEY);
  expect(cached?.data).toEqual([job]);

  act(() => MockWebSocket.instances[0].serverClose());
  expect(screen.getByTestId("connection")).toHaveTextContent("reconnecting");
  await act(async () => {
    vi.advanceTimersByTime(1_000);
    await Promise.resolve();
  });
  expect(MockWebSocket.instances).toHaveLength(2);

  view.unmount();
});
