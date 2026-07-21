import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiWebSocketUrl } from "../api/client";
import type { ApiResponse, EventEnvelope, JobSummary } from "../api/types";

export const GLOBAL_JOBS_QUERY_KEY = ["global-activity-jobs", undefined] as const;

export type RealtimeStatus = "idle" | "connecting" | "live" | "reconnecting" | "stale" | "unavailable";

export interface JobEventStreamState {
  status: RealtimeStatus;
  lastEventAt?: string;
}

const HEARTBEAT_TIMEOUT_MS = 45_000;
const MAX_RECONNECT_MS = 15_000;

function isEnvelope(value: unknown): value is EventEnvelope {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<EventEnvelope>;
  return candidate.schemaVersion === "quantagent.event.v1"
    && typeof candidate.eventType === "string"
    && typeof candidate.sequence === "number"
    && Boolean(candidate.payload && typeof candidate.payload === "object");
}

function isJob(value: unknown): value is JobSummary {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<JobSummary>;
  return typeof candidate.id === "string"
    && typeof candidate.status === "string"
    && typeof candidate.commandId === "string";
}

function jobResponse(jobs: JobSummary[]): ApiResponse<JobSummary[]> {
  return {
    status: jobs.length ? "ready" : "empty",
    data: jobs,
    issues: [],
  };
}

export function useJobEvents(enabled: boolean): JobEventStreamState {
  const queryClient = useQueryClient();
  const [state, setState] = useState<JobEventStreamState>({ status: "idle" });

  useEffect(() => {
    if (!enabled) {
      setState({ status: "idle" });
      return undefined;
    }
    if (typeof window.WebSocket !== "function") {
      setState({ status: "unavailable" });
      return undefined;
    }

    let stopped = false;
    let socket: WebSocket | undefined;
    let reconnectTimer: ReturnType<typeof window.setTimeout> | undefined;
    let heartbeatTimer: ReturnType<typeof window.setTimeout> | undefined;
    let attempts = 0;

    const clearHeartbeat = (): void => {
      if (heartbeatTimer !== undefined) window.clearTimeout(heartbeatTimer);
      heartbeatTimer = undefined;
    };

    const armHeartbeat = (): void => {
      clearHeartbeat();
      heartbeatTimer = window.setTimeout(() => {
        if (stopped) return;
        setState((current) => ({ ...current, status: "stale" }));
        socket?.close(4000, "heartbeat timeout");
      }, HEARTBEAT_TIMEOUT_MS);
    };

    const applyEvent = (event: EventEnvelope): void => {
      if (event.eventType === "system.snapshot") {
        const jobs = event.payload.jobs;
        if (Array.isArray(jobs) && jobs.every(isJob)) {
          queryClient.setQueryData(GLOBAL_JOBS_QUERY_KEY, jobResponse(jobs));
        }
      } else if (event.eventType === "job.status" && isJob(event.payload.job)) {
        queryClient.setQueryData<ApiResponse<JobSummary[]>>(
          GLOBAL_JOBS_QUERY_KEY,
          (current) => {
            const jobs = current?.data ?? [];
            const changed = event.payload.job as JobSummary;
            const updated = [changed, ...jobs.filter((job) => job.id !== changed.id)];
            return jobResponse(updated);
          },
        );
      } else if (event.eventType === "stream.gap") {
        void queryClient.invalidateQueries({ queryKey: GLOBAL_JOBS_QUERY_KEY });
      }
    };

    const connect = (): void => {
      if (stopped) return;
      setState((current) => ({ ...current, status: attempts ? "reconnecting" : "connecting" }));
      const currentSocket = new window.WebSocket(apiWebSocketUrl("/events/ws", { topics: "jobs" }));
      socket = currentSocket;
      currentSocket.onopen = () => {
        attempts = 0;
        setState((current) => ({ ...current, status: "live" }));
        armHeartbeat();
      };
      currentSocket.onmessage = (message) => {
        try {
          const event: unknown = JSON.parse(String(message.data));
          if (!isEnvelope(event)) return;
          applyEvent(event);
          setState({ status: "live", lastEventAt: event.occurredAt });
          armHeartbeat();
        } catch {
          // Invalid frames never replace persisted REST state.
        }
      };
      currentSocket.onerror = () => currentSocket.close();
      currentSocket.onclose = () => {
        clearHeartbeat();
        if (stopped) return;
        attempts += 1;
        setState((current) => ({ ...current, status: "reconnecting" }));
        const delay = Math.min(1_000 * (2 ** (attempts - 1)), MAX_RECONNECT_MS);
        reconnectTimer = window.setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      stopped = true;
      clearHeartbeat();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close(1000, "component unmounted");
    };
  }, [enabled, queryClient]);

  return state;
}
