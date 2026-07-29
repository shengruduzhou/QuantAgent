import { useEffect, useState } from "react";
import { apiEventSourceUrl } from "../api/client";
import type { JobSummary } from "../api/types";

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
const MAX_LINES = 800;

export function useJobStream(jobId: string | null): {
  job: JobSummary | null;
  lines: string[];
  connected: boolean;
  error: string;
} {
  const [job, setJob] = useState<JobSummary | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setJob(null);
    setLines([]);
    setError("");
    if (!jobId) return undefined;
    const source = new EventSource(apiEventSourceUrl(`/jobs/${jobId}/stream`));
    source.onopen = () => setConnected(true);
    source.addEventListener("log", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent<string>).data) as { line?: string };
        if (payload.line !== undefined) {
          setLines((current) => [...current, payload.line as string].slice(-MAX_LINES));
        }
      } catch {
        setError("任务日志流格式无效");
      }
    });
    source.addEventListener("status", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent<string>).data) as JobSummary;
        setJob(payload);
        if (TERMINAL.has(payload.status)) {
          setConnected(false);
          source.close();
        }
      } catch {
        setError("任务状态流格式无效");
      }
    });
    source.onerror = () => {
      setConnected(false);
      setError("实时流暂时断开；任务仍可在任务中心查询。");
    };
    return () => {
      source.close();
      setConnected(false);
    };
  }, [jobId]);

  return { job, lines, connected, error };
}
