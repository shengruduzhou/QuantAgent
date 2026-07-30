import type { ApiResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly issues: string[] = [],
  ) {
    super(message);
  }
}

function compactApiIssue(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const issue = value as { loc?: unknown; msg?: unknown };
  if (typeof issue.msg !== "string") return null;
  const field = Array.isArray(issue.loc)
    ? issue.loc.filter((item) => item !== "body").map(String).join(".")
    : "";
  const message = issue.msg.replace(/^Value error,\s*/i, "");
  return field ? `${field}: ${message}` : message;
}

export function compactApiErrorPayload(text: string, status: number): { message: string; issues: string[] } {
  let payload: unknown;
  try {
    payload = JSON.parse(text) as unknown;
  } catch {
    return { message: text || `Request failed: ${status}`, issues: [] };
  }
  if (!payload || typeof payload !== "object") {
    return { message: `Request failed: ${status}`, issues: [] };
  }
  const body = payload as {
    detail?: unknown;
    message?: unknown;
    errors?: unknown;
  };
  if (Array.isArray(body.detail)) {
    const issues = body.detail.map(compactApiIssue).filter((item): item is string => Boolean(item));
    if (issues.length) return { message: issues.join("；"), issues };
  }
  if (body.detail && typeof body.detail === "object") {
    const detail = body.detail as { message?: unknown; errors?: unknown };
    const errors = Array.isArray(detail.errors) ? detail.errors.map(String) : [];
    const heading = typeof detail.message === "string" ? detail.message : "";
    const issues = [heading, ...errors].filter(Boolean);
    if (issues.length) return { message: issues.join("；"), issues };
  }
  if (typeof body.detail === "string") {
    return { message: body.detail, issues: [body.detail] };
  }
  if (typeof body.message === "string") {
    const errors = Array.isArray(body.errors) ? body.errors.map(String) : [];
    return {
      message: [body.message, ...errors].join("；"),
      issues: [body.message, ...errors],
    };
  }
  return { message: `Request failed: ${status}`, issues: [] };
}

async function apiError(response: Response): Promise<ApiError> {
  const text = await response.text();
  const compact = compactApiErrorPayload(text, response.status);
  return new ApiError(compact.message, response.status, compact.issues);
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number | boolean | null | undefined>,
  signal?: AbortSignal,
): Promise<ApiResponse<T>> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw await apiError(response);
  }
  return response.json() as Promise<ApiResponse<T>>;
}

export async function apiPost<T>(
  path: string,
  value: unknown,
  signal?: AbortSignal,
): Promise<ApiResponse<T>> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
    signal,
  });
  if (!response.ok) {
    throw await apiError(response);
  }
  return response.json() as Promise<ApiResponse<T>>;
}

export async function apiDelete<T>(
  path: string,
  signal?: AbortSignal,
): Promise<ApiResponse<T>> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  const response = await fetch(url, { method: "DELETE", signal });
  if (!response.ok) {
    throw await apiError(response);
  }
  return response.json() as Promise<ApiResponse<T>>;
}

export function apiEventSourceUrl(path: string): string {
  return new URL(`${API_BASE}${path}`, window.location.origin).toString();
}

export function apiWebSocketUrl(
  path: string,
  params?: Record<string, string | number | boolean | null | undefined>,
): string {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

export function downloadJson(filename: string, value: unknown): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
