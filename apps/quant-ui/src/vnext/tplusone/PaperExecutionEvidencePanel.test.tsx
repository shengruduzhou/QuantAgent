import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useApi } from "../../hooks/useApi";
import {
  PaperExecutionEvidencePanel,
  type PaperExecutionEvidence,
} from "./PaperExecutionEvidencePanel";

vi.mock("../../hooks/useApi", () => ({ useApi: vi.fn() }));

const mockedUseApi = vi.mocked(useApi);

function evidence(overrides: Partial<PaperExecutionEvidence> = {}): PaperExecutionEvidence {
  const base: PaperExecutionEvidence = {
    journal: {
      state: "valid",
      verified: true,
      path: "runtime/paper/execution_journal.jsonl",
      recordCount: 2,
      terminalCount: 1,
      unresolvedCount: 0,
      reason: null,
    },
    summary: {
      attention: "ok",
      latestStatus: "execution_observed",
      latestSignalDate: "2026-08-07",
      latestExecutionDate: "2026-08-10",
      latestRecordedAt: "2026-08-10T07:01:00+00:00",
      calendarAssurance: "observed_market_panel_only",
      shadowAcceptanceCalendarEligible: false,
      productionPretradeRiskCertified: false,
      riskScope: "paper_simulator_admissibility_only",
      sessionClosed: true,
      orderCount: 1,
      fillCount: 1,
      navBefore: 1_000_000,
      navAfter: 1_001_200,
      statusCounts: { execution_started: 1, execution_observed: 1 },
    },
    records: [
      {
        sequence: 2,
        payloadSha256: "payload-a",
        signalDate: "2026-08-07",
        executionDate: "2026-08-10",
        status: "execution_observed",
        recordedAt: "2026-08-10T07:01:00+00:00",
        recordSha256: "1234567890abcdef",
        details: {},
      },
    ],
    operatorTruth: {
      paperExecutionEvidence: true,
      productionLiveCertified: false,
      authoritativeCalendarCertified: false,
      message: "Paper/shadow execution evidence is available. It is not production/live certification.",
    },
  };
  return {
    ...base,
    ...overrides,
    journal: { ...base.journal, ...(overrides.journal ?? {}) },
    summary: { ...base.summary, ...(overrides.summary ?? {}) },
    operatorTruth: { ...base.operatorTruth, ...(overrides.operatorTruth ?? {}) },
    records: overrides.records ?? base.records,
  };
}

function mockEvidence(value: PaperExecutionEvidence): void {
  mockedUseApi.mockReturnValue({
    isLoading: false,
    isError: false,
    data: { status: "ready", data: value, issues: [] },
    error: null,
    refetch: vi.fn(),
  } as never);
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("PaperExecutionEvidencePanel", () => {
  test("shows verified paper execution while keeping production and calendar certification locked", () => {
    mockEvidence(evidence());
    render(<PaperExecutionEvidencePanel />);

    const status = screen.getByRole("status");
    expect(within(status).getByText("Paper 执行已观察")).toBeInTheDocument();
    expect(screen.getByText("Paper execution evidence").parentElement).toHaveTextContent("YES");
    expect(screen.getByText("Production pre-trade certified").parentElement).toHaveTextContent("NO");
    expect(screen.getByText("Authoritative calendar certified").parentElement).toHaveTextContent("NO");
    expect(screen.getByText(/Paper\/shadow 观察结果 ≠ 实盘认证/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  test("uses an assertive alert for indeterminate execution", () => {
    mockEvidence(evidence({
      journal: {
        state: "valid",
        verified: true,
        path: "runtime/paper/execution_journal.jsonl",
        recordCount: 1,
        terminalCount: 1,
        unresolvedCount: 0,
        reason: null,
      },
      summary: {
        ...evidence().summary,
        attention: "critical",
        latestStatus: "execution_indeterminate",
      },
      records: [],
    }));
    render(<PaperExecutionEvidencePanel />);

    expect(screen.getByRole("alert")).toHaveTextContent("执行结果不确定");
    expect(screen.getByRole("alert")).toHaveAttribute("aria-live", "assertive");
  });

  test("fails closed when the journal is invalid", () => {
    mockEvidence(evidence({
      journal: {
        state: "invalid",
        verified: false,
        path: "runtime/paper/execution_journal.jsonl",
        recordCount: 0,
        terminalCount: 0,
        unresolvedCount: 0,
        reason: "execution journal hash-chain verification failed",
      },
      summary: {
        ...evidence().summary,
        attention: "critical",
        latestStatus: null,
      },
      records: [],
      operatorTruth: {
        paperExecutionEvidence: false,
        productionLiveCertified: false,
        authoritativeCalendarCertified: false,
        message: "Paper execution evidence is invalid or unverifiable; fail closed.",
      },
    }));
    render(<PaperExecutionEvidencePanel />);

    expect(screen.getByRole("alert")).toHaveTextContent("执行证据链校验失败");
    expect(screen.getByText("Production pre-trade certified").parentElement).toHaveTextContent("NO");
  });
});
