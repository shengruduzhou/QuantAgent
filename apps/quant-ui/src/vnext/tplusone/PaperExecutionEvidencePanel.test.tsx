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
    accountIdentity: {
      state: "valid",
      verified: true,
      path: "runtime/paper/account_identity.json",
      accountInstanceId: "paper-account-instance-a",
      portfolioId: "v7-paper",
      initialCashCny: "1000000.0000000000",
      payloadSha256: "abcdef1234567890",
      reason: null,
    },
    canonicalPrefix: {
      state: "valid",
      verified: true,
      boundTerminalCount: 1,
      legacyUnboundTerminalCount: 0,
      latestTerminalBound: true,
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
      accountIdentityVerified: true,
      latestTerminalCanonicalPrefixBound: true,
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
      accountIdentityVerified: true,
      canonicalExecutionPrefixCertified: true,
      productionLiveCertified: false,
      authoritativeCalendarCertified: false,
      message: "Paper/shadow execution evidence is available. It is not production/live certification.",
    },
  };
  return {
    ...base,
    ...overrides,
    journal: { ...base.journal, ...(overrides.journal ?? {}) },
    accountIdentity: { ...base.accountIdentity, ...(overrides.accountIdentity ?? {}) },
    canonicalPrefix: { ...base.canonicalPrefix, ...(overrides.canonicalPrefix ?? {}) },
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

  test("prioritizes an unresolved incident over a newer observed record", () => {
    mockEvidence(evidence({
      journal: {
        ...evidence().journal,
        recordCount: 4,
        terminalCount: 1,
        unresolvedCount: 1,
      },
      summary: {
        ...evidence().summary,
        attention: "critical",
        latestStatus: "execution_observed",
      },
    }));
    render(<PaperExecutionEvidencePanel />);

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("存在未闭合执行尝试");
    expect(alert).toHaveTextContent("禁止把最新成功记录解释为账户整体健康");
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
      accountIdentity: {
        ...evidence().accountIdentity,
        state: "invalid",
        verified: false,
        reason: "paper account identity verification failed",
      },
      canonicalPrefix: {
        ...evidence().canonicalPrefix,
        state: "invalid",
        verified: false,
        latestTerminalBound: false,
        reason: "canonical prefix verification failed",
      },
      summary: {
        ...evidence().summary,
        attention: "critical",
        latestStatus: null,
        accountIdentityVerified: false,
        latestTerminalCanonicalPrefixBound: false,
      },
      records: [],
      operatorTruth: {
        paperExecutionEvidence: false,
        accountIdentityVerified: false,
        canonicalExecutionPrefixCertified: false,
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
