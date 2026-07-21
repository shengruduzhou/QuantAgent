export type VnpyParityStatus =
  | "not_audited"
  | "missing"
  | "planned"
  | "in_progress"
  | "partial"
  | "implemented"
  | "verified"
  | "blocked"
  | "not_applicable";

export interface VnpySourceReference {
  repo: string;
  module: string;
  version: string;
  commit?: string | null;
}

export interface QuantAgentCapabilityMapping {
  modules: string[];
  api: string[];
  events: string[];
  artifacts: string[];
  frontend: string[];
}

export interface VnpyParityCapability {
  id: string;
  category: string;
  name: string;
  status: VnpyParityStatus;
  source: VnpySourceReference;
  description: string;
  quantagent: QuantAgentCapabilityMapping;
  gap: string;
  adoption: string;
  tests: string[];
  evidence: string[];
  limitations: string[];
  nextAction: string;
}

export interface VnpyParitySummary {
  total: number;
  byStatus: Record<string, number>;
  byCategory: Record<string, number>;
  verified: number;
  actionable: number;
  completionRatio: number;
}

export interface VnpyParityView {
  schemaVersion: string;
  registryVersion: string;
  title: string;
  generatedAt: string;
  sourceBaseline: {
    repo: string;
    release: string;
    commit: string;
    releaseDate: string;
    notes: string[];
  };
  completeness: string;
  verificationPolicy: {
    verifiedRequires: string[];
  };
  knownCoverageGaps: string[];
  categories: string[];
  statuses: VnpyParityStatus[];
  summary: VnpyParitySummary;
  capabilities: VnpyParityCapability[];
}
