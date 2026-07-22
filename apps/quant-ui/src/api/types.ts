export type DataStatus = "ready" | "partial" | "empty" | "error" | "stale" | "unavailable";
export type ArtifactTrustClass = "production_ready" | "paper_only" | "research_only" | "contaminated" | "unclassified";
export type ArtifactValidationStatus = "verified" | "declared" | "unverified" | "invalid";
export type ArtifactFreshnessStatus = "current" | "stale" | "unknown";

export interface DataIssue {
  code: string;
  message: string;
  path?: string | null;
  recoverable?: boolean;
}

export interface ApiResponse<T> {
  status: DataStatus;
  data: T;
  issues: DataIssue[];
  provenance?: Record<string, unknown>;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
}

export interface RuntimeArtifact {
  id: string;
  kind: string;
  name: string;
  path: string;
  extension: string;
  sizeBytes: number;
  modifiedAt: string;
  status: DataStatus;
  parser?: string | null;
  runId?: string | null;
  horizon?: string | null;
  rows?: number | null;
  dateStart?: string | null;
  dateEnd?: string | null;
  tags: string[];
  schemaVersion?: string | null;
  trustClass: ArtifactTrustClass;
  validationStatus: ArtifactValidationStatus;
  freshnessStatus: ArtifactFreshnessStatus;
  staleReason?: string | null;
  sourceTime?: string | null;
  manifestPath?: string | null;
  contentHash?: string | null;
  declaredKind?: string | null;
  kindSource?: "manifest" | "path_heuristic";
  runIdSource?: "manifest" | "path_heuristic" | null;
  producer?: string | null;
  qualityStatus?: string | null;
  dataAsOf?: string | null;
  upstreamPaths: string[];
  capabilities: string[];
  issues: DataIssue[];
}

export interface RuntimeRunSummary {
  id: string;
  artifactCount: number;
  totalSizeBytes: number;
  kinds: string[];
  trustClasses: ArtifactTrustClass[];
  validationStatuses: ArtifactValidationStatus[];
  capabilities: string[];
  issueCount: number;
  latestModifiedAt: string;
  dateStart?: string | null;
  dateEnd?: string | null;
}

export interface RuntimeCatalogSummary {
  artifactCount: number;
  totalSizeBytes: number;
  byKind: Record<string, number>;
  byTrust: Record<string, number>;
  byValidation: Record<string, number>;
  byFreshness: Record<string, number>;
  byCapability: Record<string, number>;
  byStatus: Record<string, number>;
  runCount: number;
  manifestCoverage: number;
  indexedAt: string;
}

export interface RuntimeCatalog {
  summary: RuntimeCatalogSummary;
  runs: RuntimeRunSummary[];
  roots: string[];
}

export interface RuntimeLineage {
  artifact: RuntimeArtifact;
  upstream: Array<{ reference: string; artifact?: RuntimeArtifact | null }>;
  downstream: RuntimeArtifact[];
  status: "complete" | "partial" | "undeclared";
  issues: DataIssue[];
}


export interface DataProvider {
  id: string;
  label: string;
  module: string | null;
  commandId: string | null;
  assetClasses: string[];
  intervals: string[];
  operations: string[];
  requires: string[];
  note: string;
  installed: boolean;
  configured: boolean;
  status: "ready" | "partial" | "needs_configuration" | "unavailable";
  missingRequirements: string[];
  optionalRequirements?: string[];
  missingOptionalRequirements?: string[];
}

export interface DataManagerOverview {
  providers: DataProvider[];
  constraints: string[];
  jobEndpoint: string;
  coverageEndpoint: string;
  quarantineEndpoint: string;
  supportsCancellation: boolean;
  runtimeRoot: string;
  serverPaths: { quarantine: string; imports: string; exports: string };
}

export interface QuarantineFile {
  path: string;
  name: string;
  format: string;
  sizeBytes: number;
  modifiedAt: string;
}

export interface DataCoverage {
  path: string;
  format: string;
  sizeBytes: number;
  columns: string[];
  rows: number;
  scannedKeyRows: number;
  symbolCount: number;
  dateCount: number;
  dateStart: string | null;
  dateEnd: string | null;
  duplicateKeys: number;
  duplicateMode: "exact" | "within_batch";
  missingBusinessDayCandidates: string[];
  missingBusinessDayCount: number;
  warnings: string[];
}

export interface JobSummary {
  id: string;
  type: string;
  status: string;
  commandId: string;
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  progress?: number | null;
  message?: string | null;
  outputPaths: string[];
  error?: string | null;
}

export interface EventEnvelope {
  schemaVersion: "quantagent.event.v1";
  eventId: string;
  eventType: string;
  topic: string;
  occurredAt: string;
  source: string;
  sequence: number;
  correlationId?: string | null;
  payload: Record<string, unknown>;
}

export interface BacktestSummary {
  id: string;
  name?: string | null;
  strategyVersion?: string | null;
  modelVersion?: string | null;
  factorVersion?: string | null;
  horizon?: string | null;
  startDate?: string | null;
  endDate?: string | null;
  universeSize?: number | null;
  initialCash?: number | null;
  totalReturn?: number | null;
  annualReturn?: number | null;
  maxDrawdown?: number | null;
  sharpe?: number | null;
  calmar?: number | null;
  volatility?: number | null;
  winRate?: number | null;
  profitFactor?: number | null;
  turnover?: number | null;
  tradeCount?: number | null;
  fillCount?: number | null;
  tTradeCount?: number | null;
  tContribution?: number | null;
  totalCost?: number | null;
  status: DataStatus;
  path: string;
  tags: string[];
  trustClass?: ArtifactTrustClass;
  validationStatus?: ArtifactValidationStatus;
  manifestPath?: string | null;
  capabilities?: Record<string, boolean | string | null>;
}

export interface EquityPoint {
  datetime: string;
  nav: number;
  dailyReturn?: number | null;
  drawdown?: number | null;
  benchmarkNav?: number | null;
  excessNav?: number | null;
}

export interface Trade {
  id: string;
  datetime: string;
  symbol: string;
  name?: string | null;
  action: string;
  price: number;
  quantity: number;
  amount?: number | null;
  fee?: number | null;
  commission?: number | null;
  slippage?: number | null;
  tax?: number | null;
  transferFee?: number | null;
  impactCost?: number | null;
  positionAfter?: number | null;
  positionWeightAfter?: number | null;
  cashAfter?: number | null;
  signalSource?: string | null;
  modelScore?: number | null;
  factorContributions?: Record<string, number> | null;
  riskReason?: string | null;
  pnl?: number | null;
  cumulativePnl?: number | null;
  success?: boolean | null;
  failureReason?: string | null;
  status?: string | null;
  tPairId?: string | null;
}

export interface KlineBar {
  datetime: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number | null;
  amount?: number | null;
  isSt?: boolean | null;
  isSuspended?: boolean | null;
  isLimitUp?: boolean | null;
  isLimitDown?: boolean | null;
}

export interface StockReplay {
  backtestId: string;
  symbol: string;
  name?: string | null;
  bars: KlineBar[];
  trades: Trade[];
  signals: Array<Record<string, unknown>>;
  positions: Array<Record<string, unknown>>;
  scoreSeries: Array<Record<string, unknown>>;
  equity: EquityPoint[];
  summary: Record<string, number | string | null>;
  availability: Record<string, boolean>;
  issues?: DataIssue[];
}

export interface Factor {
  name: string;
  displayName?: string | null;
  category?: string | null;
  description?: string | null;
  codeLocation?: string | null;
  formula?: string | null;
  direction: string;
  horizonDays?: number | null;
  parameters: Record<string, unknown>;
  dataSource: string[];
  requiredColumns: string[];
  frequency?: string | null;
  lookback?: number | null;
  pitSafe?: boolean | null;
  usedInTraining?: boolean | null;
  usedInSelection?: boolean | null;
  usedInTiming?: boolean | null;
  usedInRisk?: boolean | null;
  lifecycle?: string | null;
  sourceKind: string;
}

export interface ModelSummary {
  id: string;
  modelType?: string | null;
  version?: string | null;
  featureVersion?: string | null;
  createdAt?: string | null;
  trainStart?: string | null;
  trainEnd?: string | null;
  testEnd?: string | null;
  horizons: number[];
  featureCount?: number | null;
  sampleCount?: number | null;
  device?: string | null;
  gpuName?: string | null;
  productionReady?: boolean | null;
  status: DataStatus;
  path: string;
  issues: DataIssue[];
  modelFamily?: string | null;
  sourceKind?: string | null;
  verdict?: string | null;
  capabilities?: Record<string, boolean>;
}

export interface ModelMetric {
  key: string;
  label: string;
  value: number;
  source: string;
  group: "return" | "risk" | "quality" | "scale" | "other";
  unit: "ratio" | "bps" | "count" | "number";
}

export interface ModelArtifact {
  role: string;
  name: string;
  path: string;
  extension: string;
  sizeBytes: number;
  modifiedAt: string;
  previewable: boolean;
}

export interface ModelObservability extends ModelSummary {
  metrics: ModelMetric[];
  artifacts: ModelArtifact[];
  evaluations: Array<{ name: string; path: string; data: Record<string, unknown> }>;
  config: Record<string, unknown>;
  availability: Record<string, boolean>;
  checkpoint: {
    contentExposed: boolean;
    count: number;
    sizeBytes: number;
  };
}

export interface CleanupCandidate {
  id: string;
  category: string;
  label: string;
  reason: string;
  paths: string[];
  sizeBytes: number;
  itemCount: number;
  modifiedAt?: string | null;
  safeDefault: boolean;
  requiresExplicit: boolean;
}

export interface RuntimeCleanupAnalysis {
  runtimeSizeBytes: number;
  candidateSizeBytes: number;
  safeDefaultSizeBytes: number;
  candidates: CleanupCandidate[];
  protected: string[];
}

export interface CleanupResult {
  generatedAt: string;
  deleted: Array<{
    id: string;
    label: string;
    items: Array<{ path: string; sizeBytes: number }>;
    sizeBytes: number;
  }>;
  errors: Array<{ path: string; message: string }>;
  freedBytes: number;
  auditPath: string;
}

export interface SelectionRun {
  id: string;
  asOfDate?: string | null;
  candidateCount?: number | null;
  finalCount?: number | null;
  usedFallback?: boolean | null;
  noOrdersGenerated?: boolean | null;
  path: string;
  status: DataStatus;
  modifiedAt: number;
}

export interface RiskOverview {
  backtestId?: string | null;
  maxDrawdown?: number | null;
  maxSingleStockLoss?: number | null;
  maxDailyLoss?: number | null;
  consecutiveLossDays?: number | null;
  concentration?: number | null;
  sectorConcentration?: number | null;
  volatilityExposure?: number | null;
  liquidityRisk?: number | null;
  limitDownRisk?: number | null;
  suspensionRisk?: number | null;
  doTFailureRisk?: number | null;
  eventCounts: Record<string, number>;
  rules: Array<Record<string, unknown>>;
}

export interface SystemOverview {
  modelStatus: string;
  latestModel?: ModelSummary | null;
  latestBacktest?: BacktestSummary | null;
  latestSelection?: SelectionRun | null;
  stockPoolCount?: number | null;
  candidateCount?: number | null;
  signalCount?: number | null;
  buySignalCount?: number | null;
  sellSignalCount?: number | null;
  doTSignalCount?: number | null;
  riskStatus: string;
  risk: RiskOverview;
  runtime: {
    artifactCount: number;
    totalSizeBytes: number;
    byKind: Record<string, number>;
    byTrust?: Record<string, number>;
    byValidation?: Record<string, number>;
    byFreshness?: Record<string, number>;
    byCapability?: Record<string, number>;
    byStatus?: Record<string, number>;
    runCount?: number;
    manifestCoverage?: number;
    indexedAt: string;
  };
}
