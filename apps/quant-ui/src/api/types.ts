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
  /** Exact count, or null when more pages remain and the true total is unknown. */
  total: number | null;
  /** False means `total` is null and only `loadedCount` is known. */
  totalIsExact?: boolean;
  /** Rows fetched so far; a lower bound on the true total. */
  loadedCount?: number;
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

export interface JobStage {
  id: string;
  label: string;
  startedAt: string;
  completedAt?: string | null;
  status: "running" | "completed" | "stopped";
  message?: string | null;
  progress?: number | null;
}

export interface JobResources {
  cpuPercent?: number | null;
  rssBytes?: number | null;
  gpuMemoryMiB?: number | null;
  threads?: number | null;
  childProcesses?: number | null;
  sampledAt?: string | null;
}

/** A named cause with the evidence it was derived from, never a bare exit code. */
export interface JobFailure {
  code: string;
  title: string;
  detail: string;
  remediation: string;
  retryable: boolean;
  evidence?: string | null;
  logTail: string[];
  exitCode?: number | null;
  signal?: number | null;
}

/**
 * How a run concluded without succeeding.
 *
 * `rejected` — a pre-registered gate tested the candidate and refused it. That
 * is a result, not a fault: the evidence is intact and worth reading.
 * `blocked` — the run could not be executed as configured, so nothing was
 * tested. It carries a remediation instead of evidence.
 */
export interface JobVerdict {
  verdict: "rejected" | "blocked";
  code: string;
  title: string;
  stage?: string;
  reasons: string[];
  remediation?: string;
  metrics?: Record<string, unknown>;
  evidencePaths?: string[];
  decidedAt?: string;
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
  parameters?: Record<string, unknown>;
  labels?: Record<string, string>;
  retryOf?: string | null;
  attempt?: number;
  stage?: string | null;
  stages?: JobStage[];
  lastOutputAt?: string | null;
  resources?: JobResources | null;
  exitCode?: number | null;
  exitStatusObserved?: boolean;
  failure?: JobFailure | null;
  verdict?: JobVerdict | null;
  pid?: number | null;
  workerPid?: number | null;
  adopted?: boolean;
  terminal?: boolean;
  elapsedSeconds?: number | null;
  canRetry?: boolean;
}

export interface JobValidation {
  valid: boolean;
  type: string;
  commandId: string;
  entrypoint: string;
  outputPaths: string[];
  warnings: string[];
}

export interface ConnectionStatus {
  id: string;
  label: string;
  variables: string[];
  capabilities: string[];
  note: string;
  connected: boolean;
  source: "session" | "environment" | "none";
  fingerprints: Record<string, string>;
  persistence: "process_memory" | "server_environment" | "none";
}

export interface StrategyObjectiveWeights {
  excessReturn: number;
  annualReturn: number;
  drawdownControl: number;
}

export interface StrategyRiskLimits {
  maxDrawdown: number;
  maxTurnover: number;
  minSharpe: number;
}

export interface StrategyDraft {
  id?: string | null;
  name: string;
  hypothesis: string;
  invalidationCriteria: string;
  marketPanelPath: string;
  labelsPath: string;
  fundamentalsRoot?: string | null;
  valuationPath?: string | null;
  disclosuresPath?: string | null;
  sectorMapPath?: string | null;
  trainingDatasetPath?: string | null;
  synthesizedFactorsPath?: string | null;
  outputDir: string;
  factorLibrary: "all_reviewed" | "basic" | "alpha101" | "alpha181" | "cicc_ashare80";
  model: "ridge" | "ft_transformer";
  researchPreset: "stable_alpha" | "drawdown_first" | "annual_growth" | "transparent_baseline" | "custom";
  horizons: string;
  primaryHorizon: number;
  horizonBlendMethod: "adaptive_oos" | "balanced" | "short_tactical" | "long_fundamental" | "primary_only";
  splitMode: "rolling" | "expanding";
  nSplits: number;
  requireGpu: boolean;
  topK: number;
  topKCandidates: number[];
  stockSelectionModes: Array<"none" | "fundamental">;
  fundamentalSelectionMode: "auto" | "fixed" | "off";
  fundamentalSelectionThreshold: number;
  fundamentalBlendWeight: number;
  fundamentalThresholdCandidates: number[];
  fundamentalBlendCandidates: number[];
  selectionMaxCandidates: number;
  selectionMinOosDays: number;
  selectionMinHoldoutDays: number;
  maxPbo: number;
  minDsrProbability: number;
  maxSpaPValue: number;
  factorScreeningMode: "off" | "evaluate_only" | "pretrain";
  doTMode: "off" | "daily_swing" | "intraday" | "both";
  minutePanelPath?: string | null;
  maxWeightPerName: number;
  maxSectorWeight: number;
  maxTurnover: number;
  objective: "max_expected_alpha" | "mean_variance" | "min_variance";
  weighting: "equal" | "rank" | "softmax";
  initialCash: number;
  universeScope: "full" | "pilot";
  universeSymbols?: string | null;
  universeSymbolsFile?: string | null;
  benchmarkSymbol?: string | null;
  objectiveWeights: StrategyObjectiveWeights;
  riskLimits: StrategyRiskLimits;
  humanApproved: boolean;
}

export interface DecisionCouncilMember {
  id: string;
  label: string;
  responsibility: string;
  status: "ready" | "approved" | "waiting" | "blocked";
  veto: boolean;
  finding?: string;
  nextAction?: string;
  issueCount?: number;
  issueCodes?: string[];
}

export interface StrategyValidationIssue {
  code: string;
  severity: "blocking" | "warning" | "info";
  title: string;
  detail: string;
  field?: string;
  action?: {
    kind: string;
    label: string;
  };
  evidence?: {
    requestedHorizons?: number[];
    availableHorizons?: number[];
    missingHorizons?: number[];
    candidateCount?: number;
    candidateLimit?: number;
    method?: string;
    // Pre-flight arithmetic for the nested selection protocol: what the
    // configured folds will produce versus what the protocol requires.
    nSplits?: number;
    oosDaysPerFold?: number;
    projectedOosDays?: number;
    requiredOosDays?: number;
    minimumSplits?: number;
    benchmarkSymbol?: string;
    marketPanelPath?: string;
    universeSymbolsFile?: string | null;
    inlineSymbolCount?: number;
  };
}

export interface StrategyValidation {
  valid: boolean;
  errors: string[];
  warnings: string[];
  issues?: StrategyValidationIssue[];
  requestedHorizons?: number[];
  availableHorizons?: number[];
  resolvedInputs: Record<string, string>;
  launch: {
    jobType: "strategy-pipeline";
    commandId: "run-full-real-training-v7";
    parameters: Record<string, string | number | boolean | Array<string | number>>;
    armed: boolean;
  };
  decisionCouncil: DecisionCouncilMember[];
}

export interface StrategyManifestSummary {
  id: string;
  version: string;
  name: string;
  createdAt: string;
  trustClass: "research_only";
  contentHash?: string | null;
  path: string;
  valid: boolean;
  humanApproved: boolean;
  draft: StrategyDraft;
  versionCount?: number;
  firstCreatedAt?: string;
  updatedAt?: string;
  runCount?: number;
  lastRun?: StrategyRun | null;
  versions?: StrategyManifestSummary[];
  runs?: StrategyRun[];
}

/** One launched execution of one strategy version. */
export interface StrategyRun {
  runId: string;
  strategyId: string;
  strategyVersion: string;
  strategyName: string;
  jobId: string;
  outputDir: string;
  createdAt: string;
  job?: JobSummary | null;
}

export interface RunGate {
  name: string;
  passed: boolean;
  actual: unknown;
  threshold: unknown;
  reason?: string | null;
}

export interface RunConclusion {
  outcome: "accepted" | "not_accepted" | "rejected" | "incomplete" | "no_evidence";
  headline: string;
  reasons: string[];
  remediation: string;
  promotable: boolean;
}

export interface RunResult {
  status: "complete" | "rejected" | "partial" | "empty" | "absent" | "unavailable";
  outputDir: string;
  conclusion?: RunConclusion;
  verdict?: JobVerdict | null;
  acceptance?: {
    failures: string[];
    gates: RunGate[];
    passedCount: number;
    totalCount: number;
    sourcePath: string;
  } | null;
  governance?: {
    accepted: boolean;
    pbo?: number | null;
    dsrProbability?: number | null;
    spaPValue?: number | null;
    losingFoldRate?: number | null;
    observedDays?: number | null;
    cumulativeTrials?: number | null;
    selectedCandidate?: string | null;
    rejectionReasons: string[];
    sourcePath: string;
  } | null;
  training?: {
    backend?: string | null;
    rankIcMean?: number | null;
    icir?: number | null;
    hitRate?: number | null;
    foldCount?: number | null;
    featureCount?: number | null;
    evaluatedDays?: number | null;
    dataRange?: { start?: string | null; end?: string | null };
    annualisedReturn?: number | null;
    annualisedSharpe?: number | null;
    excessReturnAfterCosts?: number | null;
    adverseRegimePassed?: boolean | null;
    executable?: Record<string, unknown> | null;
    annualisationWarning?: string | null;
    sourcePath: string;
  } | null;
  backtest?: {
    navPoints: number;
    nav: Array<{ date: string; nav: number }>;
    totalReturn?: number | null;
    maxDrawdown?: number | null;
    orderCount?: number | null;
    skippedOrderCount?: number | null;
    failedOrderCount?: number | null;
    sourcePath: string;
  } | null;
  paper?: {
    status?: string | null;
    acceptanceStatus?: string | null;
    summary?: Record<string, unknown> | null;
    warnings: string[];
    sourcePath: string;
  } | null;
  candidates?: Array<{
    id: string;
    selected: boolean;
    annualisedReturn?: number | null;
    netReturnAfterCosts?: number | null;
    excessReturnAfterCosts?: number | null;
    maxDrawdown?: number | null;
    sharpe?: number | null;
    informationRatio?: number | null;
    tradeCount?: number | null;
    skippedOrderCount?: number | null;
    estimatedCosts?: number | null;
    acceptanceStatus?: string | null;
    sourcePath: string;
  }>;
  stages?: Array<{ id: string; label: string; present: boolean; path?: string | null; sizeBytes?: number | null }>;
  artifacts?: Array<{ name: string; path: string; relative: string; sizeBytes: number; extension: string }>;
  issues?: Array<{ code: string; message: string }>;
}

export interface StrategyRunDetail extends StrategyRun {
  result: RunResult;
}

export interface CouncilRunReview {
  subject: {
    type: string;
    id: string;
    path: string;
    strategyId?: string;
    strategyName?: string;
    outcome?: string | null;
  };
  roles: Array<{ id: string; label: string; domain: string; vetoScope: string; veto: boolean }>;
  findings: Array<{
    roleId: string;
    verdict: "pass" | "warn" | "blocked" | "unknown";
    headline: string;
    detail: string;
    evidence: Record<string, unknown>;
    nextAction: string;
    override?: {
      verdict: string;
      reason: string;
      author: string;
      recordedAt: string;
      replacedVerdict: string;
    };
  }>;
  decision: { state: string; summary: string; blockedRoles?: string[]; unknownRoles?: string[] };
  overrides: unknown[];
}

export interface RunComparison {
  runs: Array<{
    runId: string;
    strategyId: string;
    strategyName: string;
    strategyVersion: string;
    createdAt: string;
    outputDir: string;
    outcome?: RunConclusion["outcome"] | null;
    headline?: string | null;
    promotable: boolean;
  }>;
  metrics: Array<{
    key: string;
    label: string;
    group: string;
    direction: "higher" | "lower" | null;
    values: Array<number | string | null>;
    bestIndex: number | null;
  }>;
  gates: Array<{
    name: string;
    values: Array<{ passed: boolean; actual: unknown; threshold: unknown } | null>;
  }>;
  note?: string;
}

export interface StrategyLaunchResult {
  job: JobSummary;
  strategy: StrategyManifestSummary;
  run: StrategyRun;
}

export interface StrategyDefaults {
  selected: {
    marketPanelPath?: string | null;
    labelsPath?: string | null;
    trainingDatasetPath?: string | null;
    sectorMapPath?: string | null;
    fundamentalsRoot?: string | null;
    valuationPath?: string | null;
    disclosuresPath?: string | null;
  };
  options?: Record<string, StrategyInputOption[]>;
  evidence: Array<{ field: string; path: string; sizeBytes: number; modifiedAt: string }>;
  selectionRule: string;
}

export interface StrategyInputOption {
  field: string;
  path: string;
  exists: boolean;
  isDirectory: boolean;
  sizeBytes: number;
  modifiedAt: string | null;
  availableHorizons: number[];
}

export interface SearchEntity {
  id: string;
  kind: string;
  label: string;
  detail: string;
  path: string;
  status: string;
  source: string;
}

export interface SearchGroup {
  type: string;
  label: string;
  items: SearchEntity[];
}

export interface GlobalSearchResult {
  query: string;
  groups: SearchGroup[];
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

export interface TrainingMetricPoint {
  epoch: number;
  step?: number | null;
  loss?: number | null;
  validationLoss?: number | null;
  learningRate?: number | null;
  rankIc?: number | null;
  gradientNorm?: number | null;
  gpuMemory?: number | null;
  samplesPerSecond?: number | null;
  metrics: Record<string, number>;
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

export interface ModelComparison {
  models: Array<{
    id: string;
    version?: string | null;
    modelType?: string | null;
    modelFamily?: string | null;
    verdict?: string | null;
    status: string;
    metrics: Record<string, number>;
  }>;
  metricKeys: string[];
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

export interface SystemResources {
  cpuPercent?: number | null;
  memoryPercent?: number | null;
  memoryUsedBytes?: number | null;
  memoryTotalBytes?: number | null;
  gpus: Array<{
    index: number;
    name: string;
    utilizationPercent?: number | null;
    memoryUsedMiB?: number | null;
    memoryTotalMiB?: number | null;
    temperatureC?: number | null;
    powerDrawW?: number | null;
  }>;
}

/* ---------------------------------------------------------------- fusion --- */

/** One evaluated blend from a factor-fusion search. */
export interface FusionCandidate {
  id: string;
  label: string;
  scheme: string;
  /** Controls never read the training segment; they exist to be beaten. */
  isControl: boolean;
  weights: Record<string, number>;
  metrics: {
    observations: number;
    annualReturn: number;
    excessReturn: number;
    benchmarkAnnualReturn: number;
    maxDrawdown: number;
    sharpe: number;
    calmar: number;
    averageTurnover: number;
    costDrag: number;
    winRate: number;
    robustness: number;
  };
  robustnessBreakdown: {
    foldConsistency: number;
    overfittingResistance: number;
    deflatedSharpeProbability: number;
    regimeConsistency: number;
    pbo: number | null;
  };
  folds: Array<{
    foldIndex: number;
    trainStart: string;
    trainEnd: string;
    testStart: string;
    testEnd: string;
    weights: Record<string, number>;
    metrics: Record<string, number>;
  }>;
  onFrontier: boolean;
  preferenceRank: number | null;
  preferenceScore: number | null;
}

export interface FusionRunSummary {
  id: string;
  name: string;
  path: string;
  generatedAt: string | null;
  contentHash: string | null;
  nTrials: number | null;
  pbo: number | null;
  benchmarkMode: string | null;
  horizonDays: number | null;
  topK: number | null;
  transactionCostBps: number | null;
  factorNames: string[];
  frontierSize: number;
  candidateCount: number | null;
  evaluatedCandidateCount: number | null;
  foldCount: number;
}

export interface FusionRunDetail {
  id: string;
  path: string;
  summary: {
    generatedAt?: string;
    nTrials?: number;
    pbo?: number | null;
    benchmarkMode?: string;
    horizonDays?: number;
    topK?: number;
    transactionCostBps?: number;
    factorNames?: string[];
    foldWindows?: Array<{
      foldIndex: string;
      trainStart: string;
      trainEnd: string;
      testStart: string;
      testEnd: string;
    }>;
    frontier?: string[];
    preferenceWeights?: Record<string, number>;
    candidateCount?: number;
    evaluatedCandidateCount?: number;
  };
  manifest: Record<string, unknown>;
  ranking: Array<{
    id: string;
    preferenceScore: number;
    contributions: Record<string, number>;
  }>;
  candidates: FusionCandidate[];
}

export type FusionNavRow = Record<string, string | number | null>;

/* --------------------------------------------------------------- council --- */

export type CouncilVerdict = "pass" | "warn" | "blocked" | "unknown";

export interface CouncilRole {
  id: string;
  label: string;
  domain: string;
  /** What this agent is allowed to block — it may not block outside it. */
  vetoScope: string;
  veto: boolean;
}

export interface CouncilRoster {
  roles: CouncilRole[];
  thresholds: Record<string, number>;
  protocol: string;
}

export interface CouncilOverride {
  subjectType: string;
  subjectId: string;
  roleId: string;
  verdict: Exclude<CouncilVerdict, "unknown">;
  reason: string;
  author: string;
  recordedAt: string;
}

export interface CouncilFinding {
  roleId: string;
  verdict: CouncilVerdict;
  headline: string;
  detail: string;
  evidence: Record<string, unknown>;
  nextAction: string;
  /** Present when a human overruled this agent; the original verdict is kept. */
  override?: {
    verdict: Exclude<CouncilVerdict, "unknown">;
    reason: string;
    author: string;
    recordedAt: string;
    replacedVerdict: CouncilVerdict;
  };
}

export interface CouncilReview {
  subject: {
    type: string;
    id: string;
    path: string | null;
    candidateId: string | null;
    candidateLabel: string | null;
  };
  roles: CouncilRole[];
  thresholds: Record<string, number>;
  findings: CouncilFinding[];
  decision: {
    state: "PROMOTABLE" | "PROMOTABLE_WITH_WARNINGS" | "INSUFFICIENT_EVIDENCE" | "BLOCKED";
    summary: string;
    blockedRoles: string[];
    unknownRoles: string[];
    warnedRoles: string[];
    overriddenRoles: string[];
  };
  overrides: CouncilOverride[];
}
