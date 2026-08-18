export const jobTemplates = {
  backtest: {
    commandId: "run-strict-a-share-backtest-v8",
    parameters: {
      target_weights_path: "runtime/reports/v8/deep/v89_rankfix_20260613_1044/short_5d/target_weights.parquet",
      market_panel_path: "runtime/data/v7/silver/market_panel/market_panel.parquet",
      output_dir: "runtime/reports/quant_ui_jobs/web_backtest",
      initial_cash: 1_000_000,
      slippage_bps: 8,
    },
  },
  train: {
    commandId: "train-v8-deep",
    parameters: {
      horizon_class: "short_5d",
      // Full-universe Gold (5,790 securities, all five boards). Replaced the
      // V7 frozen cohort (3,872) once FULL_UNIVERSE_GOLD_READY was granted.
      // There is deliberately no fallback to the old path: a silently
      // narrower universe is worse than a job that fails.
      //
      // THE SAME SWITCH ALSO CUT FEATURE BREADTH FROM 348 COLUMNS TO 15, and
      // that half was never written down. This panel carries ret_{1,5,20,60}d,
      // px_to_ma_{5,20,60}, vol_{20,60}d, turnover_20d, volume_ratio_5_20,
      // amihud_20d, high_low_range_20d, gap_open and intraday_range -- zero
      // Alpha101, zero GTJA-191, zero fundamental, zero event, zero macro.
      // Three of the fifteen are moving averages, which is why the workstation
      // reads as a technical-analysis system: on this artifact it is one.
      //
      // FULL_UNIVERSE_GOLD_READY certifies structural readiness only. Its 18
      // checks (build_u0_full_universe_gold.py + readiness_tiers.py) include
      // none for feature breadth, so "certified" here does not mean "carries
      // the factors the research stack can compute".
      //
      // The richer panel (348 columns, 156 Alpha101 + 58 GTJA-191 + 21
      // fundamental + 22 macro) is
      // runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet
      // -- but it covers 3,638 symbols with zero STAR and zero BSE, on a qfq
      // basis rather than hfq, so it is not a drop-in substitute. Neither
      // artifact dominates; the default stays on universe breadth and the
      // trade-off is now stated rather than inherited.
      dataset_path: "runtime/data/gold/full_universe/dataset.parquet",
      silver_panel_path: "runtime/data/v7/silver/market_panel/market_panel.parquet",
      output_dir: "runtime/reports/quant_ui_jobs/web_train_all_symbols",
      max_epochs: 20,
      batch_size: 512,
      learning_rate: 0.0003,
      early_stopping_patience: 5,
      feature_policy: "judgment",
      require_gpu: true,
    },
  },
  "factor-discovery": {
    commandId: "synthesize-factors-v7",
    parameters: {
      market_panel_path: "runtime/data/v7/silver/market_panel/market_panel.parquet",
      labels_path: null,
      output_dir: "runtime/reports/v7/factor_synthesis_ui",
      rd_agent: true,
      label_column: "forward_return_5d",
      rounds: 4,
      factors_per_round: 3,
      top_k: 20,
      validation_fraction: 0.25,
      min_validation_rank_ic: 0.0,
      max_reference_correlation: 0.7,
      max_sota_correlation: 0.99,
      use_llm: false,
      allow_network: false,
      exclude_st: true,
    },
  },
  "factor-evaluation": {
    commandId: "evaluate-factor-library-v7",
    parameters: {
      market_panel_path: "runtime/data/v7/silver/market_panel/market_panel.parquet",
      labels_path: "runtime/data/v7/gold/labels/labels.parquet",
      output_dir: "runtime/reports/factor_evaluation/all_reviewed",
      factor_library: "all_reviewed",
      calibration_days: 252,
      holdout_days: 60,
      min_abs_rank_ic: 0.01,
      min_abs_rank_icir: 0.25,
      min_finite_ratio: 0.8,
      min_abs_monotonicity: 0.5,
      max_pairwise_correlation: 0.85,
    },
  },
  infer: {
    commandId: "predict-alpha-v7",
    parameters: {
      model_dir: "runtime/reports/v8/deep/v89_rankfix_20260613_1044/short_5d/ft",
      feature_dataset: "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus8.parquet",
      output: "runtime/predictions/quant_ui_web_predictions.parquet",
      primary_horizon: 5,
    },
  },
} as const;

export type JobType = keyof typeof jobTemplates;

export interface JobLaunchPayload {
  commandId: string;
  parameters: Record<string, string | number | boolean | string[] | null>;
}

export function isJobType(value: string | null): value is JobType {
  return value === "backtest" || value === "train" || value === "infer" || value === "factor-discovery" || value === "factor-evaluation";
}

export function templateJson(type: JobType): string {
  return JSON.stringify(jobTemplates[type], null, 2);
}

export function mutableTemplate(type: JobType): JobLaunchPayload {
  return JSON.parse(templateJson(type)) as JobLaunchPayload;
}
