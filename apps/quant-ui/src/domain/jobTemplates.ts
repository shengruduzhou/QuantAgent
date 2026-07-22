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
      dataset_path: "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus8.parquet",
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
  return value === "backtest" || value === "train" || value === "infer" || value === "factor-discovery";
}

export function templateJson(type: JobType): string {
  return JSON.stringify(jobTemplates[type], null, 2);
}

export function mutableTemplate(type: JobType): JobLaunchPayload {
  return JSON.parse(templateJson(type)) as JobLaunchPayload;
}
