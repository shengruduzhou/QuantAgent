import { useMemo, type ReactNode } from "react";
import { Info } from "@phosphor-icons/react";
import type { StrategyDefaults, StrategyInputOption } from "../../api/types";

export interface FusionSearchDraft {
  factorPanelPath: string;
  forwardReturnsPath: string;
  forwardColumn: string;
  factorNames: string;
  outputDir: string;
  horizonDays: number;
  topK: number;
  nFolds: number;
  embargoDays: number;
  minTrainDays: number;
  minTestDays: number;
  transactionCostBps: number;
  includeGenetic: boolean;
  randomControls: number;
  singleFactorBaselines: number;
  seed: number;
  benchmarkSymbol: string;
  preference: {
    excessReturn: number;
    annualReturn: number;
    drawdownControl: number;
    robustness: number;
  };
}

export const DEFAULT_SEARCH_DRAFT: FusionSearchDraft = {
  // This default panel carries 15 features, of which three are moving
  // averages: no Alpha101, no GTJA-191, no fundamental, no event, no macro.
  // A fusion search launched on it is combining fifteen price-and-volume
  // statistics, which bounds what any resulting "best blend" can mean. See
  // the note in src/domain/jobTemplates.ts for why the certified panel is
  // narrow and what the 348-column alternative costs.
  factorPanelPath: "runtime/data/gold/full_universe/dataset.parquet",
  forwardReturnsPath: "runtime/data/gold/full_universe/labels.parquet",
  forwardColumn: "forward_return_5d",
  factorNames: "",
  outputDir: "runtime/reports/fusion/search_01",
  horizonDays: 5,
  topK: 30,
  nFolds: 4,
  embargoDays: 5,
  minTrainDays: 120,
  minTestDays: 40,
  transactionCostBps: 8,
  includeGenetic: true,
  randomControls: 8,
  singleFactorBaselines: 6,
  seed: 17,
  benchmarkSymbol: "000300.SH",
  preference: {
    excessReturn: 0.4,
    annualReturn: 0.2,
    drawdownControl: 0.25,
    robustness: 0.15,
  },
};

/** The search space size, shown before launch so the cost is never a surprise. */
export function trialCount(draft: FusionSearchDraft): number {
  const fitted = draft.includeGenetic ? 4 : 3;
  return 1 + fitted + draft.randomControls + draft.singleFactorBaselines;
}

function Field({
  label,
  hint,
  wide = false,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  children: ReactNode;
}): JSX.Element {
  return (
    <label className={`atlas-field ${wide ? "wide" : ""}`.trim()}>
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

function PathField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: StrategyInputOption[];
  onChange: (value: string) => void;
}): JSX.Element {
  const listId = `fusion-path-${label.replace(/\W+/g, "-")}`;
  return (
    <Field
      label={label}
      wide
      hint={
        options.length
          ? `${options.filter((item) => item.exists).length} 个可用 / ${options.length} 个候选`
          : "无 canonical 候选，需手动输入项目内路径"
      }
    >
      <input className="mono" list={listId} value={value} onChange={(event) => onChange(event.target.value)} />
      <datalist id={listId}>
        {options.map((option) => (
          <option key={option.path} value={option.path}>
            {option.exists ? "可用" : "尚未构建"}
          </option>
        ))}
      </datalist>
    </Field>
  );
}

export function FusionSearchForm({
  draft,
  onChange,
  defaults,
  disabled,
}: {
  draft: FusionSearchDraft;
  onChange: (draft: FusionSearchDraft) => void;
  defaults?: StrategyDefaults;
  disabled: boolean;
}): JSX.Element {
  const update = <K extends keyof FusionSearchDraft>(key: K, value: FusionSearchDraft[K]): void => {
    onChange({ ...draft, [key]: value });
  };

  const updatePreference = (
    key: keyof FusionSearchDraft["preference"],
    value: number,
  ): void => {
    onChange({ ...draft, preference: { ...draft.preference, [key]: value } });
  };

  const options = (field: string): StrategyInputOption[] => defaults?.options?.[field] ?? [];
  const trials = useMemo(() => trialCount(draft), [draft]);
  const preferenceTotal =
    draft.preference.excessReturn
    + draft.preference.annualReturn
    + draft.preference.drawdownControl
    + draft.preference.robustness;

  return (
    <fieldset className="foundry-form" disabled={disabled}>
      <legend className="sr-only">因子融合搜索配置</legend>

      <PathField
        label="因子面板"
        value={draft.factorPanelPath}
        options={options("trainingDatasetPath")}
        onChange={(value) => update("factorPanelPath", value)}
      />
      <PathField
        label="前瞻收益面板"
        value={draft.forwardReturnsPath}
        options={options("labelsPath")}
        onChange={(value) => update("forwardReturnsPath", value)}
      />
      <Field label="前瞻收益列" hint="留空时回退到面板中第一个 forward_return* 列">
        <input
          className="mono"
          value={draft.forwardColumn}
          onChange={(event) => update("forwardColumn", event.target.value)}
        />
      </Field>
      <Field
        label="参与融合的因子"
        wide
        hint="逗号分隔的列名。融合前每个因子按日横截面排名归一，因此权重表示信念占比而非量纲。"
      >
        <textarea
          className="mono"
          rows={3}
          value={draft.factorNames}
          placeholder="alpha001,alpha012,gtja025,…"
          onChange={(event) => update("factorNames", event.target.value)}
        />
      </Field>
      <Field label="输出目录" wide>
        <input
          className="mono"
          value={draft.outputDir}
          onChange={(event) => update("outputDir", event.target.value)}
        />
      </Field>

      <div className="foundry-form-grid">
        <Field label="持有周期 (交易日)">
          <input
            type="number"
            min={1}
            max={252}
            value={draft.horizonDays}
            onChange={(event) => update("horizonDays", Number(event.target.value))}
          />
        </Field>
        <Field label="Top K">
          <input
            type="number"
            min={1}
            max={500}
            value={draft.topK}
            onChange={(event) => update("topK", Number(event.target.value))}
          />
        </Field>
        <Field label="折数">
          <input
            type="number"
            min={1}
            max={12}
            value={draft.nFolds}
            onChange={(event) => update("nFolds", Number(event.target.value))}
          />
        </Field>
        <Field label="Embargo (日)" hint="不小于持有周期，否则标签重叠">
          <input
            type="number"
            min={0}
            max={120}
            value={draft.embargoDays}
            onChange={(event) => update("embargoDays", Number(event.target.value))}
          />
        </Field>
        <Field label="最少训练日">
          <input
            type="number"
            min={20}
            value={draft.minTrainDays}
            onChange={(event) => update("minTrainDays", Number(event.target.value))}
          />
        </Field>
        <Field label="最少测试日">
          <input
            type="number"
            min={5}
            value={draft.minTestDays}
            onChange={(event) => update("minTestDays", Number(event.target.value))}
          />
        </Field>
        <Field label="交易成本 (bps)">
          <input
            type="number"
            min={0}
            max={500}
            step={0.5}
            value={draft.transactionCostBps}
            onChange={(event) => update("transactionCostBps", Number(event.target.value))}
          />
        </Field>
        <Field label="基准指数">
          <input
            value={draft.benchmarkSymbol}
            onChange={(event) => update("benchmarkSymbol", event.target.value)}
          />
        </Field>
        <Field label="随机对照数" hint="计入试验次数">
          <input
            type="number"
            min={0}
            max={64}
            value={draft.randomControls}
            onChange={(event) => update("randomControls", Number(event.target.value))}
          />
        </Field>
        <Field label="单因子基线数" hint="计入试验次数">
          <input
            type="number"
            min={0}
            max={64}
            value={draft.singleFactorBaselines}
            onChange={(event) => update("singleFactorBaselines", Number(event.target.value))}
          />
        </Field>
        <Field label="随机种子">
          <input
            type="number"
            value={draft.seed}
            onChange={(event) => update("seed", Number(event.target.value))}
          />
        </Field>
        <label className="atlas-field foundry-check">
          <span>遗传搜索</span>
          <input
            type="checkbox"
            checked={draft.includeGenetic}
            onChange={(event) => update("includeGenetic", event.target.checked)}
          />
          <small>多目标 GA，仅在训练段拟合</small>
        </label>
      </div>

      <div className="foundry-trials">
        <Info size={13} />
        <span>
          本次搜索将评估 <strong>{trials}</strong> 个方案。这个数字就是 <code>n_trials</code>，
          会直接用于收缩 Deflated Sharpe——增加对照只会让结论更保守，不会让它更好看。
        </span>
      </div>

      <div className="foundry-preference">
        <span className="atlas-eyebrow">目标偏好（仅影响前沿内排序）</span>
        <PreferenceSlider
          label="最大超额"
          value={draft.preference.excessReturn}
          onChange={(value) => updatePreference("excessReturn", value)}
        />
        <PreferenceSlider
          label="最大年化"
          value={draft.preference.annualReturn}
          onChange={(value) => updatePreference("annualReturn", value)}
        />
        <PreferenceSlider
          label="最小回撤"
          value={draft.preference.drawdownControl}
          onChange={(value) => updatePreference("drawdownControl", value)}
        />
        <PreferenceSlider
          label="样本外稳健"
          value={draft.preference.robustness}
          onChange={(value) => updatePreference("robustness", value)}
        />
        <small>
          权重会在服务端归一（当前合计 {preferenceTotal.toFixed(2)}）。偏好只对已在 Pareto
          前沿的候选排序，不改变哪些候选被生成，也不改变哪些进入前沿。
        </small>
      </div>
    </fieldset>
  );
}

function PreferenceSlider({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}): JSX.Element {
  return (
    <label className="foundry-slider">
      <span>{label}</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <strong className="mono">{Math.round(value * 100)}%</strong>
    </label>
  );
}
