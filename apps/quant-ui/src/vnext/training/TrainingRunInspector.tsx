import { CheckCircle, Copy, Play, ShieldCheck, Stop, WarningCircle } from "@phosphor-icons/react";
import type { JobSummary, JobValidation, ModelObservability } from "../../api/types";
import { formatBytes, formatDate } from "../../utils/format";

interface TrainingRunInspectorProps {
  model?: ModelObservability;
  job?: JobSummary;
  validation?: JobValidation;
  busy: boolean;
  armed: boolean;
  setArmed: (armed: boolean) => void;
  validate: () => void;
  start: () => void;
  cancel: () => void;
  clone: () => void;
}

export function TrainingRunInspector(props: TrainingRunInspectorProps): JSX.Element {
  return (
    <aside className="vnext-run-inspector">
      <header><span>RUN INSPECTOR</span><h2>{props.job?.id ?? props.model?.version ?? "No run selected"}</h2></header>
      <div className="vnext-run-actions">
        <button type="button" onClick={props.validate} disabled={props.busy}><ShieldCheck size={15} /> Validate</button>
        <button type="button" className="primary" onClick={props.start} disabled={props.busy || !props.validation?.valid || !props.armed}><Play size={15} weight="fill" /> Start</button>
        <button type="button" onClick={props.cancel} disabled={!props.job || !["queued", "running", "cancelling"].includes(props.job.status)}><Stop size={15} /> Cancel</button>
        <button type="button" onClick={props.clone}><Copy size={15} /> Clone</button>
      </div>
      <label className="vnext-launch-arm"><input type="checkbox" checked={props.armed} onChange={(event) => props.setArmed(event.target.checked)} /><span><strong>Arm research training launch</strong><small>确认这是高成本 GPU/Paper 研究任务；不会启用实盘。</small></span></label>

      {props.validation ? <div className="vnext-validation-pass"><CheckCircle size={17} /><span><strong>Backend validation passed</strong><small>{props.validation.outputPaths.join(", ") || "No output path"}</small></span></div> : <div className="vnext-validation-wait"><WarningCircle size={17} /><span><strong>Validation required</strong><small>Start remains disabled until backend path and allowlist checks pass.</small></span></div>}

      <section><h3>Run metadata</h3><dl><div><dt>Status</dt><dd>{props.job?.status ?? "UNAVAILABLE"}</dd></div><div><dt>Command</dt><dd>{props.job?.commandId ?? "—"}</dd></div><div><dt>Created</dt><dd>{formatDate(props.job?.createdAt)}</dd></div><div><dt>Started</dt><dd>{formatDate(props.job?.startedAt)}</dd></div><div><dt>Finished</dt><dd>{formatDate(props.job?.finishedAt)}</dd></div><div><dt>Progress</dt><dd>{props.job?.progress === null || props.job?.progress === undefined ? "—" : `${Math.round(props.job.progress * 100)}%`}</dd></div></dl></section>
      <section><h3>Model lineage</h3><dl><div><dt>Model</dt><dd>{props.model?.version ?? "—"}</dd></div><div><dt>Family</dt><dd>{props.model?.modelFamily ?? "—"}</dd></div><div><dt>Data revision</dt><dd>{props.model?.config.data_revision ? String(props.model.config.data_revision) : "UNAVAILABLE"}</dd></div><div><dt>Code commit</dt><dd>{props.model?.config.commit ? String(props.model.config.commit) : "UNAVAILABLE"}</dd></div><div><dt>Checkpoints</dt><dd>{props.model?.checkpoint.count ?? 0} · {formatBytes(props.model?.checkpoint.sizeBytes)}</dd></div><div><dt>Acceptance</dt><dd>{props.model?.verdict ?? (props.model?.productionReady ? "pass" : "not declared")}</dd></div></dl></section>
      {props.job?.error ? <div className="vnext-run-error"><WarningCircle size={16} />{props.job.error}</div> : null}
    </aside>
  );
}
