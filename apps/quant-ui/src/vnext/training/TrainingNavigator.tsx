import { MagnifyingGlass } from "@phosphor-icons/react";
import type { JobSummary, ModelSummary } from "../../api/types";
import { formatDate } from "../../utils/format";

interface TrainingNavigatorProps {
  models: ModelSummary[];
  jobs: JobSummary[];
  selectedModelId: string;
  selectedJobId: string;
  compareIds: string[];
  query: string;
  setQuery: (query: string) => void;
  selectModel: (id: string) => void;
  selectJob: (id: string) => void;
  toggleCompare: (id: string) => void;
}

export function TrainingNavigator(props: TrainingNavigatorProps): JSX.Element {
  const needle = props.query.trim().toLowerCase();
  const models = props.models.filter((model) => !needle || `${model.id} ${model.version} ${model.modelFamily} ${model.verdict}`.toLowerCase().includes(needle));
  const jobs = props.jobs.filter((job) => job.type === "train" && (!needle || `${job.id} ${job.commandId} ${job.status}`.toLowerCase().includes(needle)));
  return (
    <aside className="vnext-training-navigator">
      <header><div><span>EXPERIMENT NAVIGATOR</span><h2>Experiments & Runs</h2></div></header>
      <label className="vnext-training-search"><MagnifyingGlass size={15} /><input value={props.query} onChange={(event) => props.setQuery(event.target.value)} placeholder="Search experiment, run, tag, horizon" /></label>
      <section>
        <h3>MODEL EXPERIMENTS <em>{models.length}</em></h3>
        <div className="vnext-experiment-list">
          {models.slice(0, 80).map((model) => (
            <button type="button" key={model.id} className={props.selectedModelId === model.id ? "active" : ""} onClick={() => props.selectModel(model.id)}>
              <input type="checkbox" checked={props.compareIds.includes(model.id)} readOnly onClick={(event) => { event.stopPropagation(); props.toggleCompare(model.id); }} aria-label={`比较模型 ${model.version ?? model.id}`} />
              <span><strong>{model.version ?? model.id}</strong><small>{model.modelFamily ?? model.modelType ?? "model"} · {model.horizons.join("/") || "no horizon"}</small></span>
              <em className={`state-${model.productionReady ? "ready" : model.status}`}>{model.productionReady ? "accepted" : model.verdict ?? model.status}</em>
            </button>
          ))}
        </div>
      </section>
      <section>
        <h3>TRAINING RUNS <em>{jobs.length}</em></h3>
        <div className="vnext-run-list">
          {jobs.slice(0, 80).map((job) => (
            <button type="button" key={job.id} className={props.selectedJobId === job.id ? "active" : ""} onClick={() => props.selectJob(job.id)}>
              <i className={`state-${job.status}`} />
              <span><strong>{job.commandId}</strong><small>{job.id} · {formatDate(job.createdAt)}</small></span>
              <em>{job.progress === null || job.progress === undefined ? job.status : `${Math.round(job.progress * 100)}%`}</em>
            </button>
          ))}
          {!jobs.length ? <p>没有 persisted training jobs。</p> : null}
        </div>
      </section>
    </aside>
  );
}
