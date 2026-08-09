import { TPlusOnePage } from "../../pages/TPlusOnePage";
import { PaperExecutionEvidencePanel } from "./PaperExecutionEvidencePanel";

/**
 * Keeps continuous-account execution truth visible even when the legacy T+1
 * pair dataset is empty.  The evidence panel and research page intentionally
 * remain separate surfaces: paper execution state must not be inferred from
 * pair PnL, and pair PnL must not be inferred from an execution journal.
 */
export function TPlusOneExecutionWorkspace(): JSX.Element {
  return (
    <>
      <div className="institutional-workbench t1-execution-evidence-shell">
        <PaperExecutionEvidencePanel />
      </div>
      <TPlusOnePage />
    </>
  );
}
