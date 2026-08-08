import { GovernancePage } from "./GovernancePage";
import { ProductionReadinessPanel } from "./ProductionReadinessPanel";


export function GovernanceOperatorPage(): JSX.Element {
  return (
    <>
      <ProductionReadinessPanel />
      <GovernancePage />
    </>
  );
}
