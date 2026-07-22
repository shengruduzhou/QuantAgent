import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { StateView } from "./components/StateView";
import { resolveUiVersion } from "./vnext/featureFlags";
import { InstitutionalShell } from "./vnext/shell/InstitutionalShell";

const DashboardPage = lazy(() => import("./pages/DashboardPage").then((module) => ({ default: module.DashboardPage })));
const StockReplayPage = lazy(() => import("./pages/StockReplayPage").then((module) => ({ default: module.StockReplayPage })));
const BacktestLabPage = lazy(() => import("./pages/BacktestLabPage").then((module) => ({ default: module.BacktestLabPage })));
const TPlusOnePage = lazy(() => import("./pages/TPlusOnePage").then((module) => ({ default: module.TPlusOnePage })));
const FactorCenterPage = lazy(() => import("./pages/FactorCenterPage").then((module) => ({ default: module.FactorCenterPage })));
const SelectionLogicPage = lazy(() => import("./pages/SelectionLogicPage").then((module) => ({ default: module.SelectionLogicPage })));
const ModelLabPage = lazy(() => import("./pages/ModelLabPage").then((module) => ({ default: module.ModelLabPage })));
const RiskCenterPage = lazy(() => import("./pages/RiskCenterPage").then((module) => ({ default: module.RiskCenterPage })));
const RuntimeExplorerPage = lazy(() => import("./pages/RuntimeExplorerPage").then((module) => ({ default: module.RuntimeExplorerPage })));
const VnpyParityPage = lazy(() => import("./pages/VnpyParityPage").then((module) => ({ default: module.VnpyParityPage })));
const ReportsPage = lazy(() => import("./pages/ReportsPage").then((module) => ({ default: module.ReportsPage })));
const SettingsPage = lazy(() => import("./pages/SettingsPage").then((module) => ({ default: module.SettingsPage })));
const HelpCenterPage = lazy(() => import("./pages/HelpCenterPage").then((module) => ({ default: module.HelpCenterPage })));

function LegacyApp(): JSX.Element {
  return (
    <Suspense fallback={<div className="route-loading"><StateView state="loading" /></div>}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<DashboardPage />} />
          <Route path="stock-replay" element={<StockReplayPage />} />
          <Route path="backtests" element={<BacktestLabPage />} />
          <Route path="t-plus-one" element={<TPlusOnePage />} />
          <Route path="factors" element={<FactorCenterPage />} />
          <Route path="selection" element={<SelectionLogicPage />} />
          <Route path="models" element={<ModelLabPage />} />
          <Route path="risk" element={<RiskCenterPage />} />
          <Route path="runtime" element={<RuntimeExplorerPage />} />
          <Route path="parity" element={<VnpyParityPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="help" element={<HelpCenterPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}

export function App(): JSX.Element {
  if (resolveUiVersion() === "legacy") return <LegacyApp />;
  return (
    <Suspense fallback={<div className="route-loading"><StateView state="loading" /></div>}>
      <Routes><Route path="*" element={<InstitutionalShell />} /></Routes>
    </Suspense>
  );
}
