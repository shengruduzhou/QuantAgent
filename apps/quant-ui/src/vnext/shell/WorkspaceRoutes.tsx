import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { StateView } from "../../components/StateView";

const VNextDashboard = lazy(() => import("../dashboard/VNextDashboard").then((module) => ({ default: module.VNextDashboard })));
const TrainingLab = lazy(() => import("../training/TrainingLabPage").then((module) => ({ default: module.TrainingLabPage })));
const StockReplay = lazy(() => import("../../pages/StockReplayPage").then((module) => ({ default: module.StockReplayPage })));
const Backtests = lazy(() => import("../../pages/BacktestLabPage").then((module) => ({ default: module.BacktestLabPage })));
const TPlusOne = lazy(() => import("../../pages/TPlusOnePage").then((module) => ({ default: module.TPlusOnePage })));
const Factors = lazy(() => import("../../pages/FactorCenterPage").then((module) => ({ default: module.FactorCenterPage })));
const Selection = lazy(() => import("../../pages/SelectionLogicPage").then((module) => ({ default: module.SelectionLogicPage })));
const Models = lazy(() => import("../../pages/ModelLabPage").then((module) => ({ default: module.ModelLabPage })));
const Risk = lazy(() => import("../../pages/RiskCenterPage").then((module) => ({ default: module.RiskCenterPage })));
const Runtime = lazy(() => import("../../pages/RuntimeExplorerPage").then((module) => ({ default: module.RuntimeExplorerPage })));
const Parity = lazy(() => import("../../pages/VnpyParityPage").then((module) => ({ default: module.VnpyParityPage })));
const Governance = lazy(() => import("../governance/GovernancePage").then((module) => ({ default: module.GovernancePage })));
const Reports = lazy(() => import("../../pages/ReportsPage").then((module) => ({ default: module.ReportsPage })));
const Settings = lazy(() => import("../../pages/SettingsPage").then((module) => ({ default: module.SettingsPage })));
const Help = lazy(() => import("../../pages/HelpCenterPage").then((module) => ({ default: module.HelpCenterPage })));

export function WorkspaceRoutes({ location }: { location: string }): JSX.Element {
  return (
    <Suspense fallback={<StateView state="loading" detail="正在恢复工作区上下文。" />}>
      <Routes location={location}>
        <Route path="/" element={<VNextDashboard />} />
        <Route path="/training" element={<TrainingLab />} />
        <Route path="/stock-replay" element={<StockReplay />} />
        <Route path="/backtests" element={<Backtests />} />
        <Route path="/t-plus-one" element={<TPlusOne />} />
        <Route path="/factors" element={<Factors />} />
        <Route path="/selection" element={<Selection />} />
        <Route path="/models" element={<Models />} />
        <Route path="/risk" element={<Risk />} />
        <Route path="/runtime" element={<Runtime />} />
        <Route path="/parity" element={<Parity />} />
        <Route path="/governance" element={<Governance />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/help" element={<Help />} />
        <Route path="*" element={<StateView state="unavailable" title="工作站模块不存在" detail="请使用 Global Command Bar 打开已注册模块。" />} />
      </Routes>
    </Suspense>
  );
}
