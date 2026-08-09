import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { StateView } from "../../components/StateView";

const VNextDashboard = lazy(() => import("../dashboard/VNextDashboard").then((module) => ({ default: module.VNextDashboard })));
const TrainingLab = lazy(() => import("../training/TrainingLabPage").then((module) => ({ default: module.TrainingLabPage })));
const StrategyStudio = lazy(() => import("../strategy/StrategyStudioPage").then((module) => ({ default: module.StrategyStudioPage })));
const ResearchRuns = lazy(() => import("../runs/ResearchRunsPage").then((module) => ({ default: module.ResearchRunsPage })));
const DecisionCouncil = lazy(() => import("../council/DecisionCouncilPage").then((module) => ({ default: module.DecisionCouncilPage })));
const AlphaFoundry = lazy(() => import("../fusion/AlphaFoundryPage").then((module) => ({ default: module.AlphaFoundryPage })));
const StockReplay = lazy(() => import("../market/MarketWorkbenchPage").then((module) => ({ default: module.MarketWorkbenchPage })));
const MarketIntelligence = lazy(() => import("../market/MarketIntelligencePage").then((module) => ({ default: module.MarketIntelligencePage })));
const FuyaoBestPractices = lazy(() => import("../market/FuyaoBestPracticesPage").then((module) => ({ default: module.FuyaoBestPracticesPage })));
const MarketPlaybooks = lazy(() => import("../market/MarketPlaybooksPage").then((module) => ({ default: module.MarketPlaybooksPage })));
const FundResearch = lazy(() => import("../market/FundResearchPage").then((module) => ({ default: module.FundResearchPage })));
const Backtests = lazy(() => import("../../pages/BacktestLabPage").then((module) => ({ default: module.BacktestLabPage })));
const TPlusOne = lazy(() => import("../tplusone/TPlusOneExecutionWorkspace").then((module) => ({ default: module.TPlusOneExecutionWorkspace })));
const Factors = lazy(() => import("../../pages/FactorCenterPage").then((module) => ({ default: module.FactorCenterPage })));
const Selection = lazy(() => import("../../pages/SelectionLogicPage").then((module) => ({ default: module.SelectionLogicPage })));
const Models = lazy(() => import("../../pages/ModelLabPage").then((module) => ({ default: module.ModelLabPage })));
const Risk = lazy(() => import("../../pages/RiskCenterPage").then((module) => ({ default: module.RiskCenterPage })));
const Runtime = lazy(() => import("../../pages/RuntimeExplorerPage").then((module) => ({ default: module.RuntimeExplorerPage })));
const Parity = lazy(() => import("../../pages/VnpyParityPage").then((module) => ({ default: module.VnpyParityPage })));
const Governance = lazy(() => import("../governance/GovernanceOperatorPage").then((module) => ({ default: module.GovernanceOperatorPage })));
const Reports = lazy(() => import("../../pages/ReportsPage").then((module) => ({ default: module.ReportsPage })));
const Settings = lazy(() => import("../../pages/SettingsPage").then((module) => ({ default: module.SettingsPage })));
const Help = lazy(() => import("../../pages/HelpCenterPage").then((module) => ({ default: module.HelpCenterPage })));

export function WorkspaceRoutes({ location }: { location: string }): JSX.Element {
  return (
    <Suspense fallback={<StateView state="loading" detail="正在恢复工作区上下文。" />}>
      <Routes location={location}>
        <Route path="/" element={<VNextDashboard />} />
        <Route path="/training" element={<TrainingLab />} />
        <Route path="/strategy" element={<StrategyStudio />} />
        <Route path="/runs" element={<ResearchRuns />} />
        <Route path="/fusion" element={<AlphaFoundry />} />
        <Route path="/council" element={<DecisionCouncil />} />
        <Route path="/market-intelligence" element={<MarketIntelligence />} />
        <Route path="/fuyao-research" element={<FuyaoBestPractices />} />
        <Route path="/market-playbooks" element={<MarketPlaybooks />} />
        <Route path="/fund-research" element={<FundResearch />} />
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
