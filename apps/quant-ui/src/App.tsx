import { Suspense } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { StateView } from "./components/StateView";
import { InstitutionalShell } from "./vnext/shell/InstitutionalShell";

function WorkstationEntry(): JSX.Element {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  if (params.has("ui")) {
    params.delete("ui");
    const search = params.toString();
    return <Navigate replace to={`${location.pathname}${search ? `?${search}` : ""}`} />;
  }
  return <InstitutionalShell />;
}

export function App(): JSX.Element {
  return (
    <Suspense fallback={<div className="route-loading"><StateView state="loading" /></div>}>
      <Routes><Route path="*" element={<WorkstationEntry />} /></Routes>
    </Suspense>
  );
}
