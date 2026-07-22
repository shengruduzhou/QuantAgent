import React from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import "./styles.css";
import "./theme.css";
import "./terminal.css";
import "./monitor-table.css";
import "./kline-workstation.css";
import "./parity.css";
import "./ux-depth.css";
import "./ux-depth-fixes.css";
import "./control-depth.css";
import "./data-manager.css";
import "./workstation-v4.css";
import "./vnext/styles/tokens.css";
import "./vnext/styles/shell.css";
import "./vnext/styles/dashboard.css";
import "./vnext/styles/training.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const root = document.getElementById("root");

if (!root) {
  throw new Error("Quant UI root element is missing");
}

createRoot(root).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
