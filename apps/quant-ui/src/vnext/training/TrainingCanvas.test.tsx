import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { TrainingCanvas } from "./TrainingCanvas";

vi.mock("../../components/EChart", () => ({
  EChart: () => <div data-testid="training-chart" />,
}));

afterEach(cleanup);

test("exposes persisted run telemetry and focused metric views", () => {
  render(
    <TrainingCanvas
      job={{
        id: "run-42",
        type: "train",
        status: "running",
        commandId: "train-v8-deep",
        createdAt: "2026-07-23T00:00:00Z",
        startedAt: new Date(Date.now() - 600_000).toISOString(),
        progress: 0.42,
        message: "epoch 21/50",
        outputPaths: [],
      }}
      points={[{
        epoch: 21,
        loss: 0.05123,
        validationLoss: 0.05789,
        rankIc: 0.083,
        learningRate: 0.0001,
        gpuMemory: 18.4,
        samplesPerSecond: 128.5,
        gradientNorm: 0.73,
        metrics: {},
      }]}
    />,
  );

  expect(screen.getByText("RUNNING")).toBeInTheDocument();
  expect(screen.getByText("42%")).toBeInTheDocument();
  expect(screen.getByText("128.5 samples/s")).toBeInTheDocument();
  expect(screen.getByTestId("training-chart")).toBeInTheDocument();

  const rankIc = screen.getByRole("button", { name: "RankIC" });
  fireEvent.click(rankIc);
  expect(rankIc).toHaveAttribute("aria-pressed", "true");
});
