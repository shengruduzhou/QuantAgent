import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MonitorTable, type MonitorColumn } from "./MonitorTable";

interface Row {
  id: string;
  name: string;
  value: number;
}

const rows: Row[] = [
  { id: "b", name: "Beta", value: 2 },
  { id: "a", name: "Alpha", value: 2 },
  { id: "c", name: "Gamma", value: 1 },
];

const columns: MonitorColumn<Row>[] = [
  { id: "name", header: "Name", value: (row) => row.name, width: 120 },
  { id: "value", header: "Value", value: (row) => row.value, align: "right", width: 80 },
];

function rowNames(): string[] {
  const table = screen.getByRole("table", { name: "测试监控" });
  return within(table).getAllByRole("row").slice(1).map((row) => within(row).getAllByRole("cell")[0].textContent ?? "");
}

describe("MonitorTable", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  test("sorts stably and restores source order on the third click", () => {
    render(
      <MonitorTable
        monitorId="test-sort"
        ariaLabel="测试监控"
        rows={rows}
        columns={columns}
        rowKey={(row) => row.id}
      />,
    );

    const valueHeader = screen.getByRole("button", { name: "Value" });
    fireEvent.click(valueHeader);
    expect(rowNames()).toEqual(["Gamma", "Beta", "Alpha"]);

    fireEvent.click(valueHeader);
    expect(rowNames()).toEqual(["Beta", "Alpha", "Gamma"]);

    fireEvent.click(valueHeader);
    expect(rowNames()).toEqual(["Beta", "Alpha", "Gamma"]);
  });

  test("supports keyboard row selection", () => {
    const onSelect = vi.fn();
    render(
      <MonitorTable
        monitorId="test-keyboard"
        ariaLabel="测试监控"
        rows={rows}
        columns={columns}
        rowKey={(row) => row.id}
        onSelect={onSelect}
      />,
    );

    const tableRows = screen.getAllByRole("row").slice(1);
    fireEvent.keyDown(tableRows[0], { key: "ArrowDown" });
    expect(onSelect).toHaveBeenLastCalledWith(rows[1]);
  });

  test("exports the current monitor rows as CSV", () => {
    const createObjectURL = vi.fn(() => "blob:monitor");
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { value: createObjectURL, configurable: true });
    Object.defineProperty(URL, "revokeObjectURL", { value: revokeObjectURL, configurable: true });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);

    render(
      <MonitorTable
        monitorId="test-export"
        ariaLabel="测试监控"
        rows={rows}
        columns={columns}
        rowKey={(row) => row.id}
        exportFilename="test.csv"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /CSV/ }));
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:monitor");
    click.mockRestore();
  });
});
