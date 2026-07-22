import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test } from "vitest";
import { HelpCenterPage } from "./HelpCenterPage";

afterEach(() => cleanup());

test("renders QuantAgent help before optional external references", () => {
  render(
    <MemoryRouter>
      <HelpCenterPage />
    </MemoryRouter>,
  );

  expect(screen.getByRole("heading", { name: "帮助中心" })).toBeInTheDocument();
  expect(screen.getByText(/这里是 QuantAgent 的操作说明/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /全宇宙训练/ })).toHaveAttribute("href", "/settings?job=train&universe=all");
  expect(screen.getByRole("link", { name: /数据与删除/ })).toHaveAttribute("href", "/runtime?view=cleanup");
  expect(screen.getByRole("link", { name: /VeighNa 社区版文档/ })).toHaveAttribute("target", "_blank");
});
