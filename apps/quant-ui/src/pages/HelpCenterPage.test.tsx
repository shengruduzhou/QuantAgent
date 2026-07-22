import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test } from "vitest";
import { HelpCenterPage } from "./HelpCenterPage";

afterEach(() => cleanup());

test("keeps all help navigation inside QuantAgent", () => {
  render(
    <MemoryRouter>
      <HelpCenterPage />
    </MemoryRouter>,
  );

  expect(screen.getByRole("heading", { name: "帮助中心" })).toBeInTheDocument();
  expect(screen.getByText(/这里是 QuantAgent 的操作说明/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /全宇宙训练/ })).toHaveAttribute("href", "/settings?job=train&universe=all");
  expect(screen.getByRole("link", { name: /数据与删除/ })).toHaveAttribute("href", "/runtime?view=cleanup");
  expect(screen.queryByRole("link", { name: /VeighNa/ })).not.toBeInTheDocument();
  expect(document.querySelector('a[href^="http"]')).toBeNull();
});
