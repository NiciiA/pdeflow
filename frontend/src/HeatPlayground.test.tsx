import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, it, expect, vi } from "vitest";
import HeatPlayground from "./HeatPlayground";
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
it("renders the empty initial state before a reference or evaluated time exists", () => {
  localStorage.removeItem("pdeflow.heat.preferences");
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal(
    "fetch",
    vi.fn(() => new Promise(() => {})),
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  render(<HeatPlayground />);
  expect(screen.getByRole("button", { name: /^Play$/ })).toBeDisabled();
  expect(
    screen.getByText("Time interval: 0 to 1", { exact: true }),
  ).toBeInTheDocument();
});
