import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { ProcessControls, ProcessPanel } from "./ProcessPanel";

afterEach(cleanup);

describe("physical process status", () => {
  it("distinguishes budget exhaustion and unavailable checks from success", () => {
    render(
      <ProcessPanel
        metrics={{
          verification: {
            label: "final_field",
            mandatory_failures: ["pde.u.rmse"],
            entries: {
              "pde.u.rmse": { value: 2, units: "u/time", status: "violated" },
              boundary: {
                value: null,
                units: "unavailable",
                status: "not_applicable",
              },
            },
          },
          process: {
            stop_reason: "FINISH_BUDGET_EXHAUSTED",
            criteria_met: false,
            final_grid: { shape: [8, 8] },
            consumed: { network_evaluations: 4 },
            resampling: "FK unavailable",
            actions: [
              {
                index: 0,
                action: "REFINE_UNAVAILABLE",
                reason: "No fine checkpoint",
                budget: { attempts: 1 },
              },
            ],
          },
        }}
      />,
    );
    expect(
      screen.getByLabelText("Process completion status"),
    ).toHaveTextContent("FINISH_BUDGET_EXHAUSTED");
    expect(screen.getByText("not_applicable")).toBeVisible();
    expect(screen.getByText("REFINE_UNAVAILABLE")).toBeVisible();
  });
  it("only enables explicit configuration and exports no fake resampling", () => {
    const change = vi.fn();
    render(<ProcessControls value={null} onChange={change} />);
    fireEvent.click(screen.getByLabelText("Enable feedback controller"));
    expect(change).toHaveBeenCalledWith({});
    expect(
      screen.getByText(/FK resampling is unavailable/),
    ).toBeInTheDocument();
  });
});

it("requires an explicit native-grid budget action and preserves other limits", () => {
  const change = vi.fn();
  render(
    <ProcessControls
      value={{ max_network_evaluations: 64 }}
      nativePoints={262144}
      onChange={change}
    />,
  );
  expect(change).not.toHaveBeenCalled();
  fireEvent.click(screen.getByText("Feedback controller and budgets"));
  fireEvent.click(
    screen.getByText("Use requested native grid budget (262144)"),
  );
  expect(change).toHaveBeenCalledWith({
    max_network_evaluations: 64,
    max_grid_points: 262144,
  });
});
