import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { FKPanel, FKControls } from "./FKPanel";
describe("FK real report display", () => {
  it("labels incomplete paths and actual nonuniform weights", () => {
    render(
      <FKPanel
        metrics={{
          cost: {
            fk: {
              trace: [
                {
                  generation_position: 0.5,
                  before: { ess: [2], max_weight: [0.9] },
                  after: { ess: [4] },
                  unique_ancestors: [2],
                  resampled_problems: [true],
                  trigger: "scheduled_ESS",
                  energy_components_weighted_mean: { observation: [3] },
                },
              ],
              terminal_weights: [[0.9, 0.1]],
              terminal_target_complete: false,
              model_evaluations: 4,
              model_particle_evaluations: 16,
              energy_particle_evaluations: 16,
              config: { terminal_resample: false },
            },
          },
        }}
      />,
    );
    expect(screen.getByLabelText("FK terminal status")).toHaveTextContent(
      "Incomplete path",
    );
    expect(
      screen.getByRole("table", { name: "FK population trace" }),
    ).toHaveTextContent("scheduled_ESS");
    expect(
      screen.getByRole("img", { name: "Actual FK ESS trace" }),
    ).toBeVisible();
    // The event is at actual s=.5, not the first equally spaced array index.
    expect(
      screen
        .getByRole("img", { name: "Actual FK ESS trace" })
        .querySelector("circle"),
    ).toHaveAttribute("cx", "307");
  });
  it("states the population and compatible method", () => {
    render(<FKControls value={{}} onChange={() => {}} />);
    expect(screen.getByText(/not physical molecules/)).toBeVisible();
    expect(screen.getByText(/euler_maruyama/)).toBeVisible();
  });
});
