import { afterEach, expect, it } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MeshConvergence } from "./MeshConvergence";
afterEach(cleanup);
it("separates training and full-stage validation with honest missing breakdowns", () => {
  const { container } = render(
    <MeshConvergence
      record={{
        architecture: "M3",
        stage: "stochastic",
        history: [
          { step: 600, training_loss: 20 },
          { step: 600, training_loss: 1000, optimizer_update_skipped: true },
        ],
        training_series: [{ step: 600, mean: 18, count: 2 }],
        training_summary: {
          window: 100,
          count: 2,
          mean: 18,
          label:
            "Training objective: rolling mean, last 100 successful updates",
          family_objective: {
            heat: { mean: 2, examples: 3, missing_updates: 0 },
            wave: { mean: null, examples: 0, missing_updates: 2 },
          },
          quantity_objective:
            "Quantity-specific convergence is measured through decoded-field validation.",
        },
        validation_history: [
          { update: 0, uniform_domain_normalized_mse: 3 },
          {
            update: 500,
            uniform_domain_normalized_mse: 1,
            groups: {
              "wave/v": {
                physical_mse: 4,
                physical_rmse: 2,
                normalized_mse: 1,
                case_count: 2,
                evaluated_points: 32,
              },
            },
          },
        ],
        latest_validation: {
          update: 500,
          uniform_domain_normalized_mse: 1,
          groups: {
            "wave/v": {
              physical_mse: 4,
              physical_rmse: 2,
              normalized_mse: 1,
              case_count: 2,
              evaluated_points: 32,
            },
          },
        },
        best_validation: { update: 500, uniform_domain_normalized_mse: 1 },
      }}
    />,
  );
  expect(screen.getByText("Optimized training objective")).toBeVisible();
  expect(
    screen.getByText("Fixed-validation normalized decoded-field MSE"),
  ).toBeVisible();
  expect(screen.getByText("Wave v · velocity")).toBeVisible();
  expect(screen.getAllByText(/Not available/).length).toBeGreaterThan(0);
  expect(container.querySelector("circle title")?.textContent).not.toContain(
    "1000",
  );
  const validation = screen.getByRole("img", {
    name: /Fixed-validation normalized/,
  });
  expect(validation.querySelectorAll("circle")).toHaveLength(3);
  expect(validation.textContent).toContain("500");
});
it("does not invent validation markers or group errors for old records", () => {
  const { container } = render(
    <MeshConvergence
      record={{
        history: [{ step: 3, training_loss: 2 }],
        next_validation_update: 250,
      }}
    />,
  );
  expect(screen.getByText(/No validation has been recorded/)).toHaveTextContent(
    "250",
  );
  expect(
    screen.queryByText("Fixed-validation normalized decoded-field MSE"),
  ).not.toBeInTheDocument();
  expect(container.querySelectorAll("svg")).toHaveLength(1);
});
