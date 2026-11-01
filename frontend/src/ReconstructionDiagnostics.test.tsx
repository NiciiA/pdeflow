import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { ReconstructionDiagnostics } from "./ReconstructionDiagnostics";
afterEach(cleanup);
it("labels exact-sensor ensemble metrics independently from whole-field accuracy", () => {
  render(
    <ReconstructionDiagnostics
      metrics={{
        seconds: 1,
        observation_consistency: null,
        checkpoint_sha256: "abc123",
        checkpoint_updates: 2500,
        sensor_diagnostics: {
          count: 2,
          by_quantity: {
            v: {
              count: 2,
              ensemble_mean: {
                mse: 4,
                rmse: 2,
                mae: 1.5,
                noiseless_count: 1,
                noisy_count: 1,
              },
            },
          },
        },
      }}
    />,
  );
  expect(screen.getByText("Reconstruction diagnostics")).toBeInTheDocument();
  expect(screen.getByText(/Ensemble MSE 4.000/)).toHaveTextContent(
    "RMSE 2.000",
  );
  expect(
    screen.getByText(/conditioning-data fit, not held-out accuracy/),
  ).toBeInTheDocument();
  expect(screen.getByText("abc123")).toBeInTheDocument();
});
it("does not invent errors when sensor predictions are unavailable", () => {
  const { container } = render(
    <ReconstructionDiagnostics
      metrics={{ seconds: 0, observation_consistency: null }}
    />,
  );
  expect(container).toBeEmptyDOMElement();
});
