import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { beforeEach, afterEach, it, expect, vi } from "vitest";
import { TrainingOverview } from "./TrainingOverview";
import { AnalyticalExamples } from "./AnalyticalExamples";
import { emptyMathematics } from "./mathematics";
import type { Problem } from "./types";
const call = vi.hoisted(() => vi.fn());
vi.mock("./types", () => ({ api: call }));
const problem: Problem = {
  schema_version: 1,
  family: "heat",
  grid: emptyMathematics().domain,
  fields: [{ name: "u", units: "nondimensional", role: "unknown" }],
  known_parameters: { diffusivity: 0.07 },
  known_fields: {},
  operator_spec: {},
  unknown_parameters: [],
  initial_condition: "unknown",
};
beforeEach(() => call.mockReset());
afterEach(cleanup);
it("plans and previews without launching training or deleting artifacts", async () => {
  call.mockImplementation(async (path: string) =>
    path === "/training-coverage"
      ? {
          targets: [
            {
              id: "heat-2d-512",
              specialist: "heat",
              grid: problem.grid,
              architecture: { parameters: 73939 },
              state: "partial",
              maximum_updates: 2,
              updates: 10000,
              evaluation_available: false,
              compatible_discoverable_checkpoint: true,
              command: "pdeflow train --preset heat-2d-512",
              resolved_config: {},
            },
          ],
        }
      : path === "/training-runs"
        ? []
        : path === "/training-plan"
          ? {
              catalogue_version: "v2",
              entries: [
                {
                  preset: "heat-2d-512",
                  grid: [512, 512],
                  updates: 10000,
                  device: "mps",
                },
              ],
            }
          : path === "/storage/preview"
            ? {
                id: "test",
                review_digest: "digest",
                entries: [],
                total_bytes: 0,
              }
            : {},
  );
  render(<TrainingOverview problem={problem} />);
  await screen.findByRole("cell", { name: /^heat-2d-512/ });
  expect(screen.getByText(/10000 required \/ 2 recorded/)).toBeInTheDocument();
  fireEvent.click(screen.getByText("Plan full catalogue (no training)"));
  await screen.findByText(/no jobs launched/);
  fireEvent.click(screen.getByText("Preview cleanup (no deletion)"));
  await screen.findByText(/Nothing deleted or relocated/);
  expect(
    call.mock.calls.every(
      ([path]) =>
        !String(path).includes("/jobs") &&
        !String(path).includes("/apply") &&
        !String(path).includes("/train-start"),
    ),
  ).toBe(true);
});
it("inserts distinct editable reference and initial expressions without altering the inverse problem", async () => {
  const onInputs = vi.fn(),
    onVerify = vi.fn().mockResolvedValue({
      passed: false,
      classification: "inconsistent candidate",
    });
  const before = JSON.stringify(problem);
  call.mockResolvedValue([
    {
      id: "product",
      label: "Product",
      initial_expression: "cos(2*pi*x)*cos(2*pi*y)",
      reference_expression: "exp(-8*pi**2*D*t)*cos(2*pi*x)*cos(2*pi*y)",
      distribution: "low-mode diagnostic",
      decay_note: "Times preserved",
    },
  ]);
  render(
    <AnalyticalExamples
      problem={problem}
      ready
      inputs="{}"
      onInputs={onInputs}
      onVerify={onVerify}
    />,
  );
  fireEvent.click(
    screen.getByText(
      "Analytical example · editable initial and complete reference",
    ),
  );
  await screen.findByText(/u0 =/);
  fireEvent.click(
    screen.getByText("Replace reference inputs with reviewed example"),
  );
  const inputs = JSON.parse(onInputs.mock.calls[0][0]);
  expect(inputs.initial_u.expression).not.toContain("exp");
  expect(inputs.analytical_reference.expression).toContain("D*t");
  expect(JSON.stringify(problem)).toBe(before);
  expect(onVerify).not.toHaveBeenCalled();
});
it("shows an active preparing worker with missing telemetry without inventing a rate or crashing", async () => {
  call.mockImplementation(async (path: string) =>
    path === "/training-coverage"
      ? { targets: [] }
      : path === "/training-runs"
        ? [
            {
              id: "preparing",
              run_id: "new-run",
              name: "New worker",
              specialist: "heat",
              grid: problem.grid,
              state: "preparing",
              active: true,
              device: "mps",
              dtype: "float32",
              data_mode: "online",
              progress: 0,
              budget: 10000,
              architecture: null,
              live: {},
              history: [],
              checkpoint_updated_at_unix: null,
              resume_command: "pdeflow resume new-run",
              evaluate_command: "pdeflow evaluate --run new-run",
              resolved_config: { provider: {} },
              log_tail: "",
            },
          ]
        : {},
  );
  render(<TrainingOverview problem={problem} />);
  await screen.findByText(/1 active worker\(s\)/);
  expect(screen.getByText(/throughput unavailable/)).toBeInTheDocument();
  expect(screen.getByText(/ETA unavailable/)).toBeInTheDocument();
});
it("never moves progress or unique counts backward when a resumed checkpoint is newer than throttled telemetry", async () => {
  call.mockImplementation(async (path: string) =>
    path === "/training-coverage"
      ? { targets: [] }
      : path === "/training-runs"
        ? [
            {
              id: "resume",
              run_id: "resumed-run",
              name: "Resume",
              specialist: "heat",
              grid: problem.grid,
              state: "preparing",
              active: true,
              device: "mps",
              dtype: "float32",
              data_mode: "online",
              progress: 7,
              budget: 8,
              architecture: null,
              provider_summary: { unique_training_problems_consumed: 7 },
              live: {
                step: 5,
                unique_physical_problems: 5,
                updates_per_second: 9,
                generated_reference_problems_per_second: 3,
              },
              history: [{ step: 7, training_loss: 0.2 }],
              checkpoint_updated_at_unix: null,
              resume_command: "pdeflow resume resumed-run",
              evaluate_command: "pdeflow evaluate --run resumed-run",
              resolved_config: { provider: {} },
              log_tail: "",
            },
          ]
        : {},
  );
  render(<TrainingOverview problem={problem} />);
  await screen.findByText(/update 7\/8 · unique physical problems 7/);
  expect(screen.getByText(/throughput unavailable/)).toBeInTheDocument();
  expect(screen.queryByText(/9.00 updates\/s/)).not.toBeInTheDocument();
});
