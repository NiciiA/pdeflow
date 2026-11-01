import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { ReferenceWorkflow } from "./ReferenceWorkflow";
import { EvaluationPanel } from "./EvaluationPanel";
import { emptyMathematics } from "./mathematics";
import type { Problem } from "./types";
const call = vi.hoisted(() => vi.fn());
vi.mock("./types", () => ({ api: call }));
const math = emptyMathematics();
math.equations = ["dτ(u)=D*lap(u)"];
const problem: Problem = {
  schema_version: 1,
  family: "heat",
  grid: math.domain,
  fields: [{ name: "u", role: "unknown", units: "nondimensional" }],
  known_parameters: { diffusivity: 0.07 },
  known_fields: {},
  operator_spec: {},
  unknown_parameters: [],
  initial_condition: "unknown",
};
beforeEach(() => {
  localStorage.clear();
  call.mockReset();
});
it("solves without a neural checkpoint and keeps display choices outside inverse adoption", async () => {
  call.mockImplementation(async (path: string) =>
    path === "/references"
      ? {
          id: "a".repeat(32),
          problem,
          diagnostics: { classification: "finite Fourier" },
        }
      : {
          values: [0, 1],
          shape: [2],
          axes: ["x"],
          coordinates: { x: [0, 0.5] },
          vmin: 0,
          vmax: 1,
          observations: [],
          physical_time: 0,
          units: "nondimensional",
        },
  );
  const adopt = vi.fn();
  render(
    <ReferenceWorkflow
      math={math}
      problem={problem}
      ready
      visible
      onSample={adopt}
      onReconstruct={() => {}}
    />,
  );
  fireEvent.click(screen.getByText("Load visible reference defaults"));
  fireEvent.click(screen.getByText("Solve reference"));
  await screen.findByText(/Immutable reference/);
  const body = call.mock.calls.find((c) => c[0] === "/references")![1];
  expect(body.inputs.initial_u.fourier).toBeDefined();
  expect(body.model_id).toBeUndefined();
  fireEvent.click(screen.getByLabelText(/Show reference truth/));
  expect(adopt).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText("Complete reference inputs"), {
    target: { value: '{"initial_u":1}' },
  });
  await waitFor(() =>
    expect(screen.getByText(/Reference is stale/)).toBeVisible(),
  );
  expect(
    screen.getByRole("button", { name: "Generate observations" }),
  ).toBeDisabled();
});
it("renders undefined relative norms and distinct population summaries", () => {
  const e = {
    mse: 1,
    rmse: 1,
    mae: 1,
    relative_l2: null,
    status: "near-zero reference",
    point_count: 8,
  };
  render(
    <EvaluationPanel
      metrics={{
        evaluation: {
          fields: {
            u: {
              units: "unit",
              ensemble_mean: { full: e },
              weighted_mean_of_sample_errors: { full: e },
              per_sample: [{ full: e }],
              per_time: [{ physical_time: 0, full: e }],
              empirical_rms_spread: 0.2,
            },
          },
          summary_rule: "separate summaries",
          relative_norm_rule: "near-zero is undefined",
          observation_exclusion: "exact grid indices",
          comparison: {},
        },
      }}
    />,
  );
  expect(
    screen.getByRole("table", { name: "Reference comparison errors" }),
  ).toBeVisible();
  expect(screen.getByText(/Weighted mean \/ full/)).toBeVisible();
  expect(screen.getByText(/Mean individual errors \/ full/)).toBeVisible();
  expect(screen.getAllByText(/undefined/).length).toBeGreaterThan(0);
});
