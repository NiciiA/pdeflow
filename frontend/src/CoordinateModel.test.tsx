import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { beforeEach, afterEach, it, expect, vi } from "vitest";
import {
  CoordinateReconstruction,
  CoordinateTraining,
} from "./CoordinateModel";
import { ReferenceWorkflow } from "./ReferenceWorkflow";
import { emptyMathematics } from "./mathematics";
import type { Problem } from "./types";
const call = vi.hoisted(() => vi.fn());
vi.mock("./types", () => ({ api: call, saveFile: vi.fn() }));
const math = emptyMathematics();
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
afterEach(cleanup);
beforeEach(() => {
  localStorage.clear();
  call.mockReset();
});
it("separates query grid and scattered measurement rows from the reference and fixed grid", async () => {
  call.mockImplementation(async (path: string) =>
    path === "/coordinate/runs"
      ? [
          {
            id: "smoke-flow",
            stage: "flow",
            available: true,
            model_id: "a".repeat(32),
            progress: 2,
          },
        ]
      : path === "/coordinate/reconstruct"
        ? { id: "b".repeat(32), status: "succeeded" }
        : path.includes("/slice")
          ? {
              values: [0, 1],
              shape: [2],
              axes: ["x"],
              coordinates: { x: [0, 0.5] },
              reference: null,
              error: null,
              vmin: 0,
              vmax: 1,
              observations: [],
              physical_time: 0,
            }
          : {},
  );
  render(
    <CoordinateReconstruction
      problem={problem}
      rows={[]}
      referenceId={"c".repeat(32)}
      noiseStd={0.03}
      ready
      mathematicalRequest={math}
    />,
  );
  await screen.findByRole("option", { name: /smoke-flow/ });
  fireEvent.change(screen.getByLabelText("Coordinate query grid"), {
    target: { value: "12,12" },
  });
  fireEvent.change(screen.getByLabelText("Coordinate observation rows"), {
    target: {
      value:
        '[{"field":"u","x":0.123,"y":0.4,"time":0.03,"value":0,"sigma":0.02}]',
    },
  });
  fireEvent.click(
    screen.getByRole("button", {
      name: "Reconstruct with coordinate specialist",
    }),
  );
  await waitFor(() =>
    expect(
      call.mock.calls.some((c) => c[0] === "/coordinate/reconstruct"),
    ).toBe(true),
  );
  const request = call.mock.calls.find(
    (c) => c[0] === "/coordinate/reconstruct",
  )![1];
  expect(request.problem.grid.shape).toEqual([12, 12]);
  expect(problem.grid.shape).not.toEqual([12, 12]);
  expect(request.rows[0].value).toBe(0);
  expect(request.rows[0].sigma).toBe(0.02);
  expect(request.reference).toBeUndefined();
  expect(request.reference_id).toBeUndefined();
  expect(request.initial_u).toBeUndefined();
  expect(
    screen.getByRole("button", {
      name: "Compare coordinate result with reference",
    }),
  ).toBeEnabled();
});
it("shows a missing checkpoint without starting any training", async () => {
  call.mockResolvedValue([]);
  render(<CoordinateTraining />);
  await screen.findByText("No coordinate stages trained in this workspace.");
  expect(call.mock.calls.every((c) => c[0] === "/coordinate/runs")).toBe(true);
});
it("provides measurements only and displays precise scalar count before generation", async () => {
  call.mockImplementation(async (path: string) =>
    path === "/analytical-examples"
      ? []
      : path === "/observations/validate"
        ? {
            rows: [{ field: "u", x: 0, y: 0, time: 0, value: 0, sigma: 0.1 }],
            count: 1,
            observations: { indices: [[0, 0, 0, 0]] },
          }
        : {},
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
  fireEvent.change(screen.getByLabelText("Sampling count"), {
    target: { value: "7" },
  });
  expect(
    screen.getByText(/Preview total: 28 scalar observations/),
  ).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Input source mode"), {
    target: { value: "measurements" },
  });
  fireEvent.change(screen.getByLabelText("Measurements-only rows"), {
    target: {
      value: '[{"field":"u","x":0,"y":0,"time":0,"value":0,"sigma":0.1}]',
    },
  });
  fireEvent.click(
    screen.getByRole("button", {
      name: "Adopt measurements without reference",
    }),
  );
  await waitFor(() => expect(adopt).toHaveBeenCalled());
  expect(adopt.mock.calls[0][1]).toBeNull();
  expect(adopt.mock.calls[0][0].rows[0].value).toBe(0);
});

it("blocks a stale executable problem while the current equation is unsupported or parsing", async () => {
  call.mockResolvedValue([
    {
      id: "heat-flow",
      stage: "flow",
      available: true,
      model_id: "a".repeat(32),
      progress: 2,
    },
  ]);
  render(
    <CoordinateReconstruction
      problem={problem}
      rows={[]}
      referenceId={null}
      noiseStd={0.03}
      ready={false}
      mathematicalRequest={math}
    />,
  );
  await screen.findByRole("option", { name: /heat-flow/ });
  expect(
    screen.getByRole("button", {
      name: "Reconstruct with coordinate specialist",
    }),
  ).toBeDisabled();
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Current equation or declarations are unsupported or still parsing",
  );
  expect(call.mock.calls.every((c) => c[0] === "/coordinate/runs")).toBe(true);
});
