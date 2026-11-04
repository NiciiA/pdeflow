import { afterEach, expect, it, vi } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { MeshTraining } from "./MeshTraining";
const call = vi.hoisted(() => vi.fn());
vi.mock("./types", () => ({ api: call }));
afterEach(() => {
  cleanup();
  call.mockReset();
});
it("defaults to M1/joint/MSE and keeps train explicit while routing family/loss settings", async () => {
  call.mockImplementation(
    async (path: string, request: Record<string, unknown>) =>
      path === "/training/runs"
        ? []
        : path === "/training/plan"
          ? {
              training_settings: {
                policy: (request.advanced as Record<string, unknown>)
                  .batch_policy,
                stages: {
                  reconstruction: {
                    learning_rate: (request.advanced as Record<string, unknown>)
                      .learning_rate,
                    effective_batch: (
                      request.advanced as Record<string, unknown>
                    ).batch_size,
                    microbatch: (request.advanced as Record<string, unknown>)
                      .microbatch,
                    accumulation: Math.ceil(
                      Number(
                        (request.advanced as Record<string, unknown>)
                          .batch_size,
                      ) /
                        Number(
                          (request.advanced as Record<string, unknown>)
                            .microbatch,
                        ),
                    ),
                    last_microbatch: 2,
                    optimizer: { schedule: "constant", warmup: 0 },
                    schedule_horizon: 2,
                  },
                },
              },
              model_label: "M1 mesh",
              scope: "Initial only",
              stages: ["reconstruction"],
              updates: { reconstruction: 2 },
              config: { node_shape: [4, 4], data: { query_points: 64 } },
              architecture_parameters: { total: 121355 },
              training_physics: {
                equations: { heat: "u_t=D lap u", wave: "u_tt=c² lap u" },
                family_weights: { heat: 0.5, wave: 0.5 },
                state_schemas: { heat: ["u"], wave: ["u", "v"] },
              },
              resolved_device: "cpu",
              request,
            }
          : {},
  );
  render(<MeshTraining onRefresh={() => {}} onLegacy={() => {}} />);
  await waitFor(() =>
    expect(screen.getByTestId("mesh-resolved-summary")).toBeVisible(),
  );
  expect(screen.getByLabelText("New model candidate")).toHaveValue(
    "mesh_m1_v1",
  );
  expect(screen.getByLabelText("Field reconstruction loss")).toHaveValue("mse");
  expect(call.mock.calls.some(([p]) => p === "/training/start")).toBe(false);
  fireEvent.change(screen.getByLabelText("Training equations"), {
    target: { value: "wave-2d" },
  });
  fireEvent.change(screen.getByLabelText("Field reconstruction loss"), {
    target: { value: "region_balanced_mse" },
  });
  await waitFor(() =>
    expect(
      call.mock.calls.filter(([p]) => p === "/training/plan").at(-1)?.[1],
    ).toMatchObject({
      problem: "wave-2d",
      model: "mesh_m1_v1",
      advanced: { field_loss: { type: "region_balanced_mse" } },
    }),
  );
  expect(screen.getByLabelText("Feature balance: α")).toHaveValue(0.5);
  expect(screen.queryByText(/A1 trains/)).not.toBeInTheDocument();
  expect(screen.getByLabelText("Mesh learning rate")).toBeVisible();
  expect(screen.getByLabelText("Mesh effective batch")).toBeVisible();
  expect(screen.getByLabelText("Mesh forward microbatch")).toBeVisible();
  fireEvent.change(screen.getByLabelText("Mesh learning rate"), {
    target: { value: "0.0001" },
  });
  fireEvent.change(screen.getByLabelText("Mesh effective batch"), {
    target: { value: "5" },
  });
  fireEvent.change(screen.getByLabelText("Mesh forward microbatch"), {
    target: { value: "2" },
  });
  fireEvent.change(screen.getByLabelText("Mesh batch budget policy"), {
    target: { value: "keep_examples" },
  });
  await waitFor(() =>
    expect(
      call.mock.calls.filter(([p]) => p === "/training/plan").at(-1)?.[1],
    ).toMatchObject({
      advanced: {
        learning_rate: 0.0001,
        batch_size: 5,
        microbatch: 2,
        batch_policy: "keep_examples",
      },
    }),
  );
  expect(screen.getByText(/Derived accumulation: 3/)).toBeVisible();
  fireEvent.click(screen.getByText("Advanced model and per-stage settings"));
  fireEvent.click(screen.getByLabelText("Override reconstruction"));
  fireEvent.change(
    await screen.findByLabelText("reconstruction Learning rate"),
    {
      target: { value: "0.0002" },
    },
  );
  await waitFor(() =>
    expect(
      call.mock.calls.filter(([p]) => p === "/training/plan").at(-1)?.[1],
    ).toMatchObject({
      advanced: {
        stage_settings: { reconstruction: { learning_rate: 0.0002 } },
      },
    }),
  );
  fireEvent.change(screen.getByLabelText("Training budget"), {target:{value:"pipeline-check"}});
  expect(screen.getByLabelText("Mesh effective batch")).toHaveValue(5);
  expect(screen.getByLabelText("Mesh forward microbatch")).toHaveValue(2);
  expect(screen.getByLabelText("Model capacity")).toHaveValue("xsmall");
  expect(screen.getByLabelText("Mesh learning rate")).toHaveValue(0.0001);
  expect(call.mock.calls.some(([p]) => p === "/training/start")).toBe(false);
});
