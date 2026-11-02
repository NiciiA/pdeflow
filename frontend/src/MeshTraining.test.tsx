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
});
