import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { TrainModel } from "./TrainModel";
import { emptyMathematics } from "./mathematics";
const call = vi.hoisted(() => vi.fn());
vi.mock("./types", () => ({ api: call }));
const grid = emptyMathematics().domain;
const options = {
  problems: ["heat-2d", "darcy-2d"],
  models: {},
  budgets: {
    "pipeline-check": "Verifies execution; unsuitable for judging accuracy.",
    development: "Bounded learning experiment",
    extended: "No guarantee of accuracy",
  },
  devices: { cpu: true, mps: false, cuda: false },
  auto_device: "cpu",
  auto_explanation: "Auto selects NVIDIA, then Apple, then CPU.",
};
function plan(request: Record<string, unknown> = {}) {
  return {
    problem: "heat-2d",
    model: "fixed-grid",
    model_label: "Existing fixed-grid model",
    equation: "∂u/∂τ = D Δu",
    varies: "Random initial fields and known D.",
    resolved_device: "cpu",
    grid,
    updates: { fixed: 2 },
    examples: { validation: 2, untouched_test: 2 },
    distribution: { diffusivity_range: [0.03, 0.12] },
    config: { width: 24, ae_steps: 4, flow_steps: 4 },
    budget_meaning: "Execution only",
    pipeline_check: true,
    stages: ["prepare", "fixed", "model_validation", "register"],
    ...request,
  };
}
function run(state = "completed") {
  return {
    id: "run-1",
    state,
    stage: "register",
    stage_label: "Make model available",
    stage_meaning: "Explicit selection required",
    progress: 0,
    stage_budget: 0,
    elapsed_seconds: 1.4,
    eta_seconds: null,
    history: [],
    finished_stages: [],
    plan: plan(),
    validation: { split: "validation" },
    model_id: state === "completed" ? "a".repeat(32) : null,
    can_resume: state === "needs_review",
    stop_requested: false,
    log_tail: "Saved checkpoint",
    meaning: "Pipeline check; not evidence of accuracy.",
  };
}
const model = {
  id: "a".repeat(32),
  model: "fixed-grid",
  type: "Existing fixed-grid model",
  problem: { family: "heat", grid },
  status: "completed",
  updates: { fixed: 2 },
  validation: { split: "validation" },
  quality: "Execution only",
  pipeline_check: true,
  compatibility: { compatible: true, summary: "Matches" },
};
const slice = {
  values: [
    [1, 2],
    [3, 4],
  ],
  reference: [
    [1, 1],
    [2, 3],
  ],
  error: [
    [0, 1],
    [1, 1],
  ],
  shape: [2, 2],
  axes: ["x", "y"],
  coordinates: { x: [0, 0.5], y: [0, 0.5] },
  vmin: 0,
  vmax: 4,
  physical_time: 0,
  observations: [],
  units: "nondimensional",
};
afterEach(cleanup);
beforeEach(() => {
  localStorage.clear();
  call.mockReset();
  call.mockImplementation(
    async (path: string, body: Record<string, unknown>) => {
      if (path === "/training/options") return options;
      if (path === "/training/plan") return plan(body);
      if (path === "/training/models") return [];
      if (path === "/training/runs") return [];
      if (path.includes("/preview")) return slice;
      if (path === "/training/start") return run();
      return {};
    },
  );
});
it("opens passively and shows understandable budgets and unavailable devices", async () => {
  render(<TrainModel onRefresh={vi.fn()} />);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Train model" })).toBeEnabled(),
  );
  expect(screen.getByText(/unsuitable for judging accuracy/)).toBeVisible();
  expect(screen.getByRole("option", { name: /Apple GPU/ })).toBeDisabled();
  expect(screen.getByRole("option", { name: /NVIDIA GPU/ })).toBeDisabled();
  expect(call.mock.calls.some((c) => c[0] === "/training/start")).toBe(false);
  expect(
    screen.getByText(/separate from the reference currently displayed/),
  ).toBeVisible();
});
it("submits both-stage intent once and requires explicit model use afterward", async () => {
  const onUse = vi.fn();
  call.mockImplementation(
    async (path: string, body: Record<string, unknown>) => {
      if (path === "/training/options") return options;
      if (path === "/training/plan") return plan(body);
      if (path === "/training/models") return [model];
      if (path === "/training/runs") return [];
      if (path === "/training/start") {
        await new Promise((resolve) => setTimeout(resolve, 20));
        return run();
      }
      if (path.includes("/preview")) return slice;
      if (path === "/training/use")
        return {
          id: model.id,
          model: "fixed-grid",
          adopt_training_problem: false,
        };
      return {};
    },
  );
  render(<TrainModel onRefresh={vi.fn()} onUse={onUse} />);
  fireEvent.change(screen.getByLabelText("Training model"), {
    target: { value: "resolution-flexible" },
  });
  fireEvent.change(screen.getByLabelText("Training device"), {
    target: { value: "cpu" },
  });
  const train = screen.getByRole("button", { name: "Train model" });
  await waitFor(() => expect(train).toBeEnabled());
  fireEvent.click(train);
  fireEvent.click(train);
  await screen.findByRole("button", { name: "Use this model" });
  expect(
    call.mock.calls.filter((c) => c[0] === "/training/start"),
  ).toHaveLength(1);
  expect(
    call.mock.calls.find((c) => c[0] === "/training/start")?.[1],
  ).toMatchObject({
    model: "resolution-flexible",
    device: "cpu",
    budget: "pipeline-check",
  });
  expect(call.mock.calls.some((c) => c[0] === "/training/use")).toBe(false);
  expect(screen.getByText(/has not been automatically selected/)).toBeVisible();
  expect(
    await screen.findByRole("img", { name: /Absolute error/ }),
  ).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Use this model" }));
  await waitFor(() =>
    expect(onUse).toHaveBeenCalledWith(
      expect.objectContaining({ id: model.id }),
    ),
  );
});
it("shows a poor representation and records explicit resume override intent", async () => {
  const row = {
    ...run("needs_review"),
    representation_quality: {
      passed: false,
      validation_mse: 1.2,
      constant_baseline_mse: 1,
      mse_ratio: 1.2,
    },
  };
  call.mockImplementation(
    async (path: string, body: Record<string, unknown>) =>
      path === "/training/options"
        ? options
        : path === "/training/plan"
          ? plan(body)
          : path === "/training/runs"
            ? [row]
            : path === "/training/models"
              ? []
              : {},
  );
  render(<TrainModel onRefresh={vi.fn()} />);
  await screen.findByText(/Poor representation: minimal baseline check failed/);
  fireEvent.click(
    screen.getByLabelText(
      /Continue this run despite representation distortion/,
    ),
  );
  fireEvent.click(
    screen.getByRole("button", {
      name: /Resume training with recorded quality override/,
    }),
  );
  await waitFor(() =>
    expect(call).toHaveBeenCalledWith("/training/runs/run-1/resume", {
      allow_poor_representation: true,
    }),
  );
});
it("can safely stop an active run and prevents a second submission", async () => {
  call.mockImplementation(
    async (path: string, body: Record<string, unknown>) =>
      path === "/training/options"
        ? options
        : path === "/training/plan"
          ? plan(body)
          : path === "/training/runs"
            ? [run("running")]
            : path === "/training/models"
              ? []
              : {},
  );
  render(<TrainModel onRefresh={vi.fn()} />);
  const stop = await screen.findByRole("button", { name: "Stop safely" });
  expect(screen.getByRole("button", { name: "Train model" })).toBeDisabled();
  fireEvent.click(stop);
  await waitFor(() =>
    expect(call).toHaveBeenCalledWith("/training/runs/run-1/stop", {}),
  );
});
it("shows independent equation intent in the focused page and preserves loss when switching", async () => {
  call.mockImplementation(
    async (path: string, body: Record<string, unknown>) => {
      if (path === "/training/options")
        return { ...options, problems: ["heat-2d", "wave-2d", "heat-wave-2d"] };
      if (path === "/training/runs" || path === "/training/models") return [];
      if (path === "/training/plan") {
        const keys =
          body.problem === "heat-2d"
            ? ["heat"]
            : body.problem === "wave-2d"
              ? ["wave"]
              : ["heat", "wave"];
        return plan({
          ...body,
          training_physics: {
            family_weights: Object.fromEntries(
              keys.map((k) => [k, 1 / keys.length]),
            ),
            state_schemas: Object.fromEntries(
              keys.map((k) => [k, k === "heat" ? ["u"] : ["u", "v"]]),
            ),
            equations: Object.fromEntries(
              keys.map((k) => [
                k,
                k === "heat" ? "∂u/∂t = D Δu" : "∂²u/∂t² = c² Δu; v = ∂u/∂t",
              ]),
            ),
            parameter_ranges: Object.fromEntries(
              keys.map((k) => [
                k === "heat" ? "diffusivity" : "wave_speed",
                k === "heat" ? [0.03, 0.12] : [0.2, 0.6],
              ]),
            ),
          },
        });
      }
      return {};
    },
  );
  render(
    <TrainModel
      focused
      problem={{ family: "heat", grid } as never}
      onRefresh={vi.fn()}
    />,
  );
  expect(screen.getByLabelText("Training equations")).toHaveValue(
    "heat-wave-2d",
  );
  await screen.findByText(/Wave state: displacement u and velocity v/);
  expect(screen.getByText(/Heat 50% · Wave 50%/)).toBeVisible();
  expect(screen.getByLabelText("Field reconstruction loss")).toHaveValue("mse");
  expect(screen.queryByLabelText("Feature balance: α")).toBeNull();
  fireEvent.change(screen.getByLabelText("Field reconstruction loss"), {
    target: { value: "region_balanced_mse" },
  });
  await screen.findByLabelText("Feature balance: α");
  fireEvent.change(screen.getByLabelText("Feature balance: α"), {
    target: { value: "0.6" },
  });
  fireEvent.change(screen.getByLabelText("Training equations"), {
    target: { value: "wave-2d" },
  });
  await waitFor(() =>
    expect(
      call.mock.calls.filter((c) => c[0] === "/training/plan").at(-1)?.[1],
    ).toMatchObject({
      problem: "wave-2d",
      advanced: { field_loss: { type: "region_balanced_mse", alpha: 0.6 } },
    }),
  );
  await screen.findByText(/Wave 100%/);
  expect(screen.queryByText(/Heat state: u/)).not.toBeInTheDocument();
  expect(screen.getByText(/wave speed: 0.2–0.6/)).toBeVisible();
  expect(screen.queryByText(/diffusivity: 0.03–0.12/)).not.toBeInTheDocument();
  expect(screen.getByLabelText("Feature balance: α")).toHaveValue(0.6);
});
it("blocks invalid selections without presenting the old summary as resolved", async () => {
  call.mockImplementation(
    async (path: string, body: Record<string, unknown>) => {
      if (path === "/training/options") return options;
      if (path === "/training/runs" || path === "/training/models") return [];
      if (path === "/training/plan") {
        if (body.problem === "wave-2d")
          throw new Error(
            "Training equations conflict with advanced family_weights",
          );
        return plan(body);
      }
      return {};
    },
  );
  render(<TrainModel focused onRefresh={vi.fn()} />);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Train model" })).toBeEnabled(),
  );
  fireEvent.change(screen.getByLabelText("Training equations"), {
    target: { value: "wave-2d" },
  });
  await screen.findByText(
    "Training configuration is invalid. Resolve the error before starting.",
  );
  expect(screen.getByTestId("resolved-training-summary")).not.toHaveTextContent(
    "∂u/∂τ",
  );
  expect(screen.getByRole("button", { name: "Train model" })).toBeDisabled();
});
