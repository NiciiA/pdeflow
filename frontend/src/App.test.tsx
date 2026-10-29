import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  render,
  screen,
  waitFor,
  cleanup,
  fireEvent,
  act,
} from "@testing-library/react";
import App from "./App";
import { editDimension, previewEquation } from "./types";
import type { Problem } from "./types";
// Unit seam only. Real MathLive editing is exercised by the Chromium suite.
vi.mock("./MathInput", () => ({
  MathInput: ({
    value,
    onChange,
    index,
  }: {
    value: string;
    onChange: (s: string) => void;
    index: number;
  }) => (
    <input
      aria-label={`Equation ${index + 1}`}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}));
const heatPreset = {
  id: "heat",
  label: "Heat / diffusion",
  static: false,
  equations: [String.raw`\frac{\partial u}{\partial \tau}=D\Delta u`],
  declarations: [
    {
      name: "u",
      kind: "field",
      rank: 0,
      status: "unknown",
      value: null,
      units: "nondimensional",
      meaning: "signed scalar field",
    },
    {
      name: "D",
      kind: "parameter",
      rank: 0,
      status: "known",
      value: 0.07,
      units: "nondimensional",
      meaning: "diffusivity",
    },
  ],
};
function mathResponse(url: string, init?: RequestInit) {
  if (url.endsWith("/mathematics/presets")) return [heatPreset];
  if (url.endsWith("/mathematics/parse")) {
    const r = JSON.parse(String(init?.body)).request;
    return {
      parsed: [],
      status: {
        editable: true,
        parsed: !!r.equations[0],
        residual: true,
        data_generator: true,
        checkpoint: true,
        reconstruction: false,
      },
      reasons: [],
      canonical_source: r.equations,
      recognized_family: "heat",
      problem: r.equations[0]
        ? {
            ...problem,
            grid: r.domain,
            known_parameters: {
              diffusivity:
                r.declarations.find((d: { name: string }) => d.name === "D")
                  ?.value ?? 0.07,
            },
          }
        : null,
    };
  }
  return undefined;
}
async function insertHeat() {
  fireEvent.click(
    await screen.findByRole("button", { name: /Heat \/ diffusion/ }),
  );
  await screen.findByLabelText("Diffusivity");
  await waitFor(() =>
    expect(screen.getByText("checkpoint: yes")).toBeVisible(),
  );
}
const problem: Problem = {
  schema_version: 1,
  family: "heat",
  grid: {
    shape: [8, 8],
    times: [0, 0.05, 0.1, 0.15],
    convention: "periodic_nodes_endpoint_excluded",
    domain: {
      lengths: [1, 1],
      origins: [0, 0],
      boundary: "periodic",
      geometry: "cartesian",
    },
  },
  fields: [{ name: "u", units: "nondimensional", role: "unknown" }],
  known_parameters: { diffusivity: 0.07 },
  operator_spec: { op: "field", name: "u", args: [], value: 0 },
  unknown_parameters: [],
  initial_condition: "unknown",
};
const fetcher = vi.fn();
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", fetcher);
  fetcher.mockImplementation(async (url: string, init?: RequestInit) => ({
    ok: true,
    json: async () =>
      mathResponse(url, init) ??
      (url.endsWith("/input-preview")
        ? {
            field: "u",
            choices: [],
            shape: [8, 8],
            axes: ["x", "y"],
            coordinates: { x: [0, 0.875], y: [0, 0.875] },
            values: null,
            points: [],
            point_count: 0,
            total_observations: 0,
            physical_time: 0,
            slice_coordinate: null,
            units: "nondimensional",
            role: "unknown",
            valid: true,
            issues: [],
            boundary: "periodic",
          }
        : undefined) ??
      (url.endsWith("/models")
        ? [
            {
              id: "a".repeat(32),
              name: "Heat 2D test",
              problem,
              config: { inference: { steps: 3, particles: 2, seed: 37 } },
              training: {
                updates: 30,
                status: "smoke",
                diffusivity_range: [0.03, 0.12],
              },
            },
          ]
        : url.endsWith("/health")
          ? { backends: { cpu: true, mps: false, cuda: false } }
          : url.endsWith("/jobs")
            ? []
            : {}),
  }));
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
describe("problem editing", () => {
  it("synchronizes form and canonical spec", async () => {
    render(<App />);
    await insertHeat();
    const input = await screen.findByLabelText("Diffusivity");
    fireEvent.change(input, { target: { value: ".09" } });
    fireEvent.click(screen.getByText("Request JSON, import & export"));
    const spec = screen.getByLabelText(
      "Complete mathematical request JSON",
    ) as HTMLTextAreaElement;
    expect(
      JSON.parse(spec.value).declarations.find(
        (d: { name: string }) => d.name === "D",
      ).value,
    ).toBe(0.09);
    const changed = JSON.parse(spec.value);
    changed.declarations.find((d: { name: string }) => d.name === "D").value =
      0.11;
    fireEvent.change(spec, { target: { value: JSON.stringify(changed) } });
    expect(screen.getByLabelText("Diffusivity")).toHaveValue(0.11);
  });
  it("preserves invalid advanced draft", async () => {
    render(<App />);
    await insertHeat();
    fireEvent.click(screen.getByText("Request JSON, import & export"));
    const editor = await screen.findByLabelText(
      "Complete mathematical request JSON",
    );
    fireEvent.change(editor, { target: { value: "{bad" } });
    expect(editor).toHaveValue("{bad");
    expect(screen.getByRole("alert")).toBeVisible();
  });
  it("adapts coordinates to dimension", () => {
    expect(editDimension(problem, 3).grid.shape).toEqual([8, 8, 8]);
    expect(editDimension(problem, 1).grid.domain.lengths).toEqual([1]);
  });
  it("updates readable equation from the operator", () => {
    expect(
      previewEquation({ op: "laplacian", args: [{ op: "field", name: "v" }] }),
    ).toContain("Δ(v)");
  });
  it("prevents submission before compatibility validation", async () => {
    render(<App />);
    await insertHeat();
    fireEvent.click(screen.getByRole("button", { name: /3Reconstruct/ }));
    expect(
      screen.getByRole("button", { name: "Run reconstruction" }),
    ).toBeDisabled();
  });
  it("shows imported observation errors", async () => {
    render(<App />);
    await insertHeat();
    fireEvent.click(screen.getByRole("button", { name: /2Add information/ }));
    fetcher.mockImplementationOnce(async () => ({
      ok: false,
      json: async () => ({ error: { message: "required columns missing" } }),
    }));
    const file = new File(["bad"], "bad.csv", { type: "text/csv" });
    Object.defineProperty(file, "text", { value: async () => "bad" });
    fireEvent.change(screen.getByLabelText("Import observations"), {
      target: { files: [file] },
    });
    await waitFor(() =>
      expect(screen.getByText("required columns missing")).toBeVisible(),
    );
  });
});

it("prevents duplicate submissions while the first request is pending", async () => {
  render(<App />);
  await insertHeat();
  fireEvent.click(screen.getByRole("button", { name: /3Reconstruct/ }));
  let posts = 0;
  fetcher.mockImplementation(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/reconstruction/validate"))
      return {
        ok: true,
        json: async () => ({
          valid: true,
          sample_storage_bytes: 128,
          warnings: [],
          resolved: {},
        }),
      };
    if (url.endsWith("/jobs") && init?.method === "POST") {
      posts++;
      return new Promise(() => {});
    }
    return { ok: true, json: async () => [] };
  });
  fireEvent.click(
    screen.getByRole("button", { name: "Validate complete request" }),
  );
  await screen.findByText("Compatible request");
  const run = screen.getByRole("button", { name: "Run reconstruction" });
  fireEvent.click(run);
  fireEvent.click(run);
  await waitFor(() => expect(posts).toBe(1));
  expect(screen.getByRole("button", { name: "Working…" })).toBeDisabled();
});

it("reopens a failed job with its real error state", async () => {
  const job = {
    id: "b".repeat(32),
    status: "failed",
    stage: "worker_exit",
    progress: 0,
    seed: 37,
    model_id: "a".repeat(32),
    error: { message: "Worker exited unexpectedly" },
  };
  fetcher.mockImplementation(async (url: string, init?: RequestInit) => ({
    ok: true,
    json: async () =>
      mathResponse(url, init) ??
      (url.endsWith("/input-preview")
        ? {
            field: "u",
            choices: [],
            shape: [8, 8],
            axes: ["x", "y"],
            coordinates: { x: [0, 0.875], y: [0, 0.875] },
            values: null,
            points: [],
            point_count: 0,
            total_observations: 0,
            physical_time: 0,
            slice_coordinate: null,
            units: "nondimensional",
            role: "unknown",
            valid: true,
            issues: [],
            boundary: "periodic",
          }
        : undefined) ??
      (url.endsWith("/models")
        ? [
            {
              id: "a".repeat(32),
              name: "Heat 2D",
              problem,
              config: { inference: { steps: 3, particles: 2 } },
              training: {
                updates: 30,
                status: "smoke",
                diffusivity_range: [0.03, 0.12],
              },
            },
          ]
        : url.endsWith("/health")
          ? { backends: { cpu: true } }
          : url.endsWith("/jobs")
            ? [job]
            : {}),
  }));
  render(<App />);
  await insertHeat();
  fireEvent.click(screen.getByRole("button", { name: /^Runs$/ }));
  fireEvent.click(await screen.findByRole("button", { name: "Open run" }));
  expect(await screen.findByText("Job failed")).toBeVisible();
  expect(screen.getByText("Worker exited unexpectedly")).toBeVisible();
});

it("starts with an empty equation and marks a visually edited preset custom", async () => {
  render(<App />);
  expect(await screen.findByLabelText("Equation 1")).toHaveValue("");
  await insertHeat();
  fireEvent.change(screen.getByLabelText("Equation 1"), {
    target: { value: "u_τ=D lap(u)+u" },
  });
  fireEvent.click(screen.getByText("Request JSON, import & export"));
  const current = JSON.parse(
    (
      screen.getByLabelText(
        "Complete mathematical request JSON",
      ) as HTMLTextAreaElement
    ).value,
  );
  expect(current.equations).toEqual(["u_τ=D lap(u)+u"]);
  expect(current.preset).toBeNull();
});
it("requires a visible review before replacing a nonempty preset and retains parameters", async () => {
  render(<App />);
  await insertHeat();
  fireEvent.change(screen.getByLabelText("Diffusivity"), {
    target: { value: ".09" },
  });
  fireEvent.click(screen.getByRole("button", { name: /Heat \/ diffusion/ }));
  expect(screen.getByRole("dialog")).toBeVisible();
  fireEvent.click(
    screen.getByRole("button", { name: "Apply reviewed changes" }),
  );
  expect(screen.getByLabelText("Diffusivity")).toHaveValue(0.09);
});
it("invalid advanced source cannot revalidate the last successfully parsed request", async () => {
  render(<App />);
  await insertHeat();
  fireEvent.click(screen.getByText("Request JSON, import & export"));
  fireEvent.change(
    screen.getByLabelText("Complete mathematical request JSON"),
    { target: { value: "{bad" } },
  );
  expect(
    screen.getByRole("button", { name: "Validate problem" }),
  ).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: /3Reconstruct/ }));
  fireEvent.click(
    screen.getByRole("button", { name: "Validate complete request" }),
  );
  await screen.findByText(/preserved request JSON is incomplete or invalid/i);
  expect(
    screen.getByRole("button", { name: "Run reconstruction" }),
  ).toBeDisabled();
});
it("reviews conflicts when a blank equation already has user declarations", async () => {
  render(<App />);
  fireEvent.click(
    screen.getByText("Edit declarations, domain, conditions & model override"),
  );
  fireEvent.click(
    await screen.findByRole("button", { name: "Add declaration" }),
  );
  expect(screen.getByLabelText("Symbol 1")).toHaveValue("q");
  fireEvent.click(
    await screen.findByRole("button", { name: /Heat \/ diffusion/ }),
  );
  expect(screen.getByRole("dialog")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Keep current draft" }));
  expect(screen.getByLabelText("Symbol 1")).toHaveValue("q");
  expect(screen.getByLabelText("Equation 1")).toHaveValue("");
});

it("explains a disconnected backend with the actual URL and startup command", async () => {
  fetcher.mockRejectedValue(new TypeError("Failed to fetch"));
  render(<App />);
  const notice = await screen.findByRole("alert", {
    name: "Backend connection",
  });
  expect(notice).toHaveTextContent("Backend unavailable.");
  expect(notice).toHaveTextContent("http://127.0.0.1:8000");
  expect(notice).toHaveTextContent("pdeflow serve");
  expect(
    screen.getByRole("button", { name: "Refresh checkpoints" }),
  ).toBeVisible();
});

it("finishes current equation validation when navigating immediately after a parameter edit", async () => {
  render(<App />);
  await insertHeat();
  fireEvent.change(screen.getByLabelText("Diffusivity"), {
    target: { value: ".095" },
  });
  fireEvent.click(screen.getByRole("button", { name: /2\s*Add information/ }));
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Generate synthetic observations" }),
    ).toBeEnabled(),
  );
  expect(
    screen.getByRole("heading", { name: "Bring your measurements" }),
  ).toBeVisible();
  expect(screen.getByLabelText("Equation 1")).not.toBeVisible();
  expect(
    screen.getByRole("button", { name: "Edit Equation" }),
  ).toHaveTextContent("Parsed");
});

it("resetting an unfinished draft returns to definition and persists the selected workspace stage", async () => {
  render(<App />);
  await insertHeat();
  fireEvent.click(screen.getByRole("button", { name: /2\s*Add information/ }));
  fireEvent.click(screen.getByRole("button", { name: "Reset draft" }));
  fireEvent.click(screen.getByRole("button", { name: "Clear current draft" }));
  expect(
    screen.getByRole("heading", { name: "Define the physical problem" }),
  ).toBeVisible();
  expect(screen.getByLabelText("Equation 1")).toHaveValue("");
  await waitFor(() =>
    expect(localStorage.getItem("pdeflow.view.v1")).toBe("Problem"),
  );
});

it("manual validation supersedes an older pending debounce without clearing its completed report", async () => {
  render(<App />);
  await insertHeat();
  fireEvent.click(
    screen.getByText("Edit declarations, domain, conditions & model override"),
  );
  fireEvent.change(screen.getByLabelText("Diffusivity"), {
    target: { value: ".096" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Validate problem" }));
  await screen.findByText("checkpoint: yes");
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 350));
  });
  expect(screen.getByText("checkpoint: yes")).toBeVisible();
  expect(screen.queryByText(/Parsing current draft/)).toBeNull();
});
