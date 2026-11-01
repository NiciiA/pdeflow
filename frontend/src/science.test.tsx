import { FieldDiagnostics } from "./FieldDiagnostics";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { emptyMathematics } from "./mathematics";
import { requestItems, symbolName, readDraft } from "./science";
import { RequestSummary } from "./RequestSummary";
import { SparsePlot } from "./InputPreview";
import type { Preview } from "./InputPreview";
import { ObservationCharts, LineChart } from "./ScienceCharts";
afterEach(() => {
  cleanup();
  localStorage.clear();
});
describe("scientific summaries", () => {
  it("distinguishes defaults, unknown initial targets and static non-applicability", () => {
    const v = emptyMathematics();
    expect(requestItems(v, null).find((i) => i.symbol === "Ω")?.status).toBe(
      "Review defaults",
    );
    expect(
      requestItems(v, null, true).find((i) => i.symbol === "Ω")?.status,
    ).toBe("Reviewed");
    expect(requestItems(v, null).find((i) => i.symbol === "ℐ")?.status).toBe(
      "Unknown target",
    );
    v.domain.temporal_kind = "static";
    v.domain.times = [];
    expect(requestItems(v, null).find((i) => i.symbol === "ℐ")?.status).toBe(
      "Not applicable",
    );
  });
  it("resolves the same symbol from its actual specialist role", () => {
    const d = {
      name: "a",
      kind: "field" as const,
      rank: 0,
      status: "unknown" as const,
      value: null,
      units: "nondimensional",
      meaning: "",
    };
    expect(symbolName("darcy", d)).toBe("Permeability");
    expect(symbolName("chemistry", d)).toBe("Concentration of A");
    expect(symbolName("unsupported", d)).toBe("a");
  });
  it("provides direct keyboard-accessible request edits without declaring defaults complete", () => {
    const edit = vi.fn();
    render(
      <RequestSummary
        value={emptyMathematics()}
        report={null}
        reviewed={false}
        onEdit={edit}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Edit Measurements" }));
    expect(edit).toHaveBeenCalledWith("Observations", "observation-editor");
    expect(screen.getByText("What will be reconstructed?")).toBeVisible();
    expect(screen.getAllByText(/Review default/).length).toBeGreaterThan(0);
  });
  it("does not restore unknown draft schema versions or corrupt JSON", () => {
    localStorage.setItem("pdeflow.draft.v2", "{bad");
    expect(readDraft()).toBeNull();
    localStorage.setItem(
      "pdeflow.draft.v2",
      JSON.stringify({ version: 9, math: emptyMathematics() }),
    );
    expect(readDraft()).toBeNull();
  });
  it("coordinates actual observation selection and labels unknown regions", () => {
    const data: Preview = {
      field: "u",
      choices: [],
      shape: [8, 8],
      axes: ["x", "y"],
      coordinates: { x: [0, 0.875], y: [0, 0.875] },
      values: null,
      points: [{ row_id: 4, field: "u", x: 0, y: 0.875, value: 2 }],
      point_count: 1,
      total_observations: 1,
      physical_time: 0,
      slice_coordinate: null,
      units: "nondimensional",
      role: "unknown target",
      valid: true,
      issues: [],
      downsampled: false,
      boundary: "periodic",
      interpretation: "",
    };
    const select = vi.fn();
    render(<SparsePlot data={data} selected={4} onSelect={select} />);
    fireEvent.keyDown(
      screen.getByRole("button", { name: "Select observation 5: u = 2" }),
      { key: "Enter" },
    );
    expect(select).toHaveBeenCalledWith(4);
    expect(
      screen.getByText(/No interpolation or inferred field/),
    ).toBeVisible();
    expect(screen.getByRole("img")).toHaveAccessibleName(
      /unknown regions with measured points/,
    );
  });
  it("labels actual observation predictions and unavailable history without fabricated curves", () => {
    render(
      <>
        <ObservationCharts
          points={[
            { index: 0, field: "u", observed: 1, predicted: 2, residual: 1 },
          ]}
        />
        <LineChart
          title="No training"
          xlabel="Update"
          ylabel="CFM"
          series={[]}
        />
      </>,
    );
    expect(
      screen.getByRole("img", {
        name: /predicted versus observed with identity line/,
      }),
    ).toBeVisible();
    expect(
      screen.getByText(
        "No recorded values are available. No curve is inferred.",
      ),
    ).toBeVisible();
  });
});

it("keeps absent truth unavailable and gives baseline error tables distinct accessible names", () => {
  const view = render(<FieldDiagnostics metrics={{}} />);
  expect(
    screen.getByText(
      "Reconstruction errors unavailable without an evaluation reference.",
    ),
  ).toBeVisible();
  expect(screen.queryByRole("table")).toBeNull();
  view.rerender(
    <>
      <FieldDiagnostics
        metrics={{ fields: { u: { relative_l2_unobserved: null, mae: 1 } } }}
      />
      <FieldDiagnostics
        label="Baseline Fourier: per-field reconstruction errors"
        metrics={{ fields: { u: { relative_l2_unobserved: 0.5, mae: 0.2 } } }}
      />
    </>,
  );
  expect(
    screen.getByRole("table", { name: "Per-field reconstruction errors" }),
  ).toHaveTextContent("Unavailable");
  expect(
    screen.getByRole("table", {
      name: "Baseline Fourier: per-field reconstruction errors",
    }),
  ).toHaveTextContent("0.5000");
});
