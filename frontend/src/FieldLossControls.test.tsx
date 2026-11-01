import { afterEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { FieldLossControls } from "./FieldLossControls";
afterEach(cleanup);
it("selects the experimental blend with alpha .5 and keeps advanced controls collapsed", () => {
  const change = vi.fn();
  render(
    <FieldLossControls
      value={{ type: "region_balanced_mse", alpha: 0.5 }}
      onChange={change}
      request={{}}
      queryPoints={256}
    />,
  );
  expect(screen.getByLabelText("Field reconstruction loss")).toHaveValue(
    "region_balanced_mse",
  );
  expect(screen.getByLabelText("Feature balance: α")).toHaveValue(0.5);
  expect(
    screen.getByText("Advanced region analysis").closest("details"),
  ).not.toHaveAttribute("open");
  fireEvent.change(screen.getByLabelText("Feature balance: α"), {
    target: { value: ".7" },
  });
  expect(change).toHaveBeenCalledWith({
    type: "region_balanced_mse",
    alpha: 0.7,
  });
  expect(
    screen.getByText(
      /256 whole-domain queries per supervised quantity\/time \+ 256 additional region-training/,
    ),
  ).toBeVisible();
});
it("shows only the selected loss controls and submits its explicit type", () => {
  const change = vi.fn();
  render(
    <FieldLossControls
      value={{ type: "weighted_mse", contrast_strength: 2 }}
      onChange={change}
      request={{}}
      queryPoints={256}
    />,
  );
  expect(screen.queryByLabelText("Feature balance: α")).toBeNull();
  expect(screen.getByLabelText("Contrast emphasis: λ")).toHaveValue(2);
  fireEvent.change(screen.getByLabelText("Field reconstruction loss"), {
    target: { value: "mse" },
  });
  expect(change).toHaveBeenCalledWith({ type: "mse", contrast_strength: 2 });
});
it("selects spectral supervision explicitly and discloses the structured query cost", () => {
  const change = vi.fn();
  const props = { onChange: change, request: {}, queryPoints: 256 };
  const view = render(<FieldLossControls {...props} value={{ type: "mse" }} />);
  fireEvent.change(screen.getByLabelText("Field reconstruction loss"), {
    target: { value: "spectral_mse" },
  });
  expect(change).toHaveBeenLastCalledWith({
    type: "spectral_mse",
    spectral_weight: 0.1,
  });
  view.rerender(
    <FieldLossControls
      {...props}
      value={{ type: "spectral_mse", spectral_weight: 0.1 }}
    />,
  );
  expect(
    screen.getByLabelText("Structured spectral grid per axis"),
  ).toHaveValue(16);
  expect(
    screen.getByText(
      /Extra structured periodic-grid queries per quantity: 256/,
    ),
  ).toBeVisible();
  fireEvent.change(screen.getByLabelText("Field reconstruction loss"), {
    target: { value: "region_balanced_mse" },
  });
  expect(change).toHaveBeenLastCalledWith({
    type: "region_balanced_mse",
    spectral_weight: 0,
  });
});

it("omitted loss is Standard MSE with no region controls or extra queries", () => {
  render(
    <FieldLossControls
      value={{}}
      onChange={vi.fn()}
      request={{}}
      queryPoints={256}
    />,
  );
  expect(screen.getByLabelText("Field reconstruction loss")).toHaveValue("mse");
  expect(
    screen.getByRole("option", { name: "Standard MSE — default" }),
  ).toBeInTheDocument();
  expect(screen.queryByLabelText("Feature balance: α")).toBeNull();
  expect(screen.queryByText("Advanced region analysis")).toBeNull();
  expect(screen.queryByText("Preview regions on training examples")).toBeNull();
  expect(
    screen.getByText(/0 additional region-training queries/),
  ).toBeVisible();
  expect(
    screen.getByText(
      "Training loss controls learning. Display metrics only change evaluation.",
    ),
  ).toBeVisible();
});
