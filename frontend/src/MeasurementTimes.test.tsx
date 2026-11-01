import { render, screen, fireEvent } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { MeasurementTimes, MeasurementSummary } from "./MeasurementTimes";
import { temporalPayload } from "./measurementTimeConfig";
const c = {
  start: 0,
  end: 1,
  manual: 0.237,
  mixture: false,
  early: 0.25,
  initial: 0.1,
  counts: "",
};
test("default whole interval and explicit counts retain total semantics", () => {
  expect(temporalPayload("continuous", "0.1,0.2", c)).toMatchObject({
    start: 0,
    end: 1,
    times: [],
    temporal_policy: { policy: "continuous_uniform" },
  });
  expect(temporalPayload("chosen", "0.02,0.2,0.47,0.83", c).times).toEqual([
    0.02, 0.2, 0.47, 0.83,
  ]);
  expect(
    temporalPayload("chosen", ".2,.8", { ...c, counts: "3,2" }).time_counts,
  ).toEqual([3, 2]);
  const change = vi.fn();
  render(
    <MeasurementTimes
      mode="chosen"
      onMode={vi.fn()}
      value={c}
      onChange={change}
      times=".02,.2,.47,.83"
      onTimes={vi.fn()}
      total={7}
      support={1}
      manual={false}
    />,
  );
  expect(screen.getByText(/Resolved total/)).toHaveTextContent(
    "7 measurements",
  );
  fireEvent.change(screen.getByLabelText("Counts per measurement time"), {
    target: { value: "1,2,3,4" },
  });
  expect(change).toHaveBeenCalledWith({ ...c, counts: "1,2,3,4" });
});
test("manual control is explicit, time summary exposes exact stored timestamps", () => {
  render(
    <>
      <MeasurementTimes
        mode="continuous"
        onMode={vi.fn()}
        value={c}
        onChange={vi.fn()}
        times=""
        onTimes={vi.fn()}
        total={2}
        support={1}
        manual={true}
      />
      <MeasurementSummary
        rows={[
          { field: "v", x: 0.2, y: 0.3, time: 0.237, value: 1, sigma: 0.1 },
        ]}
        policy="chosen"
        end={1}
      />
    </>,
  );
  expect(screen.getByLabelText("Active measurement time")).toHaveValue(0.237);
  expect(screen.getByText("0.237")).toBeInTheDocument();
  expect(
    screen.getByRole("img", { name: /Measurement time histogram/ }),
  ).toBeInTheDocument();
});
