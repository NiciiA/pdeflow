import { beforeEach, afterEach, it, expect, vi } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { BatchControls, type BatchStage } from "./BatchControls";
const call = vi.hoisted(() => vi.fn());
vi.mock("./types", () => ({ api: call }));
const stages = ["autoencoder", "evolution", "flow"];
const batches = Object.fromEntries(
  stages.map((s) => [
    s,
    {
      label: s,
      microbatch: 2,
      requested_microbatch: 2,
      accumulation: 1,
      effective_batch: 2,
      last_microbatch: 2,
      planned_successful_updates: 2,
      approximate_examples: 4,
    },
  ]),
) as Record<string, BatchStage>;
afterEach(cleanup);
beforeEach(() => call.mockReset());
it("applies explicit batch controls through the authoritative API and reports old/new", async () => {
  const apply = vi.fn(),
    dirty = vi.fn();
  call.mockResolvedValue({
    request: {
      advanced: { batch_size: 7, training_options: { microbatch: 3 } },
    },
    application: {
      before: batches,
      after: Object.fromEntries(
        stages.map((s) => [
          s,
          { ...batches[s], microbatch: 3, accumulation: 3, effective_batch: 7 },
        ]),
      ),
    },
    state: "Applied; training has not started",
  });
  render(
    <BatchControls
      request={{}}
      batches={batches}
      onApply={apply}
      onDirty={dirty}
    />,
  );
  fireEvent.change(screen.getByLabelText("Effective batch size"), {
    target: { value: "7" },
  });
  fireEvent.change(screen.getByLabelText("GPU microbatch", { exact: true }), {
    target: { value: "manual" },
  });
  fireEvent.change(screen.getByLabelText("Manual GPU microbatch"), {
    target: { value: "3" },
  });
  fireEvent.click(
    screen.getByRole("button", { name: "Apply selected batch settings" }),
  );
  await waitFor(() =>
    expect(apply).toHaveBeenCalledWith({
      batch_size: 7,
      training_options: { microbatch: 3 },
    }),
  );
  expect(
    call.mock.calls.find((c) => c[0] === "/training/batch/apply")![1].settings,
  ).toMatchObject({
    effective_batch: 7,
    microbatch: 3,
  });
  expect(screen.getByRole("status").textContent).toContain(
    "training has not started",
  );
  expect(dirty).toHaveBeenCalledWith(true);
});
it("requires an estimate and reports efficiency/learning modes separately", async () => {
  call.mockResolvedValue({
    rows: [{}, {}],
    estimate: {
      seconds: null,
      runtime_cap_seconds: 120,
      meaning: "Pilot required",
    },
  });
  render(
    <BatchControls
      request={{}}
      batches={batches}
      onApply={vi.fn()}
      onDirty={vi.fn()}
    />,
  );
  const start = screen.getByRole("button", { name: "Run batch benchmark" });
  expect((start as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Estimate experiment" }));
  await waitFor(() =>
    expect((start as HTMLButtonElement).disabled).toBe(false),
  );
  fireEvent.change(screen.getByLabelText("Batch experiment mode"), {
    target: { value: "learning" },
  });
  expect((start as HTMLButtonElement).disabled).toBe(true);
  expect(
    screen.getByText(/Equal updates are unequal data exposure/),
  ).toBeTruthy();
});
