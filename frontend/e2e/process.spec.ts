import { experts, physical } from "./disclosure";
import { test, expect } from "@playwright/test";

test("real feedback budgets, criteria status, live decisions and durable reopening", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("4");
  await page
    .getByText("Feedback controller and budgets", { exact: true })
    .click();
  await experts(page);
  await expect(page.getByLabel("Enable feedback controller")).toBeChecked();
  const policy = {
    check_interval: 1,
    refine: true,
    request_resampling: true,
    max_attempts: 1,
    thresholds: {},
  };
  await page
    .getByLabel("Process configuration JSON")
    .fill(JSON.stringify(policy));
  await page
    .getByRole("button", { name: "Apply process configuration" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await physical(page);
  await expect(page.getByLabel("Process completion status")).toHaveText(
    "FINISH_BUDGET_EXHAUSTED",
    { timeout: 30000 },
  );
  await physical(page);
  await expect(
    page.getByRole("table", { name: "Execution decisions" }),
  ).toContainText("RESAMPLE_UNAVAILABLE");
  await physical(page);
  await expect(
    page.getByRole("table", { name: "Execution decisions" }),
  ).toContainText("REFINE_UNAVAILABLE");
  await expect(
    page.getByRole("table", { name: "Component verification" }),
  ).toContainText("unconfigured");
  const before = await page
    .getByRole("table", { name: "Execution decisions" })
    .textContent();
  // The displayed history is loaded again from durable Python results.
  await page.reload();
  await page.getByRole("button", { name: /^Runs$/ }).click();
  await experts(page);
  await page.getByRole("button", { name: /Open/ }).first().click();
  await experts(page);
  await physical(page);
  await expect(
    page.getByRole("table", { name: "Execution decisions" }),
  ).toHaveText(before!);
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/process-budget.png",
    fullPage: true,
  });
  // Explicit permissive test thresholds demonstrate state-machine criteria status,
  // not scientific accuracy; every mandatory heat component must be configured.
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("4");
  await page
    .getByText("Feedback controller and budgets", { exact: true })
    .click();
  await experts(page);
  await page.getByLabel("Process configuration JSON").fill(
    JSON.stringify({
      check_interval: 1,
      thresholds: {
        "pde.u.rmse": 1e8,
        "observation.u.standardized_rmse": 1e8,
        "conservation.u.max_drift": 1e8,
      },
    }),
  );
  await page
    .getByRole("button", { name: "Apply process configuration" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await physical(page);
  await expect(page.getByLabel("Process completion status")).toHaveText(
    "FINISH_CRITERIA_MET",
    { timeout: 30000 },
  );
  await expect(
    page.getByText(/reconstruction accuracy is not certified/),
  ).toBeVisible();
});

test("real fine heat checkpoint rechecks layout and returns a physical 16x16 grid", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  const models = await (await page.request.get("/api/models")).json();
  const fine = models.find(
    (m: { specialist: string; problem: { grid: { shape: number[] } } }) =>
      m.specialist === "heat" &&
      m.problem.grid.shape.length === 2 &&
      m.problem.grid.shape[0] === 16,
  );
  expect(fine).toBeTruthy();
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("3");
  await page
    .getByText("Feedback controller and budgets", { exact: true })
    .click();
  await experts(page);
  await page.getByLabel("Process configuration JSON").fill(
    JSON.stringify({
      check_interval: 1,
      refine: true,
      fine_model_id: fine.id,
      max_attempts: 2,
      max_sampling_steps: 6,
      thresholds: {},
    }),
  );
  await page
    .getByRole("button", { name: "Apply process configuration" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await physical(page);
  await expect(page.getByLabel("Process completion status")).toHaveText(
    "FINISH_BUDGET_EXHAUSTED",
    { timeout: 30000 },
  );
  await expect(page.getByText(/Final physical grid: 16 × 16/)).toBeVisible();
  await physical(page);
  await expect(
    page.getByRole("table", { name: "Execution decisions" }),
  ).toContainText("REFINE");
  const records = await (await page.request.get("/api/jobs")).json();
  const completed = records.find(
    (r: { status: string }) => r.status === "succeeded",
  );
  const saved = await (
    await page.request.get("/api/jobs/" + completed.id + "/result")
  ).json();
  expect(saved.request.model_id).toBe(fine.id);
  expect(saved.request.problem.grid.shape).toEqual([16, 16]);
  expect(saved.request.observations.coordinates).toEqual(
    saved.original_request.observations.coordinates,
  );
  expect(saved.request.mathematical_request.domain.shape).toEqual([16, 16]);
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/process-refinement.png",
    fullPage: true,
  });
});
