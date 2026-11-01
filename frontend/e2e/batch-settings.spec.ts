import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
const out = path.resolve("../outputs/batch-settings-20261031");
test("real rendered configure benchmark apply tiny-training chain", async ({
  page,
}) => {
  const errors: string[] = [];
  const evidence: unknown[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Application" })
    .getByRole("button", { name: "Train model", exact: true })
    .click();
  await page
    .getByLabel("Training budget", { exact: true })
    .selectOption("pipeline-check");
  const settings = page.getByRole("region", { name: "Batch settings" });
  await expect(settings).toBeVisible();
  await settings.getByLabel("Effective batch size", { exact: true }).fill("4");
  await settings
    .getByLabel("GPU microbatch", { exact: true })
    .selectOption("manual");
  await settings.getByLabel("Manual GPU microbatch", { exact: true }).fill("2");
  await settings
    .getByText("Advanced per-stage GPU microbatches", { exact: true })
    .click();
  await settings
    .getByLabel("A2 evolution GPU microbatch", { exact: true })
    .fill("1");
  const applied = page.waitForResponse((r) =>
    r.url().endsWith("/training/batch/apply"),
  );
  await settings
    .getByRole("button", { name: "Apply selected batch settings", exact: true })
    .click();
  const response = await applied;
  expect(response.ok()).toBeTruthy();
  const value = await response.json();
  evidence.push(value);
  expect(value.resolved.batch_settings.autoencoder).toMatchObject({
    microbatch: 2,
    accumulation: 2,
    effective_batch: 4,
  });
  expect(value.resolved.batch_settings.evolution).toMatchObject({
    microbatch: 1,
    accumulation: 4,
  });
  await expect(settings.getByRole("status")).toContainText(
    "training has not started",
  );
  await expect(page.getByTestId("resolved-training-summary")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await settings
    .getByText("Benchmark batch sizes (optional)", { exact: true })
    .click();
  await settings
    .getByRole("button", { name: "Estimate experiment", exact: true })
    .click();
  await expect(settings.getByText(/runtime cap 120s/)).toBeVisible();
  await settings
    .getByRole("button", { name: "Run batch benchmark", exact: true })
    .click();
  await expect(
    settings.getByText(
      "Benchmark completed; not applied; actual training has not started.",
      { exact: true },
    ),
  ).toBeVisible({ timeout: 100000 });
  const recommended = page.waitForResponse((r) =>
    r.url().endsWith("/training/batch/apply"),
  );
  await settings
    .getByRole("button", { name: "Apply recommendation", exact: true })
    .click();
  const appliedRecommendation = await (await recommended).json();
  evidence.push(appliedRecommendation);
  expect(
    appliedRecommendation.resolved.batch_settings.evolution.microbatch,
  ).toBe(1);
  await expect(settings.getByRole("status")).toContainText(
    "training has not started",
  );
  await expect(page.getByTestId("resolved-training-summary")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await settings.screenshot({ path: path.join(out, "batch-application.png") });
  const request = page.waitForResponse((r) =>
    r.url().endsWith("/training/start"),
  );
  await page
    .getByRole("main")
    .getByRole("button", { name: "Train model", exact: true })
    .click();
  const started = await request;
  expect(started.ok()).toBeTruthy();
  const run = await started.json();
  evidence.push({
    start_payload: started.request().postDataJSON(),
    started: run,
  });
  expect(started.request().postDataJSON().advanced.batch_size).toBe(4);
  let final: Record<string, unknown>;
  await expect
    .poll(
      async () => {
        final = await (
          await page.request.get("/api/training/runs/" + run.id)
        ).json();
        return final.state;
      },
      { timeout: 120000 },
    )
    .toBe("completed");
  evidence.push({ completed: final! });
  fs.writeFileSync(
    path.join(out, "browser-evidence.json"),
    JSON.stringify(evidence, null, 2),
  );
  expect(errors).toEqual([]);
});

test("inspect persisted results and apply manual settings without rerunning benchmark", async ({
  page,
}) => {
  const saved = JSON.parse(
    fs.readFileSync(path.join(out, "browser-evidence.json"), "utf8"),
  );
  const id =
    saved[1]?.request?.experiment_id || saved[1]?.application?.experiment_id;
  // Select a real persisted result via the supported list API, never inject UI data.
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Application" })
    .getByRole("button", { name: "Train model", exact: true })
    .click();
  await page
    .getByLabel("Training budget", { exact: true })
    .selectOption("pipeline-check");
  const settings = page.getByRole("region", { name: "Batch settings" });
  await expect(settings).toBeVisible();
  let starts = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/training/batch/benchmark")) starts++;
  });
  await settings
    .getByText("Benchmark batch sizes (optional)", { exact: true })
    .click();
  const rows = await (
    await page.request.get("/api/training/batch/benchmarks")
  ).json();
  expect(rows.length).toBeGreaterThan(0);
  await settings
    .getByLabel("Saved batch experiment")
    .selectOption(id || rows[0].id);
  await expect(
    settings.getByRole("table", { name: "Batch benchmark results" }),
  ).toBeVisible();
  await settings
    .getByText("Benchmark batch sizes (optional)", { exact: true })
    .click();
  await settings.getByLabel("Effective batch size", { exact: true }).fill("4");
  await settings
    .getByLabel("GPU microbatch", { exact: true })
    .selectOption("manual");
  await settings.getByLabel("Manual GPU microbatch", { exact: true }).fill("2");
  await settings
    .getByText("Advanced per-stage GPU microbatches", { exact: true })
    .click();
  await settings
    .getByLabel("A2 evolution GPU microbatch", { exact: true })
    .fill("1");
  const applying = page.waitForResponse((r) =>
    r.url().endsWith("/training/batch/apply"),
  );
  await settings
    .getByRole("button", { name: "Apply selected batch settings", exact: true })
    .click();
  const response = await applying;
  expect(response.ok()).toBeTruthy();
  const result = await response.json();
  await expect(
    page
      .getByRole("main")
      .getByRole("button", { name: "Train model", exact: true }),
  ).toBeEnabled();
  await expect(
    settings.getByRole("table", { name: "Authoritative batch configuration" }),
  ).toContainText("8");
  await settings.screenshot({
    path: path.join(out, "batch-settings-current.png"),
  });
  fs.writeFileSync(
    path.join(out, "browser-final-application.json"),
    JSON.stringify(result, null, 2),
  );
  expect(starts).toBe(0);
});
